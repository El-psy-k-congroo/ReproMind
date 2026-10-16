"""Bounded HTTP downloads with durable progress and cooperative cancellation."""
import asyncio
import hashlib
import json
import re
import shutil
import time
from html.parser import HTMLParser
from urllib.parse import urlparse, parse_qs, urlencode
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ..config import get_settings


class _DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__(); self.action = ''; self.fields = {}; self.inside = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form' and attrs.get('id') == 'download-form':
            self.action = attrs.get('action', ''); self.inside = True
        if self.inside and tag == 'input' and attrs.get('name'):
            self.fields[attrs['name']] = attrs.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form': self.inside = False


def drive_url(url):
    parsed = urlparse(url)
    if parsed.hostname not in {'drive.google.com', 'docs.google.com'}:
        return url
    match = re.search(r'/file/d/([A-Za-z0-9_-]+)', parsed.path)
    file_id = match.group(1) if match else (parse_qs(parsed.query).get('id') or [''])[0]
    return 'https://drive.google.com/uc?' + urlencode({'export': 'download', 'id': file_id}) if file_id else url


def verify_file(path, checksum):
    if not checksum: return
    algorithm, expected = checksum.split(':', 1)
    if algorithm not in {'md5', 'sha256', 'sha512'}: raise ValueError('不支持的校验算法')
    digest = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
    if digest.hexdigest().lower() != expected.lower(): raise ValueError('文件校验和不匹配')


class _SlowMirror(Exception):
    pass


def slow_mirror(elapsed, received, remaining, has_alternative):
    """Allow a slow sole source; switch only after a meaningful observation window."""
    rate = received / max(elapsed, .001)
    return bool(has_alternative and elapsed >= 30 and rate < 128 * 1024
                and remaining is not None and remaining / max(rate, 1) > 90)


async def download_file(session, job, workspace: Path, dataset: str, urls: list[str],
                        destination: Path, check_pause, *, max_bytes=None,
                        timeout_seconds=None, allowed_hosts=None, resume=False, checksum=''):
    settings = get_settings()
    if timeout_seconds is None:
        timeout_seconds = settings.dataset_download_timeout_seconds
    read_timeout = settings.dataset_download_read_timeout_seconds
    connect_timeout = settings.dataset_download_connect_timeout_seconds
    # Never publish a half-written file as a usable dataset or overwrite old data.
    if destination.exists() or destination.is_symlink():
        raise ValueError("下载目标已存在，拒绝覆盖")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if max_bytes is None: max_bytes = getattr(settings,'dataset_max_download_bytes',12_000_000_000)
    max_bytes = min(max_bytes,max(0,shutil.disk_usage(destination.parent).free-2_000_000_000))
    if max_bytes <= 0: raise ValueError('磁盘剩余空间不足，需保留至少 2 GB 工作空间')
    partial = destination.with_name(destination.name + '.part')
    metadata = partial.with_name(partial.name + '.json')
    if partial.is_symlink() or metadata.is_symlink():
        raise ValueError('临时下载目标不能是符号链接')
    if partial.exists() and not resume:
        raise ValueError("临时下载目标已存在，拒绝覆盖")
    log = workspace / 'runs' / job.id / 'operations' / f'download-{time.time_ns()}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch(exist_ok=False)
    operation = {'kind': 'dataset_download', 'dataset': dataset, 'step': 1, 'total_steps': 1,
                 'started_at': datetime.now(timezone.utc).isoformat(), 'log_path': str(log),
                 'command': '', 'downloaded_bytes': 0, 'total_bytes': None, 'state': 'connecting',
                 'timeout_seconds': timeout_seconds, 'read_timeout_seconds': read_timeout,
                 'connect_timeout_seconds': connect_timeout}

    async def publish(message):
        operation['updated_at'] = datetime.now(timezone.utc).isoformat()
        with log.open('a', encoding='utf-8') as writer:
            writer.write(f"{operation['updated_at']} {message}\n")
        job.result_json = {**(getattr(job, 'result_json', None) or {}), 'active_operation': dict(operation)}
        if session:
            await session.commit()

    async def transfer():
        last_error = None
        sources = list(dict.fromkeys(urls))
        def saved_partial(url):
            return partial.parent / '.repro-download-partials' / (hashlib.sha256(url.encode()).hexdigest()[:16] + '-' + partial.name)
        max_attempts = 6 if resume and checksum else 2
        for source_index, url in enumerate(sources):
            for attempt in range(1, max_attempts+1):
                check_pause(job.id)
                operation.update(command=f'GET {url}', attempt=attempt, state='connecting',
                                 downloaded_bytes=0, total_bytes=None, error='')
                await publish(f'{dataset}: 连接下载源，第 {attempt}/{max_attempts} 次尝试：{url}')
                try:
                    async with httpx.AsyncClient(follow_redirects=True,
                            timeout=httpx.Timeout(read_timeout, connect=connect_timeout)) as client:
                        saved = json.loads(metadata.read_text()) if metadata.is_file() else {}
                        if resume and partial.is_file() and saved.get('url') and saved.get('url') != url and saved.get('validator'):
                            archived = saved_partial(saved['url']); archived.parent.mkdir(exist_ok=True)
                            partial.replace(archived)
                            metadata.replace(archived.with_name(archived.name+'.json'))
                            saved = {}
                        archived = saved_partial(url)
                        archived_metadata = archived.with_name(archived.name+'.json')
                        if resume and not partial.exists() and archived.is_file() and archived_metadata.is_file():
                            archived.replace(partial); archived_metadata.replace(metadata)
                            saved = json.loads(metadata.read_text())
                        # Immutable author checksums also identify a resumable
                        # resource when the server supplies no ETag/Last-Modified.
                        # Final verification remains mandatory; partials are never
                        # exposed as datasets or weights before that verification.
                        identity = saved.get('validator') or (checksum and saved.get('checksum','') in {'',checksum})
                        offset = partial.stat().st_size if resume and partial.is_file() and saved.get('url') == url and identity else 0
                        if not offset and partial.exists(): partial.unlink()
                        headers = {'Accept-Encoding': 'identity'}
                        if offset:
                            headers['Range'] = f'bytes={offset}-'
                            if saved.get('validator'): headers['If-Range'] = saved['validator']
                        endpoint = drive_url(url)
                        # Public Drive confirmation is a normal download form, never a login/auth bypass.
                        if endpoint != url or urlparse(url).hostname == 'drive.google.com':
                            async with client.stream('GET', endpoint) as landing:
                                landing.raise_for_status()
                                if 'text/html' in landing.headers.get('content-type', ''):
                                    body = bytearray()
                                    async for chunk in landing.aiter_bytes():
                                        body.extend(chunk)
                                        if len(body)>200_000: raise ValueError('公开下载确认页面超过解析上限')
                                    form = _DownloadForm(); form.feed(body.decode('utf-8',errors='replace'))
                                    parsed = urlparse(form.action)
                                    if parsed.scheme != 'https' or parsed.hostname not in {'drive.usercontent.google.com', 'drive.google.com'} or not form.fields.get('id'):
                                        raise ValueError('Google Drive 未提供公开下载确认表单，可能需要访问许可')
                                    endpoint = form.action + '?' + urlencode(form.fields)
                        async with client.stream('GET', endpoint, headers=headers) as response:
                            response.raise_for_status()
                            if allowed_hosts and (response.url.scheme != 'https' or response.url.host not in allowed_hosts):
                                raise ValueError('下载重定向到了未批准的来源')
                            size = response.headers.get('content-length', '')
                            total = int(size) if size.isdigit() else None
                            if offset and response.status_code == 206:
                                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('content-range', ''))
                                if not match or int(match[1]) != offset or int(match[2]) < offset:
                                    raise ValueError('续传 Content-Range 与本地偏移不一致')
                                validator = response.headers.get('etag') or response.headers.get('last-modified')
                                if validator and saved.get('validator') and validator != saved['validator']:
                                    raise ValueError('续传资源版本发生变化')
                                total = int(match[3])
                            else:
                                offset = 0
                            if response.status_code == 206 and not offset:
                                raise ValueError('服务器返回了未请求的分段内容')
                            if 'text/html' in response.headers.get('content-type', '').lower():
                                raise ValueError('下载返回 HTML 页面，不能作为数据或模型文件')
                            if total and total > max_bytes:
                                raise ValueError('数据文件超过自动下载体积限制')
                            operation.update(state='downloading', total_bytes=total, downloaded_bytes=offset, resumed_bytes=offset)
                            metadata.write_text(json.dumps({'url': url, 'checksum':checksum,
                                'validator': response.headers.get('etag') or response.headers.get('last-modified') or ''}), encoding='utf-8')
                            await publish(f'{dataset}: 已连接，预计字节数 {total if total else "未知"}')
                            last_publish = time.monotonic()
                            transfer_start = last_publish
                            with partial.open('ab' if offset else 'wb') as writer:
                                async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                                    check_pause(job.id)
                                    if operation['downloaded_bytes'] == 0 and re.match(br'\s*(?:<!doctype\s+html|<html|<head)',chunk[:512],re.I):
                                        raise ValueError('下载内容是 HTML 页面，不能作为数据或模型文件')
                                    operation['downloaded_bytes'] += len(chunk)
                                    if operation['downloaded_bytes'] > max_bytes:
                                        raise ValueError('数据文件超过自动下载体积限制')
                                    writer.write(chunk)
                                    writer.flush()
                                    if time.monotonic() - last_publish >= 1:
                                        elapsed = time.monotonic() - transfer_start
                                        rate = (operation['downloaded_bytes'] - offset) / max(elapsed, .001)
                                        remaining = max(0,total-operation['downloaded_bytes']) if total is not None else None
                                        operation.update(bytes_per_second=round(rate,1), eta_seconds=round(remaining/max(rate,1)) if remaining is not None else None)
                                        await publish(f"{dataset}: 已下载 {operation['downloaded_bytes']} 字节，{rate/1024:.1f} KiB/s，预计剩余 {operation['eta_seconds']} 秒")
                                        last_publish = time.monotonic()
                                        if slow_mirror(elapsed,operation['downloaded_bytes']-offset,remaining,source_index+1<len(sources)):
                                            raise _SlowMirror('下载源持续低速，保留分段并切换下一已知来源')
                            if not operation['downloaded_bytes']:
                                raise ValueError('下载文件为空')
                            if total is not None and not response.headers.get('content-encoding') and operation['downloaded_bytes'] != total:
                                raise ValueError('下载长度与服务器声明不一致')
                    check_pause(job.id)
                    await asyncio.to_thread(verify_file, partial, checksum)
                    partial.replace(destination)
                    metadata.unlink(missing_ok=True)
                    for candidate in sources:
                        old = saved_partial(candidate)
                        old.unlink(missing_ok=True); old.with_name(old.name+'.json').unlink(missing_ok=True)
                    operation['state'] = 'downloaded'
                    await publish(f"{dataset}: 文件下载完成，{operation['downloaded_bytes']} 字节；待校验和适配")
                    return
                except _SlowMirror as exc:
                    last_error = exc
                    operation.update(state='switching_source',error=str(exc))
                    await publish(f'{dataset}: {exc}')
                    break
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    last_error = exc
                    operation.update(state='retrying', error=f'{type(exc).__name__}: {exc}')
                    await publish(f'{dataset}: 本次下载失败：{operation["error"]}')
                    # Unsafe content / permanent HTTP errors are not retried.
                    if isinstance(exc, ValueError) or (isinstance(exc, httpx.HTTPStatusError)
                            and exc.response.status_code not in {408, 429, 500, 502, 503, 504}):
                        partial.unlink(missing_ok=True); metadata.unlink(missing_ok=True)
                        break
                finally:
                    if not resume and partial.is_file():
                        partial.unlink()
                    if not resume: metadata.unlink(missing_ok=True)
        raise RuntimeError(f'所有已批准下载源获取失败：{type(last_error).__name__}: {last_error}')

    async def bounded():
        task = asyncio.create_task(transfer())
        try:
            while not task.done():
                check_pause(job.id)
                await asyncio.wait({task}, timeout=.25)
            return await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    try:
        await asyncio.wait_for(bounded(), timeout=timeout_seconds)
    except BaseException as exc:
        operation.update(state='interrupted' if not isinstance(exc, (RuntimeError, TimeoutError)) else 'failed',
                         error=f'{type(exc).__name__}: {str(exc) or f"下载超过 {timeout_seconds} 秒总时限"}')
        await publish(f'{dataset}: 下载停止：{operation["error"]}')
        raise
    finally:
        result = dict(job.result_json or {})
        result['last_operation'] = dict(operation)
        result.pop('active_operation', None)
        job.result_json = result
        job.logs = f"{getattr(job, 'logs', '') or ''}\n{log.read_text(encoding='utf-8')[-5000:]}"[-60000:]
        if session:
            await session.commit()


def operation_log_tail(workspace: Path, job_id: str, operation: dict, limit=2400) -> str:
    """Read only this run's operations; never follow another run's log path."""
    path = Path(str(operation.get('log_path') or '')).resolve()
    parent = (workspace / 'runs' / job_id / 'operations').resolve()
    if path.parent != parent or not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode('utf-8', errors='replace')
