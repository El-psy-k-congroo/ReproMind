"""Parallel ranges of one versioned URL; never substitute a dataset or source."""
import asyncio
import hashlib
import re
import shutil
import time
from pathlib import Path

import httpx


class RangeUnavailable(Exception):
    pass


async def transfer_ranges(client, url: str, partial: Path, total: int | None,
                          offset: int, validator: str, workers: int, progress, check_pause) -> bool:
    # Large weights retain the streaming path to avoid doubling their disk use.
    if (workers < 2 or not total or not 2*1024**2 <= total <= 256*1024**2
            or total-offset < 1024**2 or not validator or validator.startswith('W/')):
        return False
    key = hashlib.sha256(f'{url}\0{validator}\0{total}'.encode()).hexdigest()
    root = partial.parent / '.repro-download-ranges'
    cache = root / key
    if root.is_symlink() or cache.is_symlink():
        raise ValueError('分段缓存不能是符号链接')
    cache.mkdir(parents=True, exist_ok=True)
    span = (total-offset+workers-1)//workers
    ranges = [(start, min(start+span, total)-1) for start in range(offset, total, span)]
    paths = [cache/f'{start}-{end}.part' for start, end in ranges]
    for path, (start, end) in zip(paths, ranges):
        if path.is_symlink() or path.exists() and (not path.is_file() or path.stat().st_size > end-start+1):
            raise ValueError('分段缓存文件不合法')
    counts = [path.stat().st_size if path.exists() else 0 for path in paths]
    initial = offset + sum(counts)
    started = time.monotonic()

    async def fetch(index):
        first, end = ranges[index]
        for attempt in range(2):
            check_pause()
            current = first + counts[index]
            if current > end:
                return
            try:
                async with client.stream('GET', url, headers={'Range': f'bytes={current}-{end}',
                        'If-Range': validator, 'Accept-Encoding': 'identity'}) as response:
                    response.raise_for_status()
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('content-range', ''))
                    identity = response.headers.get('etag') or response.headers.get('last-modified')
                    if (response.status_code != 206 or not match
                            or tuple(map(int, match.groups())) != (current, end, total)
                            or identity != validator or response.headers.get('content-encoding')):
                        raise RangeUnavailable('服务器没有确认同一版本的准确字节范围')
                    if 'text/html' in response.headers.get('content-type', '').lower():
                        raise ValueError('分段下载返回了HTML页面')
                    with paths[index].open('ab') as writer:
                        async for data in response.aiter_bytes(chunk_size=64*1024):
                            check_pause()
                            if counts[index] + len(data) > end-first+1:
                                raise ValueError('分段长度超出请求范围')
                            if first == 0 and counts[index] == 0 and re.match(br'\s*(?:<!doctype\s+html|<html|<head)', data[:512], re.I):
                                raise ValueError('分段下载内容是HTML页面')
                            writer.write(data)
                            counts[index] += len(data)
                    if counts[index] != end-first+1:
                        raise httpx.ReadError('分段下载不完整')
                return
            except httpx.HTTPError:
                if attempt:
                    raise

    tasks = [asyncio.create_task(fetch(i)) for i in range(len(ranges))]
    last_publish = 0.0
    try:
        pending = set(tasks)
        while pending:
            check_pause()
            done, pending = await asyncio.wait(pending, timeout=.25, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            now = time.monotonic()
            if now-last_publish >= 1 or not pending:
                received = offset + sum(counts)
                rate = (received-initial)/max(now-started, .001)
                await progress(received, rate, len(ranges), initial)
                last_publish = now
    except RangeUnavailable:
        return False
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    check_pause()
    merged = partial.with_name(partial.name+'.merge')
    if merged.is_symlink():
        raise ValueError('合并目标不能是符号链接')
    with merged.open('wb') as writer:
        if offset:
            if not partial.is_file() or partial.stat().st_size != offset:
                raise ValueError('原续传分段在下载期间发生变化')
            with partial.open('rb') as reader:
                shutil.copyfileobj(reader, writer)
        for path in paths:
            with path.open('rb') as reader:
                shutil.copyfileobj(reader, writer)
    if merged.stat().st_size != total:
        raise ValueError('合并后长度与源文件不一致')
    merged.replace(partial)
    return True
