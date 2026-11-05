"""Bounded, verified downloads of public author repositories from GitHub."""
import asyncio
import hashlib
import os
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import httpx

MAX_ARCHIVE_BYTES = 120_000_000


def git_network_environment(use_system_proxy: bool, repository_url: str) -> dict[str, str]:
    """Give this Git process the same proxy choice as the archive downloader."""
    proxies = urllib.request.getproxies() if use_system_proxy else {}
    proxy = proxies.get('https') or proxies.get('http') or proxies.get('all') or ''
    if use_system_proxy and not proxy:
        return {}
    if proxy and '://' not in proxy:
        proxy = 'http://' + proxy
    # Environment-only Git configuration preserves global configuration and
    # keeps any proxy credentials out of argv, progress messages and logs.
    count = int(os.environ.get('GIT_CONFIG_COUNT') or 0)
    settings = {
        'http.proxy': proxy,
        f'http.{repository_url.rstrip("/")}.proxy': proxy,
        'http.lowSpeedLimit': '1024', 'http.lowSpeedTime': '12',
    }
    env = {}
    for key, value in settings.items():
        env[f'GIT_CONFIG_KEY_{count}'] = key
        env[f'GIT_CONFIG_VALUE_{count}'] = value
        count += 1
    env['GIT_CONFIG_COUNT'] = str(count)
    if not use_system_proxy:
        env.update({key: '' for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')})
        env.update(NO_PROXY='*', no_proxy='*')
    return env


class RepositoryDownloadError(RuntimeError):
    def __init__(self, message: str, diagnostics: list[str]):
        super().__init__(message)
        self.user_message = message
        self.diagnostics = "\n".join(diagnostics)


def _verify_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 12_000 or sum(entry.file_size for entry in entries) > 500_000_000:
            raise ValueError("仓库压缩包文件数量或解压体积超出限制")
        if archive.testzip():
            raise zipfile.BadZipFile("源码压缩包 CRC 校验失败")


async def download_github_archive(
    owner: str, repository: str, destination: Path, *, timeout_seconds: float = 90,
    trust_env: bool = True, progress=None, check_pause=lambda: None,
) -> dict:
    if destination.exists() or destination.is_symlink():
        raise ValueError("源码归档目标已存在，拒绝覆盖")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ValueError("源码归档临时文件已存在，拒绝覆盖")
    # HEAD and the ref-less REST endpoint both follow the repository default branch.
    # No assumption about main/master, and no third-party mirrors or TLS bypass.
    endpoints = [f"https://codeload.github.com/{owner}/{repository}/zip/HEAD",
                 f"https://api.github.com/repos/{owner}/{repository}/zipball"]
    modes = list(dict.fromkeys((trust_env, False)))
    attempts = [(url, mode) for url in endpoints for mode in modes]
    terminal = set()
    diagnostics = []
    deadline = time.monotonic() + timeout_seconds

    async def notify(message):
        check_pause()
        if progress:
            await progress(message)

    async def official_request(request):
        if request.url.scheme != "https" or request.url.host not in {
            "github.com", "api.github.com", "codeload.github.com",
        }:
            raise ValueError("源码下载被重定向到非 GitHub 官方地址")
        check_pause()

    for index, (url, mode) in enumerate(attempts, 1):
        if url in terminal:
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            diagnostics.append("源码下载总时限已耗尽")
            break
        remaining_attempts = sum(endpoint not in terminal for endpoint, _ in attempts[index-1:])
        attempt_budget = remaining / remaining_attempts
        channel = "系统代理配置" if mode else "直连"
        await notify(f"正在获取作者默认分支源码包：{channel}，尝试 {index}/{len(attempts)}")
        received = 0
        try:
            # The read timeout only limits inactivity between chunks. A slow
            # drip must not consume every fallback's share of the total budget.
            async with asyncio.timeout(attempt_budget):
                async with httpx.AsyncClient(
                    follow_redirects=True, max_redirects=5, trust_env=mode,
                    timeout=httpx.Timeout(min(15, attempt_budget), connect=min(8, attempt_budget)),
                    event_hooks={"request": [official_request]},
                    headers={"User-Agent": "ReproMind", "Accept-Encoding": "identity"},
                ) as client:
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()
                        expected = int(response.headers.get("Content-Length", 0))
                        if expected > MAX_ARCHIVE_BYTES:
                            raise ValueError("仓库 ZIP 超过 120 MB 下载限制")
                        digest, received, last_update = hashlib.sha256(), 0, time.monotonic()
                        with partial.open("xb") as output:
                            async for chunk in response.aiter_bytes(64 * 1024):
                                check_pause()
                                received += len(chunk)
                                if received > MAX_ARCHIVE_BYTES:
                                    raise ValueError("仓库 ZIP 超过 120 MB 下载限制")
                                output.write(chunk)
                                digest.update(chunk)
                                if time.monotonic() - last_update >= 3:
                                    await notify(f"正在下载作者源码：已接收 {received / 1024 / 1024:.1f} MB")
                                    last_update = time.monotonic()
                        if expected and not response.headers.get("Content-Encoding") and received != expected:
                            raise httpx.ReadError("源码下载长度与服务器声明不一致")
                        final_url = str(response.url.copy_with(query=None))
                await asyncio.to_thread(_verify_archive, partial)
                check_pause()
                partial.replace(destination)
                commit = urlparse(final_url).path.rsplit("/", 1)[-1]
                await notify(f"作者源码包下载并校验完成：{received / 1024 / 1024:.1f} MB")
                return {"transport": "github_archive", "requested_ref": "HEAD",
                        "archive_url": final_url, "archive_sha256": digest.hexdigest(),
                        "archive_bytes": received, "diagnostics": diagnostics,
                        "resolved_commit": commit if len(commit) in (40, 64) and all(c in '0123456789abcdef' for c in commit.lower()) else ""}
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            diagnostics.append(f"{channel} {url}: HTTP {code}")
            rate_limited = code == 403 and (exc.response.headers.get('X-RateLimit-Remaining') == '0' or exc.response.headers.get('Retry-After'))
            if code not in {408, 429, 500, 502, 503, 504} and not rate_limited:
                terminal.add(url)
        except TimeoutError:
            diagnostics.append(f"{channel} {url}: 通道时限 {attempt_budget:.1f} 秒已用尽，接收 {received} bytes；保留后续通道预算")
        except (httpx.TransportError, httpx.TooManyRedirects, zipfile.BadZipFile) as exc:
            diagnostics.append(f"{channel} {url}: {type(exc).__name__}: {exc}")
        finally:
            # Only this call's exclusive temporary file is removed; old archives stay intact.
            partial.unlink(missing_ok=True)
        if time.monotonic() < deadline and any(endpoint not in terminal for endpoint, _ in attempts[index:]):
            await notify("当前源码下载通道未完成，正在切换官方通道重试")
    raise RepositoryDownloadError(
        "作者源码包未能完整获取，请检查网络、核对仓库地址，或上传作者源码。", diagnostics)
