import asyncio
import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from app.services import reproduction_engine as engine
from app.services import repository_download as downloads


def archive_bytes(root='model-research'):
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as archive:
        archive.writestr(f'{root}/train.py', 'print("author source")\n')
        archive.writestr(f'{root}/README.md', 'Author repository\n')
    return data.getvalue()


def mock_http(monkeypatch, handler):
    original = httpx.AsyncClient
    modes = []
    def client(**kwargs):
        modes.append(kwargs.get('trust_env'))
        kwargs['transport'] = httpx.MockTransport(handler)
        return original(**kwargs)
    monkeypatch.setattr(downloads.httpx, 'AsyncClient', client)
    return modes


def test_default_branch_archive_precedes_git_and_verified_cache_survives_retry(tmp_path, monkeypatch):
    payload = archive_bytes()
    calls = []
    def respond(request):
        calls.append(str(request.url))
        assert request.url.path.endswith('/zip/HEAD')
        return httpx.Response(200, content=payload)
    mock_http(monkeypatch, respond)
    async def forbidden(*args, **kwargs):
        raise AssertionError('complete source must not start Git, installation or training')
    monkeypatch.setattr(engine, '_run_process', forbidden)
    source, _ = asyncio.run(engine._clone_repository('https://github.com/author/model', tmp_path, tmp_path))
    assert (source / 'train.py').read_text() == 'print("author source")\n'
    marker = json.loads((source / '.repromind-repository.json').read_text())
    assert marker['acquisition']['archive_sha256'] == hashlib.sha256(payload).hexdigest()
    assert marker['acquisition']['requested_ref'] == 'HEAD'
    assert not marker['acquisition']['resolved_commit']
    reused, _ = asyncio.run(engine._clone_repository('https://github.com/author/model', tmp_path, tmp_path))
    assert reused == source and len(calls) == 1


def test_read_timeout_retries_same_author_source_by_direct_connection(tmp_path, monkeypatch):
    calls = []
    payload = archive_bytes()
    def respond(request):
        calls.append(str(request.url))
        if len(calls) == 1:
            raise httpx.ReadTimeout('interrupted transfer', request=request)
        return httpx.Response(200, content=payload)
    modes = mock_http(monkeypatch, respond)
    target = tmp_path / 'source.zip'
    result = asyncio.run(downloads.download_github_archive('author', 'model', target))
    assert modes == [True, False] and calls[0] == calls[1]
    assert target.read_bytes() == payload
    assert 'ReadTimeout' in result['diagnostics'][0]
    assert not list(tmp_path.glob('*.part'))


def test_slow_first_body_cannot_consume_the_direct_fallback_budget(tmp_path, monkeypatch):
    calls = []
    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'partial author archive'
            await asyncio.Event().wait()
    def respond(request):
        calls.append(str(request.url))
        return httpx.Response(200, stream=SlowBody()) if len(calls) == 1 else httpx.Response(200, content=archive_bytes())
    modes = mock_http(monkeypatch, respond)
    target = tmp_path / 'source.zip'
    result = asyncio.run(downloads.download_github_archive('author', 'model', target, timeout_seconds=.6))
    assert target.is_file() and zipfile.is_zipfile(target)
    assert modes == [True, False] and calls[0] == calls[1]
    assert '保留后续通道预算' in result['diagnostics'][0]


def test_direct_only_setting_does_not_repeat_identical_routes(tmp_path, monkeypatch):
    calls = []
    def respond(request):
        calls.append(str(request.url))
        raise httpx.ConnectError('offline', request=request)
    modes = mock_http(monkeypatch, respond)
    with pytest.raises(downloads.RepositoryDownloadError):
        asyncio.run(downloads.download_github_archive('author', 'model', tmp_path / 'source.zip', trust_env=False))
    assert modes == [False, False] and len(set(calls)) == 2


def test_git_proxy_configuration_is_scoped_and_retains_inherited_pairs(monkeypatch):
    monkeypatch.setenv('GIT_CONFIG_COUNT', '1')
    monkeypatch.setenv('GIT_CONFIG_KEY_0', 'http.userAgent')
    monkeypatch.setenv('GIT_CONFIG_VALUE_0', 'existing-agent')
    monkeypatch.setattr(downloads.urllib.request, 'getproxies', lambda: {'https': 'http://localhost:7890'})
    env = downloads.git_network_environment(True, 'https://github.com/author/model')
    assert env['GIT_CONFIG_COUNT'] == '5' and env['GIT_CONFIG_KEY_1'] == 'http.proxy'
    assert env['GIT_CONFIG_VALUE_1'] == 'http://localhost:7890'
    assert 'GIT_CONFIG_KEY_0' not in env
    assert downloads.os.environ['GIT_CONFIG_COUNT'] == '1'
    direct = downloads.git_network_environment(False, 'https://github.com/author/model')
    assert direct['GIT_CONFIG_VALUE_1'] == '' and direct['GIT_CONFIG_VALUE_2'] == ''
    assert direct['NO_PROXY'] == '*' and direct['HTTPS_PROXY'] == ''


def test_git_fallback_switches_network_route_without_overwriting_source(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv('GIT_CONFIG_COUNT', '0')
    monkeypatch.setattr(downloads.urllib.request, 'getproxies', lambda: {'https': 'http://localhost:7890'})
    async def no_archive(*args): raise RuntimeError('archive unavailable')
    async def git(argv, *args, **kwargs):
        calls.append((argv, kwargs['extra_env']))
        if len(calls) == 1:
            target = Path(argv[-1]); target.mkdir(parents=True)
            (target / 'partial.txt').write_text('preserved failed transfer')
            return 1, 'connection reset'
        target = Path(argv[-1]); target.mkdir(parents=True)
        (target / 'train.py').write_text('author code')
        return 0, 'complete'
    monkeypatch.setattr(engine, '_download_repository_archive', no_archive)
    monkeypatch.setattr(engine, '_run_process', git)
    source, _ = asyncio.run(engine._clone_repository('https://github.com/author/model', tmp_path, tmp_path))
    assert calls[0][1]['GIT_CONFIG_VALUE_0'] == 'http://localhost:7890'
    assert calls[1][1]['GIT_CONFIG_VALUE_0'] == ''
    assert source != Path(calls[0][0][-1])
    assert (Path(calls[0][0][-1]) / 'partial.txt').read_text() == 'preserved failed transfer'
    assert (source / '.repromind-repository.json').is_file()


def test_official_api_redirect_resolves_default_branch_without_main_master_guesses(tmp_path, monkeypatch):
    commit = 'a' * 40
    calls = []
    def respond(request):
        calls.append(str(request.url))
        if request.url.path.endswith('/zip/HEAD'):
            return httpx.Response(404)
        if request.url.host == 'api.github.com':
            return httpx.Response(302, headers={'Location': f'https://codeload.github.com/author/model/legacy.zip/{commit}'})
        return httpx.Response(200, content=archive_bytes())
    mock_http(monkeypatch, respond)
    result = asyncio.run(downloads.download_github_archive('author', 'model', tmp_path / 'source.zip'))
    assert result['resolved_commit'] == commit
    assert len(calls) == 3 and not any('/master' in url or '/main' in url for url in calls)


def test_corrupt_zip_is_not_published_and_next_attempt_starts_cleanly(tmp_path, monkeypatch):
    payload = archive_bytes()
    corrupt = payload.replace(b'author source', b'broken source', 1)
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, content=corrupt if len(calls) == 1 else payload)
    mock_http(monkeypatch, respond)
    target = tmp_path / 'source.zip'
    result = asyncio.run(downloads.download_github_archive('author', 'model', target))
    assert len(calls) == 2 and target.read_bytes() == payload
    assert 'BadZipFile' in result['diagnostics'][0]
    assert not list(tmp_path.glob('*.part'))


def test_missing_repository_exhausts_distinct_endpoints_without_repeating_404(tmp_path, monkeypatch):
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(404)
    mock_http(monkeypatch, respond)
    with pytest.raises(downloads.RepositoryDownloadError) as failure:
        asyncio.run(downloads.download_github_archive('author', 'missing', tmp_path / 'source.zip'))
    assert len(calls) == 2 and failure.value.diagnostics.count('HTTP 404') == 2
    assert not list(tmp_path.iterdir())


def test_total_download_deadline_stops_a_stalled_body_and_cleans_partial(tmp_path, monkeypatch):
    class StalledBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'partial'
            await asyncio.Event().wait()
    mock_http(monkeypatch, lambda request: httpx.Response(200, stream=StalledBody()))
    with pytest.raises(downloads.RepositoryDownloadError):
        asyncio.run(downloads.download_github_archive('author', 'model', tmp_path / 'source.zip', timeout_seconds=.05))
    assert not list(tmp_path.iterdir())


def test_download_size_limit_is_enforced_while_streaming_without_content_length(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, 'MAX_ARCHIVE_BYTES', 10)
    mock_http(monkeypatch, lambda request: httpx.Response(200, content=b'x' * 20))
    with pytest.raises(ValueError, match='120 MB'):
        asyncio.run(downloads.download_github_archive('author', 'model', tmp_path / 'source.zip'))
    assert not list(tmp_path.iterdir())


def test_incomplete_clone_with_origin_and_one_file_is_not_reused(tmp_path, monkeypatch):
    url = 'https://github.com/author/model'
    partial = tmp_path / 'sources' / engine._repository_cache_name(url)
    (partial / '.git').mkdir(parents=True)
    (partial / '.git/config').write_text('[remote "origin"]\nurl = ' + url)
    (partial / 'train.py').write_text('incomplete source')
    mock_http(monkeypatch, lambda request: httpx.Response(200, content=archive_bytes()))
    source, _ = asyncio.run(engine._clone_repository(url, tmp_path, tmp_path))
    assert source != partial
    assert (partial / 'train.py').read_text() == 'incomplete source'
    assert 'author source' in (source / 'train.py').read_text()


def test_failed_git_transfer_does_not_attempt_archive_of_missing_head(tmp_path, monkeypatch):
    commands = []
    async def no_archive(*args):
        raise RuntimeError('offline')
    async def git(argv, *args, **kwargs):
        commands.append(argv)
        (Path(argv[-1]) / '.git').mkdir(parents=True)
        return 124, 'Cloning timed out'
    monkeypatch.setattr(engine, '_download_repository_archive', no_archive)
    monkeypatch.setattr(engine, '_run_process', git)
    with pytest.raises(downloads.RepositoryDownloadError):
        asyncio.run(engine._clone_repository('https://github.com/author/model', tmp_path, tmp_path))
    assert len(commands) == 2 and all('archive' not in argv for argv in commands)
    assert 'http.version=HTTP/1.1' in commands[1]


def test_pause_during_source_fetch_never_falls_through_to_git(tmp_path, monkeypatch):
    async def paused(*args):
        raise engine.ReproductionPaused()
    async def forbidden(*args):
        raise AssertionError('paused task started another transport')
    monkeypatch.setattr(engine, '_download_repository_archive', paused)
    monkeypatch.setattr(engine, '_run_process', forbidden)
    with pytest.raises(engine.ReproductionPaused):
        asyncio.run(engine._clone_repository('https://github.com/author/model', tmp_path, tmp_path))


def test_missing_git_reports_recoverable_source_error_instead_of_crashing(tmp_path, monkeypatch):
    async def no_archive(*args):
        raise RuntimeError('archive channel unavailable')
    async def no_git(*args, **kwargs):
        raise FileNotFoundError('git executable unavailable')
    monkeypatch.setattr(engine, '_download_repository_archive', no_archive)
    monkeypatch.setattr(engine, '_run_process', no_git)
    with pytest.raises(downloads.RepositoryDownloadError) as failure:
        asyncio.run(engine._clone_repository('https://github.com/author/model', tmp_path, tmp_path))
    assert 'FileNotFoundError' in failure.value.diagnostics


def test_success_after_partial_extraction_is_reused_without_overwriting_failed_source(tmp_path, monkeypatch):
    url = 'https://github.com/author/model'
    failed = tmp_path / 'sources' / (engine._repository_cache_name(url) + '-archive')
    failed.mkdir(parents=True)
    (failed / 'train.py').write_text('unfinished old extraction')
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, content=archive_bytes())
    mock_http(monkeypatch, respond)
    source, _ = asyncio.run(engine._clone_repository(url, tmp_path, tmp_path))
    reused, _ = asyncio.run(engine._clone_repository(url, tmp_path, tmp_path))
    assert source == reused and source.parent != failed and len(calls) == 1
    assert (failed / 'train.py').read_text() == 'unfinished old extraction'
