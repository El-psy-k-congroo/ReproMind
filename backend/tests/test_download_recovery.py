"""Offline failure injection; no model calls, Conda changes or real training."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.api import copilot, reproductions
from app.config import Settings
from app.models import ReproductionStatus
from app.services import download_activity as downloads, reproduction_engine as engine


class Session:
    def __init__(self, job): self.job = job
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def get(self, *args): return self.job
    async def scalar(self, *args): return self.job
    async def commit(self): pass


def job_fixture(tmp_path):
    workspace = tmp_path / 'source-project1'
    workspace.mkdir()
    return SimpleNamespace(id='run-1', project_id='project1', workspace_path=str(workspace),
        status=ReproductionStatus.executing, current_stage='data_downloading', progress=90,
        message='正在下载未就绪的数据', events=[], logs='', result_json={}, dataset_spec={},
        environment_spec={}, code_spec={}, paper_spec={}, execution_command={},
        execution_mode='full', model_id='offline', repository_url='', completed_at=None), workspace


def mock_client(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(downloads.httpx, 'AsyncClient',
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def test_download_timeout_defaults_and_environment_overrides(monkeypatch):
    keys = ['DATASET_DOWNLOAD_TIMEOUT_SECONDS', 'DATASET_DOWNLOAD_READ_TIMEOUT_SECONDS',
            'DATASET_DOWNLOAD_CONNECT_TIMEOUT_SECONDS']
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    defaults = Settings(_env_file=None)
    assert defaults.dataset_download_timeout_seconds == 1800
    assert defaults.dataset_download_read_timeout_seconds == 180
    assert defaults.dataset_download_connect_timeout_seconds == 60
    for key, value in zip(keys, ['3600', '240', '90']):
        monkeypatch.setenv(key, value)
    configured = Settings(_env_file=None)
    assert configured.dataset_download_timeout_seconds == 3600
    assert configured.dataset_download_read_timeout_seconds == 240
    assert configured.dataset_download_connect_timeout_seconds == 90


@pytest.mark.parametrize('explicit_total', [None, 7])
def test_download_uses_configured_network_and_total_timeouts(tmp_path, monkeypatch, explicit_total):
    job, workspace = job_fixture(tmp_path)
    settings = SimpleNamespace(dataset_download_timeout_seconds=3600,
        dataset_download_read_timeout_seconds=240, dataset_download_connect_timeout_seconds=90)
    monkeypatch.setattr(downloads, 'get_settings', lambda: settings)
    original = httpx.AsyncClient
    seen = []
    def client(**kwargs):
        seen.append(kwargs['timeout'])
        return original(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b'valid')), **kwargs)
    monkeypatch.setattr(downloads.httpx, 'AsyncClient', client)
    target = workspace / 'data' / 'fixture'
    asyncio.run(downloads.download_file(Session(job), job, workspace, 'Fixture',
        ['https://example.org/file'], target, engine._raise_if_paused, timeout_seconds=explicit_total))
    assert target.read_bytes() == b'valid'
    assert seen[0].read == 240 and seen[0].connect == 90
    assert job.result_json['last_operation']['timeout_seconds'] == (3600 if explicit_total is None else 7)
    assert job.result_json['last_operation']['read_timeout_seconds'] == 240
    assert job.result_json['last_operation']['connect_timeout_seconds'] == 90


def test_download_live_progress_preserves_result_and_approved_source_retry(tmp_path, monkeypatch):
    job, workspace = job_fixture(tmp_path)
    job.result_json['dataset_results'] = [{'dataset': 'old', 'metrics': {'accuracy': .7}}]
    calls = []
    def handler(request):
        calls.append(str(request.url))
        if len(calls) == 1:
            raise httpx.ReadTimeout('injected idle timeout', request=request)
        return httpx.Response(200, content=b'valid-payload')
    mock_client(monkeypatch, handler)
    destination = workspace / 'data' / 'fixture.tar.gz'
    asyncio.run(downloads.download_file(Session(job), job, workspace, 'AnyDataset', ['https://example.org/file'],
                                      destination, engine._raise_if_paused))
    assert len(calls) == 2 and calls[0] == calls[1]
    assert destination.read_bytes() == b'valid-payload'
    assert not destination.with_name(destination.name + '.part').exists()
    assert 'active_operation' not in job.result_json
    assert job.result_json['last_operation']['downloaded_bytes'] == 13
    assert job.result_json['last_operation']['state'] == 'downloaded'
    assert job.result_json['dataset_results'][0]['dataset'] == 'old'
    assert 'injected idle timeout' in job.logs and '第 2/2' in job.logs


@pytest.mark.parametrize('pause', [False, True])
def test_stalled_download_deadline_and_pause_cleanup_even_without_new_bytes(tmp_path, monkeypatch, pause):
    job, workspace = job_fixture(tmp_path)
    reached_stream = asyncio.Event()
    class Stalled(httpx.AsyncByteStream):
        async def __aiter__(self):
            reached_stream.set()
            yield b'partial'
            await asyncio.sleep(30)
    mock_client(monkeypatch, lambda request: httpx.Response(200, stream=Stalled()))
    destination = workspace / 'data' / 'fixture'
    async def run():
        task = asyncio.create_task(downloads.download_file(Session(job), job, workspace, 'AnyDataset',
            ['https://example.org/file'], destination, engine._raise_if_paused,
            timeout_seconds=2 if pause else .1))
        await reached_stream.wait()
        try:
            if pause:
                engine.request_process_pause(job.id)
            with pytest.raises(engine.ReproductionPaused if pause else TimeoutError):
                await task
        finally:
            engine.clear_process_pause(job.id)
    asyncio.run(run())
    assert not destination.exists() and not destination.with_name(destination.name + '.part').exists()
    assert 'active_operation' not in job.result_json
    assert job.result_json['last_operation']['error']
    assert '下载停止' in job.logs


def test_current_download_is_visible_before_completion_not_historical_training(tmp_path, monkeypatch):
    job, workspace = job_fixture(tmp_path)
    session = Session(job)
    reached = asyncio.Event()
    release = asyncio.Event()
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            reached.set()
            yield b'data'
            await release.wait()
    mock_client(monkeypatch, lambda request: httpx.Response(200, stream=Stream()))
    monkeypatch.setattr(reproductions, 'get_settings', lambda: SimpleNamespace(paper_projects_dir=tmp_path))
    monkeypatch.setattr(copilot, '_safe_workspace', lambda job: workspace)
    telemetry = workspace / 'runs' / job.id / 'telemetry'
    telemetry.mkdir(parents=True)
    (telemetry / 'old-original.log').write_text('Epoch 50 train_loss=0.1 val_acc=0.8')
    async def run():
        task = asyncio.create_task(downloads.download_file(session, job, workspace, 'UnrelatedDataset',
            ['https://example.org/file'], workspace / 'data' / 'file', engine._raise_if_paused))
        try:
            await reached.wait()
            assert not task.done()
            activity = await reproductions.current_activity(job.project_id, job.id, session)
            assert activity['active'] and 'UnrelatedDataset' in activity['log_tail']
            assert activity['operation']['kind'] == 'dataset_download'
            assert 'log_path' not in activity['operation']
            answer = await copilot._status_answer(session, SimpleNamespace(id=job.project_id), job.id, '下载数据集出错了吗？')
            assert '下载数据集' in answer and '不是训练' in answer and 'UnrelatedDataset' in answer
            assert 'epoch 50' not in answer and '没有出错' not in answer
        finally:
            release.set()
            await task
    asyncio.run(run())


@pytest.mark.parametrize('payload,status', [(b'', 200), (b'too-large', 200), (b'not found', 404)])
def test_invalid_download_not_promoted_or_retried_blindly(tmp_path, monkeypatch, payload, status):
    job, workspace = job_fixture(tmp_path)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, content=payload)
    mock_client(monkeypatch, handler)
    target = workspace / 'data' / 'file'
    with pytest.raises(RuntimeError):
        asyncio.run(downloads.download_file(Session(job), job, workspace, 'dataset', ['https://example.org/file'],
            target, engine._raise_if_paused, max_bytes=4))
    assert len(calls) == 1 and not target.exists()


@pytest.mark.parametrize('failed', ['first', 'middle', 'last', 'all', 'alias'])
def test_data_failure_continues_ready_datasets_without_training_or_repair_of_missing_data(tmp_path, monkeypatch, failed):
    job, workspace = job_fixture(tmp_path)
    source = workspace / 'source'
    source.mkdir()
    datasets = ['CIFAR-10'] if failed=='alias' else ['first', 'middle', 'last']
    job.execution_command = {'cwd': str(source), 'commands': [
        {'dataset': name, 'argv': [sys.executable, 'train.py', '--dataset', name], 'cwd': str(source)} for name in datasets]}
    failed_names = datasets if failed in {'all','alias'} else [failed]
    job.dataset_spec['download_errors'] = [f'{name.replace("-", "") if failed=="alias" else name}: injected download timeout' for name in failed_names]
    job.result_json['old_evidence'] = 'preserved-before-training'
    calls = []
    async def empty(*args): return []
    async def ensure(*args): return Path(sys.executable)
    async def advance(session, row, stage, progress, message, **kwargs):
        row.current_stage, row.progress, row.message = stage, progress, message
        row.events.append({'stage': stage, 'message': message})
    async def process(argv, cwd, timeout, env, **kwargs):
        calls.append(argv[-1])
        return 0, 'Test set results: accuracy=0.75'
    async def forbidden(*args): pytest.fail('Missing data must not invoke source repair')
    monkeypatch.setattr(engine, 'AsyncSessionLocal', lambda: Session(job))
    monkeypatch.setattr(engine, '_ensure_environment', ensure)
    monkeypatch.setattr(engine, '_install_uploaded_datasets', empty)
    monkeypatch.setattr(engine, '_download_small_datasets', empty)
    monkeypatch.setattr(engine, '_align_dataset_commands', lambda job: [])
    monkeypatch.setattr(engine, '_preflight_source', lambda *args, **kwargs: {'issues': [], 'created_output_dirs': []})
    monkeypatch.setattr(engine, '_run_process', process)
    monkeypatch.setattr(engine, '_advance', advance)
    monkeypatch.setattr(engine, '_agent_repair_code', forbidden)
    monkeypatch.setattr(engine, '_write_reproduction_report', lambda *args: None)
    asyncio.run(engine.execute_reproduction(job.id))
    assert calls == [name for name in datasets if name not in failed_names]
    assert job.status == ReproductionStatus.awaiting_data
    assert job.dataset_spec['missing_datasets'] == failed_names
    rows = job.result_json['dataset_results']
    assert len(rows) == len(datasets) and not job.result_json['reproduction_success']
    assert job.result_json['old_evidence'] == 'preserved-before-training'
    assert all(row['execution_state'] == 'not_started' and row['metrics'] == {}
               for row in rows if row['dataset'] in failed_names)


@pytest.mark.parametrize('stage', ['environment_building', 'data_downloading'])
def test_preparation_without_live_log_cannot_claim_normal_or_training(tmp_path, monkeypatch, stage):
    job, workspace = job_fixture(tmp_path)
    job.current_stage = stage
    job.execution_command = {'commands': [{'dataset': 'fixture'}]}
    job.result_json = {'dataset_results': [{'dataset': 'old', 'metrics': {'accuracy': .8}}]}
    monkeypatch.setattr(copilot, '_safe_workspace', lambda job: workspace)
    answer = asyncio.run(copilot._status_answer(Session(job), SimpleNamespace(id='project1'), job.id, '现在到哪一步了'))
    assert '不是训练' in answer and '不能据此认定正常或没有出错' in answer
    assert '正在运行的数据集' not in answer


@pytest.mark.parametrize('question', ['下载数据集出错了吗？？', '环境构建卡住了吗？', '现在在干什么', '进行道路哪一步了'])
def test_preparation_queries_use_lightweight_current_evidence(question):
    assert copilot._status_question(question)


def test_downloader_records_one_failed_source_and_continues_next_request(tmp_path, monkeypatch):
    job, workspace = job_fixture(tmp_path)
    job.dataset_spec = {'automatic_download': True, 'download_requests': [
        {'dataset': 'FirstDataset', 'url': 'https://example.org/fail.tar.gz'},
        {'dataset': 'SecondDataset', 'url': 'https://example.org/good.tar.gz'}]}
    job.execution_command = {'cwd': str(workspace), 'pythonpath': str(workspace)}
    async def advance(session, row, *args, **kwargs): pass
    calls = []
    def handler(request):
        calls.append(str(request.url))
        if 'fail' in str(request.url):
            raise httpx.ReadTimeout('fixture timeout', request=request)
        return httpx.Response(200, content=b'fixture-archive')
    mock_client(monkeypatch, handler)
    monkeypatch.setattr(engine, '_advance', advance)
    monkeypatch.setattr(engine, '_extract_dataset_archive', lambda *args: None)
    monkeypatch.setattr(engine, '_materialize_extracted_dataset', lambda *args: (['data/fixture'], True))
    monkeypatch.setattr(engine, 'inspect_local_datasets', lambda root, names: {names[0]: {'ready': True}})
    asyncio.run(engine._download_small_datasets(Session(job), job, Path(sys.executable), workspace))
    assert len(calls) == 3 and calls[-1].endswith('good.tar.gz')
    assert job.dataset_spec['missing_datasets'] == ['FirstDataset']
    assert not job.dataset_spec['all_datasets_resolved']
    assert [r['download_status'] for r in job.dataset_spec['download_requests']] == ['failed', 'ready']


def test_real_local_http_transfer_produces_durable_evidence_without_training(tmp_path, monkeypatch):
    original_client = httpx.AsyncClient
    # Local test traffic must not be routed through the host's external HTTP proxy.
    monkeypatch.setattr(downloads.httpx, 'AsyncClient', lambda **kwargs: original_client(trust_env=False, **kwargs))
    job, workspace = job_fixture(tmp_path)
    payload = b'harmless-test-payload'
    async def run():
        async def handle(reader, writer):
            await reader.readuntil(b'\r\n\r\n')
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Length: ' + str(len(payload)).encode() + b'\r\nConnection: close\r\n\r\n' + payload)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        server = await asyncio.start_server(handle, '127.0.0.1', 0)
        try:
            port = server.sockets[0].getsockname()[1]
            target = workspace / 'data' / 'test.bin'
            await downloads.download_file(Session(job), job, workspace, 'NetworkFixture',
                [f'http://127.0.0.1:{port}/fixture'], target, engine._raise_if_paused)
            assert target.read_bytes() == payload
            assert job.result_json['last_operation']['downloaded_bytes'] == len(payload)
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(run())


def test_checksum_bound_partial_resumes_without_server_etag(tmp_path,monkeypatch):
    import hashlib,json
    job,workspace=job_fixture(tmp_path)
    payload=b'verified-immutable-author-content'
    target=workspace/'public.bin';offset=10
    target.with_name(target.name+'.part').write_bytes(payload[:offset])
    url='https://example.org/public.bin'
    target.with_name(target.name+'.part.json').write_text(json.dumps({'url':url,'validator':''}))
    def handler(request):
        assert request.headers['Range']==f'bytes={offset}-'
        assert 'If-Range' not in request.headers
        return httpx.Response(206,headers={'Content-Range':f'bytes {offset}-{len(payload)-1}/{len(payload)}'},content=payload[offset:])
    mock_client(monkeypatch,handler)
    asyncio.run(downloads.download_file(Session(job),job,workspace,'Fixture',[url],target,
        engine._raise_if_paused,resume=True,checksum='sha256:'+hashlib.sha256(payload).hexdigest()))
    assert target.read_bytes()==payload and job.result_json['last_operation']['resumed_bytes']==offset
