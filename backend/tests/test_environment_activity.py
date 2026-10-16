import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api import reproductions
from app.models import ReproductionStatus
from app.services import reproduction_engine as engine


class Session:
    def __init__(self, job):
        self.job = job
        self.commits = 0

    async def commit(self):
        self.commits += 1

    async def scalar(self, _statement):
        return self.job
    async def refresh(self, *_args, **_kwargs):
        pass


def fixture_job(tmp_path):
    workspace = tmp_path / 'gcn-project1'
    log = workspace / 'runs' / 'run-1' / 'operations' / 'install.log'
    log.parent.mkdir(parents=True)
    job = SimpleNamespace(id='run-1', project_id='project1', workspace_path=str(workspace),
                          status=ReproductionStatus.executing, logs='', result_json={'other': 1})
    return job, workspace, log


@pytest.mark.parametrize('kind', ['tensorflow_gpu_migration', 'pytorch_gpu_migration',
                                 'dependency_install', 'environment_install', 'environment_install_retry'])
def test_live_operation_published_before_process_and_activity_reads_live_tail(tmp_path, monkeypatch, kind):
    job, workspace, log = fixture_job(tmp_path)
    session = Session(job)
    monkeypatch.setattr(reproductions, 'get_settings', lambda: SimpleNamespace(paper_projects_dir=tmp_path))

    async def process(command, cwd, timeout, output_path):
        assert session.commits == 1
        assert cwd == workspace and output_path == log
        operation = job.result_json['active_operation']
        assert operation['kind'] == kind and operation['step'] == 2 and operation['total_steps'] == 4
        assert operation['started_at'] and 'pip install' in operation['command']
        log.write_bytes(b'fetching CUDA dependency 20%')
        activity = await reproductions.current_activity(job.project_id, job.id, session)
        assert activity['active'] and activity['log_tail'] == 'fetching CUDA dependency 20%'
        assert 'log_path' not in activity['operation']
        return 0, 'installed'

    monkeypatch.setattr(engine, '_run_process', process)
    result = asyncio.run(engine._run_environment_operation(
        session, job, workspace, ['python', '-m', 'pip', 'install', 'package'], 90, log,
        kind=kind, step=2, total_steps=4))
    assert result == (0, 'installed') and session.commits == 2
    assert job.result_json == {'other': 1}
    assert 'installed' in job.logs
    assert asyncio.run(reproductions.current_activity(job.project_id, job.id, session))['active'] is False


@pytest.mark.parametrize('exception', [RuntimeError, engine.ReproductionPaused, asyncio.CancelledError])
def test_pause_error_and_cancellation_preserve_log_and_clear_current_operation(tmp_path, monkeypatch, exception):
    job, workspace, log = fixture_job(tmp_path)
    session = Session(job)

    async def process(*args, **kwargs):
        log.write_bytes(b'downloading before interruption')
        raise exception('interrupted')

    monkeypatch.setattr(engine, '_run_process', process)
    with pytest.raises(exception):
        asyncio.run(engine._run_environment_operation(session, job, workspace, ['conda', 'install'],
                    90, log, kind='tensorflow_gpu_migration', step=1, total_steps=4))
    assert job.result_json == {'other': 1}
    assert 'downloading before interruption' in job.logs
    assert session.commits == 2


def test_activity_rejects_other_runs_log_path(tmp_path, monkeypatch):
    job, workspace, log = fixture_job(tmp_path)
    other_log = workspace / 'runs' / 'other-run' / 'operations' / 'secret.log'
    other_log.parent.mkdir(parents=True)
    other_log.write_bytes(b'not this run')
    job.result_json['active_operation'] = {'log_path': str(other_log)}
    monkeypatch.setattr(reproductions, 'get_settings', lambda: SimpleNamespace(paper_projects_dir=tmp_path))
    assert asyncio.run(reproductions.current_activity(job.project_id, job.id, Session(job))) == {
        'active': False, 'log_tail': ''}


def test_live_activity_tail_is_bounded(tmp_path, monkeypatch):
    job, workspace, log = fixture_job(tmp_path)
    log.write_bytes(b'x' * 10000 + b'last output')
    job.result_json['active_operation'] = {'log_path': str(log), 'command': 'conda install'}
    monkeypatch.setattr(reproductions, 'get_settings', lambda: SimpleNamespace(paper_projects_dir=tmp_path))
    result = asyncio.run(reproductions.current_activity(job.project_id, job.id, Session(job)))
    assert len(result['log_tail']) == 5000 and result['log_tail'].endswith('last output')


def test_real_stdout_is_readable_before_command_finishes(tmp_path, monkeypatch):
    """Harmless local subprocess, not an install, model request, or experiment."""
    job, workspace, log = fixture_job(tmp_path)
    settings = SimpleNamespace(paper_projects_dir=tmp_path, hf_endpoint='', reproduction_log_chars=5000)
    monkeypatch.setattr(engine, 'get_settings', lambda: settings)
    monkeypatch.setattr(reproductions, 'get_settings', lambda: settings)
    session = Session(job)

    async def verify():
        command = [sys.executable, '-c', "import time; print('live-output', flush=True); time.sleep(0.5)"]
        task = asyncio.create_task(engine._run_environment_operation(
            session, job, workspace, command, 5, log, kind='diagnostic', step=1, total_steps=1))
        try:
            for _ in range(200):
                if log.is_file() and b'live-output' in log.read_bytes():
                    break
                await asyncio.sleep(0.01)
            assert not task.done(), 'Log should be visible before process completion'
            activity = await reproductions.current_activity(job.project_id, job.id, session)
            assert activity['active'] and 'live-output' in activity['log_tail']
            assert (await task)[0] == 0
        finally:
            await task
        assert 'active_operation' not in job.result_json

    asyncio.run(verify())
