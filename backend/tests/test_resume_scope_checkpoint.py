import asyncio
from types import SimpleNamespace
import pytest
from fastapi import BackgroundTasks, HTTPException
from app.api import reproductions
from app.models import ReproductionStatus
from app.services import reproduction_engine as engine
from datetime import datetime,timezone

@pytest.mark.parametrize('applied,allowed', [('2026-10-15T15:00:00+08:00',True),('2026-10-15T15:02:00+08:00',False),('',False)])
def test_code_applied_before_initial_approval_does_not_block_pause_resume(monkeypatch,applied,allowed):
    job=SimpleNamespace(id='run',status=ReproductionStatus.paused,source_mode='paper_repository',
        approved_at=datetime(2026,10,15,7,1,tzinfo=timezone.utc),execution_command={'commands':[{'dataset':'Alpaca'}]},
        result_json={'paused_from_status':'executing','code_revisions':[{'applied_at':applied}]})
    class Session:
        async def scalar(self,_):return job
        async def commit(self):pass
        async def refresh(self,_):pass
    monkeypatch.setattr(reproductions,'has_active_work',lambda _:False)
    monkeypatch.setattr(reproductions,'clear_process_pause',lambda _:None)
    tasks=BackgroundTasks()
    if allowed:
        asyncio.run(reproductions.resume_paused('p','run',tasks,Session()))
        assert job.status==ReproductionStatus.executing and tasks.tasks
    else:
        with pytest.raises(HTTPException):asyncio.run(reproductions.resume_paused('p','run',tasks,Session()))
        assert not tasks.tasks

@pytest.mark.parametrize('checkpoint', [{}, {'paused_from_status': 'running', 'paused_from_stage': 'source_acquisition'}])
def test_approved_revision_resumes_execution_not_repository_preparation(monkeypatch, checkpoint):
    job = SimpleNamespace(id='run', status=ReproductionStatus.paused, source_mode='paper_repository',
                          approved_at='approved', execution_command={'commands': [{'dataset': 'Dataset-X'}]},
                          result_json={**checkpoint, 'revision': {'datasets': ['Dataset-X']}})
    class Session:
        async def scalar(self, _): return job
        async def commit(self): pass
        async def refresh(self, _): pass
    monkeypatch.setattr(reproductions, 'has_active_work', lambda _: False)
    monkeypatch.setattr(reproductions, 'clear_process_pause', lambda _: None)
    tasks = BackgroundTasks()
    asyncio.run(reproductions.resume_paused('project', 'run', tasks, Session()))
    assert job.status == ReproductionStatus.executing
    assert job.source_mode == 'copilot_revision'
    assert len(tasks.tasks) == 1 and tasks.tasks[0].func is engine.execute_reproduction

def test_lost_unapproved_checkpoint_does_not_default_to_full_preparation(monkeypatch):
    job = SimpleNamespace(id='run', status=ReproductionStatus.paused, result_json={}, approved_at=None,
                          source_mode='paper_repository', execution_command={})
    class Session:
        async def scalar(self, _): return job
    monkeypatch.setattr(reproductions, 'has_active_work', lambda _: False)
    tasks = BackgroundTasks()
    with pytest.raises(HTTPException, match='检查点缺失'):
        asyncio.run(reproductions.resume_paused('p', 'run', tasks, Session()))
    assert not tasks.tasks

def test_expanded_command_scope_blocks_resume(monkeypatch):
    job = SimpleNamespace(id='run', status=ReproductionStatus.paused, source_mode='copilot_revision',
                          approved_at='approved', execution_command={'commands': [{'dataset': 'Dataset-X'}, {'dataset': 'Other'}]},
                          result_json={'revision': {'datasets': ['Dataset-X']}})
    class Session:
        async def scalar(self, _): return job
    monkeypatch.setattr(reproductions, 'has_active_work', lambda _: False)
    tasks = BackgroundTasks()
    with pytest.raises(HTTPException, match='范围不一致'):
        asyncio.run(reproductions.resume_paused('p', 'run', tasks, Session()))
    assert not tasks.tasks

def test_cleanup_refreshes_pause_checkpoint_before_updating_logs(tmp_path, monkeypatch):
    job = SimpleNamespace(id='run', logs='', result_json={})
    class Session:
        async def commit(self): pass
        async def refresh(self, item, **kwargs):
            item.result_json = {**item.result_json, 'paused_from_status': 'executing', 'paused_from_stage': 'environment_building'}
    async def process(*args, **kwargs): raise engine.ReproductionPaused()
    monkeypatch.setattr(engine, '_run_process', process)
    with pytest.raises(engine.ReproductionPaused):
        asyncio.run(engine._run_environment_operation(Session(), job, tmp_path, ['conda'], 1, tmp_path / 'op.log',
                                                     kind='install', step=1, total_steps=1))
    assert job.result_json['paused_from_status'] == 'executing'
    assert 'active_operation' not in job.result_json
