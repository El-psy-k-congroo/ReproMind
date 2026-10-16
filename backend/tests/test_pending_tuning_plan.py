import asyncio
from types import SimpleNamespace
import pytest
from fastapi import HTTPException
from app.api import reproductions


def test_readonly_plan_for_unprepared_job_is_pending_without_filesystem_or_search(monkeypatch):
    class Session:
        async def scalar(self,*args): return SimpleNamespace(workspace_path='',execution_command={})
    monkeypatch.setattr(reproductions,'reviewed_plans',lambda *args:pytest.fail('No workspace has been planned yet'))
    response=asyncio.run(reproductions.tuning_plan('project','queued',Session()))
    assert response['pending'] and response['plans']==[] and not response['executed']


def test_invalid_workspace_remains_blocked_but_is_not_an_internal_server_error(monkeypatch):
    class Session:
        async def scalar(self,*args): return SimpleNamespace(workspace_path='foreign',execution_command={'commands':[{}]})
    def denied(*args):raise ValueError('工作区归属不合法')
    monkeypatch.setattr(reproductions,'reviewed_plans',denied)
    with pytest.raises(HTTPException) as exc:asyncio.run(reproductions.tuning_plan('project','run',Session()))
    assert exc.value.status_code==409
