import asyncio
from datetime import datetime,timezone
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks,HTTPException

from app.api import reproductions
from app.models import ReproductionStatus
from app.services import reproduction_engine as engine
from app.services.gpu_preparation import TF_WINDOWS_PROFILE


def job_fixture(tmp_path,status=ReproductionStatus.failed):
    return SimpleNamespace(id='repair-run',project_id='project',status=status,
        approved_at=datetime.now(timezone.utc),environment_spec={
            'prefix':str(tmp_path/'env'),'gpu_environment_migration':{'approved':True,'profile':TF_WINDOWS_PROFILE}},
        workspace_path=str(tmp_path),result_json={},error_message='previous network failure',
        logs='original logs',events=[],progress=86,current_stage='failed',message='failed')


class Session:
    def __init__(self,job): self.job=job;self.reads=0
    async def __aenter__(self):return self
    async def __aexit__(self,*args):pass
    async def scalar(self,*args):
        self.reads+=1
        return self.job if self.reads==1 else None
    async def execute(self,*args):pass
    async def scalars(self,*args):return SimpleNamespace(all=lambda:[self.job])
    async def commit(self):pass
    async def refresh(self,*args,**kwargs):pass


def test_environment_repair_requires_prior_migration_approval(tmp_path):
    job=job_fixture(tmp_path);job.environment_spec['gpu_environment_migration']['approved']=False
    with pytest.raises(HTTPException,match='单独批准'):
        asyncio.run(reproductions.repair_environment('project',job.id,BackgroundTasks(),Session(job)))
    assert job.status==ReproductionStatus.failed and job.error_message=='previous network failure'


@pytest.mark.parametrize('active_field',['tuning_active','ablation_active'])
def test_auxiliary_experiment_work_blocks_shared_environment_changes(tmp_path,monkeypatch,active_field):
    job=job_fixture(tmp_path)
    job.result_json={active_field:{'dataset':'example'}}
    monkeypatch.setattr(reproductions,'has_active_work',lambda _:False)
    with pytest.raises(HTTPException,match='活动任务'):
        asyncio.run(reproductions.repair_environment('project',job.id,BackgroundTasks(),Session(job)))
    assert job.status==ReproductionStatus.failed


def test_repair_endpoint_preserves_failure_and_only_schedules_environment_worker(tmp_path,monkeypatch):
    job=job_fixture(tmp_path)
    monkeypatch.setattr(reproductions,'has_active_work',lambda _:False)
    tasks=BackgroundTasks()
    asyncio.run(reproductions.repair_environment('project',job.id,tasks,Session(job)))
    assert job.status==ReproductionStatus.queued
    assert job.result_json['environment_repair_history'][0]['error_message']=='previous network failure'
    assert [task.func for task in tasks.tasks]==[engine.repair_environment_only]


def test_successful_environment_repair_pauses_before_training(tmp_path,monkeypatch):
    job=job_fixture(tmp_path,ReproductionStatus.queued)
    job.result_json={'environment_repair':{'state':'queued','training_started':False}}
    async def prepare(session,current,workspace,python,spec):
        spec['gpu_runtime_verified']={'cuda_available':True,'compute_checked':True,'version':'2.10.0'}
    async def forbidden(*args,**kwargs):raise AssertionError('environment-only repair started training')
    async def check(session,current,workspace,command,*args,**kwargs):
        assert command[1:]==['-m','pip','check']
        return 0,'No broken requirements found.'
    monkeypatch.setattr(engine,'AsyncSessionLocal',lambda:Session(job))
    monkeypatch.setattr(engine,'_prepare_tensorflow_gpu',prepare)
    monkeypatch.setattr(engine,'_run_environment_operation',check)
    monkeypatch.setattr(engine,'execute_reproduction',forbidden)
    monkeypatch.setattr(engine,'_run_author_process',forbidden)
    asyncio.run(engine.repair_environment_only(job.id))
    assert job.status==ReproductionStatus.paused
    assert job.result_json['environment_repair']['state']=='completed'
    assert job.result_json['environment_repair']['training_started'] is False
    assert job.environment_spec['gpu_runtime_verified']['compute_checked']
    assert job.result_json['paused_from_status']=='executing'
    assert job.logs=='original logs' and not engine.has_active_work(job.id)


def test_failed_environment_repair_does_not_resume_training(tmp_path,monkeypatch):
    job=job_fixture(tmp_path,ReproductionStatus.queued)
    job.result_json={'environment_repair':{'state':'queued'}}
    async def failure(*args):raise RuntimeError('dependency download failed')
    monkeypatch.setattr(engine,'AsyncSessionLocal',lambda:Session(job))
    monkeypatch.setattr(engine,'_prepare_tensorflow_gpu',failure)
    asyncio.run(engine.repair_environment_only(job.id))
    assert job.status==ReproductionStatus.failed
    assert job.result_json['environment_repair']['state']=='failed'
    assert 'dependency download failed' in job.error_message


@pytest.mark.parametrize('state',['queued','running','paused'])
def test_resuming_paused_environment_repair_keeps_environment_only_intent(tmp_path,monkeypatch,state):
    job=job_fixture(tmp_path,ReproductionStatus.paused)
    job.result_json={'environment_repair':{'state':state},'paused_from_status':'running',
                    'revision':{'datasets':['example']}}
    monkeypatch.setattr(reproductions,'has_active_work',lambda _:False)
    monkeypatch.setattr(reproductions,'_source_changed_after_approval',lambda _:False)
    tasks=BackgroundTasks()
    asyncio.run(reproductions.resume_paused('project',job.id,tasks,Session(job)))
    assert [task.func for task in tasks.tasks]==[engine.repair_environment_only]
    assert job.result_json['environment_repair']['state']=='queued'


def test_gpu_enumeration_does_not_hide_dependency_integrity_failure(tmp_path,monkeypatch):
    job=job_fixture(tmp_path,ReproductionStatus.queued)
    job.result_json={'environment_repair':{'state':'queued'}}
    async def prepare(*args):pass
    async def failed_check(*args,**kwargs):return 1,'dependency conflict'
    monkeypatch.setattr(engine,'AsyncSessionLocal',lambda:Session(job))
    monkeypatch.setattr(engine,'_prepare_tensorflow_gpu',prepare)
    monkeypatch.setattr(engine,'_run_environment_operation',failed_check)
    asyncio.run(engine.repair_environment_only(job.id))
    assert job.status==ReproductionStatus.failed and '依赖完整性检查未通过' in job.error_message
