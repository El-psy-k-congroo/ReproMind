"""Offline environment-plan contracts; no environment installation or training."""
import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from app.services import reproduction_engine as engine
from app.services import host_environment


def test_reviewed_dependency_change_rebuilds_plan_preserving_environment(tmp_path, monkeypatch):
    source = tmp_path/'source';source.mkdir()
    requirements = source/'requirements.txt';requirements.write_bytes(b'numpy==1.26.4\r\n')
    (source/'run.py').write_text('import numpy\n')
    old = tmp_path/'envs'/'original';fresh = tmp_path/'envs'/'new'
    python = str(engine._environment_python(old))
    fingerprint = hashlib.sha256(b'numpy==1.26.4\n').hexdigest()
    job = SimpleNamespace(id='run', execution_command={'pythonpath':str(source),'entrypoint_candidates':['run.py']},
        paper_spec={'model_name':'Example'},result_json={},environment_spec={
            'prefix':str(old),'environment_name':'original','recheck_on_run':True,
            'host_compatibility':{'issues':[{'requirement':'withdrawn==1'}]},
            'dependency_revision':{'sha256':{'requirements.txt':fingerprint}}})
    plans=[]
    def plan(src,manifest,*args,**kwargs):
        plans.append(manifest)
        return {'prefix':str(fresh),'environment_name':'new','python':'3.11',
            'create_commands':[['conda','create','--name','new','python=3.11']],
            'install_commands':[[str(engine._environment_python(fresh)),'-m','pip','install','-r',str(requirements)]],
            'resolved_requirements':['numpy==1.26.4']}
    async def existing(job,spec):
        assert spec['prefix']==str(old)
        return {**spec,'create_commands':[],'python':'3.9'}
    async def inspect(source,spec):
        assert spec['python']=='3.9'
        assert spec['resolved_requirements']==['numpy==1.26.4']
        return {'issues':[]}
    monkeypatch.setattr(engine,'_environment_plan',plan)
    monkeypatch.setattr(engine,'_existing_environment_plan',existing)
    monkeypatch.setattr(host_environment,'inspect_host_requirements',inspect)
    assert asyncio.run(engine._refresh_reviewed_dependencies(job,tmp_path))
    assert job.environment_spec['install_commands'][0][0]==python
    assert job.environment_spec['create_commands']==[]
    assert job.environment_spec['host_compatibility']=={'issues':[]}
    assert job.environment_spec['dependency_revision_pending_install']
    assert job.environment_spec['recheck_on_run']
    assert not job.environment_spec['ready']
    assert plans[0]['entrypoints']==['run.py']
    assert not asyncio.run(engine._refresh_reviewed_dependencies(job,tmp_path))
    requirements.write_text('numpy==2.0.0\n')
    with pytest.raises(ValueError,match='偏离审查'):
        asyncio.run(engine._refresh_reviewed_dependencies(job,tmp_path))


def test_unreviewed_dependency_does_not_replan_or_install(tmp_path):
    job=SimpleNamespace(environment_spec={'prefix':'existing'},result_json={})
    assert not asyncio.run(engine._refresh_reviewed_dependencies(job,tmp_path))


@pytest.mark.parametrize('cached_issue', [False, True])
def test_ready_marker_does_not_skip_reviewed_dependency_install(tmp_path, monkeypatch, cached_issue):
    prefix=tmp_path/'env';prefix.mkdir()
    python=engine._environment_python(prefix);python.touch()
    (prefix/'.repromind-ready').write_text('old dependency snapshot')
    conda=tmp_path/'conda.exe';conda.touch()
    command=[str(python),'-m','pip','install','numpy==1.26.4']
    source=tmp_path/'source';source.mkdir()
    job=SimpleNamespace(id='run',project_id='project',result_json={},events=[],logs='',execution_command={'pythonpath':str(source)},
        environment_spec={'prefix':str(prefix),'environment_name':'original','conda_executable':str(conda),
            'host_compatibility':{'issues':[{'reason':'cached false rejection'}] if cached_issue else []},
            'recheck_on_run':True,'dependency_revision_pending_install':True,
            'create_commands':[['conda','create']], 'install_commands':[command]})
    calls=[]
    class Session:
        async def commit(self):pass
    async def refresh(*args):return False
    async def install(*args,**kwargs):calls.append(args[3]);return 0,'installed'
    async def advance(*args,**kwargs):pass
    async def existing(job,spec):return {**spec,'python':'3.7.16'}
    async def inspect(source,spec):
        assert spec['python']=='3.7.16'
        return {'issues':[]}
    monkeypatch.setattr(engine,'_refresh_reviewed_dependencies',refresh)
    monkeypatch.setattr(engine,'_named_environment_prefix',lambda *args:prefix)
    monkeypatch.setattr(engine,'_environment_owned_by_job',lambda *args:True)
    monkeypatch.setattr(engine,'_recover_environment_operation',install)
    monkeypatch.setattr(engine,'_run_environment_operation',install)
    monkeypatch.setattr(engine,'_advance',advance)
    monkeypatch.setattr(engine,'_existing_environment_plan',existing)
    monkeypatch.setattr(host_environment,'inspect_host_requirements',inspect)
    assert asyncio.run(engine._ensure_environment(Session(),job,tmp_path))==python
    assert calls[-1]==command
    assert len(calls)==2 and '--dry-run' in calls[0]
    assert job.environment_spec['ready']
    assert not job.environment_spec['dependency_revision_pending_install']
