"""Offline contracts; live browser/model/experiment evidence is recorded separately."""
from types import SimpleNamespace
from uuid import uuid4
import hashlib
import json

from app.api.experiment_branches import _recover_preview
from app.services.task_context import rank_entries, question_entrypoints
from app.services.environment_recovery import pytorch_cuda_plan
from app.services import reproduction_engine as engine
import asyncio


def test_unnamed_question_preserves_selected_entry_but_named_entry_can_change(tmp_path):
    for name in ['transformer.py','mlp.py']:
        (tmp_path/name).write_text('import torch\n')
    manifest={'entrypoints':['transformer.py','mlp.py']}
    assert rank_entries(tmp_path,manifest,'当前 Adult 的训练入口、forward和损失在哪里？')[0]=='transformer.py'
    assert rank_entries(tmp_path,manifest,'请解释mlp的实现')[0]=='mlp.py'
    (tmp_path/'mlp.py').write_text('def forward(x): return x\n')
    assert question_entrypoints(tmp_path,manifest,'当前 Adult 的forward和优化器在哪里？不运行。')[0]=='transformer.py'
    assert question_entrypoints(tmp_path,manifest,'请解释mlp.py的forward')[0]=='mlp.py'


def test_preview_survives_navigation_but_not_source_changes_or_application(tmp_path):
    source=tmp_path/'source';source.mkdir()
    original='n=1\n';updated='n=2\n';(source/'train.py').write_text(original)
    directory=tmp_path/'runs'/'run'/'proposals';directory.mkdir(parents=True)
    saved={'base_run_id':'run','files':['train.py'],
           'instruction':'set n to two',
           'source_sha256':{'train.py':hashlib.sha256(original.encode()).hexdigest()},
           'updated_sha256':{'train.py':hashlib.sha256(updated.encode()).hexdigest()},
           'changes':[{'file':'train.py','old_text':'n=1','new_text':'n=2'}]}
    p=directory/(str(uuid4())+'.json');p.write_text(json.dumps(saved))
    base=SimpleNamespace(id='run')
    restored=_recover_preview(tmp_path,base,source)
    assert restored['instruction']=='set n to two' and '+n=2' in restored['diffs']['train.py']
    assert (source/'train.py').read_text()==original
    (source/'train.py').write_text('n=3\n')
    assert _recover_preview(tmp_path,base,source) is None
    (source/'train.py').write_text(original)
    p.write_text(json.dumps({**saved,'applied_at':'already-applied'}))
    assert _recover_preview(tmp_path,base,source) is None


def test_archived_cuda_build_preserves_old_api_and_platform_abi():
    runtime={'python_version':[3,9,21],'platform':'win32','machine':'AMD64','version':'1.13.1'}
    plan=pytorch_cuda_plan(runtime,{'torch':'1.13.1'})
    assert plan['supported'] and plan['packages']==['torch==1.13.1+cu117']
    assert plan['candidates'][0]['index_url']=='https://download.pytorch.org/whl/cu117'
    assert not pytorch_cuda_plan({**runtime,'python_version':[3,11,0]}, {'torch':'1.13.1'})['supported']


def test_removed_submodule_is_routed_to_source_repair_without_reinstall(tmp_path,monkeypatch):
    calls=[]
    async def probe(argv,*args,**kwargs):
        calls.append(argv)
        return 0,'True\n'
    monkeypatch.setattr(engine,'_run_process',probe)
    result=asyncio.run(engine._repair_missing_dependency(None,SimpleNamespace(execution_command={}),
        tmp_path/'python.exe',tmp_path,"ModuleNotFoundError: No module named 'somepackage.removed.api'"))
    assert result[0] is False and result[2]=='source_api'
    assert len(calls)==1 and calls[0][1]=='-c'


def test_existing_interpreter_abi_replaces_new_environment_default(tmp_path,monkeypatch):
    python=tmp_path/'python.exe';python.touch()
    monkeypatch.setattr(engine,'_environment_python',lambda prefix:python)
    monkeypatch.setattr(engine,'_environment_owned_by_job',lambda *args:True)
    async def actual(_):return {'python_version':[3,9],'platform':'win32','machine':'AMD64'}
    monkeypatch.setattr(engine,'probe_interpreter',actual)
    spec=asyncio.run(engine._existing_environment_plan(None,{'prefix':str(tmp_path),'python':'3.11','create_commands':[['conda','create','python=3.11']]}))
    assert spec['python']=='3.9' and not spec['create_commands']
