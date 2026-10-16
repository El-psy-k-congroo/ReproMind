"""Offline transition/router tests. No installed packages, training or model calls."""
import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from app.api import experiment_branches as branches, model_routing
from app.config import Settings
from app.core.model_policy import read_policy, write_policy, task_model_id
from app.core.model_router import ModelRole, ModelRouter
from app.models import ReproductionStatus


@pytest.mark.parametrize('state',[ReproductionStatus.awaiting_approval,ReproductionStatus.needs_input])
def test_idle_author_source_can_be_revised_before_first_run(tmp_path,monkeypatch,state):
    original='epochs = 200\n';(tmp_path/'train.py').write_text(original)
    base=SimpleNamespace(id='base',status=state,events=[],progress=82,result_json={})
    proposal={'base_run_id':'base','files':['train.py'],'source_sha256':{'train.py':hashlib.sha256(original.encode()).hexdigest()},
              'changes':[{'file':'train.py','old_text':'epochs = 200','new_text':'epochs = 2'}],'instruction':'本次仅训练两轮'}
    proposal_file=tmp_path/'proposal.json';proposal_file.write_text('{}')
    async def idle(*_):pass
    async def job(*_):return base
    class DB:
        async def commit(self):pass
    monkeypatch.setattr(branches,'_ensure_idle',idle)
    monkeypatch.setattr(branches,'_job',job)
    monkeypatch.setattr(branches,'_scope',lambda *_:(tmp_path,tmp_path))
    monkeypatch.setattr(branches,'_proposal',lambda *_:(proposal_file,proposal))
    result=asyncio.run(branches.apply_revision('p','base',branches.BranchConfirmRequest(proposal_id='proposal'),DB()))
    assert result['status']=='applied_not_run' and (tmp_path/'train.py').read_text()=='epochs = 2\n'
    assert (tmp_path/'runs/base/revisions/proposal/original/train.py').read_text()==original


def fixture_proposal(tmp_path):
    body = 'import tensorflow as tf\n'
    (tmp_path / 'train.py').write_text(body, encoding='utf-8')
    base = SimpleNamespace(id='base', execution_command={'commands': [{'dataset': 'Dataset-A'}]})
    proposal = {'base_run_id': 'base', 'applied_at': 'now', 'instruction': '请在GPU上重新跑一下实验',
                'updated_sha256': {'train.py': hashlib.sha256(body.encode()).hexdigest()}}
    return base, proposal


@pytest.mark.parametrize('action, migration, runtime', [
    ('prepare_code', {'needed': True, 'supported': True, 'source_changes_needed': True}, {'checked': True, 'cuda_available': False}),
    ('approve_environment', {'needed': True, 'supported': True, 'source_changes_needed': False}, {'checked': True, 'cuda_available': False}),
    ('blocked', {'needed': True, 'supported': False, 'reason': 'ABI incompatible'}, {'checked': True, 'cuda_available': False}),
    ('ready', {'needed': False}, {'checked': True, 'cuda_available': True}),
    ('blocked', {'needed': False}, {'checked': False, 'error': 'probe failed'}),
])
def test_readonly_plan_classifies_next_transition(tmp_path, monkeypatch, action, migration, runtime):
    base, proposal = fixture_proposal(tmp_path)
    async def readiness(_):
        return {'framework': 'tensorflow', 'gpu_code_issues': [], 'gpu_environment_plan': migration, 'runtime': runtime}
    monkeypatch.setattr(branches, 'inspect_run_readiness', readiness)
    result = asyncio.run(branches.inspect_revision_plan(base, tmp_path, proposal, 'proposal'))
    assert result['action'] == action and result['gpu_requested']
    assert (tmp_path / 'train.py').read_text() == 'import tensorflow as tf\n'
    assert 'run_id' not in proposal


def test_stale_digest_and_unapplied_stop_before_runtime_probe(tmp_path, monkeypatch):
    base, proposal = fixture_proposal(tmp_path)
    async def forbidden(_): pytest.fail('No probe for a stale/unapplied proposal')
    monkeypatch.setattr(branches, 'inspect_run_readiness', forbidden)
    for invalid in [{**proposal, 'applied_at': ''}, {**proposal, 'updated_sha256': {'train.py': 'wrong'}}]:
        assert asyncio.run(branches.inspect_revision_plan(base, tmp_path, invalid, 'proposal'))['action'] == 'blocked'
    existing = {**proposal, 'run_id': 'created'}
    assert asyncio.run(branches.inspect_revision_plan(base, tmp_path, existing, 'proposal'))['action'] == 'already_created'


def test_gpu_preview_uses_general_planner_and_preserves_source(tmp_path, monkeypatch):
    from app.services import project_coding
    original = 'import tensorflow as tf\n'.replace('\\n', '\n')
    (tmp_path / 'train.py').write_text(original, encoding='utf-8')
    base = SimpleNamespace(id='base', paper_spec={}, environment_spec={}, execution_command={}, result_json={})
    async def job(*_): return base
    async def readiness(_): return {'gpu_environment_plan': {}}
    async def propose(source, instruction, model, **kwargs):
        assert instruction == '请在 GPU 上运行'
        assert source == tmp_path
        return {'train.py': original}, {'train.py': 'changed'}, [], SimpleNamespace(data={}, model='real-model'), 7, [{'agent': 'Code Planning Agent'}]
    monkeypatch.setattr(branches, '_job', job)
    monkeypatch.setattr(branches, '_scope', lambda _: (tmp_path, tmp_path))
    monkeypatch.setattr(branches, 'inspect_run_readiness', readiness)
    monkeypatch.setattr(project_coding, 'propose_revision', propose)
    saved = tmp_path / 'p.json'
    saved.write_text('{}')
    monkeypatch.setattr(branches, '_store_preview', lambda *_: {'proposal_id': 'p', 'files': ['train.py']})
    monkeypatch.setattr(branches, '_proposal', lambda *_: (saved, {}))
    result = asyncio.run(branches.preview_branch('project', 'base', branches.BranchPreviewRequest(
        instruction='请在 GPU 上运行'), None))
    assert result['coding_trace'][0]['agent'] == 'Code Planning Agent'
    assert (tmp_path / 'train.py').read_text() == original


def test_gpu_code_gap_blocks_approval_before_copy_or_scheduling(tmp_path, monkeypatch):
    base, proposal = fixture_proposal(tmp_path)
    async def job(*_): return base
    async def idle(*_): pass
    async def readiness(_):
        return {'framework': 'tensorflow', 'gpu_code_issues': [],
                'runtime': {'checked': True, 'cuda_available': False},
                'gpu_environment_plan': {'needed': True, 'supported': True, 'source_changes_needed': True}}
    monkeypatch.setattr(branches, '_job', job)
    monkeypatch.setattr(branches, '_scope', lambda _: (tmp_path, tmp_path))
    monkeypatch.setattr(branches, '_proposal', lambda *_: (tmp_path / 'p.json', proposal))
    monkeypatch.setattr(branches, '_ensure_idle', idle)
    monkeypatch.setattr(branches, '_branch_environment', lambda *_: {})
    monkeypatch.setattr(branches, '_hardware_plan', lambda *_: {})
    monkeypatch.setattr(branches, '_infer_python_dependencies', lambda *_: [])
    monkeypatch.setattr(branches, 'inspect_run_readiness', readiness)
    monkeypatch.setattr(branches, '_copy_bounded', lambda *_: pytest.fail('Must not copy or schedule'))
    tasks = BackgroundTasks()
    with pytest.raises(HTTPException, match='源码兼容修改'):
        asyncio.run(branches._run_revision_once('p', 'base', branches.RevisionRunRequest(
            proposal_id='p', isolated_copy=True, gpu_environment_migration=True), tasks, None))
    assert not tasks.tasks and 'run_id' not in proposal


def test_groups_share_models_persist_and_preserve_defaults(tmp_path):
    settings = Settings(_env_file=None, model_provider='moma', moma_model_chat='current',
                        moma_model_coder='current', moma_model_reasoning='current',
                        model_routing_path=tmp_path / 'routing.json')
    router = ModelRouter(settings)
    assert router.select(ModelRole.code).model == router.select(ModelRole.paper).model == 'current'
    write_policy(settings, {'router': '', 'chat': 'reader', 'coder': 'coder', 'reasoning': 'reviewer'})
    assert router.select(ModelRole.code).model == router.select(ModelRole.reproduction).model == 'coder'
    assert router.select(ModelRole.paper).model == router.select(ModelRole.report).model == 'reader'
    assert router.select(ModelRole.audit).model == 'reviewer'
    assert router.select(ModelRole.router).model == 'current'
    assert task_model_id(ModelRole.code, 'historical-project-model', settings) == 'coder'
    assert ModelRouter(settings).select(ModelRole.code).model == 'coder'
    assert 'api_key' not in settings.model_routing_path.read_text()
    settings.model_provider = 'openai_compatible'
    assert read_policy(settings)['models'] == {}


def test_explicit_one_call_override_does_not_mutate_policy(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, model_provider='moma', moma_api_key='fake', moma_model_chat='current',
                        model_routing_path=tmp_path / 'routing.json')
    write_policy(settings, {'router': '', 'chat': '', 'coder': 'group-coder', 'reasoning': ''})
    router = ModelRouter(settings)
    used = []
    async def request(selection, *_args, **_kwargs):
        used.append(selection.model); return {'ok': True}, {}
    monkeypatch.setattr(router, '_request_json', request)
    asyncio.run(router.complete_json(ModelRole.code, 'system', 'user', model_override='one-off'))
    assert used == ['one-off'] and router.select(ModelRole.code).model == 'group-coder'


def test_invalid_group_and_stale_revision_do_not_write_policy(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, model_routing_path=tmp_path / 'routing.json')
    monkeypatch.setattr(model_routing, 'get_settings', lambda: settings)
    async def validate(_): pytest.fail('No remote validation for defaults')
    monkeypatch.setattr(model_routing, 'validate_model', validate)
    with pytest.raises(HTTPException, match='四个'):
        asyncio.run(model_routing.update_routing(model_routing.RoutingChoice(models={'coder': ''})))
    write_policy(settings, {'router': '', 'chat': '', 'coder': '', 'reasoning': ''})
    before = settings.model_routing_path.read_bytes()
    with pytest.raises(HTTPException, match='已更新'):
        asyncio.run(model_routing.update_routing(model_routing.RoutingChoice(
            models={'router': '', 'chat': '', 'coder': '', 'reasoning': ''}, revision='stale')))
    assert settings.model_routing_path.read_bytes() == before


def test_policy_save_storage_failure_is_visible_and_keeps_original(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, model_routing_path=tmp_path / 'routing.json')
    monkeypatch.setattr(model_routing, 'get_settings', lambda: settings)
    models = {'router': '', 'chat': '', 'coder': '', 'reasoning': ''}
    policy = write_policy(settings, models)
    before = settings.model_routing_path.read_bytes()
    def fail_write(*_):
        raise OSError('read-only filesystem')
    monkeypatch.setattr(model_routing, 'write_policy', fail_write)
    with pytest.raises(HTTPException, match='保存失败') as failure:
        asyncio.run(model_routing.update_routing(model_routing.RoutingChoice(
            models=models, revision=policy['revision'])))
    assert failure.value.status_code == 503
    assert settings.model_routing_path.read_bytes() == before
