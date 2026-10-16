"""Offline GPU preparation contracts. All installation/training boundaries mocked."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import copilot, experiment_branches as branches
from app.services import reproduction_engine as engine
from app.services.gpu_preparation import gpu_preparation_state


def evidence(framework='tensorflow', *, cuda=False, checked=True, source=False, supported=True):
    return {'framework': framework, 'runtime': {'checked': checked, 'cuda_available': cuda},
            'gpu_code_ready': True, 'gpu_code_issues': [],
            'gpu_environment_plan': {'needed': not cuda, 'supported': supported,
                                     'source_changes_needed': source, 'source_files': ['model.py', 'train.py'],
                                     'packages': ['framework==reviewed'], 'candidates': [
                                         {'profile': 'reviewed', 'packages': ['framework==reviewed']}],
                                     'reason': 'incompatible interpreter'}}


@pytest.mark.parametrize('question', [
    '请把实验在gpu上跑一遍', '把实验在GPU上跑一遍', '请将实验在 CUDA 上运行一遍',
    '请在GPU上运行一下实验', '请把模型在GPU上重新跑一遍',
])
def test_plain_gpu_rerun_is_not_a_code_generation_request(question):
    assert copilot._rerun_requested(question)
    assert not copilot._code_action_requested(question)
    assert not copilot._gpu_capability_question(question)


@pytest.mark.parametrize('question', ['请把训练脚本改成GPU版本', '把模型迁移到PyTorch并在GPU上运行'])
def test_explicit_edits_still_use_general_coding(question):
    assert copilot._code_action_requested(question)


@pytest.mark.parametrize('framework', ['tensorflow', 'pytorch'])
@pytest.mark.parametrize('cuda,source,expected', [(True, False, 'ready'),
                                                (False, False, 'approve_environment'),
                                                (False, True, 'prepare_code')])
def test_framework_agnostic_preparation_states(framework, cuda, source, expected):
    assert gpu_preparation_state(evidence(framework, cuda=cuda, source=source))['action'] == expected


@pytest.mark.parametrize('variant', ['unchecked', 'unsupported', 'mixed'])
def test_unknown_or_unsafe_environment_never_reaches_installation(variant):
    data = evidence(checked=variant != 'unchecked', supported=variant != 'unsupported')
    if variant == 'mixed': data['framework'] = 'mixed'
    assert gpu_preparation_state(data)['action'] == 'blocked'


def test_missing_framework_can_be_configured_only_with_verified_interpreter():
    data = evidence('pytorch', checked=False)
    data['runtime'].update(error_kind='framework_missing', interpreter_checked=True)
    assert gpu_preparation_state(data)['action'] == 'approve_environment'
    data['runtime']['interpreter_checked'] = False
    assert gpu_preparation_state(data)['action'] == 'blocked'


def test_existing_gpu_does_not_bypass_source_compatibility():
    data = evidence('pytorch', cuda=True)
    data['gpu_code_issues'] = ['model remains on CPU']
    assert gpu_preparation_state(data)['action'] == 'prepare_code'


@pytest.mark.parametrize('framework,source,expected', [('tensorflow', True, 'gpu_revision'),
                                                    ('tensorflow', False, 'rerun_confirm'),
                                                    ('pytorch', False, 'rerun_confirm')])
def test_project_rerun_preserves_all_datasets_and_environment_plan(monkeypatch, framework, source, expected):
    job = SimpleNamespace(id='run', status=SimpleNamespace(value='completed'), result_json={},
                          environment_spec={}, execution_command={'commands': [
                              {'dataset': 'Dataset-A'}, {'dataset': 'Dataset-B'}]})
    class Session:
        async def scalar(self, _): return job
    async def inspect(_): return evidence(framework, source=source)
    monkeypatch.setattr(copilot, 'inspect_run_readiness', inspect)
    result = asyncio.run(copilot._rerun_action_plan(Session(), SimpleNamespace(id='p'), 'run',
                                                  '请把实验在gpu上跑一遍'))
    assert result['kind'] == expected
    assert result['selected_datasets'] == ['Dataset-A', 'Dataset-B']
    assert result['gpu_environment_plan']['packages'] == ['framework==reviewed']
    assert result['preparation']['action'] == ('prepare_code' if source else 'approve_environment')


@pytest.mark.parametrize('framework', ['tensorflow', 'pytorch'])
def test_environment_approval_required_for_both_frameworks(monkeypatch, framework):
    async def inspect(_): return evidence(framework)
    monkeypatch.setattr(branches, 'inspect_run_readiness', inspect)
    spec = {}
    with pytest.raises(HTTPException, match='单独批准'):
        asyncio.run(branches._approve_gpu_environment(None, spec, False))
    assert spec == {}
    asyncio.run(branches._approve_gpu_environment(None, spec, True))
    assert spec['gpu_environment_migration']['framework'] == framework
    assert spec['gpu_environment_migration']['approved'] is True


def test_gpu_ready_rerun_never_requests_dependency_installation(monkeypatch):
    async def inspect(_): return evidence('pytorch', cuda=True)
    monkeypatch.setattr(branches, 'inspect_run_readiness', inspect)
    spec = {}
    asyncio.run(branches._approve_gpu_environment(None, spec, False))
    assert spec == {}


def test_pytorch_rerun_cannot_install_with_only_training_approval(tmp_path, monkeypatch):
    async def probe(*args, **kwargs): return {'checked': True, 'cuda_available': False}
    async def process(*args, **kwargs): pytest.fail('Must not execute installation without environment approval')
    monkeypatch.setattr(engine, 'probe_framework', probe)
    monkeypatch.setattr(engine, '_environment_owned_by_job', lambda *args: True)
    monkeypatch.setattr(engine, '_run_process', process)
    job = SimpleNamespace(approved_at='approved', source_mode='copilot_rerun')
    with pytest.raises(RuntimeError, match='单独批准'):
        asyncio.run(engine._prepare_pytorch_gpu(None, job, tmp_path, tmp_path / 'python.exe',
                                               {'gpu_requested': True}))


def test_reviewed_tf_preparation_generates_all_import_changes_without_model(tmp_path, monkeypatch):
    from app.services import project_coding
    originals = {'train.py': 'import tensorflow as tf\nlr = 0.01\n',
                 'model.py': 'import tensorflow as tf\nhidden = 16\n',
                 'data.py': 'import numpy as np\n'}
    for name, body in originals.items(): (tmp_path / name).write_text(body, encoding='utf-8')
    job = SimpleNamespace(id='base')
    async def find(*args): return job
    async def inspect(*args): return evidence(source=True)
    async def forbidden(*args, **kwargs): pytest.fail('Mechanical compatibility does not need remote generation')
    monkeypatch.setattr(branches, '_job', find)
    monkeypatch.setattr(branches, '_scope', lambda job: (tmp_path, tmp_path))
    monkeypatch.setattr(branches, 'inspect_run_readiness', inspect)
    monkeypatch.setattr(project_coding, 'propose_revision', forbidden)
    result = asyncio.run(branches.preview_branch('p', 'base', branches.BranchPreviewRequest(
        instruction='请把实验在GPU上跑一遍', preparation_only=True), None))
    assert result['files'] == ['model.py', 'train.py']
    assert result['coding_trace'][0]['model'] == '未调用模型'
    _, saved = branches._proposal(tmp_path, 'base', result['proposal_id'])
    assert saved['usage']['total_tokens'] == 0
    assert all((tmp_path / name).read_text() == body for name, body in originals.items())
    for row in saved['changes']:
        assert row['new_text'] == 'import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()'
        assert row['old_text'] == 'import tensorflow as tf'
