import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import copilot, experiment_branches as branches
from app.services import reproduction_engine as engine
from app.services.gpu_preparation import tensorflow_compat_changes, tensorflow_gpu_plan, TF_WINDOWS_PROFILE


@pytest.mark.parametrize('question', [
    '你可不可以将模型放到gpu上重新跑一下',
    '你能不能把模型在GPU上重新跑一遍？',
    '可以帮我重新跑这个模型吗？',
    '请帮我在GPU上重新运行这个项目',
    '请把模型在GPU上重新跑一遍',
    '你可不可以将模型放到GPU上重新跑一下，不要退回CPU',
    '请用GPU重新跑模型，不要用CPU',
])
def test_polite_concrete_rerun_is_confirmation_not_capability(question):
    assert copilot._rerun_requested(question)
    assert not copilot._gpu_capability_question(question)


@pytest.mark.parametrize('question', [
    '现在的代码可以迁移到GPU上吗？',
    '能不能把训练脚本改为GPU版本？',
    '如果你把模型在GPU上重新跑一遍会更快吗？',
    '你能不能解释为什么模型在GPU上重新跑会出错？',
    '不要在GPU上重新跑模型，只解释原因',
])
def test_capability_and_negative_requests_never_prepare_mutations(question):
    assert not copilot._rerun_requested(question)
    assert not copilot._code_action_requested(question)


@pytest.mark.parametrize('question', ['现在还有数据集在训练吗？', '现在还在跑吗', '当前在运行吗？', '目前跑的是哪个数据集？'])
def test_common_current_status_queries_use_project_evidence_without_model(question):
    assert copilot._status_question(question)


def test_tensorflow_checker_does_not_demand_torch_cuda(tmp_path):
    (tmp_path / 'train.py').write_text('import tensorflow as tf\nx = tf.placeholder(tf.float32)\n')
    assert branches._gpu_preview_issues(tmp_path, {}) == []
    (tmp_path / 'metrics.py').write_text("import tensorflow as tf\nwith tf.device('/CPU:0'):\n    x = tf.constant(1)\n")
    assert any('CPU' in issue for issue in branches._gpu_preview_issues(tmp_path, {}))


def test_tensorflow_compat_preview_covers_related_modules_without_writes(tmp_path):
    bodies = {'train.py': 'import tensorflow as tf\nflags=tf.app.flags\n',
              'layers.py': 'import tensorflow as tf # author graph code\ndef f(x): return tf.nn.dropout(x, 0.5)\n',
              'data.py': 'import numpy as np\n'}
    for name, body in bodies.items():
        (tmp_path / name).write_text(body)
    revisions = dict(bodies)
    changes = tensorflow_compat_changes(bodies)
    assert {row['file'] for row in changes} == {'train.py', 'layers.py'}
    for row in changes:
        revisions[row['file']] = branches._apply_text(revisions[row['file']], row['old_text'], row['new_text'], row['file'])
    assert 'tf.app.flags' in revisions['train.py']
    assert 'tf.nn.dropout(x, 0.5)' in revisions['layers.py']
    assert branches._gpu_preview_issues(tmp_path, revisions) == []
    assert all((tmp_path / name).read_text() == body for name, body in bodies.items())


@pytest.mark.parametrize('body', ['import tensorflow as tf\nx=tf.contrib.layers\n',
                                 'import tensorflow as tensor\n', 'from tensorflow import Session\n',
                                 'import tensorflow as tf; x=1\n'])
def test_unsupported_tf_imports_require_review(body):
    with pytest.raises(ValueError):
        tensorflow_compat_changes({'model.py': body})


def test_tf_migration_plan_is_pinned_and_does_not_install(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'train.py').write_text('import tensorflow as tf\n')
    conda = tmp_path / 'conda.exe'; conda.touch()
    prefix = tmp_path / 'env'; (prefix / 'conda-meta').mkdir(parents=True)
    runtime = {'checked': True, 'cuda_available': False, 'platform': 'win32', 'python_version': [3, 7]}
    env = {'conda_executable': str(conda), 'prefix': str(prefix)}
    plan = tensorflow_gpu_plan(source, runtime, env)
    assert plan['supported'] and plan['source_changes_needed']
    assert plan['packages'] == ['tensorflow==2.10.0', 'cudatoolkit=11.2', 'cudnn=8.1.0']
    assert not tensorflow_gpu_plan(source, {**runtime, 'python_version': [3, 11]}, env)['supported']
    assert not tensorflow_gpu_plan(source, {**runtime, 'checked': False}, env)['supported']
    assert not tensorflow_gpu_plan(source, {**runtime, 'platform': 'linux'}, env)['supported']


def test_tf_model_rerun_uses_actual_runtime_and_prepares_all_datasets(monkeypatch):
    job = SimpleNamespace(id='run-1', environment_spec={}, result_json={}, status=SimpleNamespace(value='completed'),
                          execution_command={'commands': [{'dataset': 'Cora'}, {'dataset': 'NELL'}]})
    class Session:
        async def scalar(self, _statement): return job
    async def readiness(_job):
        return {'framework': 'tensorflow', 'gpu_code_ready': True, 'gpu_code_issues': [], 'torch_runtime': {},
                'runtime': {'checked': True, 'version': '1.15.5', 'cuda_available': False},
                'gpu_environment_plan': {'needed': True, 'supported': True, 'source_changes_needed': True}}
    monkeypatch.setattr(copilot, 'inspect_run_readiness', readiness)
    plan = asyncio.run(copilot._rerun_action_plan(Session(), SimpleNamespace(id='p'), 'run-1',
                                                '你可不可以将模型放到gpu上重新跑一下'))
    assert plan['kind'] == 'gpu_revision'
    assert plan['selected_datasets'] == ['Cora', 'NELL']
    assert 'TensorFlow' in copilot._rerun_answer(plan)
    assert 'PyTorch' not in copilot._rerun_answer(plan)
    assert '当前没有改代码或运行实验' in copilot._rerun_answer(plan)


def test_tf_upgrade_requires_separate_approval_before_job_creation(monkeypatch):
    async def readiness(_job):
        return {'framework': 'tensorflow', 'gpu_code_issues': [],
                'runtime': {'checked': True, 'cuda_available': False},
                'gpu_environment_plan': {'needed': True, 'supported': True, 'source_changes_needed': False,
                                         'profile': TF_WINDOWS_PROFILE}}
    monkeypatch.setattr(branches, 'inspect_run_readiness', readiness)
    environment = {}
    with pytest.raises(HTTPException, match='单独批准'):
        asyncio.run(branches._approve_gpu_environment(None, environment, False))
    assert not environment
    asyncio.run(branches._approve_gpu_environment(None, environment, True))
    assert environment['gpu_environment_migration']['approved']


def test_tf1_gpu_probe_failure_never_installs_without_approval(tmp_path, monkeypatch):
    async def probe(*_args, **_kwargs): return {'checked': True, 'version': '1.15.5', 'cuda_available': False}
    async def process(*_args, **_kwargs): pytest.fail('No installation or experiment without separate approval')
    monkeypatch.setattr(engine, 'probe_framework', probe)
    monkeypatch.setattr(engine, '_run_process', process)
    with pytest.raises(RuntimeError, match='尚未单独批准'):
        asyncio.run(engine._prepare_tensorflow_gpu(None, None, tmp_path, tmp_path / 'python.exe', {}))


def test_tf_gpu_runtime_requires_real_compute_not_just_enumeration(tmp_path, monkeypatch):
    probes = []
    async def probe(_python, _framework, verify_compute=False):
        probes.append(verify_compute)
        return {'checked': True, 'cuda_available': True, 'compute_checked': verify_compute}
    monkeypatch.setattr(engine, 'probe_framework', probe)
    spec = {}
    asyncio.run(engine._prepare_tensorflow_gpu(None, None, tmp_path, tmp_path / 'python.exe', spec))
    assert probes == [False, True]
    assert spec['gpu_runtime_verified']['compute_checked']


@pytest.mark.parametrize('failure_step', [0, 1, 2, 3])
def test_approved_migration_stops_at_failure_and_preserves_snapshot(tmp_path, monkeypatch, failure_step):
    source = tmp_path / 'source'; source.mkdir()
    calls = []
    async def probe(*_args, **_kwargs): return {'checked': True, 'cuda_available': False}
    async def process(command, *_args, **_kwargs):
        calls.append(command)
        if len(calls) == failure_step + 3:
            return 1, 'installation failed'
        return (0, '{"success":true,"actions":{"LINK":[{"name":"cudnn"}]}}') if '--dry-run' in command else (0, 'snapshot or install success')
    async def advance(*_args, **_kwargs): pass
    class Session:
        async def commit(self): pass
        async def refresh(self, *_args, **_kwargs): pass
    monkeypatch.setattr(engine, 'probe_framework', probe)
    monkeypatch.setattr(engine, '_run_process', process)
    monkeypatch.setattr(engine, '_advance', advance)
    monkeypatch.setattr(engine, '_environment_owned_by_job', lambda *_args: True)
    monkeypatch.setattr(engine, 'tensorflow_gpu_plan', lambda *_args: {'supported': True, 'source_changes_needed': False})
    monkeypatch.setattr(engine, 'protected_conda_specs', lambda _: ['python=3.7.16=existing'])
    spec = {'conda_executable': 'conda.exe', 'gpu_environment_migration': {'approved': True, 'profile': TF_WINDOWS_PROFILE}}
    job = SimpleNamespace(id='run-1', logs='', result_json={}, execution_command={'pythonpath': str(source)})
    with pytest.raises(RuntimeError, match='失败'):
        asyncio.run(engine._prepare_tensorflow_gpu(Session(), job, tmp_path, tmp_path / 'env' / 'python.exe', spec))
    snapshot = tmp_path / 'runs' / 'run-1' / 'operations' / 'environment-before-tensorflow-migration.json'
    assert snapshot.is_file()
    assert len(calls) == failure_step + 3
    assert all('create' not in command for command in calls)
    if failure_step > 0:
        assert '--freeze-installed' in calls[2]
        assert '--dry-run' in calls[2]
    if failure_step > 2:
        assert calls[4][-1] == '--install'
        assert '2.10.0' in calls[4] and calls[4][1].endswith('registry_wheel_download.py')


def test_tf_gpu_enumerated_but_compute_fails_blocks_training(tmp_path, monkeypatch):
    async def probe(_python, _framework, verify_compute=False):
        return {'checked': not verify_compute, 'cuda_available': True, 'compute_checked': False,
                'error': 'GPU kernel unavailable'}
    monkeypatch.setattr(engine, 'probe_framework', probe)
    with pytest.raises(RuntimeError, match='MatMul'):
        asyncio.run(engine._prepare_tensorflow_gpu(None, None, tmp_path, tmp_path / 'python.exe', {}))


def test_cpu_rerun_and_revision_keep_host_gpu_absence_as_fact(tmp_path, monkeypatch):
    base=SimpleNamespace(id='run-a',status=SimpleNamespace(value='completed'),result_json={},environment_spec={},
        execution_command={'commands':[{'dataset':'DatasetA'}]})
    class Session:
        async def scalar(self,_): return base
    question='当前服务器无 GPU，请在 CPU 上重新运行 DatasetA'
    plan=asyncio.run(copilot._rerun_action_plan(Session(),SimpleNamespace(id='project-a'),question,'run-a'))
    assert plan['kind']=='rerun_confirm' and not plan['gpu_requested']
    proposal={'instruction':'无 GPU，请修改 CPU 实验的学习率','base_run_id':'run-a'}
    revision=asyncio.run(branches.inspect_revision_plan(base,tmp_path,proposal,'proposal-a'))
    assert not revision['gpu_requested']
    base.environment_spec={'gpu_requested':True}
    async def unavailable(_): return {}
    monkeypatch.setattr(copilot,'inspect_run_readiness',unavailable)
    inherited=asyncio.run(copilot._rerun_action_plan(Session(),SimpleNamespace(id='project-a'),question,'run-a'))
    assert inherited['gpu_requested']  # Do not silently weaken an approved GPU-only run.
