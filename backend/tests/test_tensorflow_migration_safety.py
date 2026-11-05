import asyncio
import json
from types import SimpleNamespace

import pytest

from app.services import reproduction_engine as engine
from app.services.gpu_preparation import TF_WINDOWS_PROFILE, validate_cuda_dependency_plan


@pytest.mark.parametrize('payload', [
    'truncated {', 'null', '[]', '{}', '{"success":false}', '{"success":true}',
    '{"success":true,"actions":[]}', '{"success":true,"actions":{"LINK":"invalid"}}',
    '{"success":true,"actions":{"UNLINK":[{}]}}',
])
def test_unverifiable_dry_run_cannot_install(payload):
    with pytest.raises(ValueError):
        validate_cuda_dependency_plan(payload)


@pytest.mark.parametrize('package', ['python', 'openssl', 'numpy', 'certifi', 'pip'])
def test_protected_dependency_changes_are_rejected(package):
    with pytest.raises(ValueError, match=package):
        validate_cuda_dependency_plan(json.dumps({'success': True, 'actions': {
            'UNLINK': [{'name': package, 'version': 'old'}],
            'LINK': [{'name': package, 'version': 'new'}, {'name': 'cudnn'}]}}))


@pytest.mark.parametrize('actions', [{}, {'LINK': [{'name': 'cudnn'}]},
                                    {'UNLINK': [{'name': 'cudatoolkit'}], 'LINK': [{'name': 'cudatoolkit'}]}])
def test_only_cuda_dependency_replacement_can_pass(actions):
    validate_cuda_dependency_plan(json.dumps({'success': True, 'actions': actions}))


def test_already_installed_is_valid_but_unknown_message_is_not():
    validate_cuda_dependency_plan('{"success":true,"message":"All requested packages already installed."}')
    with pytest.raises(ValueError):
        validate_cuda_dependency_plan('{"success":true,"message":"solver finished"}')


def setup_migration(tmp_path, monkeypatch, *, return_code=0, unsafe=False):
    source = tmp_path / 'source'; source.mkdir()
    spec = {'conda_executable': 'conda.exe',
            'gpu_environment_migration': {'approved': True, 'profile': TF_WINDOWS_PROFILE}}
    job = SimpleNamespace(id='run-1', logs='', result_json={}, environment_spec=spec,
                          execution_command={'pythonpath': str(source)})
    calls = []
    async def probe(*args, verify_compute=False, **kwargs):
        return {'checked': True, 'cuda_available': verify_compute, 'compute_checked': verify_compute}
    async def process(command, *args, **kwargs):
        calls.append(command)
        if 'list' in command or 'freeze' in command:
            return 0, 'original dependency inventory'
        if '--dry-run' in command:
            return 0, json.dumps({'success': True, 'actions': {'UNLINK': [{'name': 'python'}] if unsafe else []}})
        return return_code, 'downloading cudnn 46%\n命令超过 2700 秒，已终止。' if return_code else 'installed'
    async def advance(*args, **kwargs): pass
    class Session:
        async def commit(self): pass
        async def refresh(self, *args, **kwargs): pass
    monkeypatch.setattr(engine, 'probe_framework', probe)
    monkeypatch.setattr(engine, '_run_process', process)
    monkeypatch.setattr(engine, '_advance', advance)
    monkeypatch.setattr(engine, '_environment_owned_by_job', lambda *args: True)
    monkeypatch.setattr(engine, 'tensorflow_gpu_plan', lambda *args: {'supported': True, 'source_changes_needed': False})
    monkeypatch.setattr(engine, 'protected_conda_specs', lambda _: ['python=3.7.16=existing'])
    return Session(), job, spec, calls


def test_unsafe_solver_stops_before_real_install(tmp_path, monkeypatch):
    session, job, spec, calls = setup_migration(tmp_path, monkeypatch, unsafe=True)
    with pytest.raises(RuntimeError, match='python'):
        asyncio.run(engine._prepare_tensorflow_gpu(session, job, tmp_path, tmp_path / 'env' / 'python.exe', spec))
    assert len(calls) == 3 and '--dry-run' in calls[-1]
    assert spec['last_failure']['kind'] == 'unsafe_dependency_plan'


def test_timeout_is_environment_failure_and_never_reaches_pip_or_training(tmp_path, monkeypatch):
    session, job, spec, calls = setup_migration(tmp_path, monkeypatch, return_code=124)
    with pytest.raises(RuntimeError, match='训练尚未启动'):
        asyncio.run(engine._prepare_tensorflow_gpu(session, job, tmp_path, tmp_path / 'env' / 'python.exe', spec))
    assert len(calls) == 4 and all('tensorflow==2.10.0' not in command for command in calls)
    assert spec['last_failure']['kind'] == 'timeout' and spec['last_failure']['return_code'] == 124


@pytest.mark.parametrize('changed', [False, True])
def test_retry_preserves_snapshot_and_requires_exact_inventory(tmp_path, monkeypatch, changed):
    session, job, spec, calls = setup_migration(tmp_path, monkeypatch)
    snapshot_file = tmp_path / 'runs' / job.id / 'operations' / 'environment-before-tensorflow-migration.json'
    snapshot_file.parent.mkdir(parents=True)
    original = json.dumps({'conda_explicit': 'old versions' if changed else 'original dependency inventory',
                           'pip_freeze': 'original dependency inventory'})
    snapshot_file.write_text(original, encoding='utf-8')
    if changed:
        with pytest.raises(RuntimeError, match='快照不一致'):
            asyncio.run(engine._prepare_tensorflow_gpu(session, job, tmp_path, tmp_path / 'env' / 'python.exe', spec))
        assert len(calls) == 2
    else:
        asyncio.run(engine._prepare_tensorflow_gpu(session, job, tmp_path, tmp_path / 'env' / 'python.exe', spec))
        assert len(calls) == 6 and spec['gpu_runtime_verified']['compute_checked']
    assert snapshot_file.read_text(encoding='utf-8') == original


def test_cold_channel_cache_is_loaded_before_install_and_install_follows_verified_plan(tmp_path, monkeypatch):
    session, job, spec, calls = setup_migration(tmp_path, monkeypatch)
    original_process = engine._run_process
    index_loaded = False
    async def cold_cache(command, *args, **kwargs):
        nonlocal index_loaded
        if '--dry-run' in command:
            if '--use-index-cache' in command:
                return 1, 'WARNING: No repodata found for channel conda-forge.\n{"success":false,"exception_name":"PackagesNotFoundError"}'
            index_loaded = True
        elif command[:2] == ['conda.exe', 'install']:
            assert index_loaded, 'installation started without obtaining channel metadata'
            assert '--use-index-cache' in command
        return await original_process(command, *args, **kwargs)
    monkeypatch.setattr(engine, '_run_process', cold_cache)
    asyncio.run(engine._prepare_tensorflow_gpu(session, job, tmp_path, tmp_path / 'env' / 'python.exe', spec))
    assert index_loaded and spec['gpu_runtime_verified']['compute_checked']
    assert '--use-index-cache' not in calls[2] and '--dry-run' in calls[2]
    assert '--use-index-cache' in calls[3] and '--dry-run' not in calls[3]


def test_warning_prefixed_unsafe_plan_still_blocks_real_install(tmp_path, monkeypatch):
    session, job, spec, calls = setup_migration(tmp_path, monkeypatch, unsafe=True)
    original_process = engine._run_process
    async def warned_plan(command, *args, **kwargs):
        code, output = await original_process(command, *args, **kwargs)
        return code, 'WARNING conda: package metadata warning\n' + output if '--dry-run' in command else output
    monkeypatch.setattr(engine, '_run_process', warned_plan)
    with pytest.raises(RuntimeError, match='python'):
        asyncio.run(engine._prepare_tensorflow_gpu(session, job, tmp_path, tmp_path / 'env' / 'python.exe', spec))
    assert len(calls) == 3 and spec['last_failure']['kind'] == 'unsafe_dependency_plan'


@pytest.mark.parametrize('payload', [None, 'WARNING solver\n{"success":true}\ntruncated',
                                     'WARNING solver\n{"success":false}\n{"success":true,"actions":{}}'])
def test_incomplete_or_multiple_conda_results_cannot_be_treated_as_success(payload):
    with pytest.raises(ValueError):
        validate_cuda_dependency_plan(payload)


def test_missing_index_is_not_reported_as_an_incompatible_cuda_version():
    from app.services.environment_recovery import failure_kind
    from app.services.gpu_preparation import cuda_failure_detail
    output = 'WARNING: No repodata found for channel conda-forge.\n{"exception_name":"PackagesNotFoundError","packages":["cudnn=8.1.0*"]}'
    kind = failure_kind(1, output)
    assert kind == 'index_unavailable'
    assert '索引未加载' in cuda_failure_detail(output, kind)


def test_warning_prefixed_success_is_accepted_with_protected_dependencies_unchanged():
    validate_cuda_dependency_plan('WARNING conda: metadata notice\n{"success":true,"actions":{"LINK":[{"name":"cudnn"}]}}')
