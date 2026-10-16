"""Offline environment recovery regressions: no network, real install or training."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import reproduction_engine as engine
from app.services.environment_recovery import pytorch_cuda_plan, validate_pip_report, protected_constraints, failure_kind


@pytest.mark.parametrize('minor,version,index', [(7, '1.13.1', 'cu117'), (8, '2.4.1', 'cu121'),
                                                (9, '2.6.0', 'cu126'), (11, '2.6.0', 'cu126'), (13, '2.6.0', 'cu126')])
def test_profile_uses_actual_python_abi(minor, version, index):
    plan = pytorch_cuda_plan({'python_version': [3, minor], 'platform': 'win32', 'machine': 'AMD64'})
    assert plan['supported'] and plan['packages'] == [f'torch=={version}+{index}']
    assert len(plan['candidates']) <= 2


@pytest.mark.parametrize('runtime', [{}, {'python_version': [3, 6]},
    {'python_version': [3, 14], 'platform': 'win32', 'machine': 'AMD64'},
    {'python_version': [3, 7], 'platform': 'win32', 'machine': 'x86'},
    {'python_version': [3, 11], 'platform': 'darwin', 'machine': 'arm64'}])
def test_unknown_runtime_never_blindly_installs(runtime):
    assert not pytorch_cuda_plan(runtime)['supported']


def test_existing_api_is_not_downgraded_to_satisfy_wheel_profile():
    plan = pytorch_cuda_plan({'python_version': [3, 11], 'platform': 'win32', 'machine': 'AMD64'}, {'torch': '2.9.0'})
    assert plan['supported'] and plan['packages']==['torch==2.9.0+cu126']
    assert all('2.6.0' not in package for candidate in plan['candidates'] for package in candidate['packages'])


@pytest.mark.parametrize('code,text,kind', [(1, 'No matching distribution found for torch', 'incompatible_package'),
    (1, 'ReadTimeout; No matching distribution found for torch', 'network'),
    (1, "ProxyError: Cannot connect to proxy; No matching distribution found for torch", 'network'),
    (124, 'command timeout', 'timeout'), (1, 'ResolutionImpossible: conflicting dependencies', 'dependency_conflict')])
def test_failure_diagnosis_distinguishes_transport_and_compatibility(code, text, kind):
    assert failure_kind(code, text) == kind


def report_item(name, version, host='download.pytorch.org'):
    return {'metadata': {'name': name, 'version': version},
            'download_info': {'url': f'https://{host}/whl/{name}.whl'}}


@pytest.mark.parametrize('name', ['numpy', 'tensorflow', 'pip', 'setuptools', 'typing-extensions'])
def test_existing_non_torch_dependencies_cannot_be_replaced(name):
    candidate = {'packages': ['torch==1.13.1+cu117']}
    report = {'install': [report_item('torch', '1.13.1+cu117'), report_item(name, '2.0')]}
    with pytest.raises(ValueError, match='未批准'):
        validate_pip_report(report, {name: '1.0'}, candidate)


def test_only_verified_binary_sources_and_requested_cuda_build_are_allowed():
    candidate = {'packages': ['torch==1.13.1+cu117']}
    assert validate_pip_report({'install': [report_item('torch', '1.13.1+cu117')]}, {}, candidate) == ['torch==1.13.1+cu117']
    for row in [report_item('torch', '1.13.1+cpu'), report_item('torch', '1.13.1+cu117', 'untrusted.example')]:
        with pytest.raises(ValueError):
            validate_pip_report({'install': [row]}, {}, candidate)
    assert protected_constraints({'numpy': '1.18.5', 'tensorflow': '1.15.5', 'torch': '1.13.1+cpu'}) == ['numpy==1.18.5', 'tensorflow==1.15.5']


def setup_recovery(tmp_path, monkeypatch, *, minor=7, blocked=False, gpu_after=True, fallback=False, network=False):
    job = SimpleNamespace(id='job', approved_at='approved', environment_spec={})
    spec = {'inferred_dependencies': ['torch'], 'dependency_source': 'original TensorFlow'}
    calls = []
    count = 0
    async def probe(*args, **kwargs):
        nonlocal count
        count += 1
        return {'checked': count > 1, 'cuda_available': count > 1 and gpu_after,
                'compute_checked': count > 1 and gpu_after}
    async def interpreter(*args): return {'python_version': [3, minor], 'platform': 'win32', 'machine': 'AMD64'}
    async def process(argv, *args):
        if 'list' in argv: return 0, json.dumps([{'name': 'numpy', 'version': '1.18.5'}])
        if '--help' in argv: return 0, '--dry-run --report'
        pytest.fail(f'Unexpected process: {argv}')
    async def operation(session, job, workspace, argv, timeout, log, **kwargs):
        calls.append(argv)
        if '--dry-run' in argv:
            if network: return 1, 'ReadTimeout'
            if fallback and 'https://download.pytorch.org/whl/cu126' in argv: return 1, 'No matching distribution found for torch'
            requirement = next(arg for arg in argv if arg.startswith('torch=='))
            rows = [report_item('torch', requirement.split('==')[1])]
            if blocked: rows.append(report_item('numpy', '2.0'))
            Path(argv[argv.index('--report') + 1]).write_text(json.dumps({'install': rows}), encoding='utf-8')
        return 0, 'ok'
    async def noop(*args): pass
    monkeypatch.setattr(engine, 'probe_framework', probe)
    monkeypatch.setattr(engine, 'probe_interpreter', interpreter)
    monkeypatch.setattr(engine, '_environment_owned_by_job', lambda *args: True)
    monkeypatch.setattr(engine, '_run_process', process)
    monkeypatch.setattr(engine, '_run_environment_operation', operation)
    monkeypatch.setattr(engine, '_advance', noop)
    session = SimpleNamespace(commit=noop)
    return job, spec, session, calls


def test_python37_migration_preflights_then_installs_without_force_reinstall(tmp_path, monkeypatch):
    job, spec, session, calls = setup_recovery(tmp_path, monkeypatch)
    asyncio.run(engine._prepare_pytorch_gpu(session, job, tmp_path, tmp_path / 'env' / 'python.exe', spec))
    assert '--dry-run' in calls[0] and '--no-deps' in calls[1]
    assert all('--force-reinstall' not in call and '--upgrade' not in call for call in calls)
    assert all('https://download.pytorch.org/whl/cu126' not in call for call in calls)
    assert spec['gpu_runtime']['compute_checked']


def test_dependency_replacement_stops_before_mutation(tmp_path, monkeypatch):
    job, spec, session, calls = setup_recovery(tmp_path, monkeypatch, blocked=True)
    with pytest.raises(RuntimeError, match='未修改环境'):
        asyncio.run(engine._prepare_pytorch_gpu(session, job, tmp_path, tmp_path / 'env' / 'python.exe', spec))
    assert len(calls) == 1 and '--dry-run' in calls[0]


def test_compatible_alternative_is_bounded_and_does_not_change_api(tmp_path, monkeypatch):
    job, spec, session, calls = setup_recovery(tmp_path, monkeypatch, minor=11, fallback=True)
    asyncio.run(engine._prepare_pytorch_gpu(session, job, tmp_path, tmp_path / 'env' / 'python.exe', spec))
    assert len(calls) == 3 and '--dry-run' in calls[0] and '--dry-run' in calls[1]
    assert 'torch==2.6.0+cu118' in calls[2]
    assert spec['recovery_attempts'][0]['kind'] == 'incompatible_package'


def test_network_retries_once_without_install_or_changing_mirror(tmp_path, monkeypatch):
    job, spec, session, calls = setup_recovery(tmp_path, monkeypatch, network=True)
    with pytest.raises(RuntimeError, match='network'):
        asyncio.run(engine._prepare_pytorch_gpu(session, job, tmp_path, tmp_path / 'env' / 'python.exe', spec))
    assert len(calls) == 2 and calls[0] == calls[1] and all('--dry-run' in call for call in calls)


def test_gpu_compute_failure_blocks_training(tmp_path, monkeypatch):
    job, spec, session, calls = setup_recovery(tmp_path, monkeypatch, gpu_after=False)
    with pytest.raises(RuntimeError, match='不回退 CPU'):
        asyncio.run(engine._prepare_pytorch_gpu(session, job, tmp_path, tmp_path / 'env' / 'python.exe', spec))
    assert len(calls) == 2


def test_without_experiment_authority_no_install(tmp_path, monkeypatch):
    job, spec, session, calls = setup_recovery(tmp_path, monkeypatch)
    job.approved_at = None
    with pytest.raises(RuntimeError, match='未批准实验'):
        asyncio.run(engine._prepare_pytorch_gpu(session, job, tmp_path, tmp_path / 'env' / 'python.exe', spec))
    assert calls == []


def test_retry_classifies_latest_error_not_old_network_failure(tmp_path, monkeypatch):
    results = iter([(1, 'ReadTimeout'), (1, 'No matching distribution found for torch')])
    async def operation(*args, **kwargs): return next(results)
    async def noop(*args): pass
    monkeypatch.setattr(engine, '_run_environment_operation', operation)
    monkeypatch.setattr(engine, '_advance', noop)
    code, output = asyncio.run(engine._recover_environment_operation(None, None, tmp_path, ['python'],
        tmp_path / 'log.txt', 'test', 1, 1))
    assert failure_kind(code, output) == 'incompatible_package'


def test_metadata_unsupported_downloads_once_then_preflights_and_installs_local(tmp_path, monkeypatch):
    import hashlib
    job, spec, session, calls = setup_recovery(tmp_path, monkeypatch)
    wheel = None
    async def operation(session, job, workspace, argv, timeout, log, **kwargs):
        nonlocal wheel
        calls.append(argv)
        if 'pip_metadata_probe.py' in str(argv[1]):
            return 86, 'REPROMIND_METADATA_ONLY: full wheel needed'
        if 'download' in argv:
            wheel = Path(argv[argv.index('--dest') + 1]) / 'torch.whl'
            wheel.write_bytes(b'offline fixture, not executable package')
        elif '--dry-run' in argv:
            row = report_item('torch', '1.13.1+cu117')
            row['download_info'] = {'url': wheel.as_uri(), 'archive_info': {'hashes': {
                'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}}}
            Path(argv[argv.index('--report') + 1]).write_text(json.dumps({'install': [row]}), encoding='utf-8')
        elif '--no-deps' in argv:
            assert str(wheel) in argv and '--no-index' in argv
        else:
            pytest.fail(f'Unexpected operation: {argv}')
        return 0, 'ok'
    monkeypatch.setattr(engine, '_run_environment_operation', operation)
    asyncio.run(engine._prepare_pytorch_gpu(session, job, tmp_path, tmp_path / 'env' / 'python.exe', spec))
    assert len(calls) == 4 and sum('download' in call for call in calls) == 1
