"""Offline regressions, no model requests, real package install or experiment training."""
import asyncio
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from app.api import reproductions, experiment_branches
from app.models import ReproductionStatus
from app.services import tuning_sessions, tuning, reproduction_engine as engine
from app.services.repair_isolation import isolate_failed_command
from app.services.source_interfaces import interface_issues, provider_context
from app.services.environment_recovery import validate_pip_report


def job_fixture(tmp_path, monkeypatch, metric='accuracy'):
    workspace = tmp_path / 'paper-12345678'
    source = workspace / 'sources' / 'author'
    source.mkdir(parents=True)
    (source / 'entry.py').write_text("parser.add_argument('--lr', type=float, default=.01)\n"
        "parser.add_argument('--dropout', type=float, default=.5)\n"
        "parser.add_argument('--batch_size', type=int, default=32)\n", encoding='utf-8')
    commands, rows = [], []
    for dataset in ['graph', 'tabular', 'text', 'regression']:
        command = {'dataset': dataset, 'argv': [sys.executable, 'entry.py', '--data', dataset, '--lr', '.01'],
                   'cwd': str(source), 'pythonpath': str(source)}
        log = workspace / 'runs' / 'run' / 'baseline_evidence' / (dataset + '.txt')
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f'val_{metric}=0.7\nTest results: {metric}=0.6', encoding='utf-8')
        rows.append({'dataset': dataset, 'return_code': 0, 'verdict': 'deviated', 'metric_key': metric,
            'metrics': {metric: .6}, 'baseline_argv': command['argv'],
            'baseline_fingerprint': tuning.command_fingerprint(command, source),
            'baseline_log': 'baseline_evidence/' + log.name})
        commands.append(command)
    job = SimpleNamespace(id='run', project_id='12345678-project', workspace_path=str(workspace),
        status=ReproductionStatus.completed, current_stage='completed', progress=100, events=[],
        execution_command={'commands': commands, 'entrypoint_candidates': ['entry.py'], 'pythonpath': str(source)},
        code_spec={'source_path': str(source)}, result_json={'dataset_results': rows, 'claim_results': [{'status': 'not_evaluated'}]},
        environment_spec={'prefix': str(Path(sys.executable).parent)}, paper_spec={})
    monkeypatch.setattr(tuning_sessions, 'get_settings', lambda: SimpleNamespace(paper_projects_dir=tmp_path, experiment_timeout_seconds=10))
    return job, workspace, source


class Session:
    def __init__(self, job): self.job = job
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def get(self, *args): return self.job
    async def scalar(self, *args): return self.job
    async def refresh(self, *args): pass
    async def commit(self): pass


def approve(job, monkeypatch, datasets):
    plans = [p for p in tuning_sessions.reviewed_plans(job) if p['dataset'] in datasets]
    assert all(p['ready'] for p in plans)
    async def idle(*args): pass
    monkeypatch.setattr(experiment_branches, '_ensure_idle', idle)
    tasks = BackgroundTasks()
    payload = reproductions.TuningApproval(datasets=datasets, review_tokens={p['dataset']: p['review_token'] for p in plans})
    result = asyncio.run(reproductions.start_tuning_session(job.project_id, job.id, payload, tasks, Session(job)))
    assert result['run_id'] == job.id and len(tasks.tasks) == 1
    return result['session_id']


@pytest.mark.parametrize('metric,best', [('accuracy', .9), ('auc', .9), ('rmse', .3)])
def test_inplace_search_preserves_all_baselines_claims_and_selects_validation(tmp_path, monkeypatch, metric, best):
    job, workspace, source = job_fixture(tmp_path, monkeypatch, metric)
    original = copy.deepcopy(job.result_json)
    session_id = approve(job, monkeypatch, ['graph', 'regression'])
    monkeypatch.setattr(tuning_sessions, 'AsyncSessionLocal', lambda: Session(job))
    async def process(argv, cwd, timeout, env, output_path):
        assert argv[0] == sys.executable and '--data' in argv and '--lr' in argv
        text = f'val_{metric}={best}\nTest results: {metric}=0.1'
        output_path.write_text(text, encoding='utf-8')
        return 0, text
    monkeypatch.setattr(engine, '_run_process', process)
    monkeypatch.setattr(engine, '_write_reproduction_report', lambda *args: None)
    asyncio.run(tuning_sessions.execute_tuning_session(job.id, session_id))
    assert job.result_json['dataset_results'] == original['dataset_results']
    assert job.result_json['claim_results'] == original['claim_results']
    assert job.status == ReproductionStatus.completed and job.progress == 100
    assert 'tuning_active' not in job.result_json
    assert len(job.result_json['tuning_attempts']) == 12
    assert sum(r['selected'] for r in job.result_json['tuning_attempts']) == 2
    assert job.result_json['tuning_sessions'][0]['status'] == 'completed'
    assert not engine.has_active_work(job.id)


def test_changed_source_and_stale_review_cannot_launch(tmp_path, monkeypatch):
    job, _, source = job_fixture(tmp_path, monkeypatch)
    plans = tuning_sessions.reviewed_plans(job)
    (source / 'entry.py').write_text('changed = True', encoding='utf-8')
    assert not tuning_sessions.reviewed_plans(job)[0]['ready']
    async def idle(*args): pass
    monkeypatch.setattr(experiment_branches, '_ensure_idle', idle)
    tasks = BackgroundTasks()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(reproductions.start_tuning_session(job.project_id, job.id,
            reproductions.TuningApproval(datasets=['graph'], review_tokens={'graph': plans[0]['review_token']}), tasks, Session(job)))
    assert exc.value.status_code == 409 and not tasks.tasks


@pytest.mark.parametrize('verdict', ['matched', 'result_produced'])
def test_explicit_validation_optimization_is_not_triggered_by_test_deviation(tmp_path, monkeypatch, verdict):
    job, _, _ = job_fixture(tmp_path, monkeypatch)
    for row in job.result_json['dataset_results']:
        row['verdict'] = verdict
        row['baseline_verdict'] = verdict
    original = copy.deepcopy(job.result_json['dataset_results'])
    recovery = tuning_sessions.reviewed_plans(job)
    assert not any(plan['ready'] for plan in recovery)
    optimized = tuning_sessions.reviewed_plans(job, purpose='validation_optimization')
    assert all(plan['ready'] for plan in optimized)
    assert all(a['review_token'] != b['review_token'] for a, b in zip(recovery, optimized))
    async def idle(*args): pass
    monkeypatch.setattr(experiment_branches, '_ensure_idle', idle)
    tasks = BackgroundTasks()
    payload = reproductions.TuningApproval(purpose='validation_optimization', datasets=['graph'],
        review_tokens={'graph': optimized[0]['review_token']})
    result = asyncio.run(reproductions.start_tuning_session(job.project_id, job.id, payload, tasks, Session(job)))
    assert len(tasks.tasks) == 1 and job.result_json['tuning_sessions'][0]['purpose'] == 'validation_optimization'
    monkeypatch.setattr(tuning_sessions, 'AsyncSessionLocal', lambda: Session(job))
    async def process(argv, cwd, timeout, env, output_path):
        output_path.write_text('val_acc=0.8\nTest accuracy=0.5', encoding='utf-8')
        return 0, output_path.read_text()
    monkeypatch.setattr(engine, '_run_process', process)
    monkeypatch.setattr(engine, '_write_reproduction_report', lambda *args: None)
    asyncio.run(tuning_sessions.execute_tuning_session(job.id, result['session_id']))
    assert job.result_json['tuning_sessions'][0]['status'] == 'completed'
    assert job.result_json['dataset_results'] == original


def test_validation_optimization_still_requires_real_validation_log_and_unchanged_source(tmp_path, monkeypatch):
    job, workspace, source = job_fixture(tmp_path, monkeypatch)
    log=workspace/'runs'/'run'/job.result_json['dataset_results'][0]['baseline_log']
    log.write_text('Test accuracy=0.9', encoding='utf-8')
    plan=tuning_sessions.reviewed_plans(job,purpose='validation_optimization')[0]
    assert not plan['ready'] and '验证集' in plan['blocked_reason']
    (source/'entry.py').write_text('changed=True', encoding='utf-8')
    plan=tuning_sessions.reviewed_plans(job,purpose='validation_optimization')[0]
    assert not plan['ready'] and '源码已改变' in plan['blocked_reason']


@pytest.mark.parametrize('failure', ['process', 'no_validation', 'paused'])
def test_search_error_or_pause_keeps_original_results(tmp_path, monkeypatch, failure):
    job, _, _ = job_fixture(tmp_path, monkeypatch)
    original = copy.deepcopy(job.result_json['dataset_results'])
    session_id = approve(job, monkeypatch, ['graph'])
    monkeypatch.setattr(tuning_sessions, 'AsyncSessionLocal', lambda: Session(job))
    async def process(*args, **kwargs):
        if failure == 'paused':
            job.status = ReproductionStatus.paused
            job.result_json = {**job.result_json, 'paused_from_status': 'completed'}
            raise engine.ReproductionPaused()
        return (1, 'runtime error') if failure == 'process' else (0, 'Test accuracy=0.99')
    monkeypatch.setattr(engine, '_run_process', process)
    asyncio.run(tuning_sessions.execute_tuning_session(job.id, session_id))
    assert job.result_json['dataset_results'] == original
    assert not job.result_json.get('tuning_active') and not engine.has_active_work(job.id)
    assert job.result_json['tuning_sessions'][0]['status'] in {'failed', 'cancelled'}


def test_shared_source_is_copied_only_for_failed_dataset(tmp_path, monkeypatch):
    job, workspace, source = job_fixture(tmp_path, monkeypatch)
    others = copy.deepcopy(job.execution_command['commands'][1:])
    command = job.execution_command['commands'][0]
    old_bytes = (source / 'entry.py').read_bytes()
    argv, cwd, env, output, record = isolate_failed_command(job, workspace, 'graph', command['argv'], source,
        {'PYTHONPATH': str(source)}, f'File "{source / "entry.py"}", line 1')
    assert record and cwd != source and str(cwd) in output and env['PYTHONPATH'] == str(cwd)
    assert job.execution_command['commands'][1:] == others and argv == command['argv']
    (cwd / 'entry.py').write_text('fixed = True', encoding='utf-8')
    assert (source / 'entry.py').read_bytes() == old_bytes
    assert engine._repair_source_scope(job, workspace, cwd, cwd / 'entry.py') == ''


def test_generic_tuple_contract_is_detected_across_modules(tmp_path):
    (tmp_path / 'transform.py').write_text('def encode(x):\n    return x, 1, 2\n', encoding='utf-8')
    entry = tmp_path / 'entry.py'
    entry.write_text('from transform import encode\nx = encode(data)\nx.toarray()\n', encoding='utf-8')
    assert len(interface_issues(tmp_path)) == 1
    assert 'transform.py:encode' in provider_context(tmp_path, entry)
    assert interface_issues(tmp_path, {'entry.py': 'from transform import encode\nx = encode(data)\ncoords, vals, shape = x\n'}) == []
    assert interface_issues(tmp_path, {'entry.py': 'from transform import encode\nx = encode(data)\nif isinstance(x, tuple):\n    x = sparse_matrix(x)\nx.toarray()\n'}) == []


def test_generic_nested_list_or_tuple_converter_contract(tmp_path):
    (tmp_path / 'features.py').write_text('def pack(x):\n'
        '    def convert(y):\n        return y, 1, 2\n'
        '    if isinstance(x, list):\n        for i in range(len(x)):\n            x[i] = convert(x[i])\n'
        '    else:\n        x = convert(x)\n    return x\n'
        'def normalize(x):\n    return pack(x)\n', encoding='utf-8')
    (tmp_path / 'entry.py').write_text('from features import *\nx = normalize(data)\nx = Tensor(x.toarray())\n', encoding='utf-8')
    assert len(interface_issues(tmp_path)) == 1


def test_joint_search_includes_every_supported_parameter(tmp_path, monkeypatch):
    job, _, _ = job_fixture(tmp_path, monkeypatch)
    plan = tuning.build_tuning_plans(job)[0]
    assert len(plan['trials'][-1]) == len(plan['space']) == 3


def test_local_wheel_report_requires_owned_file_and_matching_hash(tmp_path):
    wheel = tmp_path / 'torch.whl'
    wheel.write_bytes(b'offline test, not a real package')
    row = {'metadata': {'name': 'torch', 'version': '1.13.1+cu117'},
           'download_info': {'url': wheel.as_uri(), 'archive_info': {'hashes': {'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}}}}
    report = {'install': [row]}
    assert validate_pip_report(report, {}, {'packages': ['torch==1.13.1+cu117']}, tmp_path) == [str(wheel)]
    row['download_info']['archive_info']['hashes']['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='SHA256'):
        validate_pip_report(report, {}, {'packages': ['torch==1.13.1+cu117']}, tmp_path)


def test_metadata_probe_rejects_non_preflight_invocation():
    from app.services.pip_metadata_probe import main
    assert main(['torch']) == 86


@pytest.mark.parametrize('status,content_range,expected', [(200, '', 86), (206, 'bytes 0-99/1000', 0),
                                                         (206, 'bytes 0-999/1000', 86)])
def test_metadata_probe_never_reads_full_response_when_range_is_ignored(monkeypatch, status, content_range, expected):
    import importlib
    from pip._internal.operations.prepare import RequirementPreparer
    from pip._internal.network import download
    Downloader = download.Downloader
    BatchDownloader = getattr(download, 'BatchDownloader', None)
    from pip._internal.network.lazy_wheel import LazyZipOverHTTP
    from app.services.pip_metadata_probe import main
    closed = []
    response = SimpleNamespace(status_code=status, headers={'Content-Range': content_range}, close=lambda: closed.append(True))
    monkeypatch.setattr(LazyZipOverHTTP, '_stream_response', lambda *args: response)
    monkeypatch.setattr(RequirementPreparer, 'prepare_linked_requirements_more', RequirementPreparer.prepare_linked_requirements_more)
    monkeypatch.setattr(Downloader, '__call__', Downloader.__call__)
    if BatchDownloader is not None:
        monkeypatch.setattr(BatchDownloader, '__call__', BatchDownloader.__call__)
    if hasattr(Downloader, 'batch'):
        monkeypatch.setattr(Downloader, 'batch', Downloader.batch)
    module = importlib.import_module('pip._internal.cli.main')
    def fake_pip(argv):
        assert '--use-feature=fast-deps' in argv
        LazyZipOverHTTP._stream_response(None, 0, 99)
        return 0
    monkeypatch.setattr(module, 'main', fake_pip)
    assert main(['--dry-run', '--report', 'offline.json', '--only-binary=:all:']) == expected
    assert bool(closed) == (expected == 86)


def test_repeat_approval_does_not_queue_another_worker(tmp_path, monkeypatch):
    job, _, _ = job_fixture(tmp_path, monkeypatch)
    approve(job, monkeypatch, ['graph'])
    tasks = BackgroundTasks()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(reproductions.start_tuning_session(job.project_id, job.id,
            reproductions.TuningApproval(datasets=['graph'], review_tokens={}), tasks, Session(job)))
    assert exc.value.status_code == 409 and not tasks.tasks


def test_nested_logs_use_real_subprocess_without_training(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'get_settings', lambda: SimpleNamespace(paper_projects_dir=tmp_path,
        hf_endpoint='https://huggingface.co', reproduction_log_chars=5000))
    log = tmp_path / 'paper' / 'runs' / 'run' / 'telemetry' / 'tuning' / 'session' / 'offline.log'
    token = engine.CURRENT_JOB_ID.set('run')
    try:
        code, output = asyncio.run(engine._run_process([sys.executable, '-c', "print('offline process check')"],
            tmp_path, 10, output_path=log))
        assert code == 0 and 'offline process check' in output and log.is_file()
        with pytest.raises(ValueError, match='日志路径'):
            asyncio.run(engine._run_process([sys.executable, '-c', 'pass'], tmp_path, 10,
                output_path=tmp_path / 'paper' / 'runs' / 'other-run' / 'telemetry' / 'bad.log'))
    finally:
        engine.CURRENT_JOB_ID.reset(token)


def test_many_search_trials_do_not_evict_baseline_curves(tmp_path):
    from app.services.telemetry import read_job_telemetry
    root = tmp_path / 'paper' / 'runs' / 'run' / 'telemetry'
    root.mkdir(parents=True)
    for dataset in ['graph', 'tabular', 'text', 'regression']:
        path = root / f'{dataset}-original.log'
        path.write_text('Epoch 1 val_acc=0.7', encoding='utf-8')
        os.utime(path, (1, 1))
    search = root / 'tuning' / 'session'
    search.mkdir(parents=True)
    for trial in range(60):
        (search / f'graph-tuning-{trial + 1}.log').write_text('Epoch 1 val_acc=0.8', encoding='utf-8')
    series = read_job_telemetry(tmp_path / 'paper', 'run')['series']
    assert {r['dataset'] for r in series if r['attempt'] == 'original'} == {'graph', 'tabular', 'text', 'regression'}


def test_sqlite_search_results_are_persisted_without_extra_job(tmp_path, monkeypatch):
    import shutil
    from uuid import uuid4
    from sqlalchemy import select, func
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from app.database import Base
    from app.models import Project, ReproductionJob
    job, workspace, _ = job_fixture(tmp_path, monkeypatch)
    identity = str(uuid4())
    shutil.copytree(workspace / 'runs' / 'run', workspace / 'runs' / identity)
    job.id = identity
    job.project_id = '12345678-1234-1234-1234-123456789abc'
    original = copy.deepcopy(job.result_json['dataset_results'])
    async def process(*args, output_path, **kwargs):
        output_path.write_text('val_accuracy=0.9\nTest accuracy=0.1', encoding='utf-8')
        return 0, 'val_accuracy=0.9\nTest accuracy=0.1'
    monkeypatch.setattr(engine, '_run_process', process)
    monkeypatch.setattr(engine, '_write_reproduction_report', lambda *args: None)
    async def exercise():
        database = create_async_engine('sqlite+aiosqlite:///' + str(tmp_path / 'offline.sqlite'))
        sessions = async_sessionmaker(database, expire_on_commit=False)
        monkeypatch.setattr(tuning_sessions, 'AsyncSessionLocal', sessions)
        async with database.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with sessions() as db:
            db.add(Project(id=job.project_id, name='Offline regression'))
            stored = ReproductionJob(**{key: value for key, value in vars(job).items()
                if key in ReproductionJob.__table__.columns})
            db.add(stored)
            await db.commit()
            plans = tuning_sessions.reviewed_plans(stored)
            tasks = BackgroundTasks()
            response = await reproductions.start_tuning_session(stored.project_id, stored.id,
                reproductions.TuningApproval(datasets=['graph'], review_tokens={'graph': plans[0]['review_token']}), tasks, db)
        await tuning_sessions.execute_tuning_session(job.id, response['session_id'])
        async with sessions() as db:
            stored = await db.get(ReproductionJob, job.id)
            assert stored.result_json['dataset_results'] == original
            assert len(stored.result_json['tuning_attempts']) == 6
            assert stored.result_json['tuning_sessions'][0]['status'] == 'completed'
            assert (await db.scalar(select(func.count()).select_from(ReproductionJob))) == 1
        await database.dispose()
    asyncio.run(exercise())
