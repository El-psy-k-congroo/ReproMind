"""Offline regressions. No remote model, package install, or author training."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from app.api import copilot, experiment_branches as branches
from app.models import ReproductionStatus
from app.services import reproduction_engine as engine
from app.services.gpu_preparation import requested_framework, source_framework
from app.services.tuning import command_fingerprint, recovery_datasets, tuning_gate


@pytest.mark.parametrize('verdict,approved,changed,expected', [
    ('matched', True, False, '不需要调参'),
    ('deviated', False, False, '未获批准'),
    ('deviated', True, True, '旧调参批准'),
    ('deviated', True, False, ''),
])
def test_tuning_is_recovery_not_blanket_search(verdict, approved, changed, expected):
    policy = {'enabled': approved, 'approved_at': 'now', 'recovery_datasets': ['NELL']}
    plan = {'fingerprint': 'before', 'trials': [{'lr': .1}]}
    row = {'dataset': 'NELL', 'return_code': 0, 'verdict': verdict}
    assert expected in tuning_gate(policy, row, plan, 'after' if changed else 'before')


def test_initial_approval_cannot_search_and_errors_require_repair():
    policy = {'enabled': True, 'approved_at': 'now', 'recovery_datasets': []}
    row = {'dataset': 'NELL', 'return_code': 0, 'verdict': 'deviated'}
    assert '另行批准' in tuning_gate(policy, row, {}, 'fingerprint')
    row['return_code'] = 1
    assert '先纠错' in tuning_gate(policy, row, {}, 'fingerprint')
    assert recovery_datasets([row], ['NELL']) == []


def test_only_reviewed_deviated_baselines_are_recovery_candidates():
    rows = [{'dataset': name, 'return_code': code, 'verdict': verdict}
            for name, code, verdict in [('Cora', 0, 'matched'), ('NELL', 0, 'deviated'),
                                        ('broken', 1, 'failed'), ('unknown', 0, 'configuration_unverified')]]
    assert recovery_datasets(rows, ['Cora', 'NELL', 'broken', 'unknown']) == ['NELL']


def test_baseline_claim_survives_tuned_result_and_context_is_not_refutation():
    row = {'dataset': 'Cora', 'return_code': 0, 'metric_key': 'accuracy', 'target': .8,
           'metrics': {'accuracy': .84}, 'baseline_metrics': {'accuracy': .81},
           'baseline_verdict': 'matched', 'verdict': 'tuned_result', 'tuning_selected': True}
    claims = engine._claim_results({'claims': ['Scales linearly']}, [row])
    assert claims[0]['supported'] and '0.8100' in claims[0]['evidence']
    assert not claims[1]['supported'] and '非作者原配置' in claims[1]['claim']
    assert claims[2]['status'] == 'not_evaluated'


@pytest.mark.parametrize('text', [
    '请把 TensorFlow 迁移到 PyTorch，然后在 GPU 上运行',
    '你可不可以用pytorch复现这篇论文？注意：是gpu版本的pytorch',
])
def test_explicit_framework_change_routes_to_code_preview(text):
    assert requested_framework(text) == 'pytorch'
    assert copilot._code_action_requested(text)


def test_capability_question_does_not_authorize_changes():
    assert not copilot._code_action_requested('当前代码可以迁移到pytorch吗？')


def test_source_is_readable_before_execution_command_exists(tmp_path, monkeypatch):
    root = tmp_path / 'projects'
    workspace = root / 'gcn-12345678'
    source = workspace / 'sources' / 'gcn'
    source.mkdir(parents=True)
    monkeypatch.setattr(branches, 'get_settings', lambda: SimpleNamespace(paper_projects_dir=root))
    job = SimpleNamespace(project_id='12345678-rest', workspace_path=str(workspace),
                          execution_command={}, code_spec={'source_path': str(source)})
    assert branches._scope(job) == (workspace, source)


def test_migration_candidates_include_related_modules_not_just_entry(tmp_path):
    for name in ['train', 'models', 'layers', 'metrics', 'utils', 'inits']:
        (tmp_path / f'{name}.py').write_text('import tensorflow as tf\n', encoding='utf-8')
    (tmp_path / '__init__.py').write_text('', encoding='utf-8')
    paths = branches._candidate_code_files(tmp_path, '请用pytorch复现并在gpu运行')
    assert {p.stem for p in paths} >= {'train', 'models', 'layers', 'metrics', 'utils', 'inits'}


def test_complete_file_rewrite_compiles_but_invalid_code_is_rejected():
    old = '# original\n' * 800
    assert branches._apply_text(old, old, 'import torch\n', 'layers.py') == 'import torch\n'
    with pytest.raises(HTTPException):
        branches._apply_text(old, old, 'def invalid(', 'layers.py')


def test_existing_revision_run_is_idempotent_even_when_active(monkeypatch, tmp_path):
    base = SimpleNamespace(id='base')
    existing = SimpleNamespace(id='created', status=ReproductionStatus.executing, result_json={'revision': {}})
    async def get_job(db, project, run): return base if run == 'base' else existing
    async def idle(*args): pytest.fail('Existing run is returned before idle checks')
    monkeypatch.setattr(branches, '_job', get_job)
    monkeypatch.setattr(branches, '_scope', lambda job: (tmp_path, tmp_path))
    monkeypatch.setattr(branches, '_proposal', lambda *args: (tmp_path, {'base_run_id': 'base', 'applied_at': 'now', 'run_id': 'created'}))
    monkeypatch.setattr(branches, '_ensure_idle', idle)
    tasks = BackgroundTasks()
    result = asyncio.run(branches.run_revision('project', 'base', branches.RevisionRunRequest(proposal_id='proposal'), tasks, None))
    assert result['run_id'] == 'created' and result['already_created'] and not tasks.tasks


def test_partial_applied_migration_is_blocked_before_install_or_copy(monkeypatch, tmp_path):
    base = SimpleNamespace(id='base')
    (tmp_path / 'train.py').write_text('import torch\n', encoding='utf-8')
    (tmp_path / 'models.py').write_text('import tensorflow as tf\n', encoding='utf-8')
    async def get_job(*args): return base
    async def idle(*args): pass
    monkeypatch.setattr(branches, '_job', get_job)
    monkeypatch.setattr(branches, '_ensure_idle', idle)
    monkeypatch.setattr(branches, '_scope', lambda job: (tmp_path, tmp_path))
    monkeypatch.setattr(branches, '_proposal', lambda *args: (tmp_path, {'base_run_id': 'base', 'applied_at': 'now', 'instruction': '请用pytorch复现并用gpu'}))
    with pytest.raises(HTTPException, match='多文件迁移'):
        asyncio.run(branches.run_revision('p', 'base', branches.RevisionRunRequest(proposal_id='partial'), BackgroundTasks(), None))
    assert source_framework({'train.py': 'import torch', 'models.py': 'import tensorflow'}) == 'mixed'
    assert '混用' in branches._gpu_preview_issues(tmp_path, {})[0]


def test_pending_old_migration_is_read_only_and_explicitly_not_runnable(monkeypatch, tmp_path):
    base = SimpleNamespace(id='base')
    directory = tmp_path / 'runs' / 'base' / 'proposals'
    directory.mkdir(parents=True)
    proposal = {'instruction': '请用pytorch复现并在gpu运行', 'applied_at': 'now', 'files': ['train.py']}
    (directory / 'proposal.json').write_text(json.dumps(proposal), encoding='utf-8')
    (tmp_path / 'train.py').write_text('import torch\n', encoding='utf-8')
    (tmp_path / 'models.py').write_text('import tensorflow\n', encoding='utf-8')
    async def get_job(*args): return base
    monkeypatch.setattr(branches, '_job', get_job)
    monkeypatch.setattr(branches, '_scope', lambda job: (tmp_path, tmp_path))
    result = asyncio.run(branches.pending_revision('p', 'base', None))
    assert result['runnable'] is False and '旧提案' in result['validation_error']
    assert json.loads((directory / 'proposal.json').read_text()) == proposal


def test_all_baselines_precede_recovery_and_matched_dataset_never_tunes(tmp_path, monkeypatch):
    workspace = tmp_path / 'project'
    source = workspace / 'source'
    source.mkdir(parents=True)
    (source / 'run.py').write_text("print('fixture')\n", encoding='utf-8')
    commands = [{'dataset': d, 'argv': [sys.executable, 'run.py', '--dataset', d], 'cwd': str(source)} for d in ['Cora', 'NELL']]
    plans = [{'dataset': c['dataset'], 'fingerprint': command_fingerprint(c, source), 'trials': [{'lr': .1}],
              'max_trials': 1} for c in commands]
    job = SimpleNamespace(id='offline-flow', project_id='p', status=ReproductionStatus.executing,
        current_stage='executing', progress=90, message='', events=[], logs='',
        workspace_path=str(workspace), environment_spec={}, dataset_spec={}, code_spec={},
        paper_spec={'target_results': {'Cora': {'accuracy': .8}, 'NELL': {'accuracy': .66}}},
        execution_mode='full', model_id='test', repository_url='repo', completed_at=None,
        execution_command={'commands': commands, 'pythonpath': str(source), 'cwd': str(source)},
        result_json={'tuning_policy': {'enabled': True, 'approved_at': 'now', 'recovery_datasets': ['NELL']}, 'tuning_plans': plans})
    calls = []
    class Session:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, *args): return job
        async def commit(self): pass
    async def ensure(*args): return Path(sys.executable)
    async def empty(*args): return []
    async def advance(session, row, stage, progress, message, **kwargs):
        row.current_stage = stage
        row.events.append({'stage': stage})
    async def process(argv, cwd, timeout, env, *, output_path=None):
        dataset = argv[argv.index('--dataset') + 1]
        is_tuning = '--lr' in argv
        calls.append((dataset, is_tuning))
        return 0, f"val_acc={0.8 if is_tuning else 0.7}\nTest set results: accuracy={0.81 if dataset == 'Cora' else 0.5}"
    monkeypatch.setattr(engine, 'AsyncSessionLocal', Session)
    monkeypatch.setattr(engine, '_ensure_environment', ensure)
    monkeypatch.setattr(engine, '_install_uploaded_datasets', empty)
    monkeypatch.setattr(engine, '_download_small_datasets', empty)
    monkeypatch.setattr(engine, '_align_dataset_commands', lambda job: [])
    monkeypatch.setattr(engine, '_preflight_source', lambda *args, **kwargs: {'issues': [], 'created_output_dirs': []})
    monkeypatch.setattr(engine, '_run_process', process)
    monkeypatch.setattr(engine, '_advance', advance)
    monkeypatch.setattr(engine, '_write_reproduction_report', lambda *args: None)
    asyncio.run(engine.execute_reproduction(job.id))
    assert calls == [('Cora', False), ('NELL', False), ('NELL', True)]
    assert job.result_json['dataset_results'][0]['verdict'] == 'matched'
    assert job.result_json['dataset_results'][1]['baseline_verdict'] == 'deviated'
    assert len(job.result_json['tuning_attempts']) == 1


def test_resume_reuses_completed_trials_instead_of_training_twice(tmp_path, monkeypatch):
    (tmp_path / 'nell-tuning-1.log').write_text('val_acc=0.8\nTest set results: accuracy=0.5', encoding='utf-8')
    prior = {'dataset': 'NELL', 'trial': 1, 'parameters': {'lr': .1}, 'return_code': 0,
             'validation_score': .8, 'log': 'nell-tuning-1.log'}
    async def forbidden(*args, **kwargs): pytest.fail('Completed trial must not run twice')
    monkeypatch.setattr(engine, '_run_process', forbidden)
    plan = {'dataset': 'NELL', 'fingerprint': 'approved', 'trials': [{'lr': .1}], 'max_trials': 1}
    rows, output, note = asyncio.run(engine._run_tuning(plan, [], tmp_path, {}, 100, tmp_path,
                                'val_acc=0.7', 'accuracy', existing_records=[prior]))
    assert len(rows) == 1 and rows[0]['selected'] and output
