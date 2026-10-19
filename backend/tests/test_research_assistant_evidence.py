import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api import copilot, reproductions
from fastapi import BackgroundTasks
from app.models import ArtifactKind, ReproductionStatus
from app.services import runtime_evidence, tuning, reproduction_engine as engine
from app.services.telemetry import parse_experiment_log
from app.services.dataset_discovery import inspect_local_datasets, PLANETOID_PARTS


def fixture_job(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'train.py').write_text(
        "flags.DEFINE_float('learning_rate', 0.01, 'rate')\n"
        "flags.DEFINE_float('dropout', 0.5, 'dropout')\n"
        "flags.DEFINE_float('weight_decay', 5e-4, 'regularization')\n"
        "flags.DEFINE_integer('hidden1', 16, 'width')\n"
        "flags.DEFINE_integer('epochs', 200, 'epochs')\n", encoding='utf-8')
    (source / 'models.py').write_text('class GCN:\n    def loss(self):\n        return cross_entropy\n', encoding='utf-8')
    (source / 'README.md').write_text('Cora dataset. Two graph convolution layers.', encoding='utf-8')
    command = {'dataset': 'Cora', 'argv': ['python', '-m', 'train', '--learning_rate', '0.02'],
               'cwd': str(source), 'pythonpath': str(source)}
    return SimpleNamespace(id='run-a', project_id='project-a', workspace_path=str(tmp_path),
        status=ReproductionStatus.completed, current_stage='completed', message='Complete', error_message='',
        paper_spec={'datasets': ['Cora'], 'metrics': ['Accuracy']},
        code_spec={'source_path': str(source)}, dataset_spec={}, environment_spec={'hardware_plan': {'framework': 'tensorflow'}},
        execution_command={'commands': [command], 'pythonpath': str(source), 'entrypoint_candidates': ['train.py']},
        result_json={'dataset_results': [{'dataset': 'Cora', 'metrics': {'accuracy': .81}}]},
        source_mode='paper_repository', repository_url='https://github.com/author/gcn', events=[], logs='')


def test_network_closure_keeps_short_class_names_loss_aliases_and_decode(tmp_path,monkeypatch):
    job=fixture_job(tmp_path);source=Path(job.code_spec['source_path'])
    job.code_spec['entrypoints']=['train.py']
    (source/'train.py').write_text('from network import CAE\nimport torch\nlf=torch.nn.L1Loss()\noptim=torch.optim.Adam(model.parameters())\n')
    (source/'network.py').write_text('class CAE:\n    def __init__(self):\n        self.encoder = Encoder()\n        self.decoder = Decoder()\n    def encode(self,x):\n        return self.encoder(x)\n    def decode(self,z):\n        return self.decoder(z)\n    def forward(self,x):\n        return self.decode(self.encode(x))\n')
    (source/'unrelated_model.py').write_text('class Other:\n    def forward(self,x):\n        return x + 999\n')
    class Session:
        async def scalars(self,*_):return SimpleNamespace(all=lambda:[job])
        async def scalar(self,*_):return job
    monkeypatch.setattr(copilot,'_safe_workspace',lambda _:tmp_path)
    project=SimpleNamespace(id=job.project_id,name='Example',description='',artifacts=[])
    question='解释当前模型的编码器/解码器、网络 forward、损失函数和优化器，不运行。'
    context=asyncio.run(copilot._context(Session(),project,question,job.id))
    packet=json.loads(copilot._bounded_prompt(question,context,[],''))['selected_source_evidence_not_execution_proof']
    assert {row['symbol'] for row in packet['named_implementations']}=={'CAE.encode','CAE.decode','CAE.forward'}
    assert any('L1Loss' in row['code'] for row in packet['loss_assignments'])
    assert any('optim.Adam' in row['code'] for row in packet['optimizer_assignments'])
    assert '999' not in json.dumps(context['source_relevant_excerpts'])


def test_execution_selected_network_outranks_other_imports_and_keeps_default_branch(tmp_path,monkeypatch):
    job=fixture_job(tmp_path);source=Path(job.code_spec['source_path'])
    job.code_spec['entrypoints']=['train.py']
    job.execution_command['commands'][0]['argv']=['python','train.py','--arch','chosen']
    (source/'train.py').write_text('from alternative import Other\nfrom chosen import Chosen\n')
    (source/'alternative.py').write_text('\n'.join(f'class Other{i}:\n    def forward(self,x):\n        return x + {i}' for i in range(5)))
    (source/'chosen.py').write_text('class Chosen:\n    def __init__(self, residual=False):\n        self.residual=residual\n    def forward(self,x):\n        if self.residual:\n            x=x+1\n        return x\n')
    class Session:
        async def scalars(self,*_):return SimpleNamespace(all=lambda:[job])
        async def scalar(self,*_):return job
    monkeypatch.setattr(copilot,'_safe_workspace',lambda _:tmp_path)
    project=SimpleNamespace(id=job.project_id,name='Example',description='',artifacts=[])
    context=asyncio.run(copilot._context(Session(),project,'当前网络的forward机制是什么，不运行。',job.id))
    packet=context['source_task_evidence']['named_implementations'][0]
    assert packet['symbol']=='Chosen.forward'
    assert 'residual=False' in packet['constructor_signature_from_static_ast']
    assert 'if self.residual' in packet['code']


def test_question_control_flag_retrieves_its_calculation_block():
    from app.services.source_calculations import calculation_packets
    body='''def train(x):
    if args.patchmix_prob > 0:
        weight = draw()
        left, right = region(weight)
        x[left:right] = alternate[left:right]
        weight = 1 - (right-left)/len(x)
        loss = weight * loss_a + (1-weight) * loss_b
    return loss
'''
    packets=calculation_packets(body,'解释PatchMix的裁剪和面积校正','train.py')
    assert packets and 'weight = 1 - (right-left)/len(x)' in packets[0]['code']
    assert 'if args.patchmix_prob' in packets[0]['code']


def test_search_space_reads_real_flags_and_current_argv_not_fixed_rates(tmp_path):
    job = fixture_job(tmp_path)
    plan = tuning.build_tuning_plans(job)[0]
    space = {row['parameter']: row for row in plan['space']}
    assert set(space) == {'learning_rate', 'dropout', 'weight_decay', 'hidden1'}
    assert space['learning_rate']['baseline'] == .02
    assert space['learning_rate']['values'] == [.01, .04]
    assert len(plan['trials']) <= 6 and any(len(row) > 1 for row in plan['trials'])
    assert 'epochs' not in space and plan['requires_approval']
    changed = dict(job.execution_command['commands'][0], argv=['python', '-m', 'train'])
    assert tuning.command_fingerprint(changed, Path(job.code_spec['source_path'])) != plan['fingerprint']


def test_variable_mention_is_not_a_supported_cli_flag():
    assert tuning.numeric_flags('learning_rate = .01\ndropout = .5') == {}
    assert tuning.numeric_flags("parser.add_argument('--lr', type=float, default=.02)") == {'lr': .02}


def test_parameter_override_does_not_append_duplicate_flags():
    assert tuning.parameter_argv(['python', '--lr', '.01', '--dropout=.5'], {'lr': .02, 'dropout': .4}) == [
        'python', '--lr', '0.02', '--dropout=0.4']


def test_selection_ignores_test_score_and_keeps_only_one_final_winner():
    records = [{'return_code': 0, 'validation_score': .81, 'test_accuracy': .5},
               {'return_code': 0, 'validation_score': .79, 'test_accuracy': 1.0},
               {'return_code': 1, 'validation_score': .99, 'test_accuracy': 1.0}]
    assert tuning.best_trial(records, .8, 'accuracy') == 0
    assert tuning.best_trial(records, .85, 'accuracy') is None
    assert tuning.best_trial([{'return_code': 0, 'validation_score': .02}], .1, 'loss') == 0


@pytest.mark.parametrize('text,metric,expected', [
    ('val_acc=0.0', 'accuracy', 0), ('Validation Accuracy: 0.8', 'accuracy', .8),
    ('val_rmse=0.2', 'rmse', .2), ('test_accuracy=0.99', 'accuracy', None),
    ('val_loss=1e-3', 'loss', .001)])
def test_validation_parser_never_uses_test_metrics(text, metric, expected):
    assert tuning.validation_score(text, metric) == expected


@pytest.mark.parametrize('baseline', ['Test accuracy=0.99', 'Test accuracy=0.01'])
def test_no_validation_means_no_trials_regardless_of_test_result(tmp_path, monkeypatch, baseline):
    async def forbidden(*args, **kwargs): raise AssertionError('Must not train without validation evidence')
    monkeypatch.setattr(engine, '_run_process', forbidden)
    records, output, reason = asyncio.run(engine._run_tuning({'fingerprint': 'x'}, [], tmp_path, {}, 100,
                                                           tmp_path, baseline, 'accuracy'))
    assert not records and output is None and '验证集' in reason


def test_trials_are_bounded_fail_closed_and_report_exactly_one_winner(tmp_path, monkeypatch):
    source=tmp_path/'author';source.mkdir()
    counter = []
    async def process(argv, *args, **kwargs):
        counter.append(argv)
        value = '.8' if len(counter) == 1 else '.85'
        return 0, f'val_acc={value}\nTest set results: accuracy=0.6'
    monkeypatch.setattr(engine, '_run_process', process)
    plan = {'dataset': 'Cora', 'fingerprint': 'x', 'max_trials': 2,
            'trials': [{'lr': .1}, {'lr': .2}, {'lr': .3}]}
    records, output, _ = asyncio.run(engine._run_tuning(plan, ['python'], source, {'PYTHONPATH':str(source)}, 100,
                                                       tmp_path, 'val_acc=.75', 'accuracy'))
    # Parser intentionally requires digits before decimal; use author-style numbers below.
    assert not counter
    records, output, _ = asyncio.run(engine._run_tuning(plan, ['python'], source, {'PYTHONPATH':str(source)}, 100,
                                                       tmp_path, 'val_acc=0.75', 'accuracy'))
    assert len(counter) == 1  # malformed validation in the first trial stops further search
    assert records[0]['validation_score'] is None and output is None


def test_tuned_result_cannot_claim_exact_paper_reproduction():
    claims = engine._claim_results({'target_results': {'Cora': .8}}, [
        {'dataset': 'Cora', 'metrics': {'accuracy': .8}, 'metric_key': 'accuracy',
         'target': .8, 'verdict': 'tuned_result', 'tuning_selected': True}])
    assert not claims[0]['supported'] and claims[0]['status'] == 'unverified'
    assert '原超参数' in claims[0]['evidence']


def test_log_gpu_initialization_is_not_training_proof(tmp_path):
    logs = [{'log': 'cora-original.log', 'head': 'Created TensorFlow device /device:GPU:0', 'tail': 'epoch 200'}]
    assert runtime_evidence.device_log_evidence(logs)[0]['kind'] == 'gpu_initialized_not_training_proof'
    assert not runtime_evidence.device_log_evidence([{'log': 'log', 'head': 'NVIDIA RTX 3090 Ti GPU available: True'}])
    logs[0]['tail'] = 'Training device: cuda:0'
    assert runtime_evidence.device_log_evidence(logs)[-1]['kind'] == 'training_device_gpu'


def test_tensorflow_probe_uses_legacy_compatible_api_not_torch(tmp_path, monkeypatch):
    python = tmp_path / 'python.exe'
    python.touch()
    def fake_run(argv, **kwargs):
        assert 'tensorflow' in argv[-1] and 'device_lib.list_local_devices' in argv[-1] and 'torch' not in argv[-1]
        assert 'allow_growth=True' in argv[-1]
        return SimpleNamespace(returncode=0, stdout=json.dumps({'version': '1.15.5', 'cuda_build': False,
                                                              'cuda_available': False, 'device_count': 0}), stderr='')
    monkeypatch.setattr(runtime_evidence.subprocess, 'run', fake_run)
    result = asyncio.run(runtime_evidence.probe_framework(python, 'tensorflow'))
    assert result['checked'] and not result['cuda_available']


@pytest.mark.parametrize('question', ['当前的实验环境是在gpu上进行的吗', '模型主要网络结构是什么？',
    '损失函数怎么定义？', '调参为什么只调学习率？', '数据集文件在哪里？', '实验指标和论文有什么差别？'])
def test_common_research_questions_retrieve_current_source_logs_and_results(tmp_path, monkeypatch, question):
    job = fixture_job(tmp_path)
    log_dir = tmp_path / 'runs' / job.id / 'telemetry'
    log_dir.mkdir(parents=True)
    (log_dir / 'cora-original.log').write_text('Training device: cpu\nEpoch 1 val_acc=0.8', encoding='utf-8')
    class Session:
        async def scalars(self, *args): return SimpleNamespace(all=lambda: [job])
        async def scalar(self, *args): return job
    paper = SimpleNamespace(kind=ArtifactKind.paper, filename='paper.pdf',
                            content_text='The graph convolution model uses cross_entropy loss on Cora.')
    project = SimpleNamespace(id=job.project_id, name='GCN', description='', artifacts=[paper])
    monkeypatch.setattr(copilot, '_safe_workspace', lambda job: tmp_path)
    async def readiness(*args): return {'runtime': {'checked': True, 'cuda_available': False}, 'framework': 'tensorflow'}
    monkeypatch.setattr(copilot, 'inspect_run_readiness', readiness)
    context = asyncio.run(copilot._context(Session(), project, question, job.id))
    assert context['job']['run_id'] == 'run-a'
    if 'gpu' in question:
        assert copilot._gpu_capability_question(question)
        assert context['gpu_readiness']['framework'] == 'tensorflow'
        assert 'Training device: cpu' in context['execution_log_tails'][0]['head']
    if '网络' in question or '损失' in question:
        assert any('models.py' in row['file'] for row in context['source_primary_outline'])
        assert context['source_relevant_excerpts']
    if '调参' in question:
        assert len(context['proposed_tuning_plans_not_executed'][0]['space']) == 4


def test_large_context_is_valid_json_preserves_question_and_current_run():
    context = {'job': {'run_id': 'selected-run', 'status': 'completed', 'source': 'x'*20000},
               'paper_excerpt': 'large'*5000, 'source_relevant_excerpts': [{'file': 'models.py', 'excerpts': ['x'*8000]}]*20}
    prompt = copilot._bounded_prompt('解释当前模型', context, [], '')
    assert len(prompt) <= 15000
    parsed = json.loads(prompt)
    assert parsed['question'] == '解释当前模型' and parsed['project_evidence']['job']['run_id'] == 'selected-run'


def test_new_tuning_logs_have_separate_curve_identity(tmp_path):
    log = tmp_path / 'cora-tuning-2.log'
    log.write_text('Epoch 1 train_loss=0.5 val_acc=0.7', encoding='utf-8')
    assert parse_experiment_log(log)['attempt'] == 'tuning-2'


def test_successful_search_keeps_one_final_winner_and_test_does_not_choose_it(tmp_path, monkeypatch):
    source=tmp_path/'author';source.mkdir()
    values = iter(['val_acc=0.81\nTest accuracy=0.99', 'val_acc=0.85\nTest accuracy=0.5'])
    persisted = []
    async def process(*args, **kwargs): return 0, next(values)
    async def record(value): persisted.append(dict(value))
    monkeypatch.setattr(engine, '_run_process', process)
    plan = {'dataset': 'Cora', 'fingerprint': 'x', 'max_trials': 2, 'trials': [{'lr': .1}, {'lr': .2}]}
    records, output, _ = asyncio.run(engine._run_tuning(plan, ['python'], source, {'PYTHONPATH':str(source)}, 100, tmp_path,
                                                       'val_acc=0.8', 'accuracy', on_result=record))
    assert len(persisted) == 2
    assert [row['selected'] for row in records] == [False, True]
    assert 'Test accuracy=0.5' in output


def test_trial_failure_stops_search_without_repairing_shared_source(tmp_path, monkeypatch):
    source=tmp_path/'author';source.mkdir()
    calls = []
    async def process(*args, **kwargs): calls.append(1); return 1, 'Out of memory'
    monkeypatch.setattr(engine, '_run_process', process)
    plan = {'dataset': 'Cora', 'fingerprint': 'x', 'max_trials': 6, 'trials': [{'lr': .1}]*6}
    records, output, _ = asyncio.run(engine._run_tuning(plan, ['python'], source, {'PYTHONPATH':str(source)}, 100,
                                                       tmp_path, 'val_acc=0.8', 'accuracy'))
    assert len(calls) == 1 and records[0]['return_code'] == 1 and output is None


@pytest.mark.parametrize('budget', [0, 6])
def test_approval_tuning_is_explicit_and_preserves_old_results(tmp_path, budget):
    job = fixture_job(tmp_path)
    job.status = ReproductionStatus.awaiting_approval
    previous = job.result_json['dataset_results']
    class Session:
        async def scalar(self, *args): return job
        async def commit(self): pass
        async def refresh(self, value): pass
    tasks = BackgroundTasks()
    payload = reproductions.ApprovalOptions(tuning_budget=budget) if budget else None
    asyncio.run(reproductions.approve_reproduction(job.project_id, job.id, tasks, Session(), payload))
    assert job.result_json['tuning_policy']['enabled'] == bool(budget)
    assert job.result_json['tuning_policy']['test_feedback_allowed'] is False
    assert job.result_json['dataset_results'] == previous
    assert bool(job.result_json['tuning_plans']) == bool(budget)
    assert len(tasks.tasks) == 1  # queued only, no training in this test


def test_tuning_proposal_endpoint_does_not_approve_or_mutate(tmp_path, monkeypatch):
    workspace = tmp_path / 'test-12345678'
    workspace.mkdir()
    job = fixture_job(workspace)
    from app.services import tuning_sessions
    job.project_id = '12345678-test'
    monkeypatch.setattr(tuning_sessions, 'get_settings', lambda: SimpleNamespace(paper_projects_dir=tmp_path))
    previous = json.dumps(job.result_json)
    class Session:
        async def scalar(self, *args): return job
    result = asyncio.run(reproductions.tuning_plan(job.project_id, job.id, Session()))
    assert result['executed'] is False and result['requires_approval']
    assert json.dumps(job.result_json) == previous and job.status == ReproductionStatus.completed


def test_source_rank_prefers_loss_implementation_over_import():
    source = 'import cross_entropy\n' + '\n' * 30 + 'def loss(self):\n    return cross_entropy(self.logits, self.labels)\n'
    excerpts = copilot._source_excerpts(source, ['loss', 'cross_entropy'], limit=1)
    assert 'def loss' in excerpts[0] and 'return cross_entropy' in excerpts[0]


def test_fact_budget_never_silently_turns_two_trials_and_four_parameters_into_one():
    trials = [{'dataset': 'NELL', 'value': .005}, {'dataset': 'NELL', 'value': .02}]
    plans = [{'dataset': 'NELL', 'space': [{'parameter': key, 'values': [.1, .2]}
        for key in ['lr', 'dropout', 'weight_decay', 'hidden1']], 'trials': []}]
    context = {'job': {'run_id': 'current', 'tuning_attempts': trials, 'dataset_results_count': 4,
        'paper_spec': {'huge': ['x'*8000]*40}}, 'source_relevant_excerpts': ['z'*4000]*40,
        'proposed_tuning_plans_not_executed': plans}
    parsed = json.loads(copilot._bounded_prompt('NELL tuning', context, [], '', limit=11000))
    facts = parsed['verified_run_facts']
    assert facts['tuning_attempts_total'] == 2 and len(facts['tuning_attempts']) == 2
    assert len(facts['tuning_plans_not_executed'][0]['space']) == 4


def test_matching_model_class_is_kept_with_loss_evidence():
    source = 'class Other:\n    def _loss(self):\n        return cross_entropy\n\nclass GCN:\n    def _loss(self):\n        return masked_cross_entropy\n'
    excerpt = copilot._source_excerpts(source, ['gcn', 'loss', 'cross_entropy'], limit=1)[0]
    assert '[class GCN]' in excerpt and 'masked_cross_entropy' in excerpt


def test_planetoid_dataset_versions_are_not_misread_as_missing_parts(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    for part in PLANETOID_PARTS:
        (root / ('ind.nell.0.001.' + part)).write_bytes(b'payload')
    result = inspect_local_datasets(tmp_path, ['NELL', 'nell.0.001'])
    assert result['NELL']['ready'] and result['NELL']['file_count'] == 8
    assert result['NELL']['dataset_variant'] == 'nell.0.001'
    assert result['nell.0.001']['ready']


def test_incomplete_planetoid_variants_cannot_be_combined_into_complete_dataset(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    for index, part in enumerate(sorted(PLANETOID_PARTS)):
        (root / ('ind.nell.' + ('0.001.' if index % 2 else '0.01.') + part)).write_bytes(b'payload')
    assert not inspect_local_datasets(tmp_path, ['NELL'])['NELL']['ready']


@pytest.mark.parametrize('question', ['下一次调参建议搜索哪些参数？只给方案，不启动实验。',
    '调参为什么只调学习率？', '搜索空间和预算是多少？'])
def test_tuning_read_only_queries_use_project_plan_without_model_dependency(tmp_path, question):
    assert copilot._tuning_plan_question(question)
    job = fixture_job(tmp_path)
    answer = copilot._tuning_plan_answer({'job': {}, 'proposed_tuning_plans_not_executed': tuning.build_tuning_plans(job)}, question)
    for name in ['learning_rate', 'dropout', 'weight_decay', 'hidden1']:
        assert name in answer
    assert '尚未批准或执行' in answer and '没有修改源码' in answer


def test_parameter_mutation_is_not_routed_as_readonly_plan():
    assert not copilot._tuning_plan_question('请修改代码，把调参范围扩大')


def test_fact_row_limits_explicitly_report_incomplete_not_false_complete():
    context = {'job': {'tuning_attempts': [{'value': i} for i in range(20)], 'tuning_attempts_count': 20,
        'dataset_results_count': 4, 'dataset_results_preview': [{'dataset': 'A'}, {'dataset': 'B'}]}}
    facts = json.loads(copilot._bounded_prompt('history', context, [], ''))['verified_run_facts']
    assert facts['tuning_attempts_total'] == 20 and not facts['tuning_attempt_rows_complete']
    assert not facts['dataset_result_rows_complete']


@pytest.mark.parametrize('question', ['这篇论文复现成功了吗？', '这四个数据集都完成了吗？', '你好'])
def test_context_always_supplies_all_final_results_not_two_row_preview(tmp_path, monkeypatch, question):
    job = fixture_job(tmp_path)
    job.result_json['dataset_results'] = [{'dataset':name, 'metrics':{'accuracy':score}}
        for name,score in [('Custom-A',.71),('Custom-B',.81),('Custom-C',.78),('Custom-D',.53)]]
    class Session:
        async def scalars(self,*_): return SimpleNamespace(all=lambda:[job])
        async def scalar(self,*_): return job
    monkeypatch.setattr(copilot,'_safe_workspace',lambda _:None)
    project = SimpleNamespace(id='project-a',name='Custom',description='',artifacts=[])
    context = asyncio.run(copilot._context(Session(),project,question,job.id))
    facts = json.loads(copilot._bounded_prompt(question,context,[],''))['verified_run_facts']
    assert facts['dataset_result_rows_complete']
    assert len(facts['dataset_results']) == 4
    assert facts['dataset_results'][2]['metrics']['accuracy'] == .78


def test_result_summary_intent_is_readonly_and_recognizes_reproduction_success():
    assert copilot._results_summary_question('这篇论文复现成功了吗？')
    assert copilot._results_summary_question('四个数据集都跑完了吗？')
    assert not copilot._results_summary_question('请修改代码，然后重新跑实验')


def test_result_summary_excludes_generic_paper_defaults_but_preserves_actual_results():
    original = {'source_relevant_excerpts':['default dropout=.5'],
                'proposed_tuning_plans_not_executed':[{'trials':[{}]}],
                'job':{'dataset_results':[{'dataset':'Custom','metrics':{'accuracy':.6}}],
                       'tuning_attempts':[], 'tuning_plans':[{}],
                       'paper_spec':{'hyperparameters':{'dropout':.5}, 'target_results':{'Custom':.66}}}}
    context = copilot._results_summary_context(original)
    facts = json.loads(copilot._bounded_prompt('这次实验最终取得哪些结果？有没有额外调参？',context,[],''))
    assert facts['verified_run_facts']['dataset_results'][0]['metrics']['accuracy'] == .6
    assert facts['verified_run_facts']['tuning_attempts_total'] == 0
    assert facts['verified_run_facts']['tuning_plans_not_executed'] == []
    assert 'hyperparameters' not in facts['project_evidence']['job']['paper_spec']
    assert original['job']['paper_spec']['hyperparameters'] == {'dropout':.5}


def test_six_trial_budget_covers_all_five_supported_parameter_groups(tmp_path):
    job = fixture_job(tmp_path)
    entry = Path(job.code_spec['source_path']) / 'train.py'
    entry.write_text(entry.read_text(encoding='utf-8') + "flags.DEFINE_integer('batch_size', 32, 'batch')\n", encoding='utf-8')
    plan = tuning.build_tuning_plans(job, budget=6)[0]
    assert len(plan['space']) == 5
    assert set().union(*(set(trial) for trial in plan['trials'])) == {row['parameter'] for row in plan['space']}
def test_tensorflow_sparse_and_numbered_compute_nodes_are_gpu_execution_evidence():
    from app.services import runtime_evidence
    for node in ['gcn/layer/SparseTensorDenseMatMul', 'gcn/layer/MatMul_1']:
        evidence = runtime_evidence.device_log_evidence([{'log': 'nell-original.log',
            'head': f'Op {node} on device: /job:localhost/replica:0/task:0/device:GPU:0'}])
        assert evidence[0]['kind'] == 'gpu_compute_placement'


def test_paper_parameter_values_survive_large_mixed_question_context():
    parameters=[{'name':key,'value':value} for key,value in
        [('learning_rate',0.01),('epochs',200),('hidden',16),('dropout',0.5),('weight_decay',0.0005),('early_stopping',10)]]
    context={'job':{'run_id':'current','paper_spec':{'hyperparameters':parameters}},
        'source_relevant_excerpts':[{'file':'model.py','excerpt':'x'*3000}]*30,
        'paper_excerpt':'y'*30000}
    packed=json.loads(copilot._bounded_prompt('解释论文方法、划分、超参数和设备',context,[],''))
    assert packed['verified_run_facts']['paper_parameters_not_runtime']==parameters
