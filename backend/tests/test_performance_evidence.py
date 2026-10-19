import asyncio
import json
from types import SimpleNamespace

import pytest

from app.api import copilot
from app.models import ReproductionStatus
from app.services.performance_evidence import comparison_evidence, comparison_facts_answer, render_hypotheses
from app.services.tuning import command_fingerprint


@pytest.mark.parametrize('question', [
    '为什么迁移到gpu上后性能差这么多', 'GPU上准确率为什么比CPU低？',
    '为什么训练速度变慢了？', '对比这几个实验的准确率和耗时',
    'Why is GPU performance worse than CPU?', '比较实验结果，有没有退化？',
])
def test_performance_is_analysis_not_device_template_or_mutation(question):
    assert copilot._performance_question(question)
    assert not copilot._gpu_capability_question(question)
    assert not copilot._code_action_requested(question)
    assert not copilot._rerun_requested(question)


@pytest.mark.parametrize('question', ['GPU是否可用？', '这次实际在GPU运行了吗？'])
def test_actual_gpu_capability_queries_keep_their_route(question):
    assert copilot._gpu_capability_question(question)
    assert not copilot._performance_question(question)


def test_explicit_performance_fix_still_requires_code_confirmation():
    assert copilot._code_action_requested('请修改代码，优化GPU性能')
    assert not copilot._performance_question('请修改代码，优化GPU性能')


def job(root, id, score, flags='', argv=None):
    source = root / id
    source.mkdir()
    (source / 'train.py').write_text(flags)
    argv = argv or ['python', '-m', 'train', '--dataset', 'custom-data']
    fingerprint = command_fingerprint({'argv':argv}, source)
    return SimpleNamespace(id=id, status=ReproductionStatus.completed, created_at=None,
        code_spec={}, execution_command={'pythonpath':str(source), 'entrypoint_candidates':['train.py']},
        result_json={'dataset_results':[{'dataset':'custom-data','return_code':0,
           'metrics':{'accuracy':score}, 'baseline_metrics':{'accuracy':score},
           'baseline_argv':argv,'baseline_fingerprint':fingerprint,'wall_time_seconds':10}]})


def test_cross_run_uses_exact_metric_and_verified_parameters_not_gpu_causes(tmp_path):
    old = job(tmp_path, 'old-run', .8, "parser.add_argument('--width', default=32)\n",
              ['python','-m','train','--dataset','custom-data','--dropout','0.1'])
    current = job(tmp_path, 'new-run', .5, "parser.add_argument('--width', default=16)\n")
    other = job(tmp_path, 'unrelated', .99)
    other.result_json['dataset_results'][0]['dataset'] = 'other-data'
    result = comparison_evidence([other, current, old], current, '性能为什么下降', lambda _:tmp_path)
    assert [row['run_id'] for row in result['records']] == ['new-run','old-run']
    assert result['records'][0]['datasets'][0]['parameters']['width'] == 16
    assert result['records'][1]['datasets'][0]['parameters']['dropout'] == .1
    assert result['records'][1]['datasets'][0]['metrics']['accuracy'] == .8
    assert not result['records'][1]['device_evidence']


def test_changed_historical_source_cannot_fabricate_original_defaults(tmp_path):
    old = job(tmp_path, 'old-run', .8, "parser.add_argument('--width', default=32)\n",
              ['python','--dropout=0.1'])
    (tmp_path / old.id / 'train.py').write_text("parser.add_argument('--width', default=999)\n")
    row = comparison_evidence([old], old, '性能', lambda _:tmp_path)['records'][0]['datasets'][0]
    assert row['parameters'] == {'dropout':.1}
    assert not row['source_matches_execution_fingerprint']


def test_failed_missing_and_zero_metrics_remain_distinct(tmp_path):
    current = job(tmp_path, 'run', 0)
    current.result_json['dataset_results'][0].update(return_code=1, metrics={'accuracy':0,'loss':float('nan')})
    row = comparison_evidence([current], current, '准确率', lambda _:tmp_path)['records'][0]['datasets'][0]
    assert row['return_code'] == 1 and row['metrics'] == {'accuracy':0}


def test_actual_logged_epoch_is_not_configured_epoch_limit(tmp_path):
    current = job(tmp_path,'run',.5,"parser.add_argument('--epochs', default=200)\n")
    telemetry = tmp_path/'runs'/'run'/'telemetry'
    telemetry.mkdir(parents=True)
    (telemetry/'custom-data-original.log').write_text('Epoch: 168 train_loss= 0.2 val_acc= 0.5\n')
    record = comparison_evidence([current],current,'速度',lambda _:tmp_path)['records'][0]
    assert record['datasets'][0]['parameters']['epochs'] == 200
    assert record['datasets'][0]['last_logged_epoch'] == 168
    assert '未知' in record['training_device_verdict']
    answer = comparison_facts_answer({'records':[record]})
    assert '168 / 200' in answer and '实际训练设备未知' in answer
    assert 'epochs=200' not in answer


def test_successful_retry_baseline_log_is_authoritative_not_failed_first_attempt(tmp_path):
    current = job(tmp_path,'run',.5)
    current.result_json['dataset_results'][0]['baseline_log'] = 'baseline_evidence/custom.txt'
    directory = tmp_path/'runs'/'run'/'baseline_evidence'
    directory.mkdir(parents=True)
    (directory/'custom.txt').write_text('Epoch: 129 train_loss= 0.5 val_acc= 0.6\nTest accuracy= 0.5\n')
    record = comparison_evidence([current],current,'性能',lambda _:tmp_path)['records'][0]
    assert record['datasets'][0]['last_logged_epoch'] == 129


@pytest.mark.parametrize('reason,ids', [('准确率是0.99',['run']), ('配置不同',['foreign-project-run'])])
def test_model_cannot_invent_numeric_facts_or_reference_other_project(reason,ids):
    with pytest.raises(ValueError):
        render_hypotheses({'hypotheses':[{'possible_reason':reason,'verify':'核对配置','supporting_runs':ids}]},
                         {'records':[{'run_id':'run'}]})


def test_numbered_parameter_identifier_is_not_a_numeric_value():
    comparison = {'records':[{'run_id':'run', 'datasets':[{'parameters':{'hidden1':64}}]}]}
    data = {'hypotheses':[{'possible_reason':'hidden1 可能影响容量',
                         'verify':'比较 hidden1 的配置证据', 'supporting_runs':['run']}]}
    assert 'hidden1' in render_hypotheses(data, comparison)
    data['hypotheses'][0]['possible_reason'] = 'hidden1=999'
    with pytest.raises(ValueError):
        render_hypotheses(data, comparison)


def test_known_run_identifier_in_reason_is_not_a_fabricated_numeric_value():
    id = 'a5354796-aaaa-bbbb-cccc-acc32061c6d4'
    comparison = {'records':[{'run_id':id}]}
    data = {'hypotheses':[{'possible_reason':'a5354796 的初始化可能不同',
                         'verify':'建议受控重复运行', 'supporting_runs':[id]}]}
    assert 'a5354796' in render_hypotheses(data, comparison)
    data['hypotheses'][0]['possible_reason'] = 'a5354796 的准确率为0.99'
    with pytest.raises(ValueError):
        render_hypotheses(data, comparison)


@pytest.mark.parametrize('bad_row', [
    {'possible_reason':'准确率可能为0.99', 'verify':'重复实验', 'supporting_runs':['run']},
    {'possible_reason':'未知配置可能变化', 'verify':'重复实验', 'supporting_runs':['another-project']},
])
def test_invalid_optional_hypothesis_does_not_discard_valid_analysis(bad_row):
    data = {'hypotheses':[bad_row, {'possible_reason':'随机初始化可能影响收敛',
                                 'verify':'固定种子进行受控对照', 'supporting_runs':['run']}]}
    answer = render_hypotheses(data, {'records':[{'run_id':'run'}]})
    assert '随机初始化可能影响收敛' in answer and '已省略' in answer
    assert '0.99' not in answer and 'another-project' not in answer


@pytest.mark.parametrize('other_epochs', [60, 200, None])
def test_explicit_epoch_value_must_match_every_cited_run(other_epochs):
    comparison = {'records':[
        {'run_id':'new','datasets':[{'parameters':{'epochs':60}}]},
        {'run_id':'old','datasets':[{'parameters':{'epochs':other_epochs}}]}]}
    data = {'hypotheses':[{'possible_reason':'将epoch设为60的记录可能没有充分收敛',
                         'verify':'检查验证曲线', 'supporting_runs':['new','old']}]}
    if other_epochs == 60:
        assert 'epoch设为60' in render_hypotheses(data,comparison)
    else:
        with pytest.raises(ValueError):
            render_hypotheses(data,comparison)


@pytest.mark.parametrize('changed', [False, True])
def test_verified_non_epoch_configuration_changes_are_reported_before_hypotheses(tmp_path, changed):
    old = job(tmp_path,'old-run',.1,"parser.add_argument('--width', default=16)\n")
    new = job(tmp_path,'new-run',.6,
              "parser.add_argument('--width', default=" + ('64' if changed else '16') + ")\n")
    evidence = comparison_evidence([new,old],new,'为什么只改epoch性能下降',lambda _:tmp_path)
    answer = comparison_facts_answer(evidence)
    if changed:
        assert answer.startswith('已核验的配置差异：')
        assert 'width 不同；并非仅改变 epoch' in answer
    else:
        assert '已核验的配置差异：' not in answer


def test_budget_preserves_both_runs_even_when_other_evidence_is_huge(tmp_path):
    old, new = job(tmp_path, 'old', .8), job(tmp_path, 'new', .5)
    comparison = comparison_evidence([new,old], new, 'GPU性能差', lambda _:tmp_path)
    context = {'job':{'run_id':'new'}, 'performance_comparison':comparison, 'paper_excerpt':'noise'*20000}
    prompt = json.loads(copilot._bounded_prompt('GPU性能差',context,[], '',limit=11000))
    assert prompt['verified_performance_comparison']['records'][1]['datasets'][0]['metrics']['accuracy'] == .8
    assert len(json.dumps(prompt,ensure_ascii=False)) <= 11000


def test_performance_context_loads_history_without_live_gpu_probe(tmp_path, monkeypatch):
    current = job(tmp_path, 'new', .5)
    old = job(tmp_path, 'old', .8)
    for row in [current,old]:
        row.project_id='project'; row.workspace_path=str(tmp_path)
        row.current_stage='completed'; row.message='done'; row.error_message=''
        row.paper_spec={}; row.dataset_spec={}; row.environment_spec={}; row.source_mode='paper_repository'
        row.repository_url=''; row.events=[]; row.logs=''
    class Session:
        async def scalars(self, *_): return SimpleNamespace(all=lambda:[current,old])
        async def scalar(self, *_): return current
    monkeypatch.setattr(copilot, '_safe_workspace', lambda _:tmp_path)
    async def forbidden(*_): raise AssertionError('No CUDA probe needed for historical performance')
    monkeypatch.setattr(copilot, 'inspect_run_readiness', forbidden)
    project = SimpleNamespace(id='project',name='custom',description='',artifacts=[])
    context = asyncio.run(copilot._context(Session(),project,'为什么GPU性能差这么多','new'))
    assert len(context['performance_comparison']['records']) == 2


@pytest.mark.parametrize('question', [
    '请根据已上传论文说明邻接矩阵如何归一化、训练验证测试划分和超参数。是否必须用 GPU？',
    '比较论文方法和代码实现，给出学习率和 dropout，同时说明 CPU 是否能运行。',
    'What is the algorithm and dropout? Can it run on GPU?',
])
def test_mixed_research_questions_are_not_replaced_by_device_template(question):
    assert not copilot._gpu_capability_question(question)
