"""Offline regression boundaries found by actual UI runs; not reproduction evidence."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
import pytest
from app.api import copilot
from app.models import ReproductionStatus
from app.services.task_context import rank_entries
from app.services.telemetry import parse_experiment_log
from app.services.metric_contract import dataset_targets
from app.services.editable_source import editable_file, validate_requirements
from app.services.metric_contract import target_tolerance
from app.services.command_intent import apply_goal_parameters
from app.services.source_quotes import checked_source_quotes


@pytest.mark.parametrize('question', [
    '现在请通过复现助手在 CUDA 环境运行 Cora，输出准确率和曲线。不要运行其他数据集。',
    '请通过助手现在准备并启动实验；如果有阻塞须指出，不要只回答设备状态。',
    '请用作者权重采样16张图片，不训练。',
    '请评估公开测试 tensor，计算 CR、PRD、RMSE。',
    '请在GPU上训练完整基线，不运行MLP。',
])
def test_real_operations_are_routed_to_action_plans(question):
    assert copilot._rerun_requested(question)
    assert not copilot._performance_question(question)
    assert not copilot._gpu_capability_question(question)


@pytest.mark.parametrize('question', [
    '请说明模型训练过程及损失函数，不修改、不运行。',
    '如何在GPU运行这个模型？', '请问训练可以在GPU上运行吗？',
    '不要运行实验，只说明准确率怎么算。', '请先不运行实验。',
    '请解释 RMSE、PRD 的公式及代码位置。',
])
def test_explanations_and_negations_are_readonly(question):
    assert not copilot._rerun_requested(question)
    assert not copilot._code_action_requested(question)
    assert not copilot._performance_question(question)


def test_initial_approval_does_not_demand_an_installed_runtime(monkeypatch):
    job = SimpleNamespace(id='run', status=ReproductionStatus.awaiting_approval, result_json={},
        environment_spec={'gpu_requested': True}, execution_command={'commands':[{'dataset':'Dataset-A'}]})
    class DB:
        async def scalar(self, _): return job
    async def no_probe(_): pytest.fail('Initial approval owns preparation')
    monkeypatch.setattr(copilot, 'inspect_run_readiness', no_probe)
    plan = asyncio.run(copilot._rerun_action_plan(DB(), SimpleNamespace(id='p'), 'run', '请启动实验 Dataset-A'))
    assert plan['kind'] == 'initial_run_confirm'


def test_negative_model_is_not_entrypoint_evidence(tmp_path):
    for name in ['mlp.py', 'ft_transformer.py']:
        (tmp_path/name).write_text('import torch\n')
    assert rank_entries(tmp_path, {'entrypoints':['mlp.py','ft_transformer.py']},
                        'FTTransformer Adult完整训练，不运行MLP。')[0] == 'ft_transformer.py'


def test_evaluation_discovers_main_guard_even_without_training_calls(tmp_path):
    (tmp_path/'ae_training.py').write_text("import torch\nif __name__ == '__main__':\n    train()\n")
    (tmp_path/'ae_compressor.py').write_text("import torch\nif __name__ == '__main__':\n    evaluate()\n")
    assert rank_entries(tmp_path,{'entrypoints':['ae_training.py']},'只用公开权重编码/解码，不训练')[0] == 'ae_compressor.py'


def test_split_metrics_are_not_overwritten_and_sampling_is_bounded(tmp_path):
    path=tmp_path/'cora-original.log'
    path.write_text('\n'.join(f'Epoch:{i:04d} train loss:1.952 acc:15.00 | val loss:1.933 acc:16.20' for i in range(1,654)))
    points=parse_experiment_log(path)['points']
    assert len(points) == 500 and points[-1]['epoch'] == 653
    assert points[0]['train_acc'] == 15 and points[0]['val_acc'] == 16.2


def test_flat_mean_uncertainty_binds_only_unambiguous_dataset():
    spec={'datasets':['Cora'],'metrics':['Accuracy'],'target_results':{'accuracy':'85.5 ± 0.5'}}
    assert dataset_targets(spec) == {'Cora':{'accuracy':.855}}
    assert dataset_targets({**spec,'datasets':['Cora','Pubmed']}) == {}
    assert dataset_targets({'datasets':['E2E'],'metrics':['BLEU'],'target_results':{'BLEU':70.2}}) == {'E2E':{'bleu':70.2}}


def test_dependency_files_have_a_separate_syntax_contract():
    assert editable_file('requirements.txt') and editable_file('requirements-gpu.txt')
    assert not editable_file('private-data.txt')
    validate_requirements('torch==1.13.1\nnumpy>=1.19; python_version>="3.7"\n-r base.txt\n')
    with pytest.raises(ValueError): validate_requirements('torch => 1\n')


def test_no_silent_three_percentage_point_success_margin():
    spec={'datasets':['Cora'],'metrics':['Accuracy'],'target_results_raw':{'accuracy':'85.5 ± 0.5'}}
    assert target_tolerance(spec,'Cora','accuracy')[0] == .005
    assert target_tolerance({**spec,'target_results_raw':{'Cora':'85.5 ± 0.5'}},'Cora','accuracy')[0] == .005
    assert target_tolerance({},'Data','accuracy')[0] == 0
    assert .849 < .855 - target_tolerance(spec,'Cora','accuracy')[0]


def test_explicit_short_run_uses_author_cli_without_editing_source(tmp_path):
    original="import argparse\np=argparse.ArgumentParser()\np.add_argument('--epochs',default=200,type=int)\np.add_argument('--seed',default=1,type=int)\n"
    (tmp_path/'train.py').write_text(original)
    argv,changes=apply_goal_parameters(tmp_path,['train.py'],['python','train.py','--epochs','200'], '完整训练2个epoch，seed=42，不截断batch')
    assert argv == ['python','train.py','--epochs','2','--seed','42']
    assert changes == {'epochs':2,'seed':42} and (tmp_path/'train.py').read_text()==original
    assert apply_goal_parameters(tmp_path,['train.py'],['python','train.py'],'先进行1个完整训练epoch')[1]['epochs']==1
    assert apply_goal_parameters(tmp_path,['train.py'],['python','train.py'],'先进行一个完整训练epoch，把seed设为42')[1]=={'epochs':1,'seed':42}


def test_fabricated_source_lines_are_removed_instead_of_published_as_evidence():
    context={'source_task_evidence':{'named_implementations':[{'file':'layers.py','code':'L28: theta = math.log(lamda/l+1)\nL29: hi = torch.spmm(adj, input)'}]}}
    real='```python\nL29: hi = torch.spmm(adj, input)\n```'
    assert checked_source_quotes(real,context)[1]['rejected_blocks'] == 0
    invented='```python\nL29: hi = torch.spmm(support, r)\n```'
    answer,check=checked_source_quotes(invented,context)
    assert 'torch.spmm(support' not in answer and check['rejected_blocks'] == 1


def test_public_pretrained_evaluation_does_not_require_private_training(tmp_path):
    import json
    from app.services.author_workflow import archive_resource_plan, requested_phase
    (tmp_path/'evaluate.py').write_text("weights='./outputs/weights.pt'\ntest='./data/test.pt'\nprivate='./train/private.pt'\n")
    (tmp_path/'zenodo-record.json').write_text(json.dumps({'id':123,'files':[
        {'key':'weights.pt','size':12,'checksum':'md5:a'}, {'key':'test.pt','size':5,'checksum':'md5:b'},
        {'key':'../escape.pt','size':5}]}))
    plan=archive_resource_plan(tmp_path,['evaluate.py'],['Private training study'])
    assert requested_phase('只用公开权重与测试tensor编码/解码，不训练') == 'evaluation'
    assert plan['missing_datasets'] == [] and plan['available_datasets'] == []
    assert {row['relative_path'] for row in plan['download_requests'][0]['files']} == {'outputs/weights.pt','data/test.pt'}
    assert not (tmp_path/'outputs').exists()


def test_full_metric_names_are_detected_without_unit_confusion():
    from app.services.reproduction_engine import _extract_metrics,_primary_metric
    values=_extract_metrics('CR: 3.05 | PRD: 9.993 | RMSE: .820\nTop1 error: 17.0 Top5 err: 4.0')
    assert values == {'cr':3.05,'prd':9.993,'rmse':.820,'top1error':.17,'top5error':.04}
    assert _primary_metric({'metrics':['Top-1 Error']},values)=='top1error'


def test_functional_sampling_uses_author_pretrained_command_not_training(tmp_path):
    from app.services.author_configuration import config_commands
    from app.services.paper_contract import vague_spec
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs'/'dataset-a.yml').write_text('data:\n  dataset: dataset-a\n')
    (tmp_path/'main.py').write_text("import argparse\np=argparse.ArgumentParser()\np.add_argument('--config')\np.add_argument('--doc',required=True)\np.add_argument('--steps',default=50,type=int)\np.add_argument('--eta',default=0.0,type=float)\n")
    examples=[{'args':['--config','{DATASET}.yml','--doc','{MODEL_NAME}'],'readme':'README.md'},
              {'args':['--config','{DATASET}.yml','--sample','--use_pretrained','--fid','--steps','{STEPS}','--eta','{ETA}'],'readme':'README.md'}]
    commands=config_commands(tmp_path,'main.py',['python','main.py'],['dataset-a'],examples,None,phase='sampling')
    assert '--sample' in commands[0]['argv'] and '--use_pretrained' in commands[0]['argv']
    assert '--fid' not in commands[0]['argv'] and '{STEPS}' not in commands[0]['argv']
    assert '--doc' in commands[0]['argv']
    assert not vague_spec({'datasets':['dataset-a'],'metrics':[],'execution_phase':'sampling',
                           'spec_validation':{'scope':'author_source_workflow'}})
    assert vague_spec({'datasets':['dataset-a'],'metrics':[]})
