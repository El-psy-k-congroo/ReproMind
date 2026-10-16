from types import SimpleNamespace

from app.api import copilot
from app.models import ReproductionStatus
from app.services.performance_evidence import comparison_evidence
from app.services.run_parameters import effective_numeric_parameters
from app.services.tuning import command_fingerprint


def test_runtime_parameters_follow_executed_entry_imports_not_other_candidates(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source/'citation.py').write_text('from args import get_args\nargs = get_args()\n',encoding='utf-8')
    (source/'args.py').write_text('def get_args():\n p.add_argument("--epochs", default=100)\n p.add_argument("--seed", default=42)\n',encoding='utf-8')
    nested = source/'downstream'
    nested.mkdir()
    (nested/'train.py').write_text('p.add_argument("--epochs", default=3)\np.add_argument("--embedding_dim", default=300)\n',encoding='utf-8')
    argv = ['python','citation.py','--dataset','cora']
    candidates = ['citation.py','downstream/train.py']
    assert effective_numeric_parameters(source,candidates,argv) == {'epochs':100,'seed':42}
    fingerprint = command_fingerprint({'argv':argv},source)
    job = SimpleNamespace(id='run',status=ReproductionStatus.completed,created_at=None,code_spec={},
        execution_command={'pythonpath':str(source),'entrypoint_candidates':candidates},
        result_json={'dataset_results':[{'dataset':'Cora','return_code':0,'metrics':{'accuracy':.81},
            'baseline_argv':argv,'baseline_fingerprint':fingerprint,'baseline_parameters':{}}]})
    rows = comparison_evidence([job],job,'性能',lambda _:tmp_path)['records'][0]['datasets']
    assert rows[0]['parameters']['epochs']==100 and 'embedding_dim' not in rows[0]['parameters']


def test_result_listing_is_summary_while_causal_question_is_performance():
    question = '依据本次SGC实际实验记录，列出Cora、Citeseer、Pubmed的测试准确率与论文对比，并说明哪些论文结论尚未验证。'
    assert copilot._results_summary_question(question)
    assert not copilot._performance_question(question)
    assert copilot._performance_question('为什么迁移到GPU后准确率下降？请列出证据')


def test_dynamic_tuned_parameter_is_unknown_not_static_default(tmp_path):
    (tmp_path/'train.py').write_text('p.add_argument("--weight_decay", default=5e-6)\n'
        'args.weight_decay = load_tuned_config()["weight_decay"]\n',encoding='utf-8')
    assert effective_numeric_parameters(tmp_path,['train.py'],['python','train.py','--tuned'])=={}


def test_result_listing_only_renders_supplied_metrics_targets_and_claims():
    question = '列出本次实验结果及测试准确率与论文对比，并说明哪些论文结论尚未验证'
    assert copilot._recorded_results_requested(question)
    answer = copilot._recorded_results_answer({'job':{'run_id':'current','status':'completed',
        'dataset_results':[{'dataset':'Cora','metrics':{'accuracy':0.0},'target':0.0,'verdict':'matched'}],
        'claim_results':[{'claim':'Speed advantage','status':'not_evaluated'}]}})
    assert 'accuracy=0' in answer and 'Speed advantage' in answer
    assert 'NELL' not in answer and 'CPU' not in answer and 'epochs' not in answer
    assert '论文全部结论' in answer
