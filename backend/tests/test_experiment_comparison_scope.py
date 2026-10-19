from app.services.performance_scope import comparison_scope
from app.services.reproduction_engine import _claim_results


def test_explicit_user_scope_retained_without_guessing_from_epoch_count():
    assert comparison_scope({'user_goal': '先训练2个完整epoch并测试；功能检查不作为论文性能复现。'}) == 'functional_check'
    assert comparison_scope({'user_goal': 'Run a functional check on public inputs.'}) == 'functional_check'
    assert comparison_scope({'user_goal': '训练2个epoch，目标为作者报告的准确率。'}) == 'paper_baseline'
    assert comparison_scope({'user_goal': '不要功能测试；按作者完整训练协议运行。'}) == 'paper_baseline'
    assert comparison_scope({'user_goal': 'Do not functional test; run the full protocol.'}) == 'paper_baseline'
    assert comparison_scope({'objective': 'functional_check'}) == 'paper_baseline'
    assert comparison_scope({}, 'smoke') == 'functional_check'


def test_functional_claim_cannot_support_or_reject_paper_even_if_metric_matches():
    for actual, verdict in [(.2, 'result_produced'), (.8, 'matched')]:
        row = {'dataset': 'Example', 'metric_key': 'accuracy', 'metrics': {'accuracy': actual},
               'target': .8, 'return_code': 0, 'verdict': verdict, 'comparison_scope': 'functional_check'}
        claim = _claim_results({}, [row])[0]
        assert claim['scope'] == 'functional' and not claim['supported']
        assert claim['status'] == 'functional_completed'
        assert '不作论文性能' in claim['evidence']
    row['return_code'] = 1
    assert _claim_results({}, [row])[0]['status'] == 'failed'


def test_modified_parameters_are_compared_by_recorded_identity_not_code_revision():
    from app.services.performance_scope import parameter_differences
    spec={'hyperparameters':[{'name':'hidden_width','value':16},{'name':'lr','value':'1e-2'}]}
    assert parameter_differences(spec,{'hidden_width':32,'lr':0.01})==[{'parameter':'hidden_width','paper':16.0,'actual':32.0}]
    assert not parameter_differences(spec,{'hidden_width':16,'lr':0.01})
    assert not parameter_differences(spec,{})
    row={'dataset':'Example','return_code':0,'verdict':'result_produced','comparison_scope':'modified_configuration',
         'metrics':{'accuracy':.9},'target':.8,'parameter_differences':[{'parameter':'hidden_width','paper':16,'actual':32}]}
    claim=_claim_results({},[row])[0]
    assert not claim['supported'] and claim['scope']=='modified_configuration'
