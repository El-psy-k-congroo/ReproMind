import pytest
from app.services.tuning import best_trial, validation_score
from app.services.metric_contract import metric_number, target_tolerance
from app.services.reproduction_engine import _target_for_dataset


@pytest.mark.parametrize('metric', ['top1error', 'Top-5 Error', 'PRD', 'fid', 'perplexity', 'rmse'])
def test_lower_metrics_choose_validation_improvement_even_when_test_ranking_disagrees(metric):
    rows=[{'return_code':0,'validation_score':.4,'test_metrics':{'value':.1}},
          {'return_code':0,'validation_score':.1,'test_metrics':{'value':.9}}]
    assert best_trial(rows,.3,metric)==1


@pytest.mark.parametrize('metric,text,expected', [
    ('top1error','[val] Top 1-err 12.5', .125),
    ('Top-5 Error','validation Top-5 Error: 0.5%', .005),
    ('aupr','val aupr=62.0%', .62),
    ('precision','validation precision=1%', .01),
    ('prd','val PRD=32.7%', 32.7),
    ('rmse','val RMSE=-1.0', None),
])
def test_validation_metric_aliases_and_percent_units(metric,text,expected):
    assert validation_score(text,metric)==expected


def test_small_explicit_percent_and_non_ratio_targets_retain_their_units():
    assert metric_number('0.5%', 'top1error') == .005
    assert metric_number('0.5', 'accuracy') == .5
    assert metric_number('32.7%', 'prd') == 32.7
    assert _target_for_dataset({'target_results': {'Example': {'cr': 47.073}}}, 'Example', 'cr') == 47.073
    assert _target_for_dataset({'target_results': {'Example': {'prd': 32.739}}}, 'Example', 'prd') == 32.739
    tolerance, _ = target_tolerance({'datasets':['Example'], 'metrics':['Top-1 Error'],
                                   'target_results':{'Example':{'top1error':'0.5% ±0.05%'}}}, 'Example','top1error')
    assert tolerance == .0005
