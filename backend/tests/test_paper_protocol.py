"""Offline regressions for changing scope and dataset-specific execution plans."""
import pytest
from app.services import reproduction_engine as engine
from app.services.metric_contract import target_tolerance
from app.services.paper_protocol import dataset_parameters, reusable_specification, specification_context
from app.services.performance_scope import parameter_differences


def test_scope_cache_requires_current_paper_and_goal():
    spec={'datasets':['first'], 'spec_context':specification_context('paper v1', 'first only')}
    assert reusable_specification(spec, 'paper v1', 'first   only')
    assert not reusable_specification(spec, 'paper v1', 'first and second')
    assert not reusable_specification(spec, 'paper v2', 'first only')
    assert not reusable_specification({'datasets':['first']}, 'paper v1', 'first only')


def test_dataset_specific_parameters_override_common_values_without_cross_contamination(tmp_path):
    (tmp_path/'train.py').write_text("import argparse\np=argparse.ArgumentParser()\n"
        "p.add_argument('--dataset',default='first')\n"
        "p.add_argument('--width',type=int,default=16)\n"
        "p.add_argument('--dropout',type=float,default=0.5)\n")
    spec={'datasets':['first','second'], 'hyperparameters':[
        {'name':'width','value':16}, {'name':'dropout','value':.5},
        {'name':'width','value':64,'dataset':'second'},
        {'name':'dropout','value':.1,'datasets':['second']} ]}
    assert dataset_parameters(spec,'SECOND')=={'width':64,'dropout':.1}
    assert not parameter_differences(spec,{'width':64,'dropout':.1},'second')
    assert parameter_differences(spec,{'width':64},'first')
    def plan(goal=''):
        return engine._execution_command(tmp_path,engine._repository_manifest(tmp_path),'full',
            {'prefix':str(tmp_path/'env')},{**spec,'user_goal':goal},{'available_datasets':['first','second']})
    first,second=plan()['commands']
    assert '--width' not in first['argv']
    assert second['argv'][second['argv'].index('--width')+1]=='64'
    assert second['argv'][second['argv'].index('--dropout')+1]=='0.1'
    # A deliberate user override wins over the paper's dataset-specific setting.
    second=plan('width=32')['commands'][1]
    assert second['argv'][second['argv'].index('--width')+1]=='32'
    assert parameter_differences(spec,{'width':32},'second')[0]['paper']==64


@pytest.mark.parametrize('metric,target,expected', [
    ('Accuracy','79.0%',.01), ('Top-1 Error','0.5%',.01), ('RMSE',20.,1.), ('BLEU',30.,1.5),
])
def test_practical_margin_has_explicit_units_and_case_insensitive_dataset_binding(metric,target,expected):
    spec={'datasets':['Example'],'metrics':[metric], 'target_results':{'Example':{metric:target}},
          'acceptance_policy':{'mode':'practical','ratio_tolerance':.01,'relative_tolerance':.05}}
    tolerance,basis=target_tolerance(spec,'example',metric)
    assert tolerance==pytest.approx(expected)
    assert '实用复现容差' in basis


def test_paper_uncertainty_can_widen_but_not_shrink_practical_margin():
    spec={'datasets':['Example'],'metrics':['accuracy'],
          'acceptance_policy':{'mode':'practical','ratio_tolerance':.01,'relative_tolerance':.05}}
    assert target_tolerance({**spec,'target_results':{'Example':{'accuracy':'79.0 ± 0.2'}}},'Example','accuracy')[0]==.01
    assert target_tolerance({**spec,'target_results':{'Example':{'accuracy':'79.0 ± 2.0'}}},'Example','accuracy')[0]==.02
