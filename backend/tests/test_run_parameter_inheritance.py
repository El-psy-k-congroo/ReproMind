from types import SimpleNamespace

import pytest

from app.services.run_parameters import changed_cli_parameters, inherit_dataset_parameters, effective_numeric_parameters, historical_parameter_context


def base(argv, snapshot=None):
    return SimpleNamespace(id='original', execution_command={'pythonpath':'old', 'commands':[]},
        result_json={'dataset_results':[{'dataset':'Custom', 'baseline_argv':argv,
                                        'baseline_parameters':snapshot or {}}]})


def planned():
    return {'commands':[{'dataset':'Custom','argv':['python','-m','train','--dataset','custom.0.01']}]}


def test_per_dataset_recorded_cli_is_preserved_not_replaced_by_source_defaults():
    old = base(['python','-m','train','--dataset','old-alias','--dropout','0.1','--hidden1','64',
                '--weight_decay=0.00001','--model','custom-net'])
    command = inherit_dataset_parameters(old, planned(), 'new')['commands'][0]['argv']
    assert '--dataset custom.0.01' in ' '.join(command)
    assert '--dropout 0.1' in ' '.join(command)
    assert '--hidden1 64' in ' '.join(command)
    assert '--weight_decay=0.00001' in command
    assert '--model custom-net' in ' '.join(command)


def test_deliberate_default_edit_takes_precedence_over_old_command_override():
    proposal = {'changes':[{'old_text':"flags.DEFINE_integer('epochs', 200, 'epochs')",
                            'new_text':"flags.DEFINE_integer('epochs', 60, 'epochs')"}]}
    changed = changed_cli_parameters(proposal)
    old = base(['python','train.py','--epochs','200','--width','32'])
    command = inherit_dataset_parameters(old, planned(), 'new', changed)['commands'][0]['argv']
    assert '--epochs' not in command
    assert '--width 32' in ' '.join(command)
    assert changed == {'epochs'}


def test_execution_snapshot_freezes_defaults_except_deliberate_changes():
    old = base(['python','train.py','--dataset','custom'], {'lr':.02,'epochs':200,'width':32})
    result = inherit_dataset_parameters(old, planned(), 'new', {'epochs'})
    argv = result['commands'][0]['argv']
    assert '--lr 0.02' in ' '.join(argv) and '--width 32' in ' '.join(argv)
    assert '--epochs' not in argv
    assert result['parameter_inheritance'][0]['historical_defaults_snapshotted']


def test_cli_values_override_defaults_in_execution_snapshot(tmp_path):
    (tmp_path/'train.py').write_text("p.add_argument('--width', default=16)\np.add_argument('--epochs', default=60)\n")
    assert effective_numeric_parameters(tmp_path,['train.py'],['python','train.py','--width','64']) == {'width':64,'epochs':60}


@pytest.mark.parametrize('argv', [['python','train.py','--lr','0.1','--lr','0.2'],
                                ['python','train.py','unidentified-position']])
def test_ambiguous_historical_arguments_fail_closed(argv):
    with pytest.raises(ValueError):
        inherit_dataset_parameters(base(argv), planned(), 'new')


def test_negative_values_and_switches_are_kept_as_original_tokens():
    argv = inherit_dataset_parameters(base(['python','train.py','--offset','-1','--enabled']),planned(),'new')['commands'][0]['argv']
    assert '--offset -1 --enabled' in ' '.join(argv)


def test_string_and_boolean_defaults_can_be_deliberately_changed():
    assert changed_cli_parameters({'changes':[{'old_text':"p.add_argument('--mode',default='old')\nflags.DEFINE_boolean('enabled',False,'')",
        'new_text':"p.add_argument('--mode',default='new')\nflags.DEFINE_boolean('enabled',True,'')"}]}) == {'mode','enabled'}


def test_isolated_rerun_does_not_keep_writable_paths_in_original_run(tmp_path):
    source = tmp_path/'sources'/'original'
    old = base(['python','train.py','--output',str(tmp_path/'runs'/'old'/'result.json')])
    old.workspace_path = str(tmp_path)
    old.execution_command['pythonpath'] = str(source)
    with pytest.raises(ValueError,match='可写路径'):
        inherit_dataset_parameters(old, planned(), tmp_path/'branches'/'new'/'source')


def test_planner_receives_historical_cli_not_mutable_source_defaults():
    rows = historical_parameter_context(base(['python','train.py','--width','64','--dropout','.1']))
    assert rows[0]['recorded_cli_overrides'] == {'width':64,'dropout':.1}
    assert rows[0]['execution_snapshot'] == {}


def test_verified_positional_config_remaps_to_copy_and_keeps_new_output(tmp_path):
    source=tmp_path/'source';copied=tmp_path/'branches'/'new'/'source'
    config=source/'configs'/'run.toml';new_config=copied/'configs'/'run.toml'
    output=tmp_path/'outputs'/'new'/'example'
    old=base(['python','run.py',str(config),'--output',str(tmp_path/'outputs'/'old'),'--lr','.01'])
    old.workspace_path=str(tmp_path);old.execution_command['pythonpath']=str(source)
    new={'commands':[{'dataset':'Custom','argv':['python','run.py',str(new_config),'--output',str(output)]}]}
    result=inherit_dataset_parameters(old,new,copied,fresh_output_roots=(output.parent,))
    argv=result['commands'][0]['argv']
    assert str(new_config) in argv and argv[argv.index('--output')+1]==str(output)
    assert '--lr .01' in ' '.join(argv)


def test_changed_positional_model_or_dataset_is_not_silently_inherited():
    old=base(['python','run.py','dataset-a','model-a'])
    new={'commands':[{'dataset':'Custom','argv':['python','run.py','dataset-b','model-a']}]}
    with pytest.raises(ValueError,match='位置参数'):
        inherit_dataset_parameters(old,new,'new')
