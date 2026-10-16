from pathlib import Path
from types import SimpleNamespace
import pytest
from app.services.author_configuration import config_commands
from app.services.run_parameters import inherit_dataset_parameters


def test_documented_output_role_survives_rerun_with_an_arbitrary_flag_name(tmp_path):
    old=tmp_path/'sources'/'old';old.mkdir(parents=True)
    source=tmp_path/'branches'/'fresh'/'source';source.mkdir(parents=True)
    (source/'main.py').write_text("parser.add_argument('--config')\nparser.add_argument('--experiment-home')\n")
    (source/'example.yml').write_text('data:\n  dataset: Example\n')
    run=tmp_path/'runs'/'fresh'
    planned={'commands':config_commands(source,'main.py',['python','main.py'],['Example'],
        [{'args':['--config','example.yml','--experiment-home','{PROJECT_PATH}']}],run)}
    assert planned['commands'][0]['isolated_output_bindings']=={'experiment-home':str(run/'output')}
    previous=['python','main.py','--config',str(old/'example.yml'),'--experiment-home',str(tmp_path/'runs'/'prior'/'output'),'--seed','17']
    base=SimpleNamespace(id='prior',workspace_path=str(tmp_path),execution_command={'pythonpath':str(old),
        'commands':[{'dataset':'Example','argv':previous}]},result_json={})
    actual=inherit_dataset_parameters(base,planned,source,fresh_output_roots=(run,))['commands'][0]['argv']
    assert actual[actual.index('--experiment-home')+1]==str(run/'output')
    assert actual[actual.index('--seed')+1]=='17'


def test_output_binding_cannot_authorize_a_path_outside_new_run_roots(tmp_path):
    old=tmp_path/'old';old.mkdir();source=tmp_path/'branch'/'source';source.mkdir(parents=True)
    external=str(tmp_path/'unrelated'/'output')
    previous=['python','main.py','--custom-root',str(tmp_path/'runs'/'prior'/'output')]
    base=SimpleNamespace(workspace_path=str(tmp_path),execution_command={'pythonpath':str(old),
        'commands':[{'dataset':'Example','argv':previous}]},result_json={})
    planned={'commands':[{'dataset':'Example','argv':['python','main.py','--custom-root',external],
                          'isolated_output_bindings':{'custom-root':external}}]}
    with pytest.raises(ValueError,match='不能安全隔离'):
        inherit_dataset_parameters(base,planned,source,fresh_output_roots=(tmp_path/'runs'/'new',))
