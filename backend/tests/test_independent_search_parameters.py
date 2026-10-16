from pathlib import Path
from types import SimpleNamespace
from app.services.tuning import build_tuning_plans


def test_grouped_regularizers_remain_independent_search_parameters(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'train.py').write_text('''parser.add_argument('--lr', default=0.01)
parser.add_argument('--wd1', default=0.01)
parser.add_argument('--wd2', default=0.0005)
''')
    command={'dataset':'Example','argv':['python','train.py'],'cwd':str(source),'pythonpath':str(source)}
    job=SimpleNamespace(workspace_path=str(tmp_path),code_spec={'source_path':str(source)},
        execution_command={'entrypoint_candidates':['train.py'],'commands':[command]})
    plan=build_tuning_plans(job,6)[0]
    space={row['parameter']:row for row in plan['space']}
    assert set(space)=={'lr','wd1','wd2'}
    assert space['wd1']['baseline']==.01 and space['wd2']['baseline']==.0005
    assert {name for trial in plan['trials'] for name in trial}==set(space)
    assert not plan['parameters_not_covered_by_budget']
    short=build_tuning_plans(job,1)[0]
    assert set(short['parameters_not_covered_by_budget'])=={'wd1','wd2'}
    assert 'wd2' in short['limitation']
    assert all('不是论文' in row['reason'] for row in short['space'])
