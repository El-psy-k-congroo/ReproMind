import asyncio
import copy
import json
from types import SimpleNamespace

from app.services import project_coding as coding
from app.services.run_parameters import reviewed_execution_plan
from app.services.command_intent import apply_goal_parameters


def test_reviewed_default_overrides_regenerated_old_goal_without_mutating_baseline(tmp_path):
    old="parser.add_argument('--width', default=16)\nparser.add_argument('--dropout', default=0.5)\n"
    (tmp_path/'train.py').write_text(old)
    baseline={'commands':[{'dataset':'Example','argv':['python','train.py','--width','16','--dropout','.1'],
        'user_parameter_overrides':{'width':16}}]}
    original=copy.deepcopy(baseline)
    projected=reviewed_execution_plan(baseline,tmp_path,['train.py'],{'width'},{'train.py':old.replace('16','32')})
    argv=projected['commands'][0]['argv']
    assert argv[argv.index('--width')+1]=='32' and argv[argv.index('--dropout')+1]=='.1'
    assert baseline==original and (tmp_path/'train.py').read_text()==old
    (tmp_path/'train.py').write_text(old.replace('16','32'))
    assert reviewed_execution_plan(baseline,tmp_path,['train.py'],{'width'})==projected


def test_semantic_reviewer_receives_projected_command_and_separate_history(tmp_path, monkeypatch):
    old="parser.add_argument('--width', default=16)\n"
    (tmp_path/'train.py').write_text(old)
    class Router:
        def __init__(self): self.calls=[]
        async def complete_json(self,role,system,prompt,**kwargs):
            payload=json.loads(prompt);self.calls.append(payload)
            responses=[{'read_files':['train.py'],'acceptance':['width is 32']},
                {'changes':[{'file':'train.py','old_text':'default=16','new_text':'default=32'}]},
                {'fulfilled':True,'issues':[]}]
            return SimpleNamespace(data=responses[len(self.calls)-1],error='',usage={'total_tokens':2},model='offline')
    router=Router();monkeypatch.setattr(coding,'get_model_router',lambda:router)
    contract={'goal':'width=16','entrypoints':['train.py'],'commands':[{'argv':['python','train.py','--width','16']}]}
    result=asyncio.run(coding.propose_revision(tmp_path,'change width to 32',context={'task_contract':contract}))
    reviewed=router.calls[-1]
    assert reviewed['task_contract']['commands'][0]['argv'][-1]=='32'
    assert reviewed['baseline_task_contract']['commands'][0]['argv'][-1]=='16'
    assert reviewed['task_contract']['execution_state']=='proposed_not_executed'
    assert contract['goal']=='width=16' and (tmp_path/'train.py').read_text()==old
    assert result[1]['train.py'].endswith('default=32)\n')


def test_numeric_transition_uses_new_value_and_later_explicit_value(tmp_path):
    (tmp_path/'train.py').write_text("p.add_argument('--early_stopping', default=10)\np.add_argument('--width', default=16)\n")
    argv=['python','train.py']
    new, values=apply_goal_parameters(tmp_path,['train.py'],argv,'仅将early_stopping从10增大为30，width由16改为32')
    assert values=={'early_stopping':30,'width':32}
    assert new[new.index('--early_stopping')+1]=='30'
    _, values=apply_goal_parameters(tmp_path,['train.py'],argv,'早停窗口从10增大到30，early_stopping=40。不要修改width=8')
    assert values=={'early_stopping':40}
