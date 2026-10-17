import asyncio
from types import SimpleNamespace
import pytest
from fastapi import BackgroundTasks, HTTPException

from app.services import reproduction_engine as engine
from app.services.command_intent import apply_goal_parameters
from app.api.reproductions import approve_reproduction
from app.models import ReproductionStatus


BODY="""import argparse
p=argparse.ArgumentParser()
p.add_argument('--model',default='author/large')
p.add_argument('--sample_sizes',type=int,nargs='+',default=SIZES)
p.add_argument('--datasets',nargs='+')
model=Encoder.from_pretrained(args.model)
"""


def test_declared_model_and_multivalue_samples_bind_without_importing_author_code(tmp_path):
    (tmp_path/'run.py').write_text(BODY)
    argv,overrides=apply_goal_parameters(tmp_path,['run.py'],['python','run.py','--sample_sizes','8','64'],
        '英文任务，small-author/encoder-v2，每类8个训练样本，seed=42')
    assert argv==['python','run.py','--sample_sizes','8','--model','small-author/encoder-v2']
    assert overrides=={'sample_sizes':8,'model':'small-author/encoder-v2'}
    assert 'seed' not in overrides
    argv,overrides=apply_goal_parameters(tmp_path,['run.py'],['python','run.py'],'数据集=org/reviews')
    assert overrides=={} and argv==['python','run.py']


def test_unbound_explicit_seed_blocks_plan_until_author_cli_is_revised(tmp_path):
    (tmp_path/'run.py').write_text(BODY)
    (tmp_path/'README.md').write_text('python run.py --datasets=review_en --sample_sizes=8\n')
    def plan():
        return engine._execution_command(tmp_path,engine._repository_manifest(tmp_path),'full',{'prefix':str(tmp_path/'env')},
            {'datasets':['review_en'],'user_goal':'small-author/encoder-v2，每类8样本，seed=42'},
            {'available_datasets':['review_en']})
    before=plan()
    assert before['unresolved_goal_parameters'][0]['parameter']=='seed'
    assert '种子' in before['blocked_reason']
    (tmp_path/'run.py').write_text(BODY+"p.add_argument('--seeds',type=int,nargs='+')\n")
    after=plan()
    assert not after['blocked_reason'] and not after['unresolved_goal_parameters']
    assert after['commands'][0]['argv'][-2:]==['--model','small-author/encoder-v2']
    assert after['commands'][0]['user_parameter_overrides']['seeds']==42


def test_initial_approval_rejects_unresolved_requirements_before_environment_actions():
    job=SimpleNamespace(status=ReproductionStatus.awaiting_approval,
        execution_command={'blocked_reason':'作者入口尚未绑定所要求的种子'})
    class DB:
        async def scalar(self,*args):return job
        async def commit(self):pytest.fail('Blocked plan must not be approved')
    with pytest.raises(HTTPException,match='尚未绑定') as error:
        asyncio.run(approve_reproduction('project','run',BackgroundTasks(),DB()))
    assert error.value.status_code==409
