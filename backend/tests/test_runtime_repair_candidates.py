"""Offline repair control flow: candidate evidence, rollback and duplicate suppression."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from app.services import reproduction_engine as engine


@pytest.mark.parametrize('repeat', [False,True])
def test_failed_candidate_is_retained_and_given_to_refinement(tmp_path,monkeypatch,repeat):
    source=tmp_path/'branches'/'run'/'source';source.mkdir(parents=True)
    path=source/'run.py';original="raise RuntimeError('old')\n";path.write_text(original)
    job=SimpleNamespace(id='run',project_id='project',repository_url='repo',paper_spec={},code_spec={},model_id='',
        result_json={},execution_command={'commands':[{'dataset':'Example','cwd':str(source)}]})
    prompts=[];calls=[]
    class Model:
        async def complete_json(self,role,system,prompt,**kwargs):
            prompts.append(json.loads(prompt))
            replacement="raise RuntimeError('old') # candidate" if len(prompts)==1 or repeat else "print('fixed')"
            return SimpleNamespace(data={'old_text':"raise RuntimeError('old')",'new_text':replacement,'rationale':'offline candidate'},error='',usage={},model='offline')
    async def history(*args):return []
    failure=f'File "{path}", line 1, in <module>\nRuntimeError: old'
    async def run(*args,**kwargs):
        calls.append(True)
        return (1,failure) if len(calls)==1 else (0,'Test accuracy=0.9')
    monkeypatch.setattr(engine,'get_model_router',lambda:Model())
    monkeypatch.setattr(engine,'_prior_repair_experience',history)
    monkeypatch.setattr(engine,'_run_process',run)
    first=asyncio.run(engine._agent_repair_code(job,tmp_path,'Example',failure,[sys.executable,'run.py'],source,{},1))
    record=first[2]
    assert not record['applied'] and record.get('retryable_proposal_failure'),record.get('reason')
    assert path.read_text()==original
    evidence=Path(record['candidate_evidence'])
    assert evidence.is_file() and json.loads(evidence.read_text())['new_text'].endswith('# candidate')
    job.result_json={'repair_attempts':[record]}
    second=asyncio.run(engine._agent_repair_code(job,tmp_path,'Example',failure,[sys.executable,'run.py'],source,{},2))
    assert prompts[1]['current_run_failed_candidates'][0]['retry_failure_tail']==failure
    if repeat:
        assert len(calls)==1 and '相同' in second[2]['reason'] and path.read_text()==original
    else:
        assert len(calls)==2 and second[2]['runtime_verified'] and path.read_text()=="print('fixed')\n"
