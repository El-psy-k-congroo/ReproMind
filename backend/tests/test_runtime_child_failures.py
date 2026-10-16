"""Real local child-process test, plus offline rollback/host-scope contracts."""
import asyncio
import hashlib
import sys
import time
from types import SimpleNamespace

from app.services import reproduction_engine as engine
from app.services.host_environment import spawn_safety_issues


def test_fatal_spawn_child_does_not_wait_for_parent_timeout(tmp_path,monkeypatch):
    source=tmp_path/'paper-project';source.mkdir()
    log=source/'runs'/'run'/'telemetry'/'data-original.log'
    monkeypatch.setattr(engine,'get_settings',lambda:SimpleNamespace(paper_projects_dir=tmp_path,
        hf_endpoint='https://huggingface.co',reproduction_log_chars=10000))
    script="import time;print('Traceback (most recent call last):\\nRuntimeError: An attempt has been made to start a new process before the\\ncurrent process has finished its bootstrapping phase.',flush=True);time.sleep(30)"
    start=time.monotonic()
    code,output=asyncio.run(engine._run_process([sys.executable,'-c',script],source,20,output_path=log))
    assert code not in {0,124} and '[ChildProcessFailure]' in output
    assert time.monotonic()-start<8


def test_unverified_chain_is_reversed_without_overwriting_later_changes(tmp_path):
    source=tmp_path/'train.py';source.write_text('step=3\n')
    one=tmp_path/'one.bak';one.write_text('step=1\n')
    two=tmp_path/'two.bak';two.write_text('step=2\n')
    def record(before,after):return {'dataset':'d','file':str(source),'backup':str(before),
        'updated_sha256':hashlib.sha256(after.encode()).hexdigest(),'applied':True,'runtime_verified':False}
    rows=[record(one,'step=2\n'),record(two,'step=3\n')]
    engine._settle_repair_chain(tmp_path,rows,'d',False)
    assert source.read_text()=='step=1\n' and all(not r['applied'] for r in rows)
    source.write_text('external=4\n');rows=[record(one,'step=2\n')]
    engine._settle_repair_chain(tmp_path,rows,'d',False)
    assert source.read_text()=='external=4\n'


def test_windows_spawn_validation_preserves_data_and_other_platforms(tmp_path):
    body='''from torch.utils.data import DataLoader
loader=DataLoader(dataset,num_workers=4)
for epoch in range(2):
    train(epoch)
'''
    (tmp_path/'train.py').write_text(body)
    assert spawn_safety_issues(tmp_path,entrypoints=['train.py'],system='Windows')
    assert not spawn_safety_issues(tmp_path,entrypoints=['train.py'],system='Linux')
    assert not spawn_safety_issues(tmp_path,{'train.py':body.replace('num_workers=4','num_workers=0')},['train.py'],'Windows')
    guarded=body.split('for epoch')[0]+"if __name__ == '__main__':\n    for epoch in range(2):\n        train(epoch)\n"
    assert not spawn_safety_issues(tmp_path,{'train.py':guarded},['train.py'],'Windows')
