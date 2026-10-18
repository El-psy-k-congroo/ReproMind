import json
import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from app.services import execution_sandbox as sandbox


def test_local_execution_keeps_existing_behavior(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox, 'get_settings', lambda: SimpleNamespace(execution_sandbox=False))
    argv, env = sandbox.sandbox_command(['python','train.py'], tmp_path, {'PATH':'original'})
    assert argv == ['python','train.py'] and env == {'PATH':'original'}


@pytest.mark.skipif(os.name != 'posix' or not shutil.which('bwrap'), reason='Linux bubblewrap integration')
def test_linux_worker_can_write_own_files_but_not_read_other_users_or_service_secrets(tmp_path, monkeypatch):
    root=tmp_path/'projects'; work=root/'owned-12345678'; other=root/'other-87654321'
    work.mkdir(parents=True);other.mkdir()
    secret=other/'private.txt'; secret.write_text('other-account')
    monkeypatch.setattr(sandbox, 'get_settings', lambda: SimpleNamespace(execution_sandbox=True,paper_projects_dir=root))
    script=("import json,os,pathlib; pathlib.Path('own.txt').write_text('ok'); "
            "print(json.dumps({'own':pathlib.Path('own.txt').read_text(),"
            f"'other':pathlib.Path({str(secret)!r}).exists(),"
            "'secret':'LLM_API_KEY' in os.environ,'db':pathlib.Path('/var/lib/repromind/repromind.db').exists()}))")
    argv, env=sandbox.sandbox_command(['/usr/bin/python3','-c',script],work,{'PATH':'/usr/bin:/bin','LLM_API_KEY':'do-not-pass'},workspace=work)
    result=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0, result.stderr
    assert json.loads(result.stdout)=={'own':'ok','other':False,'secret':False,'db':False}
    assert (work/'own.txt').read_text()=='ok'
    with pytest.raises(ValueError): sandbox.sandbox_command(['/bin/true'],other,{},workspace=work)
    link=work/'foreign-link';link.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError): sandbox.sandbox_command(['/bin/true'],link,{})
