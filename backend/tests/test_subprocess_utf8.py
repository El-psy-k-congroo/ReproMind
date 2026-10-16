import asyncio
import sys
from app.services import reproduction_engine as engine


def test_python_subprocess_uses_utf8_for_project_files(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONUTF8','0')
    (tmp_path/'requirements.txt').write_bytes('# 中文依赖注释\nnumpy==1.26.4\n'.encode('utf-8'))
    script="import sys,pathlib; assert sys.flags.utf8_mode == 1; assert 'numpy==1.26.4' in pathlib.Path('requirements.txt').read_text(); print('utf8 file read passed')"
    code, output=asyncio.run(engine._run_process([sys.executable,'-c',script],tmp_path,10))
    assert code==0 and 'utf8 file read passed' in output
