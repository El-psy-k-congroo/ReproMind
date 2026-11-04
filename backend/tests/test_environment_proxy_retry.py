import asyncio
from types import SimpleNamespace

import pytest

from app.services import environment_network as network, reproduction_engine as engine


def test_only_failed_local_proxy_can_use_scoped_direct_retry(monkeypatch):
    command = ['python', '-m', 'pip', 'install', 'numpy==1.26.4', '--index-url', 'https://approved.example/simple']
    monkeypatch.setattr(network, 'getproxies', lambda: {'https': 'http://127.0.0.1:7897'})
    direct = network.direct_pip_retry(command, 'ProxyError: Cannot connect to proxy')
    assert direct['NO_PROXY'] == '*' and direct['HTTPS_PROXY'] == ''
    assert 'PYTHONHTTPSVERIFY' not in direct and 'PIP_TRUSTED_HOST' not in direct
    assert network.direct_pip_retry(command, 'CERTIFICATE_VERIFY_FAILED') is None
    assert network.direct_pip_retry(['conda', 'create'], 'ProxyError') is None
    monkeypatch.setattr(network, 'getproxies', lambda: {'https': 'https://company.example:8080'})
    assert network.direct_pip_retry(command, 'ProxyError') is None


def test_failed_and_successful_install_logs_are_both_preserved(tmp_path, monkeypatch):
    job = SimpleNamespace(id='run', result_json={}, logs='')
    class Session:
        async def commit(self): pass
        async def refresh(self, *args, **kwargs): pass
    monkeypatch.setattr(network, 'getproxies', lambda: {'https': 'http://127.0.0.1:7897'})
    calls=[]
    async def process(command, cwd, timeout, output_path, extra_env=None):
        calls.append((command, extra_env))
        output_path.write_text('ProxyError' if len(calls)==1 else 'Successfully installed package',encoding='utf-8')
        return (1, 'ProxyError') if len(calls)==1 else (0, 'Successfully installed package')
    monkeypatch.setattr(engine, '_run_process', process)
    command=['python','-m','pip','install','package==1']
    code, output=asyncio.run(engine._run_environment_operation(Session(),job,tmp_path,command,10,tmp_path/'install.log',kind='dependency_install',step=1,total_steps=1))
    assert code==0 and len(calls)==2
    assert calls[0][0]==calls[1][0] and calls[1][1]['NO_PROXY']=='*'
    assert 'ProxyError' in output and 'Successfully installed' in output
    assert (tmp_path/'install.log').is_file() and (tmp_path/'install-direct.log').is_file()
    assert 'active_operation' not in job.result_json
    code, _=asyncio.run(engine._run_environment_operation(Session(),job,tmp_path,command,10,tmp_path/'next-install.log',kind='dependency_install',step=1,total_steps=1))
    assert code==0 and len(calls)==3 and calls[-1][1]['NO_PROXY']=='*'
