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
    assert 'ProxyError' not in output and 'Successfully installed' in output
    assert 'ProxyError' in (tmp_path/'install.log').read_text(encoding='utf-8')
    assert (tmp_path/'install.log').is_file() and (tmp_path/'install-direct.log').is_file()
    assert 'active_operation' not in job.result_json
    code, _=asyncio.run(engine._run_environment_operation(Session(),job,tmp_path,command,10,tmp_path/'next-install.log',kind='dependency_install',step=1,total_steps=1))
    assert code==0 and len(calls)==3 and calls[-1][1]['NO_PROXY']=='*'


def test_verified_wheel_installer_retries_dependency_resolution_with_scoped_direct_route(tmp_path, monkeypatch):
    from pathlib import Path
    installer=Path(network.__file__).with_name('registry_wheel_download.py').resolve()
    command=['python',str(installer),'tensorflow','2.10.0',str(tmp_path),'--install']
    monkeypatch.setattr(network,'getproxies',lambda:{'https':'http://127.0.0.1:7897'})
    job=SimpleNamespace(id='run',result_json={},logs='')
    class Session:
        async def commit(self): pass
        async def refresh(self,*args,**kwargs): pass
    calls=[]
    async def process(argv,cwd,timeout,output_path,extra_env=None):
        calls.append((list(argv),extra_env))
        output='ProxyError: Cannot connect to proxy; No matching distribution found for flatbuffers' if len(calls)==1 else 'Successfully installed tensorflow'
        output_path.write_text(output,encoding='utf-8')
        return (1 if len(calls)==1 else 0),output
    monkeypatch.setattr(engine,'_run_process',process)
    code,output=asyncio.run(engine._run_environment_operation(Session(),job,tmp_path,command,30,tmp_path/'wheel.log',kind='tensorflow_gpu_migration',step=3,total_steps=4))
    assert code==0 and len(calls)==2 and calls[0][0]==calls[1][0]
    assert calls[1][1]['NO_PROXY']=='*' and 'Successfully installed' in output
    assert (tmp_path/'wheel.log').is_file() and (tmp_path/'wheel-direct.log').is_file()
    assert job.result_json['pip_network_route']=='direct_local_proxy_retry'
    assert network.direct_pip_retry(['python',str(tmp_path/'registry_wheel_download.py'),'tensorflow','2.10.0',str(tmp_path),'--install'],'ProxyError') is None
    assert network.direct_pip_retry(command[:-1],'ProxyError') is None


def test_direct_retry_reports_latest_solver_failure_and_retains_failed_proxy_log(tmp_path,monkeypatch):
    monkeypatch.setattr(network,'getproxies',lambda:{'https':'http://127.0.0.1:7897'})
    job=SimpleNamespace(id='run',result_json={},logs='')
    class Session:
        async def commit(self):pass
        async def refresh(self,*args,**kwargs):pass
    calls=[]
    async def process(command,cwd,timeout,output_path,extra_env=None):
        calls.append(extra_env)
        text='ProxyError: Cannot connect to proxy' if len(calls)==1 else 'No matching distribution found for unsupported-package'
        output_path.write_text(text,encoding='utf-8')
        return 1,text
    monkeypatch.setattr(engine,'_run_process',process)
    code,output=asyncio.run(engine._run_environment_operation(Session(),job,tmp_path,['python','-m','pip','install','unsupported-package'],10,tmp_path/'failure.log',kind='dependency_install',step=1,total_steps=1))
    from app.services.environment_recovery import failure_kind
    assert failure_kind(code,output)=='incompatible_package'
    assert 'ProxyError' in (tmp_path/'failure.log').read_text(encoding='utf-8')
    assert job.result_json['pip_network_route']=='direct_local_proxy_retry'


def test_environment_transport_retry_preserves_explicit_timeout_budget(tmp_path,monkeypatch):
    calls=[]
    async def operation(session,job,workspace,command,timeout,log,**kwargs):
        calls.append(timeout)
        return (1,'curl: (28) Operation timed out') if len(calls)==1 else (0,'installed')
    async def advance(*args,**kwargs):pass
    monkeypatch.setattr(engine,'_run_environment_operation',operation)
    monkeypatch.setattr(engine,'_advance',advance)
    code,_=asyncio.run(engine._recover_environment_operation(None,SimpleNamespace(id='run'),tmp_path,['python','installer'],tmp_path/'network.log','tensorflow_gpu_migration',3,4,timeout=17))
    assert code==0 and calls==[17,17]
