import asyncio
import pytest
from app.services import reproduction_engine as engine
from app.services.runtime_outcome import process_exit_code


FAILURE='''Traceback (most recent call last):
  File "main.py", line 10, in sample
    requests.get('https://author.example/checkpoint')
requests.exceptions.SSLError: TLS connection closed
'''


def test_zero_os_exit_does_not_hide_a_reported_download_exception(monkeypatch):
    async def process(*args,**kwargs):return 0, FAILURE
    monkeypatch.setattr(engine,'_run_process',process)
    code,output=asyncio.run(engine._run_author_process([],None,10,{}))
    assert code==1 and process_exit_code(code,output)==0
    assert engine._diagnose_failure(output)['type']=='network_download'
    assert '关闭 TLS 校验' in engine._diagnose_failure(output)['suggestion']


def test_plain_exception_example_is_not_a_traceback_and_later_test_results_show_recovery(monkeypatch):
    async def process(*args,**kwargs):return 0,'ValueError: example text only\nTest accuracy=0.8'
    monkeypatch.setattr(engine,'_run_process',process)
    assert asyncio.run(engine._run_author_process([],None,10,{}))[0]==0
    async def recovered(*args,**kwargs):return 0,FAILURE+'Test accuracy=0.8\n'
    monkeypatch.setattr(engine,'_run_process',recovered)
    assert asyncio.run(engine._run_author_process([],None,10,{}))[0]==0


@pytest.mark.parametrize('kind', ['ChunkedEncodingError','ContentDecodingError','ProtocolError','HTTPError'])
def test_interrupted_author_transfer_is_a_resource_failure(kind):
    output = FAILURE.replace('SSLError',kind).replace('TLS connection closed','Connection broken: IncompleteRead(20 bytes read, 100 more expected)')
    assert engine._diagnose_failure(output)['type']=='network_download'


def test_truncated_checkpoint_is_not_a_source_bug_or_missing_training_data():
    output = 'Traceback (most recent call last):\n    model.load_state_dict(torch.load(ckpt))\nRuntimeError: unexpected EOF, expected 100 more bytes. The file might be corrupted.\n'
    assert engine._diagnose_failure(output)['type']=='model_artifact_incomplete'
    assert engine._diagnose_failure(output.replace('torch.load(ckpt)','create_model()').replace('model.load_state_dict','print'))['type']=='execution_error'
