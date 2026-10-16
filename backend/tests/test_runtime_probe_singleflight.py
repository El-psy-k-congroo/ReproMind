import asyncio
import time
from types import SimpleNamespace

from app.services import runtime_evidence


def test_first_step_device_proof_survives_long_framework_startup(tmp_path):
    telemetry = tmp_path / 'runs' / 'job' / 'telemetry'
    telemetry.mkdir(parents=True)
    proof = 'Op graphconvolution_1/SparseTensorDenseMatMul on device: /job:localhost/replica:0/task:0/device:GPU:0'
    (telemetry / 'nell.log').write_text('framework warning\n' * 300 + proof + '\n' + 'epoch\n' * 1000)
    logs = runtime_evidence.execution_logs(tmp_path, 'job')
    assert proof not in logs[0]['head'] and proof not in logs[0]['tail']
    assert runtime_evidence.device_log_evidence(logs)[0]['kind'] == 'gpu_compute_placement'
    assert len(logs[0]['device_lines']) <= 5600


def test_compute_scope_name_alone_does_not_prove_compute_placement():
    line = 'Op gradients/MatMul_grad/GatherV2/axis on device: /device:GPU:0'
    assert not runtime_evidence.device_log_evidence([{'log': 'run.log', 'head': line}])


def test_concurrent_readiness_queries_share_one_subprocess(tmp_path, monkeypatch):
    python = tmp_path / 'python.exe'
    python.write_text('fake')
    calls = []
    def run(*args, **kwargs):
        calls.append(args)
        time.sleep(.03)
        return SimpleNamespace(returncode=0, stdout='{"checked":true,"version":"2.10.0"}', stderr='')
    monkeypatch.setattr(runtime_evidence.subprocess, 'run', run)
    async def scenario():
        answers = await asyncio.gather(*(runtime_evidence.probe_framework(python, 'tensorflow') for _ in range(8)))
        assert all(answer['checked'] for answer in answers)
        assert len(calls) == 1
    asyncio.run(scenario())


def test_cancelled_client_does_not_orphan_or_duplicate_the_probe(tmp_path, monkeypatch):
    python = tmp_path / 'python.exe'
    python.write_text('fake')
    calls = []
    def run(*args, **kwargs):
        calls.append(args)
        time.sleep(.05)
        return SimpleNamespace(returncode=0, stdout='{"cuda_available":true}', stderr='')
    monkeypatch.setattr(runtime_evidence.subprocess, 'run', run)
    async def scenario():
        first = asyncio.create_task(runtime_evidence.probe_framework(python, 'tensorflow'))
        await asyncio.sleep(.01)
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass
        result = await runtime_evidence.probe_framework(python, 'tensorflow')
        assert result['cuda_available'] and len(calls) == 1
    asyncio.run(scenario())
