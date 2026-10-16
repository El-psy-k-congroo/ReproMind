"""Offline regression of chained runtime failures observed in independent UI QA."""
import asyncio
from types import SimpleNamespace

from app.services import reproduction_engine as engine
from app.services.runtime_compatibility import terminal_exception


def test_nvml_terminal_exception_is_not_missing_training_data(tmp_path):
    output = ("FileNotFoundError: Could not find module 'C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll'\n"
              "During handling of the above exception, another exception occurred:\n"
              "pynvml.nvml.NVMLError_LibraryNotFound: NVML Shared Library Not Found\n")
    assert terminal_exception(output)[0] == 'pynvml.nvml.NVMLError_LibraryNotFound'
    diagnosis = engine._diagnose_failure(output, tmp_path, 'Adult')
    assert diagnosis['type'] == 'runtime_library_missing'
    assert '上传' not in diagnosis.get('expected', '')


def test_missing_config_weights_and_data_are_different(tmp_path):
    def diagnose(path, stack=''):
        return engine._diagnose_failure(stack + f"FileNotFoundError: [Errno 2] No such file: '{path}'\n", tmp_path, 'Example')
    assert diagnose('config/run.toml')['type'] == 'missing_runtime_file'
    assert diagnose('checkpoints/model.pt', 'torch.load(checkpoint)\n')['type'] == 'missing_model_artifact'
    assert diagnose('data/train.csv')['type'] == 'missing_dataset'
    assert diagnose('C:/Runtime/backend.dll')['type'] == 'runtime_library_missing'


def test_model_failure_is_not_reported_as_invalid_patch(tmp_path, monkeypatch):
    source = tmp_path / 'sources' / 'author'; source.mkdir(parents=True)
    entry = source / 'run.py'; entry.write_text('raise RuntimeError("failure")\n')
    job = SimpleNamespace(id='r', project_id='p', goal='', paper_spec={}, code_spec={},
                          environment_spec={}, model_id='', execution_command={})
    async def no_history(*args): return []
    monkeypatch.setattr(engine, '_prior_repair_experience', no_history)
    monkeypatch.setattr(engine, 'task_contract', lambda job: {})
    monkeypatch.setattr(engine, 'task_model_id', lambda *args: 'test')
    captured = {}
    async def model(*args, **kwargs):
        import json
        captured.update(json.loads(args[2]))
        return SimpleNamespace(error='ReadTimeout: upstream', model='test', usage={}, data={})
    monkeypatch.setattr(engine, 'get_model_router', lambda: SimpleNamespace(complete_json=model))
    output = f'Traceback (most recent call last):\n  File "{entry}", line 1, in <module>\nRuntimeError: failure\n'
    result = asyncio.run(engine._agent_repair_code(job, tmp_path, 'D', output,
                        ['python', 'run.py'], source, {}, 1))
    assert 'ReadTimeout' in result[2]['reason'] and not result[2]['applied']
    assert captured['source_excerpt'] == 'raise RuntimeError("failure")'
    assert captured['source_start_line'] == 1
    assert entry.read_text() == 'raise RuntimeError("failure")\n'
