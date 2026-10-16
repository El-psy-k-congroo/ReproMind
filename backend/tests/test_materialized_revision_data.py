"""Keep isolated author data without copying platform download staging archives."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

from app.api import experiment_branches as branches
from app.services import reproduction_engine as engine


def fixture(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    (source / 'data').mkdir(); (source / 'data' / 'train.csv').write_text('x,y\n1,0\n')
    request = {'dataset': 'Example', 'download_status': 'ready', 'installed_files': ['data/train.csv']}
    job = SimpleNamespace(id='r', logs='', workspace_path=str(tmp_path),
        dataset_spec={'automatic_download': True, 'dataset_mentions': ['Example'], 'download_requests': [request]},
        execution_command={'cwd':str(source), 'pythonpath':str(source), 'commands':[]})
    return source, request, job


def test_cache_excluded_only_after_outputs_exist(tmp_path):
    source, request, job = fixture(tmp_path)
    assert branches._materialized_cache_exclusions(job, source, ['Example']) == ('author',)
    request['installed_files'] = ['../outside.csv']
    assert not branches._materialized_cache_exclusions(job, source, ['Example'])
    request['installed_files'] = ['data/missing.csv']
    assert not branches._materialized_cache_exclusions(job, source, ['Example'])
    request['installed_files'] = ['data/train.csv']
    job.execution_command['commands'] = [{'argv':['python', str(tmp_path/'data'/'author'/'train.csv')]}]
    assert not branches._materialized_cache_exclusions(job, source, ['Example'])


def test_copy_retains_actual_inputs_and_does_not_duplicate_staging(tmp_path):
    source = tmp_path / 'data'; (source/'author').mkdir(parents=True)
    (source/'author'/'archive.zip').write_bytes(b'cached original')
    (source/'actual.csv').write_bytes(b'actual dataset')
    (source/'nested'/'author').mkdir(parents=True)
    (source/'nested'/'author'/'actual.csv').write_bytes(b'not a platform root cache')
    target=tmp_path/'branch'/'data'
    branches._copy_bounded(source,target,('author',))
    assert not (target/'author').exists()
    assert (target/'actual.csv').read_bytes() == b'actual dataset'
    assert (target/'nested'/'author'/'actual.csv').read_bytes() == b'not a platform root cache'
    assert (source/'author'/'archive.zip').exists()


def test_materialized_files_skip_archive_transfer_in_new_experiment(tmp_path, monkeypatch):
    source, request, job = fixture(tmp_path)
    async def fail_transfer(*args, **kwargs): raise AssertionError('Already materialized data must not be downloaded again')
    async def advance(*args): pass
    class Session:
        async def commit(self): pass
    monkeypatch.setattr(engine,'download_file',fail_transfer)
    monkeypatch.setattr(engine,'_advance',advance)
    asyncio.run(engine._download_small_datasets(Session(),job,Path('python'),tmp_path))
    assert job.dataset_spec['all_datasets_resolved']
    assert job.dataset_spec['missing_datasets'] == []
    assert (source/'data'/'train.csv').is_file()
