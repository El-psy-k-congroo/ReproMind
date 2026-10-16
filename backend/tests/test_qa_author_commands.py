"""Regression for SGC's real layout: citation.py at root, train.py downstream."""
from pathlib import Path
from types import SimpleNamespace

from app.services import reproduction_engine as engine


def author_layout(tmp_path):
    (tmp_path / 'citation.py').write_text('from args import get_citation_args\n', encoding='utf-8')
    (tmp_path / 'requirements.txt').write_text('numpy\n', encoding='utf-8')
    (tmp_path / 'README.md').write_text('Usage:\n'
        'python citation.py --dataset cora --tuned\n'
        'python citation.py --dataset citeseer --tuned --epochs 150\n', encoding='utf-8')
    nested = tmp_path / 'downstream' / 'TextSGC'
    nested.mkdir(parents=True)
    (nested / 'train.py').write_text('print("unrelated task")\n', encoding='utf-8')
    (nested / 'requirements.txt').write_text('nltk\n', encoding='utf-8')
    return engine._repository_manifest(tmp_path)


def test_author_root_entry_and_dataset_specific_args_are_preserved(tmp_path):
    manifest = author_layout(tmp_path)
    assert manifest['entrypoints'][0] == 'citation.py'
    command = engine._execution_command(tmp_path, manifest, 'full', {'prefix': str(tmp_path / 'env')},
        {'datasets': ['Cora', 'Citeseer']}, {'available_datasets': ['Cora', 'Citeseer']})
    cora, citeseer = command['commands']
    assert cora['argv'][1:] == ['citation.py', '--dataset', 'cora', '--tuned']
    assert citeseer['argv'][1:] == ['citation.py', '--dataset', 'citeseer', '--tuned', '--epochs', '150']
    assert cora['cwd'] == str(tmp_path) and not command['unplanned_datasets']


def test_dependencies_follow_selected_author_entry_not_last_nested_file(tmp_path, monkeypatch):
    manifest = author_layout(tmp_path)
    monkeypatch.setattr(engine, '_hardware_plan', lambda *_: {'available': False, 'requirement': 'not_detected'})
    monkeypatch.setattr(engine, '_conda_executable', lambda: 'conda')
    monkeypatch.setattr(engine, '_named_environment_prefix', lambda *_: tmp_path / 'env')
    plan = engine._environment_plan(tmp_path, manifest, tmp_path / 'paper-12345678', 'SGC')
    assert plan['dependency_source'] == 'requirements.txt'
    assert plan['install_commands'][0][-1] == str(tmp_path / 'requirements.txt')


def test_readme_shell_operator_and_external_paths_are_not_executable_candidates(tmp_path):
    (tmp_path / 'train.py').write_text('print("safe")', encoding='utf-8')
    (tmp_path / 'README.md').write_text('python ../outside.py --dataset cora\n'
        'python train.py --dataset cora && other_command\n', encoding='utf-8')
    assert engine._repository_manifest(tmp_path)['readme_commands'] == []


def test_verified_codeload_cache_skips_git_and_network(tmp_path, monkeypatch):
    import asyncio
    url = 'https://github.com/Tiiiger/SGC'
    outer = tmp_path / 'sources' / f'{engine._repository_cache_name(url)}-archive'
    inner = outer / 'SGC-master'
    inner.mkdir(parents=True)
    (inner / 'citation.py').write_text('print("author source")', encoding='utf-8')
    engine._record_repository_identity(outer, url)
    async def forbidden(*_, **__):
        raise AssertionError('Cached repository must not clone or fetch again')
    monkeypatch.setattr(engine, '_run_process', forbidden)
    monkeypatch.setattr(engine.httpx.AsyncClient, 'get', forbidden)
    for _ in range(2):
        source, log = asyncio.run(engine._clone_repository(url, tmp_path, tmp_path))
        assert source == inner


def test_readme_install_and_preprocessing_commands_do_not_replace_training(tmp_path):
    for name in ['setup.py', 'preprocess.py', 'train.py']:
        (tmp_path / name).write_text('pass', encoding='utf-8')
    (tmp_path / 'README.md').write_text('python setup.py install\npython preprocess.py\npython train.py\n', encoding='utf-8')
    assert engine._repository_manifest(tmp_path)['entrypoints'][0] == 'train.py'
