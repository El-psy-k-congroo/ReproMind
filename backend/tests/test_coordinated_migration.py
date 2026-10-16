"""Offline migration contracts; no model, installation, import or training executed."""
import asyncio
import json
from types import SimpleNamespace
import pytest
from app.services import framework_migration as migration
from app.api import copilot


@pytest.mark.parametrize('question', ['现在进行到了哪一步了', '目前到了哪个阶段', '现在是什么阶段', '当前进行到哪了'])
def test_progress_paraphrases_use_local_facts(question):
    assert copilot._status_question(question)


def test_complete_bundle_repairs_actual_interface_feedback(tmp_path, monkeypatch):
    original = {'entry.py': 'import tensorflow as tf\nfrom transform import features\nx = features()\n',
                'transform.py': 'def features():\n    return (1, 2, 3)\n'}
    for name, body in original.items(): (tmp_path / name).write_text(body, encoding='utf-8')
    calls = []
    async def complete(*args, **kwargs):
        context = json.loads(args[2]); calls.append(context)
        assert set(context['files']) == set(original)
        if len(calls) == 1:
            content = 'import torch\nfrom transform import features\nx = features()\nx = x.toarray()\n'
        else:
            assert 'toarray' in str(context['validation_feedback'])
            content = 'import torch\nfrom transform import features\nx = features()\nx = torch.tensor(x[1])\n'
        return SimpleNamespace(data={'files': [{'path': 'entry.py', 'content': content}]}, error='', usage={}, model='offline')
    monkeypatch.setattr(migration, 'get_model_router', lambda: SimpleNamespace(complete_json=complete))
    originals, revisions, response, usage = asyncio.run(migration.propose_migration(tmp_path,
        [tmp_path / name for name in original], '迁移为 PyTorch', 'pytorch'))
    assert len(calls) == 2 and list(revisions) == ['entry.py']
    assert 'torch.tensor' in revisions['entry.py']
    assert all((tmp_path / name).read_text(encoding='utf-8') == body for name, body in original.items())


@pytest.mark.parametrize('files', [[{'path': '../escape.py', 'content': 'pass'}],
    [{'path': 'entry.py', 'content': ''}], [{'path': 'entry.py', 'content': 'pass'}] * 2])
def test_bundle_rejects_invalid_paths_empty_or_duplicate(files):
    with pytest.raises(ValueError): migration.parse_bundle({'files': files}, {'entry.py': 'pass'})


def test_missing_local_symbol_is_reported(tmp_path):
    (tmp_path / 'layers.py').write_text('class NewLayer: pass\n', encoding='utf-8')
    (tmp_path / 'entry.py').write_text('import torch\nfrom layers import OldLayer\n', encoding='utf-8')
    assert any('OldLayer' in item for item in migration.validate_bundle(tmp_path, {}, 'pytorch'))


def test_old_framework_cannot_pass_as_complete(tmp_path):
    (tmp_path / 'layers.py').write_text('import tensorflow as tf\n', encoding='utf-8')
    (tmp_path / 'entry.py').write_text('import torch\n', encoding='utf-8')
    assert migration.validate_bundle(tmp_path, {}, 'pytorch')


def test_repeated_bad_bundle_stops_after_one_repair(tmp_path, monkeypatch):
    path = tmp_path / 'entry.py'; path.write_text('import tensorflow as tf\n', encoding='utf-8')
    calls = []
    async def complete(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(data={'files': [{'path': 'entry.py', 'content': 'def broken('}]}, error='', usage={})
    monkeypatch.setattr(migration, 'get_model_router', lambda: SimpleNamespace(complete_json=complete))
    with pytest.raises(ValueError, match='静态校验'):
        asyncio.run(migration.propose_migration(tmp_path, [path], '迁移到 PyTorch', 'pytorch'))
    assert len(calls) == 2 and 'tensorflow' in path.read_text(encoding='utf-8')
