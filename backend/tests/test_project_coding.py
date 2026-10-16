"""Offline workflow contracts, not real-model or experiment success evidence."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from app.services import project_coding as coding
from app.api import copilot


class Router:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    async def complete_json(self, role, system, prompt, **kwargs):
        self.calls.append((role, json.loads(prompt)))
        return SimpleNamespace(data=next(self.replies), error='', usage={'total_tokens': 2}, model='offline')


def test_general_request_uses_planner_shared_snapshot_and_no_writes(tmp_path, monkeypatch):
    (tmp_path / 'util.py').write_text('def value():\n    return 1\n')
    (tmp_path / 'train.py').write_text('from util import value\nprint(value())\n')
    router = Router([
        {'read_files': ['util.py', 'train.py'], 'plan': ['change interface and consumer']},
        {'changes': [
            {'file': 'util.py', 'old_text': 'value()', 'new_text': 'value(scale)'},
            {'file': 'train.py', 'old_text': 'value()', 'new_text': 'value(2)'},
        ]},
    ])
    monkeypatch.setattr(coding, 'get_model_router', lambda: router)
    original, revised, changes, response, usage, trace = asyncio.run(coding.propose_revision(
        tmp_path, '给计算增加 scale 参数并同步调用者'))
    assert len(changes) == 2 and usage == 4
    assert 'util.py' in router.calls[1][1]['files'] and 'train.py' in router.calls[1][1]['files']
    assert all((tmp_path / n).read_text() == b for n, b in original.items())
    assert trace[0]['agent'] == 'Code Planning Agent'


def test_transient_planning_timeout_retries_within_original_budget():
    class Transient:
        calls = 0
        async def complete_json(self, *args, **kwargs):
            self.calls += 1
            assert kwargs['timeout_seconds'] == 20
            return SimpleNamespace(error='ReadTimeout' if self.calls == 1 else '', data={'read_files':['train.py']})
    router=Transient()
    result=asyncio.run(coding._bounded_json(router,None,'system','prompt',timeout_seconds=40))
    assert router.calls==2 and not result.error


def test_semantic_feedback_repairs_static_valid_but_wrong_behavior(tmp_path,monkeypatch):
    (tmp_path/'train.py').write_text('count=1\n')
    router=Router([{'read_files':['train.py'],'acceptance':['count must be 3']},
        {'changes':[{'file':'train.py','old_text':'count=1','new_text':'count=2'}]},
        {'fulfilled':False,'issues':['train.py: count=2 does not fulfill count=3']},
        {'changes':[{'file':'train.py','old_text':'count=2','new_text':'count=3'}]},
        {'fulfilled':True,'issues':[]}])
    monkeypatch.setattr(coding,'get_model_router',lambda:router)
    original,revised,_,_,_,trace=asyncio.run(coding.propose_revision(tmp_path,'set count to 3',context={'task_contract':{'entrypoints':['train.py']}}))
    assert revised['train.py']=='count=3\n' and (tmp_path/'train.py').read_text()==original['train.py']
    assert any(row['agent']=='Semantic Review Agent' and row['status']=='needs_repair' for row in trace)
    assert trace[-1]['status']=='completed'


def test_over_budget_planning_is_reduced_before_code_request(tmp_path,monkeypatch):
    for name in ['a.py','b.py']:(tmp_path/name).write_text('n = 1\n#'+'x'*35+'\n')
    router=Router([{'read_files':['a.py','b.py']},{'read_files':['a.py']},
                   {'changes':[{'file':'a.py','old_text':'n = 1','new_text':'n = 2'}]}])
    monkeypatch.setattr(coding,'MAX_CONTEXT',50)
    monkeypatch.setattr(coding,'get_model_router',lambda:router)
    original,revised,_,_,_,trace=asyncio.run(coding.propose_revision(tmp_path,'调整一个默认值'))
    assert set(revised)=={'a.py'} and trace[0]['status']=='needs_replan'
    assert router.calls[1][1]['context_budget_bytes']==50
    assert (tmp_path/'a.py').read_text()==original['a.py']


def test_additional_read_and_syntax_repair_are_bounded(tmp_path, monkeypatch):
    (tmp_path / 'a.py').write_text('n = 1\n')
    (tmp_path / 'b.py').write_text('n = 2\n')
    router = Router([
        {'read_files': ['a.py']},
        {'read_files': ['b.py', 'b.py'], 'changes': []},
        {'changes': [{'file': 'b.py', 'old_text': 'n = 2', 'new_text': 'n = ('}]},
        {'changes': [{'file': 'b.py', 'old_text': 'n = (', 'new_text': 'n = 3'}]},
    ])
    monkeypatch.setattr(coding, 'get_model_router', lambda: router)
    original, revised, *rest = asyncio.run(coding.propose_revision(tmp_path, '同步更新两处默认值'))
    assert revised == {'b.py': 'n = 3\n'}
    assert router.calls[-1][1]['validation_feedback']
    assert (tmp_path / 'b.py').read_text() == 'n = 2\n'


@pytest.mark.parametrize('selected', [['../outside.py'], ['missing.py'], []])
def test_bad_planner_paths_do_not_write(tmp_path, monkeypatch, selected):
    (tmp_path / 'a.py').write_text('n = 1\n')
    monkeypatch.setattr(coding, 'get_model_router', lambda: Router([{'read_files': selected}, {'read_files': selected}]))
    with pytest.raises(ValueError):
        asyncio.run(coding.propose_revision(tmp_path, '变更'))
    assert (tmp_path / 'a.py').read_text() == 'n = 1\n'


def test_patch_rejects_unknown_file_and_ambiguous_match():
    for row in [dict(file='../b.py', old_text='a', new_text='b'), dict(file='a.py', old_text='a', new_text='b')]:
        with pytest.raises(ValueError):
            coding.apply_changes({'changes': [row]}, {'a.py': 'a a'})


def test_unrelated_obsolete_example_does_not_block_local_revision(tmp_path):
    from app.services.framework_migration import validate_bundle
    (tmp_path/'util.py').write_text('def current(): return 1\n')
    (tmp_path/'old_example.py').write_text('from util import removed\n')
    (tmp_path/'train.py').write_text('from util import current\nepochs=20\n')
    assert validate_bundle(tmp_path,{'train.py':'from util import current\nepochs=1\n'},None)==[]
    assert any('removed' in issue for issue in validate_bundle(tmp_path,{'train.py':'from util import removed\nepochs=1\n'},None))
    assert any('current' in issue for issue in validate_bundle(tmp_path,{'util.py':'def different(): return 1\n'},None))

def test_verified_preparation_extends_model_snapshot_without_writing(tmp_path, monkeypatch):
    (tmp_path / 'train.py').write_text('epochs = 200\n')
    (tmp_path / 'layers.py').write_text('import tensorflow as tf\n')
    router = Router([{'read_files': ['train.py']}, {'changes': [
        {'file': 'train.py', 'old_text': '200', 'new_text': '58'}]}])
    monkeypatch.setattr(coding, 'get_model_router', lambda: router)
    preparation = [{'file': 'layers.py', 'old_text': 'import tensorflow as tf',
                    'new_text': 'import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()'}]
    original, revised, _, _, _, trace = asyncio.run(coding.propose_revision(
        tmp_path, '设为58并使用GPU', preparation_changes=preparation))
    assert 'layers.py' in router.calls[1][1]['files']
    assert 'compat.v1' in router.calls[1][1]['files']['layers.py']
    assert set(revised) == {'train.py', 'layers.py'}
    assert (tmp_path / 'layers.py').read_text() == original['layers.py']
    assert trace[1]['status'] == 'draft_prepared'


def test_semantic_intent_works_without_feature_keywords(monkeypatch):
    monkeypatch.setattr(copilot, 'get_model_router', lambda: Router([{'code_revision': True}]))
    assert asyncio.run(copilot._semantic_code_requested('每次保存时额外写一份耗时清单，保持现在的输出不变'))


@pytest.mark.parametrize('question', ['只解释，不要修改代码', '当前代码可以迁移到pytorch吗？'])
def test_readonly_requests_never_become_code_actions(monkeypatch, question):
    monkeypatch.setattr(copilot, 'get_model_router', lambda: pytest.fail('No classifier for explicit readonly'))
    assert not asyncio.run(copilot._semantic_code_requested(question))


def test_removed_audit_routes_are_not_registered():
    from app.main import app
    assert not any('analyses' in route.path for route in app.routes)
    assert any('reproductions' in route.path for route in app.routes)
    assert any(route.path.endswith('/stream') for route in app.routes)


@pytest.mark.parametrize('instruction, expected', [
    ('保持 TensorFlow，不切换到 PyTorch，只使用 GPU', ''),
    ('不要迁移到 PyTorch，而是迁移到 TensorFlow', 'tensorflow'),
    ('当前在 TensorFlow 上，请迁移到 PyTorch', 'pytorch'),
    ('不要使用 torch，不改框架', ''),
    ('在不影响算法的前提下迁移到 PyTorch', 'pytorch'),
    ('你可不可以用pytorch复现这篇论文？注意：是gpu版本', 'pytorch'),
])
def test_framework_target_respects_negation(instruction, expected):
    from app.services.gpu_preparation import requested_framework
    assert requested_framework(instruction) == expected
