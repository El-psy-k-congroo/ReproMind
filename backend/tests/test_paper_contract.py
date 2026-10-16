import asyncio
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from app.api import reproductions, copilot
from app.models import ArtifactKind, ReproductionStatus
from app.services import reproduction_engine as engine
from app.services.paper_contract import vague_spec, spec_ready, validate_scope


PAPER = 'The main method uses fine-tuning. We also study in-context learning.'
GOAL = '复现论文核心实验'


def specification():
    return {'datasets': ['Default'], 'metrics': ['F1'], 'reproduction_scope': {
        'required': [{'name': 'fine-tuning', 'paper_evidence': 'The main method uses fine-tuning.'}],
        'optional': [{'name': 'in-context learning', 'paper_evidence': 'We also study in-context learning.'}]}}


@pytest.mark.parametrize('spec', [None, {}, {'datasets': [], 'metrics': []},
    {'datasets': 'Default', 'metrics': ['F1']}, {'datasets': ['Default'], 'metrics': []},
    {'datasets': ['多个数据集'], 'metrics': ['F1']}])
def test_missing_or_vague_targets_stop_before_code(spec):
    assert vague_spec(spec) and not spec_ready(spec, PAPER, GOAL)


def test_scope_requires_grounded_disjoint_routes_and_matches_goal():
    spec = specification()
    assert spec_ready(spec, PAPER, GOAL)
    validated = validate_scope(spec, PAPER, GOAL)
    assert validated['goal'] == GOAL
    spec['reproduction_scope'] = validated
    assert not spec_ready(spec, PAPER, '仅复现 ICL 扩展')
    spec = specification()
    spec['reproduction_scope']['optional'][0]['paper_evidence'] = 'Invented paper text'
    assert not spec_ready(spec, PAPER, GOAL)
    spec = specification()
    spec['reproduction_scope']['optional'][0]['name'] = 'fine-tuning'
    assert not spec_ready(spec, PAPER, GOAL)


def test_paper_extraction_recovers_with_feedback_bounded_timeout_and_actual_goal(monkeypatch):
    requests = []

    async def extract(_role, _system, prompt, **kwargs):
        requests.append((prompt, kwargs))
        return SimpleNamespace(data={} if len(requests) == 1 else specification(),
                               error='ReadTimeout' if len(requests) == 1 else '', usage={'total_tokens': 5})

    monkeypatch.setattr(engine, 'get_model_router', lambda: SimpleNamespace(complete_json=extract))
    result = asyncio.run(engine._paper_spec(PAPER, {}, goal=GOAL))
    assert len(requests) == 2 and 'ReadTimeout' in requests[1][0] and GOAL in requests[0][0]
    assert requests[0][1]['timeout_seconds'] == 90 and requests[0][1]['non_thinking']
    assert result['_model_usage']['total_tokens'] == 10
    assert result['reproduction_scope']['goal'] == GOAL


def test_failed_extraction_does_not_return_generic_empty_spec(monkeypatch):
    async def extract(*args, **kwargs):
        return SimpleNamespace(data={'limitations': ['Cannot extract']}, error='', usage={'total_tokens': 5})
    monkeypatch.setattr(engine, 'get_model_router', lambda: SimpleNamespace(complete_json=extract))
    with pytest.raises(engine.PaperSpecificationFailure, match='数据集或评估指标') as caught:
        asyncio.run(engine._paper_spec(PAPER, {}))
    assert len(caught.value.attempts) == 2 and caught.value.usage['total_tokens'] == 10


def test_broken_old_spec_is_rechecked_after_author_source_and_preserves_old_draft(tmp_path, monkeypatch):
    draft = tmp_path / 'method.py'
    draft.write_text('def train():\n    return 1\n', encoding='utf-8')
    project = SimpleNamespace(id='paper-contract-project', artifacts=[SimpleNamespace(
        kind=ArtifactKind.paper, filename='paper.txt', content_text=PAPER)])
    plan = {'files': [{'path': 'method.py'}]}
    job = SimpleNamespace(id='paper-contract-run', project_id=project.id, project=project,
        status=ReproductionStatus.queued, goal=GOAL, model_id='test', events=[], repository_url='https://github.com/a/b',
        result_json={}, paper_spec={'limitations': ['Extraction failed']},
        code_spec={'paper_spec_enriched': True, 'draft_source': str(tmp_path), 'coding_plan': plan,
                   'model_usage': {'total_tokens': 10}}, dataset_spec={}, environment_spec={}, execution_command={})
    values = iter([job, None, None])

    class Session:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def scalar(self, *args): return next(values)
        async def execute(self, *args): pass
        async def commit(self): pass

    async def fail(_paper, _report, _model, **kwargs):
        assert kwargs['goal'] == GOAL
        await kwargs['progress_callback'](1)
        raise engine.PaperSpecificationFailure([{'reason': 'Missing metrics', 'attempt': 1}], {'total_tokens': 3})

    async def forbidden(*args, **kwargs):
        raise AssertionError('Invalid experiment target must not enter coding')

    async def clone(*args):
        return tmp_path, 'author source acquired'

    monkeypatch.setattr(engine, 'AsyncSessionLocal', Session)
    monkeypatch.setattr(engine, '_paper_spec', fail)
    monkeypatch.setattr(engine, '_project_workspace', lambda _project: tmp_path)
    monkeypatch.setattr(engine, '_copy_paper_sources', lambda *args: None)
    monkeypatch.setattr(engine, '_clone_repository', clone)
    monkeypatch.setattr(engine, '_generate_scaffold', forbidden)
    asyncio.run(engine.prepare_reproduction(job.id))
    assert job.status == ReproductionStatus.needs_input and job.current_stage == 'experiment_planning'
    assert job.result_json['action_required']['kind'] == 'paper_spec_failed'
    assert job.result_json['paper_spec_attempts'][0]['reason'] == 'Missing metrics'
    assert job.code_spec['model_usage']['total_tokens'] == 13
    assert job.result_json['preparation_history'][0]['code_spec']['coding_plan'] == plan
    assert job.code_spec['source_path'] == str(tmp_path)
    assert draft.read_text(encoding='utf-8') == 'def train():\n    return 1\n'


def test_paper_spec_retry_routes_to_preparation_not_previously_approved_training(monkeypatch):
    job = SimpleNamespace(id='paper-spec-retry', status=ReproductionStatus.needs_input,
        result_json={'action_required': {'kind': 'paper_spec_failed'}}, approved_at=object(), error_message='bad')
    rows = iter([job, None])

    class DB:
        async def scalar(self, *args): return next(rows)
        async def commit(self): pass
        async def refresh(self, *args): pass

    monkeypatch.setattr(reproductions, 'has_active_work', lambda _id: False)
    tasks = BackgroundTasks()
    asyncio.run(reproductions.resume_reproduction('project', job.id, tasks, DB()))
    assert job.status == ReproductionStatus.queued
    assert tasks.tasks[0].func is engine.prepare_reproduction


def test_source_choice_cannot_bypass_missing_paper_spec():
    job = SimpleNamespace(result_json={'action_required': {'kind': 'paper_spec_failed'}})
    class DB:
        async def scalar(self, *args): return job
    with pytest.raises(HTTPException) as caught:
        asyncio.run(reproductions.choose_source('p', 'r', reproductions.SourceChoice(choice='generate'), BackgroundTasks(), DB()))
    assert caught.value.status_code == 409


def test_copilot_does_not_misdiagnose_spec_failure_as_no_source_or_cannot_code():
    text = copilot._generation_diagnosis_answer({'job': {'status': 'needs_input', 'source_mode': '',
        'action_required': {'kind': 'paper_spec_failed', 'reason': 'ReadTimeout'}}})
    assert 'ReadTimeout' in text and '没有开始本轮编码' in text and '已有源码和旧草稿保留' in text
