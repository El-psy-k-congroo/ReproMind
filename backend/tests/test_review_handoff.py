import asyncio
import json
from types import SimpleNamespace

import pytest

from app.services.code_agent import CodingFailure, generate_project
from app.services.code_validation import grounded_issue
from app.services.paper_evidence import core_implementation_concerns
from app.services.review_contract import inspect_review


ORIGINAL = 'def helper():\n    return 7\n\ndef train():\n    return 0\n'
PAPER = 'Fine-tune the model before evaluating the classifier.'
GOOD = {'file': 'method.py', 'category': 'method', 'reason': 'Missing training',
        'code_evidence': 'return 0', 'paper_evidence': PAPER}
BAD = {'file': 'method.py', 'category': 'method', 'reason': 'Invented location',
       'code_evidence': 'return nonexistent', 'paper_evidence': PAPER}
PLAN = {'files': [{'path': 'method.py'}], 'entrypoint': 'method.py'}


def reply(data=None, text=''):
    return SimpleNamespace(data=data, text=text, error='', usage={}, finish_reason='stop')


def test_review_accepts_valid_findings_independently_of_bad_quotes():
    result = inspect_review({'issues': [GOOD, BAD]}, {'method.py': ORIGINAL}, PAPER, [])
    assert len(result['confirmed']) == len(result['rejected']) == 1
    assert result['confirmed'][0]['reason'] == 'Missing training'
    assert result['rejected'][0]['finding'] == BAD


def test_partial_review_hands_off_repair_then_requires_clean_verification():
    reviews, writes, snapshots = [], [], []

    async def review(_role, _system, prompt, **kwargs):
        request = json.loads(prompt)
        reviews.append(request)
        if len(reviews) <= 2:
            if len(reviews) == 2:
                assert request['accepted_findings'][0]['reason'] == 'Missing training'
                assert request['unverified_findings'][0]['finding'] == BAD
            return reply({'issues': [GOOD, BAD]})
        assert request['files']['method.py'].strip() == ORIGINAL.replace('return 0', 'return 1').strip()
        return reply({'issues': [], 'issue_checks': [{'id': row['id'], 'status': 'resolved',
                      'reason': 'Training now implemented'} for row in request['pending_issues']]})

    async def write(_role, _system, prompt, **kwargs):
        writes.append(json.loads(prompt))
        return reply(text=ORIGINAL.replace('return 0', 'return 1'))

    async def checkpoint(files, trace, plan, usage):
        snapshots.append(json.loads(json.dumps({'files': files, 'plan': plan, 'trace': trace})))

    result = asyncio.run(generate_project(SimpleNamespace(complete_json=review, complete_text=write), PAPER, {},
        initial_files={'method.py': ORIGINAL}, previous_plan=PLAN.copy(), checkpoint=checkpoint))
    assert len(writes) == 1 and len(reviews) == 3
    assert any(row['plan'].get('review_validation_pending') for row in snapshots)
    assert result['issue_ledger'][0]['status'] == 'resolved'
    assert result['coding_plan']['review_validation_pending'] == []
    assert result['validation']['method_fidelity'] == 'reviewed'
    assert all(row['status'] == 'resolved' for row in result['generation_trace'] if row['phase'] == 'review_protocol')
    assert 'return 7' in result['files'][0]['content']  # unrelated definition remains intact


def test_unverified_review_cannot_pass_after_valid_repair():
    reviews, writes = [], []

    async def review(_role, _system, prompt, **kwargs):
        request = json.loads(prompt)
        reviews.append(request)
        return reply({'issues': [GOOD, BAD] if len(reviews) <= 2 else [BAD],
                      'issue_checks': [{'id': row['id'], 'status': 'resolved', 'reason': 'Fixed'}
                                       for row in request['pending_issues']]})

    async def write(*args, **kwargs):
        writes.append(True)
        return reply(text=ORIGINAL.replace('return 0', 'return 1'))

    with pytest.raises(CodingFailure, match='证据校验失败') as caught:
        asyncio.run(generate_project(SimpleNamespace(complete_json=review, complete_text=write), PAPER, {},
            initial_files={'method.py': ORIGINAL}, previous_plan=PLAN.copy()))
    assert len(writes) == 1 and len(reviews) == 4
    assert caught.value.files['method.py'].strip() == ORIGINAL.replace('return 0', 'return 1').strip()
    assert any(row['status'] != 'resolved' for row in caught.value.trace if row['phase'] == 'review_protocol')


@pytest.mark.parametrize('check', [
    [], [{'id': 'invented', 'status': 'resolved', 'reason': 'Fixed'}],
    [{'id': 'real', 'status': 'resolved', 'reason': ''}],
    [{'id': 'real', 'status': 'resolved', 'reason': 'Fixed'}] * 2,
])
def test_review_checks_cannot_close_missing_duplicate_or_fabricated_ids(check):
    issue = {**grounded_issue(GOOD, {'method.py': ORIGINAL}, PAPER), 'id': 'real'}
    result = inspect_review({'issues': [], 'issue_checks': check}, {'method.py': ORIGINAL}, PAPER, [issue])
    assert not result['checked'] and result['rejected'][0]['kind'] == 'unverified_check'


@pytest.mark.parametrize('limitation', [
    'In-context learning (ICL-LITO) is not implemented; only fine-tuning is supported.',
    'The implementation supports fine-tuning, but does not implement ICL-LITO.',
    'ICL-LITO is missing; fine-tuning and training are implemented.',
])
def test_absent_optional_icl_does_not_mean_present_training_is_missing(limitation):
    assert core_implementation_concerns([limitation], ['LITO']) == []
    assert core_implementation_concerns([limitation], ['ICL-LITO']) == [limitation]


@pytest.mark.parametrize('limitation', [
    'The implementation lacks the fine-tuning procedure.',
    'The core algorithm is not implemented.',
    'The code does not implement the required training split.',
])
def test_selected_core_methods_cannot_be_waived_as_limitations(limitation):
    result = inspect_review({'issues': [], 'limitations': [limitation]}, {'method.py': ORIGINAL}, PAPER, [])
    assert result['rejected'][0]['kind'] == 'unlocated_core_concern'


def test_external_checkpoint_is_not_missing_training_implementation():
    assert core_implementation_concerns(['Training checkpoint is missing; download requires access.']) == []
