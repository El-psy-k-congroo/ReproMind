"""Offline state regressions; no author execution or model requests."""
import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.api import experiment_branches as branches


@pytest.mark.parametrize('approved,applied,included', [
    (None, '2026-10-17T09:30:24+08:00', False),
    (datetime(2026, 10, 17, 1, 35), '2026-10-17T09:30:24+08:00', True),
    ('2026-10-17T01:35:00Z', '2026-10-17T01:35:00', True),
    ('2026-10-17T01:35:00', '2026-10-17T09:36:00+08:00', False),
    ('invalid', '2026-10-17T09:30:24+08:00', False),
    ('2026-10-17T01:35:00', 'invalid', False),
])
def test_initial_approval_only_covers_already_applied_edits(approved, applied, included):
    assert branches._included_in_initial_approval(
        SimpleNamespace(approved_at=approved), {'applied_at': applied}) is included


def test_pending_revision_excludes_initial_edit_but_keeps_later_edit(tmp_path, monkeypatch):
    base = SimpleNamespace(id='base', approved_at=datetime(2026, 10, 17, 1, 35),
                           execution_command={}, dataset_spec={})
    directory = tmp_path / 'runs/base/proposals'
    directory.mkdir(parents=True)
    initial = {'files': ['train.py'], 'applied_at': '2026-10-17T09:30:24+08:00'}
    path = directory / 'initial.json'
    path.write_text(json.dumps(initial), encoding='utf-8')
    async def job(*args):
        return base
    monkeypatch.setattr(branches, '_job', job)
    monkeypatch.setattr(branches, '_scope', lambda row: (tmp_path, tmp_path))
    assert asyncio.run(branches.pending_revision('project', 'base', None)) is None
    assert json.loads(path.read_text()) == initial
    later = {'files': ['train.py'], 'applied_at': '2026-10-17T09:36:00+08:00'}
    (directory / 'later.json').write_text(json.dumps(later), encoding='utf-8')
    result = asyncio.run(branches.pending_revision('project', 'base', None))
    assert result['proposal_id'] == 'later'
    assert json.loads(path.read_text()) == initial
