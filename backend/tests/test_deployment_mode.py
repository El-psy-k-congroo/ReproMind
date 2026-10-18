"""Startup and uploaded-material regressions; no experiments or models."""
import asyncio
from types import SimpleNamespace
import pytest
from fastapi import HTTPException

def test_frontend_fallback_keeps_missing_api_and_assets_as_404(tmp_path, monkeypatch):
    from app import main
    (tmp_path / 'index.html').write_text('<div id="app"></div>', encoding='utf-8')
    monkeypatch.setattr(main, '_frontend_dist', tmp_path)
    assert main.frontend_page('projects/550e8400-e29b-41d4-a716-446655440000').path == tmp_path / 'index.html'
    for path in ['api/missing', 'assets/missing.js', '../outside.txt', '.env']:
        with pytest.raises(HTTPException) as error:
            main.frontend_page(path)
        assert error.value.status_code == 404


def test_uploaded_notes_reach_assistant_before_any_experiment_exists():
    import json
    from app.api import copilot
    from app.models import ArtifactKind
    class Session:
        async def scalars(self, *_):
            return SimpleNamespace(all=lambda: [])
    project = SimpleNamespace(id='example', name='说明', description='', artifacts=[
        SimpleNamespace(kind=ArtifactKind.note, filename='安装说明.md',
                        content_text='Python 3.11；默认 SQLite，无需额外数据库。')])
    context = asyncio.run(copilot._context(Session(), project, '安装说明要求哪些软件？'))
    packet = json.loads(copilot._bounded_prompt('安装说明要求哪些软件？', context, [], ''))
    excerpts = packet['uploaded_material_excerpts_not_paper_or_execution_proof']
    assert excerpts[0]['filename'] == '安装说明.md'
    assert 'SQLite' in excerpts[0]['excerpt']
    assert packet['selected_paper_evidence']['passages'] == []
