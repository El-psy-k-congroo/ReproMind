"""Cloud capability boundaries; no models, downloads, environments or experiments."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app import deployment


P = '/api/projects/550e8400-e29b-41d4-a716-446655440000'
J = P + '/reproductions/550e8400-e29b-41d4-a716-446655440001'


def check(monkeypatch, method, path, mode='demo'):
    monkeypatch.setattr(deployment, 'get_settings', lambda: SimpleNamespace(deployment_mode=mode))
    request = Request({'type': 'http', 'method': method, 'path': path, 'headers': []})
    asyncio.run(deployment.enforce_deployment_capabilities(request))


@pytest.mark.parametrize('path', [J + '/' + operation for operation in (
    'approve', 'resume', 'resume-paused', 'reruns', 'revision-runs', 'revisions',
    'tuning-sessions', 'ablations', 'source-choice', 'continue-partial', 'branch-preview')]
    + [P + '/reproductions'])
def test_demo_blocks_every_execution_and_preparation_route(monkeypatch, path):
    with pytest.raises(HTTPException) as error:
        check(monkeypatch, 'POST', path)
    assert error.value.status_code == 403


def test_demo_cannot_modify_global_model_policy(monkeypatch):
    with pytest.raises(HTTPException):
        check(monkeypatch, 'PUT', '/api/model-routing')


@pytest.mark.parametrize('method,path', [
    ('GET', '/api/projects'), ('POST', '/api/projects'), ('POST', P + '/artifacts'),
    ('POST', P + '/copilot/conversations'),
    ('POST', P + '/copilot/conversations/550e8400-e29b-41d4-a716-446655440002/stream'),
])
def test_demo_keeps_materials_and_assistant_available(monkeypatch, method, path):
    check(monkeypatch, method, path)


def test_local_keeps_complete_workflow_without_authentication(monkeypatch):
    check(monkeypatch, 'POST', J + '/approve', mode='local')


def test_unknown_new_write_routes_are_closed_in_demo(monkeypatch):
    with pytest.raises(HTTPException):
        check(monkeypatch, 'POST', '/api/new-execution-endpoint')


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
