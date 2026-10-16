"""HTTP/SQLite regressions from QA; no external models or user data."""

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import main
from app.api import projects, reproductions
from app.database import Base, get_db
from app.models import Artifact


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    engine = create_async_engine('sqlite+aiosqlite:///' + str(tmp_path / 'qa.sqlite'))
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def initialize():
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def database():
        async with sessions() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    settings = SimpleNamespace(upload_dir=tmp_path / 'uploads', max_upload_mb=1,
                               default_tenant_id='qa', paper_projects_dir=tmp_path / 'workspaces')
    monkeypatch.setattr(main, 'init_database', initialize)
    monkeypatch.setattr(main, 'close_database', engine.dispose)
    monkeypatch.setattr(main, 'get_settings', lambda: settings)
    monkeypatch.setattr(projects, 'get_settings', lambda: settings)
    main.app.dependency_overrides[get_db] = database
    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client, sessions, settings
    main.app.dependency_overrides.pop(get_db, None)


def new_project(client):
    response = client.post('/api/projects', json={'name': 'QA 文件边界'})
    assert response.status_code == 201, response.text
    return response.json()['id']


@pytest.mark.parametrize('tail', ['', '/reproductions/latest', '/copilot/conversations'])
def test_malformed_project_id_returns_client_error(api_client, tail):
    client, _, _ = api_client
    for _ in range(2):
        response = client.get('/api/projects/not-a-uuid' + tail)
        assert response.status_code == 422, response.text


def test_unknown_valid_project_returns_404(api_client):
    client, _, _ = api_client
    assert client.get('/api/projects/' + str(uuid4())).status_code == 404


@pytest.mark.parametrize('name', ['  ', '\t\n'])
def test_blank_project_name_is_rejected(api_client, name):
    client, _, _ = api_client
    assert client.post('/api/projects', json={'name': name}).status_code == 422
    assert client.get('/api/projects').json() == []


@pytest.mark.parametrize('concurrent', [False, True])
def test_upload_collision_never_overwrites_evidence(api_client, concurrent):
    client, sessions, _ = api_client
    project_id = new_project(client)

    def upload(name, data):
        response = client.post(f'/api/projects/{project_id}/artifacts',
                               data={'kind': 'note'}, files={'file': (name, data, 'text/plain')})
        assert response.status_code == 201, response.text
        return response.json()['id'], data

    created = [upload('note.txt', b'original'), upload('note-3.txt', b'must survive')]
    if concurrent:
        with ThreadPoolExecutor(max_workers=3) as pool:
            created.extend(pool.map(lambda index: upload('note.txt', f'parallel-{index}'.encode()), range(3)))
    else:
        created.extend([upload('note.txt', b'third'), upload('note.txt', b'fourth')])

    async def inspect():
        paths = []
        async with sessions() as session:
            for artifact_id, data in created:
                artifact = await session.get(Artifact, artifact_id)
                paths.append(artifact.storage_path)
                assert Path(artifact.storage_path).read_bytes() == data
                assert artifact.sha256 == hashlib.sha256(data).hexdigest()
        assert len(paths) == len(set(paths))
    asyncio.run(inspect())


@pytest.mark.parametrize('filename', ['CON.txt', 'NUL', '../', '.', 'C:\\'])
def test_unsafe_filename_is_rejected_without_server_error(api_client, filename):
    client, _, settings = api_client
    project_id = new_project(client)
    response = client.post(f'/api/projects/{project_id}/artifacts', data={'kind': 'note'},
                           files={'file': (filename, b'evidence', 'text/plain')})
    assert response.status_code == 400, response.text
    assert client.get(f'/api/projects/{project_id}').json()['artifacts'] == []
    assert not settings.upload_dir.joinpath(project_id).exists()


def test_empty_and_oversize_upload_leave_no_artifacts(api_client):
    client, _, _ = api_client
    project_id = new_project(client)
    for data, expected in [(b'', 400), (b'x' * (1024 * 1024 + 1), 413)]:
        response = client.post(f'/api/projects/{project_id}/artifacts', data={'kind': 'note'},
                               files={'file': ('test.txt', data, 'text/plain')})
        assert response.status_code == expected
    assert client.get(f'/api/projects/{project_id}').json()['artifacts'] == []


def test_corrupt_paper_cannot_start_preparation(api_client, monkeypatch):
    client, _, _ = api_client
    project_id = new_project(client)
    response = client.post(f'/api/projects/{project_id}/artifacts', data={'kind': 'paper'},
                           files={'file': ('corrupt.pdf', b'not a PDF', 'application/pdf')})
    assert response.status_code == 201
    assert response.json()['metadata_json']['parse_error']

    async def no_model(*_):
        return None
    async def no_preparation(*_):
        return None
    monkeypatch.setattr(reproductions, 'validate_model', no_model)
    monkeypatch.setattr(reproductions, 'prepare_reproduction', no_preparation)
    for _ in range(2):
        response = client.post(f'/api/projects/{project_id}/reproductions', json={})
        assert response.status_code == 400, response.text
    assert client.get(f'/api/projects/{project_id}/reproductions/latest').json() is None


def test_simultaneous_start_returns_one_active_job(api_client, monkeypatch):
    client, _, _ = api_client
    project_id = new_project(client)
    response = client.post(f'/api/projects/{project_id}/artifacts', data={'kind': 'paper'},
                           files={'file': ('paper.txt', b'A real readable paper fixture', 'text/plain')})
    assert response.status_code == 201

    async def no_model(*_):
        await asyncio.sleep(.02)
    async def no_preparation(*_):
        return None
    monkeypatch.setattr(reproductions, 'validate_model', no_model)
    monkeypatch.setattr(reproductions, 'prepare_reproduction', no_preparation)

    def start(_):
        response = client.post(f'/api/projects/{project_id}/reproductions', json={})
        assert response.status_code == 202, response.text
        return response.json()['id']
    with ThreadPoolExecutor(max_workers=3) as pool:
        ids = list(pool.map(start, range(3)))
    assert len(set(ids)) == 1
