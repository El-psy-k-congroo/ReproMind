"""Real HTTP/database account boundaries, without downloading or running papers."""
import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app import auth, main
from app.api import authentication, projects
from app.database import Base, get_db
from app.models import AuthSession, Project, ReproductionJob, User


@pytest.fixture
def account_client(tmp_path, monkeypatch):
    engine = create_async_engine('sqlite+aiosqlite:///' + str(tmp_path/'accounts.sqlite'))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async def initialize():
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    async def database():
        async with sessions() as db:
            try:
                yield db
                await db.commit()
            except Exception:
                await db.rollback()
                raise
    settings = SimpleNamespace(deployment_mode='server', public_origin='http://testserver',
        session_secure_cookie=False, default_tenant_id='default', upload_dir=tmp_path/'uploads',
        paper_projects_dir=tmp_path/'projects', max_upload_mb=2)
    monkeypatch.setattr(auth, 'get_settings', lambda: settings)
    monkeypatch.setattr(projects, 'get_settings', lambda: settings)
    monkeypatch.setattr(main, 'get_settings', lambda: settings)
    monkeypatch.setattr(main, 'init_database', initialize)
    monkeypatch.setattr(main, 'close_database', engine.dispose)
    authentication._attempts.clear()
    main.app.dependency_overrides[get_db] = database
    try:
        with TestClient(main.app, headers={'X-ReproMind-Request':'1'}) as client:
            yield client, sessions
    finally:
        main.app.dependency_overrides.pop(get_db, None)


def register(client, email='one@example.org'):
    r=client.post('/api/auth/register', json={'email':email,'username':'研究用户','password':'ResearchPass123!'})
    assert r.status_code==201, r.text
    return r.json(), client.cookies.get(auth.COOKIE)


def test_authentication_passwords_cookies_logout_and_csrf(account_client):
    c, sessions = account_client
    assert c.get('/api/projects').status_code == 401
    user, token = register(c)
    assert user['user']['email'] == 'one@example.org'
    assert c.get('/api/auth/session').json()['user']['id'] == user['user']['id']
    assert c.post('/api/projects',json={'name':'被阻止的跨站请求'},headers={'Origin':'https://evil.example'}).status_code == 403
    async def stored():
        async with sessions() as db:
            u=await db.scalar(select(User));s=await db.scalar(select(AuthSession))
            assert u.password_hash != 'ResearchPass123!' and auth.verify_password('ResearchPass123!',u.password_hash)
            assert s.token_hash==auth.token_hash(token) and s.token_hash!=token
    asyncio.run(stored())
    assert c.post('/api/auth/logout',json={}).status_code==200
    assert c.get('/api/projects',headers={'Cookie':auth.COOKIE+'='+token}).status_code==401
    assert c.post('/api/auth/login',json={'email':'one@example.org','password':'wrong'}).status_code==401
    assert c.post('/api/auth/login',json={'email':'ONE@example.org','password':'ResearchPass123!'}).status_code==200


def test_two_users_cannot_read_modify_or_observe_each_others_projects(account_client):
    c, sessions = account_client
    a, token_a=register(c)
    project_a=c.post('/api/projects',json={'name':'用户甲项目'}).json()['id']
    b, token_b=register(c,'two@example.org')
    assert c.get('/api/projects').json()==[]
    project_b=c.post('/api/projects',json={'name':'用户乙项目'}).json()['id']
    for path in ['', '/reproductions', '/reproductions/latest', '/copilot/conversations']:
        assert c.get('/api/projects/'+project_a+path).status_code==404
    assert c.delete('/api/projects/'+project_a).status_code==404
    assert c.post('/api/projects/'+project_a+'/artifacts',data={'kind':'note'},files={'file':('n.txt',b'bad')}).status_code==404
    assert c.get('/api/analytics/overview').json()['active_projects']==1
    rows=c.get('/api/projects',headers={'Cookie':auth.COOKIE+'='+token_a}).json()
    assert [x['id'] for x in rows]==[project_a]
    async def insert():
        async with sessions() as db:
            job=ReproductionJob(project_id=project_b,tenant_id='test')
            db.add(job);await db.commit();return job.id
    other_job=asyncio.run(insert())
    for suffix in ['', '/report','/telemetry','/branch-files']:
        assert c.get(f'/api/projects/{project_a}/reproductions/{other_job}'+suffix,
                     headers={'Cookie':auth.COOKIE+'='+token_a}).status_code==404
    assert c.post(f'/api/projects/{project_a}/reproductions/{other_job}/approve',json={},
                  headers={'Cookie':auth.COOKIE+'='+token_a}).status_code==404
    assert c.get('/api/projects/'+project_b).status_code==200


def test_account_recovery_is_single_use_and_revokes_old_sessions(account_client):
    c,_=account_client
    account,token=register(c)
    code=account['recovery_code']
    payload={'email':'one@example.org','recovery_code':code,'new_password':'NewResearchPass456!'}
    r=c.post('/api/auth/recover',json=payload)
    assert r.status_code==200 and r.json()['recovery_code']!=code
    assert c.get('/api/projects').status_code==401
    assert c.post('/api/auth/recover',json=payload).status_code==400
    assert c.post('/api/auth/login',json={'email':'one@example.org','password':'ResearchPass123!'}).status_code==401
    assert c.post('/api/auth/login',json={'email':'one@example.org','password':'NewResearchPass456!'}).status_code==200
    assert c.post('/api/auth/password',json={'current_password':'NewResearchPass456!','new_password':'ThirdResearchPass789!'}).status_code==200


def test_model_preferences_are_separate_files(tmp_path):
    from app.core.model_policy import write_policy,read_policy
    settings=SimpleNamespace(model_routing_path=tmp_path/'routing.json', model_provider='moma')
    first=auth.current_owner.set(str(uuid4()))
    try:
        write_policy(settings, {'chat':'model-a'})
        second=auth.current_owner.set(str(uuid4()))
        try:
            assert read_policy(settings)['models']=={}
            write_policy(settings, {'chat':'model-b'})
        finally: auth.current_owner.reset(second)
        assert read_policy(settings)['models']['chat']=='model-a'
    finally: auth.current_owner.reset(first)


def test_http_model_preferences_follow_authenticated_user(account_client,tmp_path,monkeypatch):
    from app.api import model_routing
    settings=SimpleNamespace(model_routing_path=tmp_path/'models.json',model_provider='moma')
    monkeypatch.setattr(model_routing,'get_settings',lambda:settings)
    async def available(_): pass
    monkeypatch.setattr(model_routing,'validate_model',available)
    c,_=account_client
    _,token_a=register(c)
    choices={k:'model-a' for k in ('router','chat','coder','reasoning')}
    r=c.put('/api/model-routing',json={'models':choices,'revision':''})
    assert r.status_code==200,r.text
    register(c,'two@example.org')
    assert all(not g['model'] for g in c.get('/api/model-routing').json()['groups'])
    choices={k:'model-b' for k in choices}
    assert c.put('/api/model-routing',json={'models':choices,'revision':''}).status_code==200
    state=c.get('/api/model-routing',headers={'Cookie':auth.COOKIE+'='+token_a}).json()
    assert all(g['model']=='model-a' for g in state['groups'])
