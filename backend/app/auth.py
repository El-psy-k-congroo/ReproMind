"""Server accounts, revocable sessions and project access checks. Local mode is unchanged."""
import hashlib
import hmac
import secrets
from contextvars import ContextVar
from datetime import timedelta

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select

from .config import get_settings
from .database import get_db
from .models import AuthSession, Project, ReproductionJob, User, utcnow

COOKIE = 'repromind_session'
ACCOUNT_TENANT = 'registered_accounts'
current_owner: ContextVar[str | None] = ContextVar('current_owner', default=None)


def server_accounts() -> bool:
    return get_settings().deployment_mode == 'server'


def password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return 'scrypt$' + salt.hex() + '$' + digest.hex()


def verify_password(password: str, encoded: str) -> bool:
    try:
        method, salt, expected = encoded.split('$')
        if method != 'scrypt':
            return False
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1, dklen=32)
        return hmac.compare_digest(digest.hex(), expected)
    except (ValueError, TypeError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def check_origin(request: Request) -> None:
    expected = get_settings().public_origin.rstrip('/') or str(request.base_url).rstrip('/')
    origin = request.headers.get('origin')
    if origin != expected and not (origin is None and request.headers.get('x-repromind-request') == '1'):
        raise HTTPException(status_code=403, detail='请求来源校验失败，请从本站页面重新操作。')


async def session_user(request: Request, db) -> User | None:
    token = request.cookies.get(COOKIE, '')
    if not token or len(token) > 160:
        return None
    return await db.scalar(select(User).join(AuthSession, AuthSession.user_id == User.id).where(
        AuthSession.token_hash == token_hash(token), AuthSession.expires_at > utcnow(),
        User.is_active.is_(True), User.tenant_id == ACCOUNT_TENANT))


async def create_session(db, user: User, response, remember=False) -> None:
    token = secrets.token_urlsafe(48)
    seconds = (30 if remember else 1) * 86400
    db.add(AuthSession(token_hash=token_hash(token), user_id=user.id,
                       expires_at=utcnow() + timedelta(seconds=seconds)))
    response.set_cookie(COOKIE, token, max_age=seconds if remember else None,
                        httponly=True, secure=get_settings().session_secure_cookie,
                        samesite='lax', path='/')
    response.headers['Cache-Control'] = 'no-store'


def public_user(user):
    return {'id': user.id, 'username': user.username, 'email': user.email, 'role': user.role}


async def enforce_access(request: Request, db=Depends(get_db)):
    if not server_accounts():
        yield
        return
    user = await session_user(request, db)
    if user is None:
        raise HTTPException(status_code=401, detail='请先登录，或重新登录已过期的会话。')
    if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
        check_origin(request)
    request.state.user = user
    project_id = request.path_params.get('project_id')
    if project_id:
        project = await db.scalar(select(Project.id).where(Project.id == project_id, Project.owner_id == user.id))
        if project is None:
            raise HTTPException(status_code=404, detail='项目不存在或无权访问。')
        job_id = request.path_params.get('job_id')
        if job_id and await db.scalar(select(ReproductionJob.id).where(
                ReproductionJob.id == job_id, ReproductionJob.project_id == project_id)) is None:
            raise HTTPException(status_code=404, detail='实验记录不存在。')
    owner_token = current_owner.set(user.id)
    try:
        yield
    finally:
        current_owner.reset(owner_token)
