"""Account lifecycle: registration, sign-in, logout, password change and recovery."""
import asyncio
import hmac
import re
import secrets
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from ..auth import (ACCOUNT_TENANT, COOKIE, check_origin, create_session, password_hash,
                    public_user, server_accounts, session_user, token_hash, verify_password)
from ..database import get_db
from ..models import AccountRecovery, AuthSession, User, utcnow

router = APIRouter(prefix='/auth', tags=['accounts'])
_attempts = defaultdict(deque)
_DUMMY_HASH = password_hash('unavailable-account')


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=128)
    password: str = Field(min_length=1, max_length=128)
    remember: bool = False

    @field_validator('email')
    @classmethod
    def email_address(cls, value):
        value = value.strip().lower()
        if not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', value):
            raise ValueError('请输入有效的邮箱地址')
        return value


class Registration(Credentials):
    username: str = Field(min_length=2, max_length=40)

    @field_validator('username')
    @classmethod
    def display_name(cls, value):
        value = value.strip()
        if len(value) < 2 or any(ord(c) < 32 for c in value):
            raise ValueError('昵称需为 2–40 个可见字符')
        return value


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=10, max_length=128)


class RecoveryReset(BaseModel):
    email: str = Field(min_length=3, max_length=128)
    recovery_code: str = Field(min_length=12, max_length=160)
    new_password: str = Field(min_length=10, max_length=128)


def strong_password(value):
    if len(value) < 10 or not any(c.isalpha() for c in value) or not any(c.isdigit() for c in value):
        raise HTTPException(status_code=422, detail='密码至少 10 位，并同时包含字母和数字。')


def account_operation(request: Request):
    if not server_accounts():
        raise HTTPException(status_code=404, detail='本地部署无需账号，直接进入工作台。')
    check_origin(request)
    now = time.monotonic()
    key = request.client.host if request.client else 'unknown'
    if key not in _attempts and len(_attempts) > 10000:
        _attempts.clear()
    attempts = _attempts[key]
    while attempts and now - attempts[0] > 300:
        attempts.popleft()
    if len(attempts) >= 15:
        raise HTTPException(status_code=429, detail='账号操作过于频繁，请在五分钟后重试。', headers={'Retry-After': '300'})
    attempts.append(now)


async def recovery_code(db, user):
    code = secrets.token_urlsafe(32)
    record = await db.get(AccountRecovery, user.id)
    if record is None:
        db.add(AccountRecovery(user_id=user.id, code_hash=token_hash(code)))
    else:
        record.code_hash = token_hash(code)
    return code


@router.get('/session')
async def session(request: Request, response: Response, db=Depends(get_db)):
    response.headers['Cache-Control'] = 'no-store'
    user = await session_user(request, db) if server_accounts() else None
    return {'auth_required': server_accounts(), 'user': public_user(user) if user else None}


@router.post('/register', status_code=201, dependencies=[Depends(account_operation)])
async def register(payload: Registration, response: Response, db=Depends(get_db)):
    strong_password(payload.password)
    user = User(tenant_id=ACCOUNT_TENANT, username=payload.username, email=payload.email,
                password_hash=await asyncio.to_thread(password_hash, payload.password), role='researcher')
    db.add(user)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail='该邮箱已注册，请登录或找回密码。') from exc
    code = await recovery_code(db, user)
    await create_session(db, user, response, payload.remember)
    await db.commit()
    return {'user': public_user(user), 'recovery_code': code}


@router.post('/login', dependencies=[Depends(account_operation)])
async def login(payload: Credentials, response: Response, db=Depends(get_db)):
    user = await db.scalar(select(User).where(User.tenant_id == ACCOUNT_TENANT, User.email == payload.email))
    valid = await asyncio.to_thread(verify_password, payload.password, user.password_hash if user else _DUMMY_HASH)
    if not user or not valid or not user.is_active:
        raise HTTPException(status_code=401, detail='邮箱或密码不正确。')
    await db.execute(delete(AuthSession).where(AuthSession.expires_at <= utcnow()))
    await create_session(db, user, response, payload.remember)
    await db.commit()
    return {'user': public_user(user)}


@router.post('/logout')
async def logout(request: Request, response: Response, db=Depends(get_db)):
    if server_accounts():
        check_origin(request)
    token = request.cookies.get(COOKIE, '')
    if token:
        await db.execute(delete(AuthSession).where(AuthSession.token_hash == token_hash(token)))
        await db.commit()
    response.delete_cookie(COOKIE, path='/', secure=server_accounts(), httponly=True, samesite='lax')
    response.headers['Cache-Control'] = 'no-store'
    return {'ok': True}


@router.post('/password', dependencies=[Depends(account_operation)])
async def change_password(payload: PasswordChange, request: Request, response: Response, db=Depends(get_db)):
    user = await session_user(request, db)
    if user is None:
        raise HTTPException(status_code=401, detail='请先登录。')
    if not await asyncio.to_thread(verify_password, payload.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail='当前密码不正确。')
    strong_password(payload.new_password)
    user.password_hash = await asyncio.to_thread(password_hash, payload.new_password)
    await db.execute(delete(AuthSession).where(AuthSession.user_id == user.id))
    code = await recovery_code(db, user)
    await create_session(db, user, response)
    await db.commit()
    return {'ok': True, 'recovery_code': code}


@router.post('/recover', dependencies=[Depends(account_operation)])
async def recover(payload: RecoveryReset, db=Depends(get_db)):
    user = await db.scalar(select(User).where(User.tenant_id == ACCOUNT_TENANT,
                                            User.email == payload.email.strip().lower(), User.is_active.is_(True)))
    record = await db.get(AccountRecovery, user.id) if user else None
    if record is None or not hmac.compare_digest(record.code_hash, token_hash(payload.recovery_code.strip())):
        raise HTTPException(status_code=400, detail='邮箱或恢复密钥不正确。')
    strong_password(payload.new_password)
    user.password_hash = await asyncio.to_thread(password_hash, payload.new_password)
    await db.execute(delete(AuthSession).where(AuthSession.user_id == user.id))
    # One-time key: rotate after successful use; the old key cannot be replayed.
    code = await recovery_code(db, user)
    await db.commit()
    return {'ok': True, 'recovery_code': code}
