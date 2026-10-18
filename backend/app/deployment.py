"""Resource boundaries for a shared, access-controlled review deployment.

This is a capability gate, not account authentication or a sandbox. Public demo
deployments must also use an authenticated TLS reverse proxy. Local mode keeps
the complete author-source workflow and requires no login.
"""
import re

from fastapi import HTTPException, Request
from sqlalchemy import func, select

from .config import get_settings


_PROJECT = r"/api/projects/[0-9a-fA-F-]{36}"
_SAFE_WRITES = {
    "POST": [r"/api/projects", _PROJECT + r"/artifacts",
             _PROJECT + r"/copilot/conversations",
             _PROJECT + r"/copilot/conversations/[0-9a-fA-F-]{36}/stream"],
    "DELETE": [_PROJECT, _PROJECT + r"/artifacts/[0-9a-fA-F-]{36}",
               _PROJECT + r"/copilot/conversations/[0-9a-fA-F-]{36}"],
}
DEMO_MESSAGE = "云端评审模式不执行仓库下载、环境安装、代码修改或实验。请在 Windows / Linux 本地部署后使用完整复现流程。"


async def enforce_deployment_capabilities(request: Request) -> None:
    if get_settings().deployment_mode != "demo":
        return
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    if any(re.fullmatch(pattern, request.url.path) for pattern in _SAFE_WRITES.get(request.method, [])):
        return
    raise HTTPException(status_code=403, detail=DEMO_MESSAGE)


async def check_demo_capacity(db, *, new_project=False, upload_bytes=0) -> None:
    settings = get_settings()
    if settings.deployment_mode != "demo":
        return
    from .models import Artifact, Project
    if new_project and (await db.scalar(select(func.count(Project.id))) or 0) >= settings.demo_max_projects:
        raise HTTPException(status_code=409, detail="共享演示项目数量已达上限，请删除不再使用的项目或本地部署。")
    if upload_bytes:
        stored = await db.scalar(select(func.coalesce(func.sum(Artifact.size_bytes), 0))) or 0
        if stored + upload_bytes > settings.demo_max_storage_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="共享演示上传容量已达上限，请清理材料或在本地部署。")
