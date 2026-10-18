import hashlib
import asyncio
import shutil
import subprocess
import os
import stat
import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..config import get_settings
from ..database import get_db
from ..auth import current_owner
from ..models import Artifact, ArtifactKind, Project, ReproductionJob, ReproductionStatus
from ..schemas import ArtifactRead, ProjectCreate, ProjectListItem, ProjectRead
from ..services.file_parser import parse_bytes


router = APIRouter(prefix="/projects", tags=["projects"])


def _remove_owned_tree(path: Path) -> None:
    """Remove a validated project directory, including Git's read-only pack files on Windows."""
    def onerror(function, filename, exc_info):
        if not isinstance(exc_info[1], PermissionError):
            raise exc_info[1]
        os.chmod(filename, stat.S_IWRITE | stat.S_IREAD)
        function(filename)

    shutil.rmtree(path, onerror=onerror)


async def _project_or_404(db: AsyncSession, project_id: str) -> Project:
    project = await db.scalar(
        select(Project)
        .where(Project.id == project_id)
        .options(selectinload(Project.artifacts))
    )
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


@router.get("", response_model=list[ProjectListItem])
async def list_projects(db: AsyncSession = Depends(get_db)) -> list[ProjectListItem]:
    projects = (
        await db.scalars(
            select(Project)
            .where(Project.status == "active")
            .where(Project.owner_id == current_owner.get() if current_owner.get() else True)
            .order_by(Project.updated_at.desc())
        )
    ).all()
    items: list[ProjectListItem] = []
    for project in projects:
        artifact_count = await db.scalar(select(func.count(Artifact.id)).where(Artifact.project_id == project.id)) or 0
        latest_job = await db.scalar(
            select(ReproductionJob).where(ReproductionJob.project_id == project.id)
            .order_by(ReproductionJob.created_at.desc()).limit(1)
        )
        items.append(
            ProjectListItem(
                id=project.id,
                name=project.name,
                description=project.description,
                research_domain=project.research_domain,
                artifact_count=artifact_count,
                latest_reproduction_status=latest_job.status.value if latest_job else None,
                latest_reproduction_progress=latest_job.progress if latest_job else None,
                updated_at=project.updated_at,
            )
        )
    return items


@router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
async def create_project(payload: ProjectCreate, db: AsyncSession = Depends(get_db)) -> Project:
    owner = current_owner.get()
    project = Project(tenant_id=f'user_{owner}' if owner else get_settings().default_tenant_id,
                      owner_id=owner, **payload.model_dump())
    db.add(project)
    await db.flush()
    return await _project_or_404(db, project.id)


@router.get("/{project_id}", response_model=ProjectRead)
async def get_project(project_id: str, db: AsyncSession = Depends(get_db)) -> Project:
    return await _project_or_404(db, project_id)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str, db: AsyncSession = Depends(get_db)) -> None:
    project = await _project_or_404(db, project_id)
    jobs = (await db.scalars(select(ReproductionJob).where(ReproductionJob.project_id == project_id))).all()
    from ..services.reproduction_engine import has_active_work
    if any(job.status in {ReproductionStatus.queued, ReproductionStatus.running, ReproductionStatus.executing}
           or has_active_work(job.id) or (job.result_json or {}).get('tuning_active') or (job.result_json or {}).get('ablation_active') for job in jobs):
        raise HTTPException(status_code=409, detail="复现任务正在执行，请等任务结束后删除项目")

    settings = get_settings()
    workspace_root = settings.paper_projects_dir.resolve()
    legacy_root = (workspace_root.parent / "reproductions").resolve()
    upload_root = settings.upload_dir.resolve()
    workspaces = {Path(job.workspace_path).resolve() for job in jobs if job.workspace_path}
    expected_prefix = f"-{project_id[:8]}"
    if workspace_root.is_dir():
        workspaces.update(
            path.resolve() for path in workspace_root.iterdir()
            if path.is_dir() and path.name.endswith(expected_prefix)
        )
    expected_legacy = {job.id for job in jobs}
    if any(
        not (
            (path.parent == workspace_root and path.name.endswith(expected_prefix))
            or (path.parent == legacy_root and path.name in expected_legacy)
        )
        for path in workspaces
    ):
        raise HTTPException(status_code=409, detail="工作区路径校验失败，项目未删除")
    upload_dir = (upload_root / project_id).resolve()
    if upload_dir.parent != upload_root:
        raise HTTPException(status_code=409, detail="上传目录路径校验失败，项目未删除")

    from ..services.reproduction_engine import _environment_owned_by_job, _named_environment_prefix
    environments: dict[Path, tuple[str, str, ReproductionJob]] = {}
    for job in jobs:
        spec = job.environment_spec or {}
        prefix_value = str(spec.get("prefix", ""))
        conda = str(spec.get("conda_executable", ""))
        name = str(spec.get("environment_name", ""))
        if not prefix_value or not conda or not name:
            continue
        recorded_prefix = Path(prefix_value)
        if recorded_prefix.is_symlink():
            raise HTTPException(status_code=409, detail=f"环境 {name} 使用符号链接，项目未删除")
        prefix = recorded_prefix.resolve()
        if not prefix.exists():
            continue
        named = (name.endswith(expected_prefix)
                 and prefix == _named_environment_prefix(conda, name).resolve()
                 and prefix.name == name)
        legacy_in_workspace = any(prefix == workspace / ".conda" / name for workspace in workspaces)
        if not named and not legacy_in_workspace:
            raise HTTPException(status_code=409, detail=f"环境 {name} 的目录校验失败，项目未删除")
        environments[prefix] = (conda, name, job)
    other_jobs = (await db.scalars(select(ReproductionJob).where(ReproductionJob.project_id != project_id))).all()
    referenced = {Path(str(value)).resolve() for job in other_jobs
                  if (value := (job.environment_spec or {}).get("prefix"))}
    # Validate every environment before removing any of them.
    for prefix, (conda, name, job) in environments.items():
        if prefix in referenced:
            raise HTTPException(status_code=409, detail=f"环境 {name} 仍被其他项目引用，项目未删除")
        legacy_in_workspace = any(prefix == workspace / ".conda" / name for workspace in workspaces)
        if not legacy_in_workspace and not _environment_owned_by_job(job, prefix):
            raise HTTPException(status_code=409, detail=f"环境 {name} 的项目归属无法验证，项目未删除")
        if not (prefix / "conda-meta").is_dir() and not legacy_in_workspace:
            raise HTTPException(status_code=409, detail=f"环境 {name} 缺少 Conda 历史，无法安全删除")

    for prefix, (conda, name, job) in environments.items():
        if not prefix.exists():
            continue
        if not (prefix / "conda-meta").is_dir():
            _remove_owned_tree(prefix)
            continue
        try:
            from ..services.execution_sandbox import sandbox_command
            work = Path(job.workspace_path)
            argv, env = sandbox_command([conda, "env", "remove", "--yes", "--prefix", str(prefix)], work, os.environ.copy(), workspace=work)
            result = await asyncio.to_thread(subprocess.run, argv, env=env,
                                             capture_output=True, text=True, timeout=180, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(status_code=500, detail=f"Conda 环境 {name} 删除失败：{exc}") from exc
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Conda 环境 {name} 删除失败：{result.stderr[-300:]}")
        # Conda can leave ReproMind's own marker files behind after package removal.
        if prefix.exists():
            _remove_owned_tree(prefix)

    for path in workspaces:
        if path.is_dir():
            _remove_owned_tree(path)
    if upload_dir.is_dir():
        _remove_owned_tree(upload_dir)
    await db.delete(project)
    await db.commit()


@router.post("/{project_id}/artifacts", response_model=ArtifactRead, status_code=status.HTTP_201_CREATED)
async def upload_artifact(
    project_id: str,
    kind: ArtifactKind = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> Artifact:
    project = await _project_or_404(db, project_id)
    settings = get_settings()
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件不能超过 {settings.max_upload_mb} MB")
    if not data:
        raise HTTPException(status_code=400, detail="文件为空")

    safe_name = (file.filename if file.filename is not None else "artifact.bin").replace("\\", "/").rsplit("/", 1)[-1]
    if (not safe_name or safe_name in {".", ".."} or safe_name.endswith((".", " "))
            or any(ord(char) < 32 or char in '<>:"|?*' for char in safe_name)
            or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", safe_name, re.I)):
        raise HTTPException(status_code=400, detail="文件名无效，请重命名后上传")
    target_dir = settings.upload_dir / project_id
    # Storage identity is independent of display name and stale artifact counts.
    target = target_dir / f"{uuid4().hex}{Path(safe_name).suffix}"
    created = False
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            created = True
            handle.write(data)
    except OSError as exc:
        if created:
            target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="附件写入失败，请检查存储空间和目录权限后重试") from exc

    try:
        content_text, metadata = parse_bytes(safe_name, data)
    except Exception as exc:
        content_text, metadata = "", {"parse_error": f"{type(exc).__name__}: {exc}"}

    artifact = Artifact(
        tenant_id=project.tenant_id,
        project_id=project_id,
        kind=kind,
        filename=safe_name,
        media_type=file.content_type or "application/octet-stream",
        size_bytes=len(data),
        storage_provider="local",
        storage_path=str(target),
        sha256=hashlib.sha256(data).hexdigest(),
        parse_status="failed" if metadata.get("parse_error") else "completed",
        content_text=content_text,
        metadata_json=metadata,
    )
    db.add(artifact)
    try:
        await db.flush()
        await db.refresh(artifact)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return artifact


@router.delete("/{project_id}/artifacts/{artifact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_artifact(project_id: str, artifact_id: str, db: AsyncSession = Depends(get_db)) -> None:
    artifact = await db.scalar(
        select(Artifact).where(Artifact.id == artifact_id, Artifact.project_id == project_id)
    )
    if not artifact:
        raise HTTPException(status_code=404, detail="证据文件不存在")
    path = Path(artifact.storage_path)
    if path.is_file():
        path.unlink()
    await db.delete(artifact)
