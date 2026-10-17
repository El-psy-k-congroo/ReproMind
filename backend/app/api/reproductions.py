import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database import get_db
from ..config import get_settings
from ..core.model_router import ModelRole, get_model_router
from .models_catalog import validate_model
from ..models import ArtifactKind, Project, ReproductionJob, ReproductionStatus
from ..schemas import ReproductionCreate, ReproductionRead
from ..services.reproduction_engine import (
    clear_process_pause, execute_reproduction, has_active_work, prepare_reproduction, request_process_pause,
    validate_repository_url,
)
from ..services.telemetry import read_job_telemetry
from ..services.ablations import discover_ablations, execute_ablation
from ..services.tuning import build_tuning_plans
from ..services.tuning_sessions import APPROVAL_LOCK, reviewed_plans, new_session, execute_tuning_session


router = APIRouter(prefix="/projects/{project_id}/reproductions", tags=["reproductions"])


class SourceChoice(BaseModel):
    choice: str
    repository_url: str = Field(default="", max_length=500)


class AblationChoice(BaseModel):
    dataset: str
    flag: str


class ApprovalOptions(BaseModel):
    tuning_budget: int = Field(default=0, ge=0, le=12)


class TuningApproval(BaseModel):
    purpose: Literal['recovery', 'validation_optimization'] = 'recovery'
    datasets: list[str] = Field(min_length=1, max_length=12)
    budget: int = Field(default=6, ge=1, le=12)
    review_tokens: dict[str, str]


def _source_changed_after_approval(job):
    revisions = (job.result_json or {}).get('code_revisions') or []
    if not revisions: return False
    try:
        approved = datetime.fromisoformat(str(job.approved_at))
        if approved.tzinfo is None: approved = approved.replace(tzinfo=timezone.utc)
        for revision in revisions:
            applied = datetime.fromisoformat(str(revision.get('applied_at') or ''))
            if applied.tzinfo is None: applied = applied.replace(tzinfo=timezone.utc)
            if applied > approved: return True
        return False
    except (ValueError, TypeError):
        return True


@router.get("/{job_id}/tuning-plan")
async def tuning_plan(project_id: str, job_id: str, db: AsyncSession = Depends(get_db),
                      purpose: Literal['recovery', 'validation_optimization'] = 'recovery') -> dict:
    job = await db.scalar(select(ReproductionJob).where(
        ReproductionJob.id == job_id, ReproductionJob.project_id == project_id))
    if not job:
        raise HTTPException(status_code=404, detail="复现任务不存在")
    if not job.workspace_path or not (job.execution_command or {}).get('commands'):
        return {'plans': [], 'executed': False, 'requires_approval': True, 'pending': True,
                'reason': '工作区或作者执行入口尚在准备，暂无可审查的调参计划；未启动额外搜索。'}
    try:
        plans = await asyncio.to_thread(reviewed_plans, job, 6, purpose)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"plans": plans, "executed": False, "requires_approval": True}


@router.post("/{job_id}/tuning-sessions", status_code=202)
async def start_tuning_session(project_id: str, job_id: str, payload: TuningApproval,
                               background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    async with APPROVAL_LOCK:
        job = await db.scalar(select(ReproductionJob).where(
            ReproductionJob.id == job_id, ReproductionJob.project_id == project_id))
        if not job:
            raise HTTPException(404, '复现任务不存在')
        await db.refresh(job)
        if job.status != ReproductionStatus.completed:
            raise HTTPException(409, '先结束全部基线实验；不会新建实验或重跑基线')
        if (job.result_json or {}).get('tuning_active') or (job.result_json or {}).get('ablation_active') or has_active_work(job.id):
            raise HTTPException(409, '当前实验已有额外搜索或消融正在执行，不能重复批准')
        from .experiment_branches import _ensure_idle
        await _ensure_idle(db, project_id)
        plans = await asyncio.to_thread(reviewed_plans, job, payload.budget, payload.purpose)
        selected = [p for p in plans if p['dataset'] in payload.datasets]
        if len(selected) != len(set(payload.datasets)):
            raise HTTPException(422, '数据集没有可核验的调参计划')
        for plan in selected:
            if not plan['ready'] or payload.review_tokens.get(plan['dataset']) != plan['review_token']:
                raise HTTPException(409, plan['blocked_reason'] or '搜索计划已改变，请重新审查')
        record = new_session(selected)
        job.result_json = {**(job.result_json or {}), 'tuning_active': {'id': record['id']},
                           'tuning_sessions': [*(job.result_json or {}).get('tuning_sessions', []), record]}
        clear_process_pause(job.id)
        await db.commit()
        background_tasks.add_task(execute_tuning_session, job.id, record['id'])
        return {'run_id': job.id, 'session_id': record['id'], 'status': 'approved'}


def _paper_metric_excerpt(text: str, datasets: list[str]) -> str:
    lines = text.splitlines()
    terms = [name.lower() for name in datasets if isinstance(name, str)] + [
        "result", "performance", "accuracy", "aupr", "auc", "f1-score", "precision", "recall", "table"
    ]
    hits = [index for index, line in enumerate(lines) if any(term in line.lower() for term in terms)]
    selected: set[int] = set()
    for index in hits[:100]:
        selected.update(range(max(0, index - 2), min(len(lines), index + 5)))
    return "\n".join(f"L{index + 1}: {lines[index]}" for index in sorted(selected))[:24_000]


def _verified_paper_targets(raw: object, text: str, datasets: list[str]) -> dict[str, dict[str, float]]:
    if not isinstance(raw, dict):
        return {}
    accepted: dict[str, dict[str, float]] = {}
    for dataset, values in raw.items():
        if not isinstance(dataset, str) or not isinstance(values, dict):
            continue
        matched = next((name for name in datasets if name.casefold() == dataset.casefold()), None)
        if not matched:
            continue
        for metric, value in values.items():
            if not isinstance(metric, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,30}", metric):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not 0 <= numeric <= 10000:
                continue
            candidates = {f"{numeric:g}", f"{numeric:.1f}", f"{numeric:.2f}", f"{numeric:.3f}"}
            if numeric <= 1:
                candidates.update({f"{numeric * 100:g}", f"{numeric * 100:.1f}", f"{numeric * 100:.2f}"})
            if not any(re.search(r"(?<!\d)" + re.escape(candidate) + r"(?!\d)", text) for candidate in candidates):
                continue
            accepted.setdefault(matched, {})[metric.lower()] = numeric / 100 if numeric > 1 and metric.lower() in {
                "accuracy", "acc", "auc", "auroc", "aupr", "f1", "precision", "recall"
            } else numeric
    return accepted


def _table_paper_targets(text: str, model_name: str, datasets: list[str]) -> dict[str, dict[str, float]]:
    """Parse only a clearly aligned author-model row under explicit dataset headings."""
    if not model_name or not datasets:
        return {}
    lines = text.splitlines()
    model_pattern = re.compile(r"^\s*" + re.escape(model_name) + r"\s+(.+)$", re.I)
    numeric = re.compile(r"\d+(?:\.\d+)?")
    for index, line in enumerate(lines):
        match = model_pattern.match(line)
        if not match:
            continue
        values = [float(value) for value in numeric.findall(match.group(1))]
        if len(values) < len(datasets):
            continue
        context = lines[max(0, index - 12):index]
        heading = next((row for row in reversed(context) if all(
            dataset.casefold() in row.casefold() for dataset in datasets)), "")
        metric_line = next((row for row in reversed(context) if re.match(r"^\s*(?:Model|Method|Approach)\s+", row, re.I)), "")
        if not heading or not metric_line:
            continue
        heading_positions = [heading.casefold().find(dataset.casefold()) for dataset in datasets]
        if heading_positions != sorted(heading_positions):
            continue
        labels = metric_line.split()[1:]
        if not labels or len(labels) != len(values) or len(labels) % len(datasets):
            continue
        per_dataset = len(labels) // len(datasets)
        if not 1 <= per_dataset <= 10:
            continue
        result: dict[str, dict[str, float]] = {}
        for dataset_index, dataset in enumerate(datasets):
            for offset in range(per_dataset):
                label = labels[dataset_index * per_dataset + offset].lower().replace("_", "-")
                key = {"acc": "accuracy", "auc-roc": "auc", "auroc": "auc", "roc-auc": "auc",
                       "auc-pr": "aupr", "pr-auc": "aupr", "f1-score": "f1"}.get(label, label)
                if not re.fullmatch(r"[a-z][a-z0-9-]{0,30}", key):
                    continue
                value = values[dataset_index * per_dataset + offset]
                result.setdefault(dataset, {})[key] = value / 100 if value > 1 and key in {
                    "accuracy", "auc", "aupr", "f1", "precision", "recall"
                } else value
        if result:
            return result
    return {}


@router.post("/{job_id}/paper-targets")
async def refresh_paper_targets(project_id: str, job_id: str,
                                db: AsyncSession = Depends(get_db)) -> dict:
    job = await db.scalar(select(ReproductionJob).where(
        ReproductionJob.id == job_id, ReproductionJob.project_id == project_id))
    if not job:
        raise HTTPException(status_code=404, detail="实验不存在")
    spec = dict(job.paper_spec or {})
    if spec.get("target_results") and (spec.get('target_validation') or {}).get('version')==1:
        from ..services.metric_contract import dataset_targets
        normalized = dataset_targets(spec)
        if normalized:
            return {"target_results": normalized, "source": "已保存的论文提取结果，已统一数据集、指标名与单位"}
    project = await db.scalar(select(Project).where(Project.id == project_id).options(selectinload(Project.artifacts)))
    paper = next((item for item in (project.artifacts if project else []) if item.kind == ArtifactKind.paper), None)
    if not paper or not paper.content_text.strip():
        return {"target_results": {}, "reason": "论文文本未成功解析；请上传可复制文本的论文或指标表"}
    datasets = [str(name) for name in spec.get("datasets", []) if isinstance(name, str)]
    parsed = _table_paper_targets(paper.content_text, str(spec.get("model_name") or ""), datasets)
    if parsed:
        spec["target_results"] = parsed
        spec["target_results_source"] = "论文文本中的数据集表头、指标表头和作者方法结果行"
        spec["target_results_checked_at"] = datetime.now(timezone.utc).isoformat()
        spec['target_validation']={'version':1,'status':'verified','source':'deterministic_explicit_table_headers'}
        job.paper_spec = spec
        await db.commit()
        return {"target_results": parsed, "source": spec["target_results_source"], "reason": ""}
    excerpt = _paper_metric_excerpt(paper.content_text, datasets)
    if len(excerpt) < 100:
        return {"target_results": {}, "reason": "论文解析文本中未定位到结果表或数据集名称"}
    try:
        from ..services.paper_target_evidence import TARGET_EVIDENCE_PROMPT,bind_targets,target_table_context,explicit_table_targets
        direct=explicit_table_targets(spec,paper.content_text)
        if direct['target_results']:
            direct['target_results_source']='论文原文中的唯一作者方法行与显式数据集表头'
            direct['target_results_checked_at']=datetime.now(timezone.utc).isoformat()
            job.paper_spec=direct
            await db.commit()
            return {'target_results':direct['target_results'],'source':direct['target_results_source'],'reason':''}
        excerpt=target_table_context(paper.content_text,str(spec.get('model_name') or ''),datasets) or excerpt
        response = await get_model_router().complete_json(
            ModelRole.paper,
            "从论文原文摘取作者模型在指定数据集上的评估指标数值。只返回 JSON {target_results:{数据集:{指标:数值}}}。"
            "仅选论文作者方法的主结果，不要拿基线、消融或其他模型的数字冒充作者结果。"
            "保留表格原有数字；无法确定作者结果时留空，不推断、不编造。" + TARGET_EVIDENCE_PROMPT,
            json.dumps({"datasets": datasets, 'model_name':spec.get('model_name'),
                        'metrics':spec.get('metrics'), "excerpt": excerpt}, ensure_ascii=False),
            max_tokens=2200, model_override=job.model_id,non_thinking=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"论文指标提取失败：{type(exc).__name__}: {str(exc)[:180]}") from exc
    spec=bind_targets({**spec,'target_results':response.data.get('target_results') or {},
                       'target_evidence':response.data.get('target_evidence') or []},paper.content_text)
    targets=spec['target_results']
    spec["target_results_source"] = "论文原文中的作者方法行、数据集列与报告单元格已逐项回查" if targets else '没有可核验的作者方法与数据集列对应证据'
    spec["target_results_checked_at"] = datetime.now(timezone.utc).isoformat()
    job.paper_spec = spec
    await db.commit()
    return {"target_results": targets, "source": spec["target_results_source"],
            "reason": "已检查论文文本，但没有可安全归属到作者方法的数值" if not targets else ""}


def _verified_report_ready(job: ReproductionJob) -> bool:
    return bool(
        job.current_stage in {"completed", "failed", "awaiting_data"}
        and any(event.get("stage") == "result_verification" for event in (job.events or []))
        and (job.result_json or {}).get("finished_at")
    )


@router.post("", response_model=ReproductionRead, status_code=status.HTTP_202_ACCEPTED)
async def create_reproduction(
    project_id: str,
    payload: ReproductionCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ReproductionJob:
    project = await db.scalar(
        select(Project)
        .where(Project.id == project_id)
        .options(selectinload(Project.artifacts))
    )
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    if not any(item.kind == ArtifactKind.paper for item in project.artifacts):
        raise HTTPException(status_code=400, detail="启动自主复现前请先上传论文")
    if not any(item.kind == ArtifactKind.paper
               and getattr(item, 'parse_status', 'completed') == 'completed'
               and bool(getattr(item, 'content_text', '').strip())
               and Path(item.filename).suffix.lower() != '.zip'
               for item in project.artifacts):
        raise HTTPException(status_code=400, detail="论文未成功解析出正文，请上传有效PDF或文本论文后重试")
    await validate_model(payload.model_id)
    async with APPROVAL_LOCK:
        return await _enqueue_reproduction(project, payload, background_tasks, db)


async def _enqueue_reproduction(project: Project, payload: ReproductionCreate,
                                background_tasks: BackgroundTasks, db: AsyncSession) -> ReproductionJob:
    project_id = project.id
    # PostgreSQL serializes different workers on the project row; the local lock
    # also covers SQLite's single-process fallback.
    await db.scalar(select(Project.id).where(Project.id == project_id).with_for_update())
    active = await db.scalar(
        select(ReproductionJob)
        .where(
            ReproductionJob.project_id == project_id,
            ReproductionJob.status.in_([
                ReproductionStatus.queued,
                ReproductionStatus.running,
                ReproductionStatus.executing,
            ]),
        )
        .order_by(ReproductionJob.created_at.desc())
        .limit(1)
    )
    if active:
        return active

    job = ReproductionJob(
        tenant_id=project.tenant_id,
        project_id=project_id,
        goal=payload.goal,
        repository_url=payload.repository_url,
        execution_mode=payload.execution_mode,
        model_id=payload.model_id,
        events=[{
            "stage": "queued",
            "progress": 0,
            "message": "自主复现任务已进入队列",
            "time": datetime.now(timezone.utc).isoformat(),
        }],
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    background_tasks.add_task(prepare_reproduction, job.id)
    return job


@router.get("/latest", response_model=ReproductionRead | None)
async def get_latest_reproduction(
    project_id: str, db: AsyncSession = Depends(get_db)
) -> ReproductionJob | None:
    return await db.scalar(
        select(ReproductionJob)
        .where(ReproductionJob.project_id == project_id)
        .order_by(ReproductionJob.created_at.desc())
        .limit(1)
    )


@router.get("/{job_id}", response_model=ReproductionRead)
async def get_reproduction(
    project_id: str, job_id: str, db: AsyncSession = Depends(get_db)
) -> ReproductionJob:
    job = await db.scalar(
        select(ReproductionJob).where(
            ReproductionJob.id == job_id,
            ReproductionJob.project_id == project_id,
        )
    )
    if not job:
        raise HTTPException(status_code=404, detail="复现任务不存在")
    return job


@router.get("/{job_id}/report", response_class=FileResponse)
async def download_reproduction_report(
    project_id: str, job_id: str, db: AsyncSession = Depends(get_db)
) -> FileResponse:
    job = await db.scalar(
        select(ReproductionJob).where(
            ReproductionJob.id == job_id,
            ReproductionJob.project_id == project_id,
        )
    )
    if not job:
        raise HTTPException(status_code=404, detail="复现任务不存在")
    if not _verified_report_ready(job):
        raise HTTPException(status_code=409, detail="实验尚未完成结果核验，暂不提供报告")
    report_path = Path(str((job.result_json or {}).get("report_path", ""))).resolve()
    workspace = Path(job.workspace_path).resolve() if job.workspace_path else Path()
    if not report_path.is_file() or workspace not in report_path.parents:
        raise HTTPException(status_code=404, detail="该任务尚未生成复现报告")
    return FileResponse(
        report_path,
        media_type="text/markdown; charset=utf-8",
        filename=f"repromind-reproduction-{job.id}.md",
    )


@router.get("/{job_id}/telemetry")
async def experiment_telemetry(
    project_id: str, job_id: str, db: AsyncSession = Depends(get_db),
) -> dict:
    job = await db.scalar(select(ReproductionJob).where(
        ReproductionJob.id == job_id, ReproductionJob.project_id == project_id,
    ))
    if not job:
        raise HTTPException(status_code=404, detail="复现任务不存在")
    if not job.workspace_path:
        return {"series": []}
    return read_job_telemetry(Path(job.workspace_path).resolve(), job.id)


@router.get("/{job_id}/ablations/options")
async def ablation_options(project_id: str, job_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    job = await db.scalar(select(ReproductionJob).where(
        ReproductionJob.id == job_id, ReproductionJob.project_id == project_id,
    ))
    if not job:
        raise HTTPException(status_code=404, detail="复现任务不存在")
    if job.status != ReproductionStatus.completed:
        return {"options": [], "reason": "需要先完成基线实验，才能对同一数据集进行消融。"}
    try:
        options = discover_ablations(job)
    except ValueError:
        options = []
    return {"options": options, "reason": "" if options else "作者入口脚本没有可验证的组件关闭开关，暂不把超参数变化冒充消融。"}


@router.post("/{job_id}/ablations", status_code=202)
async def start_ablation(
    project_id: str, job_id: str, payload: AblationChoice,
    background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db),
) -> dict:
    job = await db.scalar(select(ReproductionJob).where(
        ReproductionJob.id == job_id, ReproductionJob.project_id == project_id,
    ))
    if not job:
        raise HTTPException(status_code=404, detail="复现任务不存在")
    if job.status != ReproductionStatus.completed or (job.result_json or {}).get("ablation_active") or (job.result_json or {}).get('tuning_active') or has_active_work(job.id):
        raise HTTPException(status_code=409, detail="请先完成当前实验或消融任务")
    if (job.result_json or {}).get("code_revisions"):
        raise HTTPException(status_code=409, detail="原实验的源码已修改；请在修改后的新实验记录上运行消融，避免混合旧结果")
    if not any(item["dataset"] == payload.dataset and item["flag"] == payload.flag
               for item in discover_ablations(job)):
        raise HTTPException(status_code=400, detail="作者仓库中不存在此组件消融开关")
    result = dict(job.result_json or {})
    result["ablation_active"] = {"dataset": payload.dataset, "flag": payload.flag,
                                 "started_at": datetime.now(timezone.utc).isoformat()}
    job.result_json = result
    await db.commit()
    background_tasks.add_task(execute_ablation, job.id, payload.dataset, payload.flag)
    return {"status": "queued", "dataset": payload.dataset, "flag": payload.flag}


@router.get("/{job_id}/activity")
async def current_activity(
    project_id: str, job_id: str, db: AsyncSession = Depends(get_db),
) -> dict:
    job = await db.scalar(select(ReproductionJob).where(
        ReproductionJob.id == job_id, ReproductionJob.project_id == project_id,
    ))
    if not job:
        raise HTTPException(status_code=404, detail="复现任务不存在")
    operation = (job.result_json or {}).get("active_operation") or {}
    recent_operation = not bool(operation) and job.status in {ReproductionStatus.paused, ReproductionStatus.failed, ReproductionStatus.awaiting_data}
    if recent_operation:
        operation = (job.result_json or {}).get('last_operation') or {}
    ablation = (job.result_json or {}).get("ablation_active") or {}
    tuning = (job.result_json or {}).get('tuning_active') or {}
    tuning_mode = bool(tuning and not operation)
    if (not operation and not ablation and not tuning) or not job.workspace_path:
        return {"active": False, "log_tail": "", "worker_active": has_active_work(job.id)}
    workspace = Path(job.workspace_path).resolve()
    root = get_settings().paper_projects_dir.resolve()
    if workspace.parent != root or not workspace.name.endswith(f"-{project_id[:8]}"):
        return {"active": False, "log_tail": ""}
    ablation_mode = bool(ablation and not operation)
    if tuning_mode:
        path = Path(str(tuning.get('log_path') or '')).resolve()
        allowed_parent = workspace / 'runs' / job.id / 'telemetry' / 'tuning' / str(tuning.get('id', ''))
        operation = {'kind': 'tuning', 'step': tuning.get('trial', 0), 'total_steps': 0,
                     'command': f"{tuning.get('dataset', '')} {tuning.get('parameters', {})}"}
    elif ablation_mode:
        dataset = str(ablation.get("dataset", ""))
        flag = str(ablation.get("flag", ""))
        if not any(item["dataset"] == dataset and item["flag"] == flag for item in discover_ablations(job)):
            return {"active": False, "log_tail": ""}
        from ..services.reproduction_engine import _slug
        path = (workspace / "runs" / job.id / "telemetry" / f"ablation-{_slug(dataset)}-{flag.removeprefix('--')}.log").resolve()
        operation = {"kind": "ablation", "step": 1, "total_steps": 1,
                     "started_at": ablation.get("started_at", ""), "command": f"{dataset} {flag}"}
        allowed_parent = workspace / "runs" / job.id / "telemetry"
    else:
        path = Path(str(operation.get("log_path", ""))).resolve()
        allowed_parent = workspace / "runs" / job.id / "operations"
    if workspace not in path.parents or path.parent != allowed_parent:
        return {"active": False, "log_tail": ""}
    tail = ""
    if path.is_file():
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 5000))
            tail = stream.read().decode("utf-8", errors="replace")
    return {"active": not recent_operation and job.status == (ReproductionStatus.completed if ablation_mode or tuning_mode else ReproductionStatus.executing), "worker_active": has_active_work(job.id), "operation": {
        key: value for key, value in operation.items() if key != "log_path"
    }, "log_tail": tail}


@router.post("/{job_id}/pause", response_model=ReproductionRead)
async def pause_reproduction(
    project_id: str, job_id: str, db: AsyncSession = Depends(get_db),
) -> ReproductionJob:
    job = await db.scalar(select(ReproductionJob).where(
        ReproductionJob.id == job_id, ReproductionJob.project_id == project_id,
    ))
    if not job:
        raise HTTPException(status_code=404, detail="复现任务不存在")
    if job.status == ReproductionStatus.paused:
        return job
    if job.status == ReproductionStatus.completed and ((job.result_json or {}).get("ablation_active") or (job.result_json or {}).get('tuning_active')):
        pass
    elif job.status not in {
        ReproductionStatus.queued, ReproductionStatus.running, ReproductionStatus.executing,
        ReproductionStatus.awaiting_approval, ReproductionStatus.awaiting_data, ReproductionStatus.needs_input,
    }:
        raise HTTPException(status_code=409, detail="当前任务无法暂停")
    previous = job.status.value
    job.result_json = {**(job.result_json or {}), "paused_from_status": previous,
                       "paused_from_stage": job.current_stage}
    job.status = ReproductionStatus.paused
    job.current_stage = "paused"
    job.message = "用户已暂停任务；正在停止当前步骤，可从已完成的阶段继续"
    job.events = [*(job.events or []), {"stage": "paused", "progress": job.progress,
                                    "message": job.message, "time": datetime.now(timezone.utc).isoformat()}]
    # Publish the pause origin before waking the worker's cleanup transaction.
    # Otherwise its stale JSON can erase the only resume checkpoint.
    await db.commit()
    await asyncio.to_thread(request_process_pause, job.id)
    await db.refresh(job)
    return job


@router.post("/{job_id}/resume-paused", response_model=ReproductionRead, status_code=202)
async def resume_paused(
    project_id: str, job_id: str, background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ReproductionJob:
    job = await db.scalar(select(ReproductionJob).where(
        ReproductionJob.id == job_id, ReproductionJob.project_id == project_id,
    ))
    if not job:
        raise HTTPException(status_code=404, detail="复现任务不存在")
    if job.status != ReproductionStatus.paused:
        raise HTTPException(status_code=409, detail="当前任务未暂停")
    if _source_changed_after_approval(job):
        raise HTTPException(status_code=409, detail="批准运行后已修改源码；不能用新代码覆盖原实验，请通过助手创建新实验记录")
    if has_active_work(job.id):
        raise HTTPException(status_code=409, detail="正在停止上一步骤，请稍后再恢复")
    previous = str((job.result_json or {}).get("paused_from_status") or '')
    revision = (job.result_json or {}).get('revision') or (job.result_json or {}).get('rerun') or {}
    approved_revision = (bool(revision) and job.approved_at and (job.execution_command or {}).get('commands'))
    if approved_revision:
        planned = [row.get('dataset') for row in job.execution_command['commands']]
        if not revision.get('datasets') or set(planned) != set(revision['datasets']):
            raise HTTPException(status_code=409, detail='当前命令与已批准的重跑范围不一致，不能恢复；请重新核对运行范围')
        job.source_mode = 'copilot_revision' if (job.result_json or {}).get('revision') else 'copilot_rerun'
        previous = 'executing'
    if not previous:
        raise HTTPException(status_code=409, detail='暂停检查点缺失，不能默认重新准备全部实验；请核对记录后明确选择重跑范围')
    clear_process_pause(job.id)
    if previous == "completed" and (job.result_json or {}).get("ablation_active"):
        active = (job.result_json or {})["ablation_active"]
        job.status = ReproductionStatus.completed
        job.current_stage = "completed"
        background_tasks.add_task(execute_ablation, job.id, str(active["dataset"]), str(active["flag"]))
    elif previous in {"executing"} and job.approved_at:
        job.status = ReproductionStatus.executing
        job.current_stage = "environment_building"
        background_tasks.add_task(execute_reproduction, job.id)
    elif previous in {"queued", "running"}:
        job.status = ReproductionStatus.queued
        job.current_stage = "queued"
        background_tasks.add_task(prepare_reproduction, job.id)
    elif previous in {"awaiting_approval", "awaiting_data", "needs_input"}:
        job.status = ReproductionStatus(previous)
        job.current_stage = str((job.result_json or {}).get("paused_from_stage") or previous)
    else:
        raise HTTPException(status_code=409, detail="暂停前状态无法恢复")
    job.message = "已从暂停状态恢复"
    job.result_json = {key: value for key, value in (job.result_json or {}).items()
                       if key not in {"paused_from_status", "paused_from_stage"}}
    await db.commit()
    await db.refresh(job)
    return job


@router.post("/{job_id}/approve", response_model=ReproductionRead, status_code=status.HTTP_202_ACCEPTED)
async def approve_reproduction(
    project_id: str,
    job_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    payload: ApprovalOptions | None = None,
) -> ReproductionJob:
    job = await db.scalar(
        select(ReproductionJob).where(
            ReproductionJob.id == job_id,
            ReproductionJob.project_id == project_id,
        )
    )
    if not job:
        raise HTTPException(status_code=404, detail="复现任务不存在")
    if job.status != ReproductionStatus.awaiting_approval:
        raise HTTPException(status_code=409, detail="当前任务不处于待批准状态")
    if (job.execution_command or {}).get('blocked_reason'):
        raise HTTPException(status_code=409, detail=job.execution_command['blocked_reason'])
    budget = payload.tuning_budget if payload else 0
    plans = await asyncio.to_thread(build_tuning_plans, job, budget) if budget else []
    job.result_json = {**(job.result_json or {}), "tuning_policy": {
        "enabled": bool(budget), "budget_per_dataset": budget,
        "approved_at": datetime.now(timezone.utc).isoformat() if budget else None,
        "test_feedback_allowed": False, "phase": "after_all_baselines",
        "recovery_datasets": [], "requires_failure_review": True}, "tuning_plans": plans}
    job.status = ReproductionStatus.executing
    job.approved_at = datetime.now(timezone.utc)
    job.message = "执行已批准，正在启动"
    await db.commit()
    await db.refresh(job)
    background_tasks.add_task(execute_reproduction, job.id)
    return job


@router.post("/{job_id}/resume", response_model=ReproductionRead, status_code=status.HTTP_202_ACCEPTED)
async def resume_reproduction(
    project_id: str, job_id: str, background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ReproductionJob:
    job = await db.scalar(select(ReproductionJob).where(
        ReproductionJob.id == job_id, ReproductionJob.project_id == project_id,
    ))
    if not job:
        raise HTTPException(status_code=404, detail="复现任务不存在")
    if job.status not in {ReproductionStatus.awaiting_data, ReproductionStatus.needs_input, ReproductionStatus.failed}:
        raise HTTPException(status_code=409, detail="该任务当前不需要续跑")
    if _source_changed_after_approval(job):
        raise HTTPException(status_code=409, detail="原实验源码已修改；请通过助手创建新的实验记录，避免覆盖旧结果")
    if has_active_work(job.id):
        raise HTTPException(status_code=409, detail="上一步骤仍在退出，请稍后重试")
    active_id = await db.scalar(select(ReproductionJob.id).where(
        ReproductionJob.project_id == project_id,
        ReproductionJob.status.in_([ReproductionStatus.queued, ReproductionStatus.running,
                                    ReproductionStatus.executing]),
    ).limit(1))
    if active_id:
        raise HTTPException(status_code=409, detail="项目已有活动实验，请等待其结束或暂停")
    action_kind = str(((job.result_json or {}).get("action_required") or {}).get("kind", ""))
    if job.status == ReproductionStatus.needs_input:
        from ..models import Artifact
        if action_kind == "source_unavailable":
            uploaded = await db.scalar(select(Artifact.id).where(
                Artifact.project_id == project_id, Artifact.kind == ArtifactKind.code,
            ).limit(1))
            if not uploaded and not job.repository_url:
                raise HTTPException(status_code=409, detail="请先提供作者仓库地址或上传作者源码；未开源自动编码已暂缓")
        elif action_kind not in {"hardware_unavailable", "paper_spec_failed", "code_generation_failed"}:
            raise HTTPException(status_code=409, detail="当前任务需要先完成页面提示的选择")
    if job.approved_at and action_kind not in {"paper_spec_failed", "source_unavailable", "code_generation_failed"}:
        job.status = ReproductionStatus.executing
        job.current_stage = "executing"
        job.progress = 88
        job.message = "正在保留已成功结果并继续未完成的数据集"
        background_tasks.add_task(execute_reproduction, job.id)
    else:
        job.status = ReproductionStatus.queued
        job.current_stage = "queued"
        job.progress = 0
        job.message = "已收到补充证据，重新检查数据与执行计划"
        background_tasks.add_task(prepare_reproduction, job.id)
    job.error_message = ""
    await db.commit()
    await db.refresh(job)
    return job


@router.post("/{job_id}/source-choice", response_model=ReproductionRead, status_code=status.HTTP_202_ACCEPTED)
async def choose_source(
    project_id: str, job_id: str, payload: SourceChoice, background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ReproductionJob:
    job = await db.scalar(select(ReproductionJob).where(
        ReproductionJob.id == job_id, ReproductionJob.project_id == project_id,
    ))
    if not job:
        raise HTTPException(status_code=404, detail="复现任务不存在")
    if payload.choice == "generate":
        raise HTTPException(status_code=409, detail="未开源论文自动编码功能已暂缓；请提供作者仓库地址或上传作者源码")
    if payload.choice != "repository":
        raise HTTPException(status_code=400, detail="当前只支持选择作者仓库")
    if job.status not in {ReproductionStatus.needs_input, ReproductionStatus.failed} or job.approved_at:
        raise HTTPException(status_code=409, detail="已批准实验不能更换源码；请使用新的实验记录")
    try:
        repository_url = validate_repository_url(payload.repository_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if has_active_work(job.id):
        raise HTTPException(status_code=409, detail="上一步骤仍在退出，请稍后重试")
    active_id = await db.scalar(select(ReproductionJob.id).where(
        ReproductionJob.project_id == project_id,
        ReproductionJob.status.in_([ReproductionStatus.queued, ReproductionStatus.running,
                                    ReproductionStatus.executing]),
    ).limit(1))
    if active_id:
        raise HTTPException(status_code=409, detail="项目已有活动实验，请等待其结束或暂停")
    job.repository_url = repository_url
    job.result_json = {**(job.result_json or {}), "source_decision": "repository"}
    job.status = ReproductionStatus.queued
    job.current_stage = "queued"
    job.progress = 0
    job.message = "已选择作者仓库，正在重新获取源码和核验实验信息"
    job.error_message = ""
    await db.commit()
    await db.refresh(job)
    background_tasks.add_task(prepare_reproduction, job.id)
    return job


@router.post("/{job_id}/continue-partial", response_model=ReproductionRead, status_code=status.HTTP_202_ACCEPTED)
async def continue_partial(
    project_id: str, job_id: str, background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ReproductionJob:
    job = await db.scalar(select(ReproductionJob).where(
        ReproductionJob.id == job_id, ReproductionJob.project_id == project_id,
    ))
    if not job:
        raise HTTPException(status_code=404, detail="复现任务不存在")
    if job.status != ReproductionStatus.awaiting_data:
        raise HTTPException(status_code=409, detail="当前任务不在等待数据阶段")
    spec = dict(job.dataset_spec or {})
    missing = {str(value).lower() for value in spec.get("missing_datasets", [])}
    missing.update(str(value).lower() for value in spec.get("download_errors", []) if ":" not in str(value))
    if not missing:
        raise HTTPException(status_code=409, detail="没有需要跳过的缺失数据集")
    candidates = [
        str(value) for value in (job.paper_spec or {}).get("datasets", [])
        if str(value).lower() not in missing
    ]
    if not candidates:
        raise HTTPException(status_code=409, detail="没有其他可执行的数据集；请上传数据或补充可用的作者数据地址")
    prior = dict(job.result_json or {})
    skipped = sorted({*prior.get("skipped_datasets", []), *missing})
    job.result_json = {**prior, "skipped_datasets": skipped, "action_required": None}
    spec["excluded_datasets"] = skipped
    spec["missing_datasets"] = []
    spec["download_requests"] = [
        item for item in spec.get("download_requests", [])
        if str(item.get("dataset", "")).lower() not in missing
    ]
    job.dataset_spec = spec
    if job.approved_at:
        command = dict(job.execution_command or {})
        commands = [item for item in command.get("commands", []) if str(item.get("dataset", "")).lower() not in missing]
        if not commands:
            raise HTTPException(status_code=409, detail="没有可执行命令，无法继续")
        command["commands"] = commands
        command["argv"] = commands[0]["argv"]
        command["display"] = "\n".join(f"[{item['dataset']}] {' '.join(item['argv'])}" for item in commands)
        job.execution_command = command
        job.status = ReproductionStatus.executing
        job.current_stage = "executing"
        job.message = "已跳过缺失数据，继续可用数据集；最终报告将标明未完成项"
        background_tasks.add_task(execute_reproduction, job.id)
    else:
        job.status = ReproductionStatus.queued
        job.current_stage = "queued"
        job.progress = 0
        job.message = "已跳过缺失数据，正在重新规划可用数据集"
        background_tasks.add_task(prepare_reproduction, job.id)
    await db.commit()
    await db.refresh(job)
    return job
