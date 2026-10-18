from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import CopilotMessage, Project, ReproductionJob
from ..auth import current_owner

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview")
async def overview(db: AsyncSession = Depends(get_db)) -> dict:
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=30)
    owned = select(Project.id).where(Project.owner_id == current_owner.get()) if current_owner.get() else select(Project.id)
    projects = await db.scalar(select(func.count(Project.id)).where(Project.status == "active", Project.id.in_(owned))) or 0
    jobs = (await db.scalars(select(ReproductionJob).where(ReproductionJob.created_at >= since, ReproductionJob.project_id.in_(owned)))).all()
    copilot_calls = (await db.scalars(select(CopilotMessage).where(
        CopilotMessage.created_at >= since, CopilotMessage.role == "assistant", CopilotMessage.project_id.in_(owned),
    ))).all()
    reproduction_tokens = sum(
        int((job.code_spec or {}).get("model_usage", {}).get("total_tokens", 0)) for job in jobs
    )
    completed_projects = len({
        job.project_id for job in jobs if (job.result_json or {}).get("reproduction_success") is True
    })
    daily_runs = [
        {
            "date": (now.date() - timedelta(days=day)).isoformat(),
            "count": sum(job.created_at.date() == now.date() - timedelta(days=day) for job in jobs),
        }
        for day in range(29, -1, -1)
    ]
    daily_tokens = [{
        "date": row["date"],
        "count": sum(int((job.code_spec or {}).get("model_usage", {}).get("total_tokens", 0))
              for job in jobs if job.created_at.date().isoformat() == row["date"])
        + sum(message.token_usage for message in copilot_calls
              if message.created_at.date().isoformat() == row["date"]),
    } for row in daily_runs]
    status_breakdown = {key: sum(job.status.value == key for job in jobs) for key in (
        "queued", "running", "needs_input", "awaiting_data", "awaiting_approval",
        "executing", "completed", "failed",
    )}
    return {
        "period_days": 30,
        "active_projects": projects,
        "reproduced_projects": completed_projects,
        "reproduction_runs": len(jobs),
        "tokens": reproduction_tokens
        + sum(message.token_usage for message in copilot_calls),
        "daily_runs": daily_runs,
        "daily_tokens": daily_tokens,
        "status_breakdown": status_breakdown,
        "tokens_by_role": {
            "复现": reproduction_tokens,
            "项目对话": sum(message.token_usage for message in copilot_calls),
        },
        "waiting_projects": len({job.project_id for job in jobs if job.status.value in {"needs_input", "awaiting_data"}}),
        "failed_runs": status_breakdown["failed"],
    }
