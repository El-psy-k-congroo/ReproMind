from contextlib import asynccontextmanager

from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .api import analytics, copilot, experiment_branches, models_catalog, model_routing, projects, reproductions
from .config import get_settings
from .database import close_database, init_database


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_database()
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield
    await close_database()


async def validate_resource_ids(request: Request) -> None:
    """Reject malformed resource IDs before they reach UUID database columns."""
    for name in ("project_id", "job_id", "artifact_id", "conversation_id"):
        value = request.path_params.get(name)
        if value is None:
            continue
        try:
            UUID(str(value))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"{name} 必须是有效的 UUID") from exc


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Paper reproduction, project code collaboration and experiment execution API",
    lifespan=lifespan,
    dependencies=[Depends(validate_resource_ids)],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(projects.router, prefix="/api")
app.include_router(reproductions.router, prefix="/api")
app.include_router(experiment_branches.router, prefix="/api")
app.include_router(models_catalog.router, prefix="/api")
app.include_router(model_routing.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(copilot.router, prefix="/api")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "llm_enabled": settings.llm_enabled,
        "model_provider": settings.model_provider,
    }


@app.get("/api/agents/topology")
def agent_topology() -> dict:
    return {
        "framework": "ReproductionEngine + project-scoped coding workflow",
        "provider": settings.model_provider,
        "nodes": [
            {"id": "planner", "name": "Code Planning Agent", "model_role": "reasoning"},
            {"id": "paper_agent", "name": "Paper Agent", "model_role": "chat"},
            {"id": "code_agent", "name": "Code Agent", "model_role": "coder"},
            {"id": "repair_agent", "name": "Repair Agent", "model_role": "coder"},
            {"id": "result_agent", "name": "Result Agent", "model_role": "rule"},
            {"id": "report_agent", "name": "Report Agent", "model_role": "chat"},
        ],
        "routes": {
            "reproduction": ["paper_agent", "code_agent", "result_agent", "report_agent"],
            "code_revision": ["planner", "code_agent", "repair_agent"],
        },
        "context": "Project source snapshots, approved commands and reproduction records",
    }
