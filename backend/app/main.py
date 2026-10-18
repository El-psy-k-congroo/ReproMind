from contextlib import asynccontextmanager
from pathlib import Path

from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .api import analytics, authentication, copilot, experiment_branches, models_catalog, model_routing, projects, reproductions
from .config import get_settings
from .database import close_database, init_database
from .auth import enforce_access


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


@app.middleware('http')
async def private_api_responses(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=([settings.public_origin] if settings.deployment_mode == 'server' and settings.public_origin
                   else ["http://localhost:5173", "http://127.0.0.1:5173"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(authentication.router, prefix="/api")
for router in (projects.router, reproductions.router, experiment_branches.router,
               models_catalog.router, model_routing.router, analytics.router, copilot.router):
    app.include_router(router, prefix="/api", dependencies=[Depends(enforce_access)])


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "llm_enabled": settings.llm_enabled,
        "model_provider": settings.model_provider,
        "deployment_mode": settings.deployment_mode,
    }


@app.get("/api/deployment")
def deployment_info() -> dict:
    return {
        "mode": settings.deployment_mode,
        "execution_enabled": True,
        "shared_workspace": False,
        "auth_required": settings.deployment_mode == "server",
        "repository_url": "https://github.com/El-psy-k-congroo/ReproMind",
        "notice": "",
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


# Serve a built frontend on the same origin on both Windows and Linux.
_frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@app.get("/{frontend_path:path}", include_in_schema=False)
def frontend_page(frontend_path: str):
    if any(part.startswith('.') for part in Path(frontend_path).parts):
        raise HTTPException(status_code=404)
    if frontend_path == "api" or frontend_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="接口不存在")
    target = (_frontend_dist / frontend_path).resolve()
    if not target.is_relative_to(_frontend_dist.resolve()):
        raise HTTPException(status_code=404)
    if target.is_file():
        return FileResponse(target)
    if frontend_path.startswith('assets/') or Path(frontend_path).suffix:
        raise HTTPException(status_code=404)
    index = _frontend_dist / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=503, detail="前端尚未构建，请先在 frontend 目录执行 npm ci 和 npm run build。")
    return FileResponse(index, headers={"Cache-Control": "no-cache"})
