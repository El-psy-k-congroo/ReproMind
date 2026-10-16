from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..core.model_router import ModelRouter

router = APIRouter(prefix="/models", tags=["models"])
DEFAULT_MODEL = "deepseek-v4-flash-0731"
_catalog: tuple[datetime, list[str]] | None = None


async def available_models() -> list[str]:
    global _catalog
    now = datetime.now(timezone.utc)
    if _catalog and now - _catalog[0] < timedelta(minutes=5):
        return _catalog[1]
    settings = get_settings()
    if settings.model_provider.lower() != "moma" or not settings.moma_api_key:
        return [settings.llm_model_chat]
    url = f"{ModelRouter._base_url(settings.moma_base_url)}/models"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5, connect=3)) as client:
            response = await client.get(url, headers={"Authorization": f"Bearer {settings.moma_api_key}"})
            response.raise_for_status()
        items = response.json().get("data", [])
        excluded = ("flux", "seedance", "bge-", "reranker", "asr", "voice", "translate", "vl-")
        models = [str(item["id"]) for item in items if item.get("id") and not any(
            part in str(item["id"]).lower() for part in excluded
        )]
        models = list(dict.fromkeys(models))
        if not models:
            raise ValueError("模型列表为空")
        _catalog = (now, models)
        return models
    except (httpx.HTTPError, ValueError, KeyError):
        # A stale catalog is preferable to advertising invented model names.
        return _catalog[1] if _catalog else [settings.moma_model_chat or DEFAULT_MODEL]


@router.get("")
async def list_models() -> dict:
    models = await available_models()
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]
    return {
        "provider": get_settings().model_provider,
        "default": default,
        "models": [{"id": model, "label": model} for model in models],
    }


async def validate_model(model_id: str) -> None:
    settings = get_settings()
    prefix = 'moma_model_' if settings.model_provider.lower() == 'moma' else 'llm_model_'
    configured = {getattr(settings, prefix + kind, '') for kind in ('chat', 'coder', 'reasoning', 'router')}
    if model_id and model_id in configured:
        return # Explicitly configured models do not require a cold remote catalog lookup.
    models = await available_models()
    if _catalog is None and settings.model_provider.lower() == 'moma' and settings.moma_api_key:
        raise HTTPException(status_code=503, detail='暂时无法读取模型目录，不能验证所选模型；请重试或使用已配置的默认模型')
    if model_id not in models:
        raise HTTPException(status_code=400, detail="所选模型当前不在移动云可用模型列表中，请刷新模型列表")
