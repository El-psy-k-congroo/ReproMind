"""User-reviewed model choices grouped by capability; no model calls on save."""
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from ..config import get_settings
from ..core.model_policy import GROUPS, read_policy, write_policy
from ..core.model_router import ModelRole, get_model_router
from .models_catalog import validate_model

router = APIRouter(prefix='/model-routing', tags=['model-routing'])
_LOCK = asyncio.Lock()


class RoutingChoice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    models: dict[str, str] = Field(max_length=4)
    revision: str = ''


@router.get('')
def routing_state():
    settings = get_settings()
    try:
        policy = read_policy(settings)
        groups = [{'id': name, **info, 'model': policy['models'].get(name, ''),
                   'effective_model': get_model_router().select(ModelRole(info['roles'][0])).model}
                  for name, info in GROUPS.items()]
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail='模型调度配置不可读取，请检查配置文件') from exc
    return {'provider': settings.model_provider, 'groups': groups, 'revision': policy.get('revision', ''),
            'note': '按能力组路由实际模型调用；执行、下载、数据检查等规则步骤不额外调用模型。单次显式模型选择优先。'}


@router.put('')
async def update_routing(payload: RoutingChoice):
    models = {key: value.strip() for key, value in payload.models.items()}
    if set(models) != set(GROUPS) or any(len(v) > 128 for v in models.values()):
        raise HTTPException(status_code=422, detail='必须提供四个能力组；空值表示沿用默认配置')
    for model in set(models.values()) - {''}:
        await validate_model(model)
    async with _LOCK:
        settings = get_settings()
        try:
            if read_policy(settings).get('revision', '') != payload.revision:
                raise HTTPException(status_code=409, detail='模型调度配置已更新，请重新读取后保存')
            await asyncio.to_thread(write_policy, settings, models)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=503, detail='模型调度配置保存失败；原配置未被替换，请检查配置文件与目录权限') from exc
    return routing_state()
