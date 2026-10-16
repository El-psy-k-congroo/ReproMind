"""Persist capability-group choices, not credentials or per-Agent model lists."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

GROUPS = {
    'router': {'label': '编排路由', 'roles': ['router'], 'agents': ['Research Orchestrator']},
    'chat': {'label': '理解与报告', 'roles': ['paper', 'report'], 'agents': ['Paper Agent', 'Report Agent', 'Project Copilot']},
    'coder': {'label': '编码与纠错', 'roles': ['code', 'reproduction'], 'agents': ['Code Agent', 'Reproduction Agent', 'Repair Agent']},
    'reasoning': {'label': '需求规划与推理', 'roles': ['reasoning'], 'agents': ['Code Planning Agent']},
}


def read_policy(settings):
    path = Path(settings.model_routing_path)
    if not path.exists():
        return {'models': {}, 'revision': ''}
    if path.stat().st_size > 32000:
        raise ValueError('模型调度配置超过大小限制')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('models'), dict):
        raise ValueError('模型调度配置格式无效')
    if data.get('provider') != settings.model_provider.lower().strip():
        return {'models': {}, 'revision': '', 'note': '提供方已切换，未使用其他提供方的模型配置'}
    if set(data['models']) - set(GROUPS) or any(not isinstance(v, str) or len(v) > 128 for v in data['models'].values()):
        raise ValueError('模型调度配置包含无效能力组或模型名称')
    return data


def write_policy(settings, models):
    path = Path(settings.model_routing_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {'provider': settings.model_provider.lower().strip(), 'models': models,
            'revision': str(uuid4()), 'updated_at': datetime.now(timezone.utc).isoformat()}
    temporary = path.with_name(path.name + '.' + str(uuid4()) + '.tmp')
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return data


def task_model_id(role, legacy_model='', settings=None):
    """Group choice supersedes a historical project-wide choice, not API overrides."""
    from ..config import get_settings
    settings = settings or get_settings()
    group = next(name for name, info in GROUPS.items() if role.value in info['roles'])
    return read_policy(settings)['models'].get(group) or legacy_model
