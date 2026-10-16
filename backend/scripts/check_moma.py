"""使用极小请求验证 MoMA OpenAI 兼容接口；密钥仅从环境变量读取。"""

import asyncio
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import Settings  # noqa: E402
from app.core.model_router import ModelRole, ModelRouter  # noqa: E402


async def check() -> None:
    result = await ModelRouter(Settings()).complete_json(
        ModelRole.paper,
        "Return only a valid JSON object.",
        'Return {"ok": true}.',
        max_tokens=128,
    )
    print(
        {
            "provider": result.provider,
            "model": result.model,
            "usage": result.usage,
            "fallback_used": result.fallback_used,
            "error": result.error,
            "data": result.data,
        }
    )
    if result.error:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(check())
