"""跨平台启动入口；Windows 下为 psycopg 使用 SelectorEventLoop。"""

import asyncio
import sys


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn  # noqa: E402

from app.config import get_settings  # noqa: E402


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        loop="asyncio",
    )
