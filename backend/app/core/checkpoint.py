import sys
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from ..config import get_settings


_memory_saver = InMemorySaver()
_pool: Any | None = None
_checkpointer: Any = _memory_saver
_backend = "memory"
_error = ""
_worker: Any | None = None


async def _open_postgres(url):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        conninfo=url, min_size=1, max_size=10, timeout=8, open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    try:
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        return pool, saver
    except BaseException:
        await pool.close()
        raise


async def init_checkpointer() -> None:
    """初始化 LangGraph 状态存储；PostgreSQL 不可用时安全降级到内存。"""
    global _pool, _checkpointer, _backend, _error, _worker

    await close_checkpointer()
    _checkpointer, _backend, _error = _memory_saver, "memory", ""

    url = get_settings().langgraph_checkpoint_url.strip()
    if not url:
        return

    try:
        if sys.platform == "win32":
            from .checkpoint_worker import CheckpointWorker, WorkerCheckpointSaver
            _worker = CheckpointWorker()
            _pool, saver = await _worker.call(_open_postgres, url)
            _checkpointer = WorkerCheckpointSaver(saver, _worker)
        else:
            _pool, _checkpointer = await _open_postgres(url)
        _backend = "postgresql"
        _error = ""
    except BaseException as exc:
        await close_checkpointer()
        if not isinstance(exc, Exception):
            raise
        _checkpointer = _memory_saver
        _backend = "memory"
        _error = f"{type(exc).__name__}: {exc}"


async def close_checkpointer() -> None:
    global _pool, _worker, _checkpointer, _backend
    try:
        if _pool is not None:
            if _worker:
                await _worker.call(_pool.close)
            else:
                await _pool.close()
    finally:
        _pool = None
        if _worker:
            await _worker.close()
            _worker = None
        _checkpointer, _backend = _memory_saver, "memory"


def get_checkpointer() -> Any:
    return _checkpointer


def checkpoint_status() -> dict[str, Any]:
    return {"backend": _backend, "error": _error, "persistent": _backend == "postgresql",
            "event_loop": "dedicated_selector" if _worker else "in_process"}
