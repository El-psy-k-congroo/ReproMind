import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.core import checkpoint
from app.core.checkpoint_worker import CheckpointWorker, WorkerCheckpointSaver


def test_worker_preserves_exceptions_cancellation_and_closes():
    async def check():
        worker = CheckpointWorker()
        try:
            async def identity():
                return threading.get_ident(), asyncio.get_running_loop()

            thread_id, loop = await worker.call(identity)
            assert thread_id != threading.get_ident() and loop is worker.loop

            async def broken():
                raise ValueError('original failure')

            with pytest.raises(ValueError, match='original failure'):
                await worker.call(broken)

            started, cancelled = threading.Event(), threading.Event()

            async def slow():
                started.set()
                try:
                    await asyncio.sleep(60)
                finally:
                    cancelled.set()

            task = asyncio.create_task(worker.call(slow))
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await asyncio.to_thread(cancelled.wait, 2)
        finally:
            await worker.close()
        assert not worker.thread.is_alive() and worker.loop.is_closed()

    asyncio.run(check())


def test_worker_adapter_runs_real_graph_and_streams_checkpoint_history():
    async def check():
        worker = CheckpointWorker()
        saver = WorkerCheckpointSaver(InMemorySaver(), worker)
        try:
            builder = StateGraph(dict)
            builder.add_node('increment', lambda state: {'value': state['value'] + 1})
            builder.add_edge(START, 'increment')
            builder.add_edge('increment', END)
            graph = builder.compile(checkpointer=saver)
            config = {'configurable': {'thread_id': 'checkpoint-worker-test'}}
            assert await graph.ainvoke({'value': 1}, config) == {'value': 2}
            assert (await graph.aget_state(config)).values == {'value': 2}
            history = [item async for item in saver.alist(config, limit=2)]
            assert len(history) == 2
            await saver.adelete_thread('checkpoint-worker-test')
            assert await saver.aget_tuple(config) is None
        finally:
            await worker.close()

    asyncio.run(check())


def test_checkpoint_worker_does_not_break_application_subprocesses():
    async def check():
        loop = asyncio.get_running_loop()
        worker = CheckpointWorker()
        try:
            child = await asyncio.create_subprocess_exec(sys.executable, '-c', 'print("subprocess-ok")',
                                                        stdout=asyncio.subprocess.PIPE)
            output, _ = await asyncio.wait_for(child.communicate(), timeout=5)
            assert child.returncode == 0 and b'subprocess-ok' in output
            assert asyncio.get_running_loop() is loop
        finally:
            await worker.close()

    asyncio.run(check())


def test_windows_checkpoint_initialization_and_shutdown_use_worker(monkeypatch):
    calls = []

    class Pool:
        async def close(self):
            calls.append(('close', threading.get_ident()))

    async def open_pool(_url):
        calls.append(('open', threading.get_ident()))
        return Pool(), InMemorySaver()

    monkeypatch.setattr(checkpoint, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.setattr(checkpoint, 'get_settings', lambda: SimpleNamespace(langgraph_checkpoint_url='mock://test'))
    monkeypatch.setattr(checkpoint, '_open_postgres', open_pool)

    async def check():
        try:
            await checkpoint.init_checkpointer()
            assert checkpoint.checkpoint_status()['backend'] == 'postgresql'
            assert checkpoint.checkpoint_status()['event_loop'] == 'dedicated_selector'
        finally:
            await checkpoint.close_checkpointer()

    asyncio.run(check())
    assert calls[0][1] == calls[1][1] != threading.get_ident()


def test_unavailable_checkpoint_is_explicitly_degraded_and_worker_is_closed(monkeypatch):
    async def unavailable(_url):
        raise RuntimeError('database unavailable')

    monkeypatch.setattr(checkpoint, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.setattr(checkpoint, 'get_settings', lambda: SimpleNamespace(langgraph_checkpoint_url='mock://test'))
    monkeypatch.setattr(checkpoint, '_open_postgres', unavailable)
    asyncio.run(checkpoint.init_checkpointer())
    assert checkpoint.checkpoint_status()['backend'] == 'memory'
    assert checkpoint.checkpoint_status()['persistent'] is False
    assert 'database unavailable' in checkpoint.checkpoint_status()['error']
    assert checkpoint._worker is None
    # Isolate module state from the remaining graph tests.
    monkeypatch.setattr(checkpoint, '_error', '')


def test_health_no_longer_depends_on_removed_audit_checkpoint():
    from app import main

    assert main.health()['status'] == 'ok'
    assert 'checkpoint' not in main.health()
