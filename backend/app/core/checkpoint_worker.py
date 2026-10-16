"""Run psycopg on a dedicated loop without changing the server's Windows loop.

Windows subprocesses need Proactor, while psycopg requires Selector. Only
checkpoint I/O is dispatched here; model calls and experiments stay on the
application loop. This is an in-process adapter, not another agent or service.
"""

import asyncio
import threading

from langgraph.checkpoint.base import BaseCheckpointSaver


class CheckpointWorker:
    def __init__(self):
        self.loop = asyncio.SelectorEventLoop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._serve, name="repromind-checkpoint", daemon=True)
        self.thread.start()
        if not self.ready.wait(5):
            raise RuntimeError("检查点事件循环启动超时")

    def _serve(self):
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        try:
            self.loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    async def call(self, function, *args, **kwargs):
        if self.loop.is_closed() or not self.thread.is_alive():
            raise RuntimeError("检查点工作线程已关闭")

        async def invoke():
            return await function(*args, **kwargs)

        future = asyncio.run_coroutine_threadsafe(invoke(), self.loop)
        return await asyncio.wrap_future(future)

    async def close(self):
        if self.thread.is_alive():
            self.loop.call_soon_threadsafe(self.loop.stop)
            await asyncio.to_thread(self.thread.join, 5)
            if self.thread.is_alive():
                raise RuntimeError("检查点工作线程未能及时退出")


class WorkerCheckpointSaver(BaseCheckpointSaver):
    def __init__(self, saver, worker):
        super().__init__(serde=saver.serde)
        self.saver, self.worker = saver, worker

    @property
    def config_specs(self):
        return self.saver.config_specs

    def get_next_version(self, current, channel):
        return self.saver.get_next_version(current, channel)

    async def aget_tuple(self, config):
        return await self.worker.call(self.saver.aget_tuple, config)

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await self.worker.call(self.saver.aput, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await self.worker.call(self.saver.aput_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id):
        return await self.worker.call(self.saver.adelete_thread, thread_id)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        async def create():
            return self.saver.alist(config, filter=filter, before=before, limit=limit)

        async def next_item(iterator):
            try:
                return True, await anext(iterator)
            except StopAsyncIteration:
                return False, None

        iterator = await self.worker.call(create)
        try:
            while True:
                exists, item = await self.worker.call(next_item, iterator)
                if not exists:
                    break
                yield item
        finally:
            await self.worker.call(iterator.aclose)
