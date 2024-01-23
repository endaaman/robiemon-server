import asyncio
import queue


global_started = False
global_queue = asyncio.Queue()

class Worker:
    def add_task(self, task):
        global global_started
        if not global_started:
            asyncio.create_task(self.main_loop())
            asyncio.create_task(global_queue.join())
            global_started = True

        global_queue.put_nowait(task)

    async def initial_dummy_task(self):
        pass

    async def main_loop(self):
        global_queue.put_nowait(self.initial_dummy_task)
        t = await global_queue.get()
        await t()
        # DO NOT call task_done() here.

        while True:
            t = await global_queue.get()
            await t()
            global_queue.task_done()
