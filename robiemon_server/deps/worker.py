import asyncio
import queue


class Worker:
    def __init__(self):
        self.started = False
        self.queue = asyncio.Queue()

    def add_task(self, task):
        if not self.started:
            print('start worker main loop')
            asyncio.create_task(self.main_loop())
            asyncio.create_task(self.queue.join())

        self.queue.put_nowait(task)

    async def initial_dummy_task(self):
        pass

    async def main_loop(self):
        self.queue.put_nowait(self.initial_dummy_task)
        t = await self.queue.get()
        await t()
        # DO NOT call task_done() here.

        while True:
            print('enter')
            t = await self.queue.get()
            await t()
            self.queue.task_done()
            print('leave')
