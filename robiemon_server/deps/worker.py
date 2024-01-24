import asyncio
import queue

from fastapi import Depends


global_started = False
global_queue = asyncio.Queue()
global_event = asyncio.Event()


async def initial_dummy_task():
    pass

async def main_loop():
    # global_queue.put_nowait(initial_dummy_task)
    # global_queue.put_nowait(initial_dummy_task)
    # t = await global_queue.get()
    # print(t)
    # await t()
    # DO NOT call task_done() here.

    while True:
        tt = await global_queue.get()
        print(tt)
        task, args, kwargs = tt
        await task(*args, **kwargs)
        global_queue.task_done()
        global_event.set()
        global_event.clear()

    print('main_loop was quited')


class Worker:
    def add_task(self, task, *args, **kwargs):
        global global_started
        if not global_started:
            asyncio.create_task(main_loop())
            asyncio.create_task(global_queue.join())
            global_started = True

        global_queue.put_nowait([task, args, kwargs])

    async def event(self):
        await global_event.wait()

    def poll(self):
        global_event.set()
        global_event.clear()
