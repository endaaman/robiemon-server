import asyncio
import queue

from fastapi import Depends


global_started = False
global_tasks = []
global_event = asyncio.Event()


async def initial_dummy_task():
    pass

async def main_loop():
    print('start main loop')
    while len(global_tasks) > 0:
        print('process one')
        tt = global_tasks.pop(0)
        task, args, kwargs = tt
        await task(*args, **kwargs)
        global_event.set()
        global_event.clear()
    print('main_loop was quited')


class Worker:
    def add_task(self, task, *args, **kwargs):
        global_tasks.append([task, args, kwargs])
        if len(global_tasks) == 1:
            asyncio.create_task(main_loop())

    async def event(self):
        await global_event.wait()

    def poll(self):
        global_event.set()
        global_event.clear()
