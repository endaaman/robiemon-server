import asyncio
import queue

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
    print('main_loop was done')


async def wait():
    await global_event.wait()

def add_task(task, *args, **kwargs):
    global_tasks.append([task, args, kwargs])
    if len(global_tasks) == 1:
        asyncio.create_task(main_loop())

def poll():
    global_event.set()
    global_event.clear()

def unlock():
    global_event.set()
