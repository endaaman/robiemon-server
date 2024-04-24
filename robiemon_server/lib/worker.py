import asyncio
import queue
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor

from . import debounce

global_started = False
global_procs = []
global_event = asyncio.Event()


async def initial_dummy_task():
    pass

def wrap_as_sync(coro):
    def inner():
        loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # loop.run_until_complete(proc())
        # loop = asyncio.get_running_loop()
        loop.run_until_complete(coro())
        loop.close()
    return inner

def get_proc_count():
    return len(global_procs)

async def main_loop():
    print('start main loop')
    loop = asyncio.get_running_loop()
    while len(global_procs) > 0:
        proc = global_procs[0]
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, proc)
        global_procs.pop(0)
    print('main_loop was done')


async def wait():
    await global_event.wait()

def add_proc(proc):
    global_procs.append(proc)
    count = len(global_procs)
    if count == 1:
        # asyncio.create_task(main_loop())
        # asyncio.ensure_future(proc)
        # loop = asyncio.get_running_loop()
        # loop.create_task(proc)
        asyncio.create_task(main_loop())
    return count

def add_proc2(func, *arg, **kwargs):
    def proc():
        func(*arg, **kwargs)
    return add_proc(proc)

def add_coro(coro):
    proc = wrap_as_sync(coro)
    return add_proc(proc)

def add_coro2(coro, *arg, **kwargs):
    async def inner():
        await coro(*arg, **kwargs)
    return add_coro(inner)

def poll0():
    global_event.set()
    global_event.clear()

@debounce(0.1)
def poll():
    poll0()

def unlock():
    global_event.set()

