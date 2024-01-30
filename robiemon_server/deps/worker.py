from ..lib.worker import add_task, wait, poll, unlock


class Worker:
    def add_task(self, task, *args, **kwargs):
        add_task(task, *args, **kwargs)

    async def wait(self):
        await wait()

    def poll(self):
        poll()

    def unlock(self):
        unlock()
