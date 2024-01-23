import asyncio


class Task:
    def __init__(self):
        status = 'pending'

    async def run(self):
        raise NotImplementedError()

    async def __call__(self):
        status = 'processing'
        await self.run()
        status = 'done'

class SleepTask(Task):
    def __init__(self, second):
        self.second = second

    async def run(self):
        print(f'start {self.second}')
        await asyncio.sleep(self.second)
        print(f'end {self.second}')
