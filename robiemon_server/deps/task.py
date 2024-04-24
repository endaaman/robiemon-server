import time
from ..lib import asdicts

global_tasks = []

def get_last_timestamp():
    timestamp = int(time.time())
    if len(global_tasks) > 0:
        last_timestap = global_tasks[-1].timestamp
        if last_timestap >= timestamp:
            timestamp = last_timestap + 1
    return timestamp



class TaskService:
    def __init__(self):
        pass

    def add(self, task):
        global_tasks.append(task)

    def remove(self, timestamp):
        idx = -1
        print('q:', timestamp)
        for i, t in enumerate(global_tasks):
            print(t)
            if t.timestamp == timestamp:
                idx = i
                break
        if idx < 0:
            return False
        del global_tasks[idx]
        return True

    def all(self):
        return global_tasks
