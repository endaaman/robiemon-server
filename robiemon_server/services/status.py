from fastapi import Depends
from tinydb import Query, TinyDB

from .bt import BTService
from .task import TaskService
from .weight import WeightService

async def get_status(
        weight_service:WeightService=Depends(),
        bt_service:BTService=Depends(),
        task_service:TaskService=Depends()
    ):

    status = {
        'tasks': task_service.all(),
        'bt_results': task_service.all(),
        'models': models,
    }

    return status
