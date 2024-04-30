import os
import io
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, Header, File, UploadFile, Response, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image

from ..deps.task import TaskService
from ..deps.scale import ScaleService
from ..deps.bt import BTResultService, BTModelService

from ..lib import asdicts
from ..lib.worker import poll, wait
from ..constants import *

from .bt import router as bt_router

## API

router = APIRouter(
    prefix='/api',
    tags=['api'],
)


router.include_router(bt_router)

## Status


class Status:
    def __init__(self,
                 task_service=Depends(TaskService),
                 scale_service=Depends(ScaleService),
                 bt_result_service=Depends(BTResultService),
                 bt_model_service=Depends(BTModelService),
                 ):
        self.task_service = task_service
        self.scale_service = scale_service
        self.bt_result_service = bt_result_service
        self.bt_model_service = bt_model_service

    async def get(self):
        return {
            'scales': asdicts(self.scale_service.all()),
            'tasks': asdicts(self.task_service.all()),
            'bt_results': asdicts(self.bt_result_service.all()),
            'bt_models': asdicts(self.bt_model_service.all()),
        }


@router.get('/status')
async def status(response: Response, status: Status = Depends()):
    data = await status.get()
    return JSONResponse(content=data)

async def send_status(status):
    while True:
        data = await status.get()
        data_str = json.dumps(data)
        yield f'data: {data_str}\n\n'
        await asyncio.sleep(0.5)
        await wait()


@router.get('/status_sse')
async def status_sse(
    request: Request,
    response: Response,
    status: Status = Depends(),
):
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Content-Type'] = 'text/event-stream'
    response.headers['Connection'] = 'keep-alive'
    response.headers['X-Accel-Buffering'] = 'no'
    return StreamingResponse(
        send_status(status),
        media_type='text/event-stream',
    )


## Tasks

@router.delete('/tasks/{timestamp}')
async def delete_task(
    timestamp: int,
    task_service:TaskService=Depends(TaskService),
):
    ok = task_service.remove(timestamp=timestamp)
    if not ok:
        raise HTTPException(status_code=404)
    poll()
    return JSONResponse(content={
        'message': f'Task({timestamp}) deleted'
    })


## Scale
@router.delete('/scales/{index}')
async def delete_scale(
    index: int,
    scale_service:ScaleService=Depends(ScaleService),
):
    ok = scale_service.remove(i=index)
    if not ok:
        raise HTTPException(status_code=404)
    poll()
    return JSONResponse(content={
        'message': f'Scale({index}) deleted'
    })
