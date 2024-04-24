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

from .deps.task import TaskService, get_last_timestamp
from .deps.scale import ScaleService
from .deps.bt import BTResultService, BTPredictService
from .schemas import BTTask, BTResult

from .lib import asdicts, get_hash
from .lib.worker import poll, wait, add_coro2
from .constants import *

## API

router = APIRouter(
    prefix='/api',
    tags=['api'],
)



## Status

bt_weights = [
    {
        'label': 'ConvNeXt V2 Nano',
        'weight': 'convnextv2_nano_v4.pt',
    },
    {
        'label': 'ResNet RS50',
        'weight': 'resnetrs50_v4.pt',
    },
]


class Status:
    def __init__(self,
                 bt_result_service=Depends(BTResultService),
                 task_service=Depends(TaskService),
                 scale_service=Depends(ScaleService),
                 ):
        self.scale_service = scale_service
        self.task_service = task_service
        self.bt_result_service = bt_result_service

    async def get(self):
        return {
            'scales': asdicts(self.scale_service.all()),
            'tasks': asdicts(self.task_service.all()),
            'bt_results': asdicts(self.bt_result_service.all()),
            'bt_weights': bt_weights,
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
        await wait()
        await asyncio.sleep(0.5)


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



## BT Result

@router.delete('/results/bt/{timestamp}')
async def delete_bt_result(
    timestamp: int,
    bt_result_service:BTResultService=Depends(BTResultService),
):
    ok = bt_result_service.remove(timestamp=timestamp)
    if not ok:
        raise HTTPException(status_code=404)
    poll()
    return JSONResponse(content={'message': f'BT result({timestamp}) deleted'})


class PatchBTResult(BaseModel):
    name: str | None = None
    memo: str | None = None

@router.patch('/results/bt/{id}')
async def patch_bt_result(
    timestamp: int,
    q: PatchBTResult,
    bt_result_service=Depends(BTResultService),
):
    ok = bt_result_service.edit(timestamp, q.dict())
    if not ok:
        raise HTTPException(status_code=404)
    poll()
    return JSONResponse(content={
        'message': 'Editted BT result'
    })


## PREDICT

@router.post('/predict/bt')
async def predict(
    file: UploadFile = File(...),
    scale: float = Form(),
    cam: bool = Form(),
    weight: str = Form(),
    task_service = Depends(TaskService),
    bt_predict_service = Depends(BTPredictService),
):
    if not scale:
        raise HTTPException(status_code=404, detail='Scale is required')

    timestamp = get_last_timestamp()

    hash = get_hash(timestamp)
    org_img = Image.open(io.BytesIO(await file.read()))

    result_dir = f'data/results/bt/{timestamp}'
    os.makedirs(result_dir, exist_ok=True)

    def process_bt_images():
        size = (int(org_img.width*scale), int(org_img.height * scale))
        scaled_img = org_img.resize(size, Image.Resampling.LANCZOS)
        long = max(org_img.width, org_img.height)
        if long > THUMB_SIZE:
            landscape = org_img.width > org_img.height
            short = min(org_img.width, org_img.height)
            new_long = round(THUMB_SIZE * long / short)
            size = (new_long, THUMB_SIZE) if landscape else (THUMB_SIZE, new_long)
            thumb_img = org_img.resize(size, Image.Resampling.LANCZOS)
        else:
            thumb_img = scaled_img
        scaled_img.save(f'{result_dir}/original.png')
        thumb_img.convert('RGB').save(f'{result_dir}/thumb.png')

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, process_bt_images)

    task = BTTask(
        timestamp=timestamp,
        name=hash[:8],
        status=STATUS_PENDING,
        mode='bt',
        cam=cam,
        weight=weight,
    )

    task_service.add(task)
    add_coro2(bt_predict_service.predict, task)

    return JSONResponse(content={
        **task.dict()
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



## Debug

@router.post('/fake')
async def post_fake(bt_result_service:BTResultService=Depends(BTResultService)):
    timestamp = get_last_timestamp()
    result = BTResult(
        timestamp=timestamp,
        name='fake',
        with_cam=False,
        weight='fake weight',
        memo='memo',
        L=0.7,
        M=0.1,
        G=0.1,
        B=0.1,
    )
    bt_result_service.add(result)
    poll()
    return JSONResponse(content={'message': 'ok'})

@router.delete('/fake')
async def post_fake(bt_result_service:BTResultService=Depends(BTResultService)):
    last = bt_result_service.all()[-1]
    bt_result_service.remove(timestamp=last.timestamp)
    poll()
    return JSONResponse(content={'message': 'ok'})
