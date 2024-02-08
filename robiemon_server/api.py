import os
import io
import re
import json
import threading
import logging
import asyncio
import shutil
import time
import hashlib
from datetime import datetime

import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, Header, File, UploadFile, Response, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import numpy as np
import torch

from PIL import Image
from .lib.config import config
from .deps.bt import BTService
from .deps.db import get_db
from .deps.worker import Worker
from .models import BTResultDB
from .schemas import BTResult, BTTask


logger = logging.getLogger('uvicorn')

STATUS_PENDING = 'pending'
STATUS_PROCESSING = 'processing'
STATUS_DONE = 'done'
STATUS_TOO_LARGE = 'too_large'
STATUS_ERROR = 'error'



async def process_bt_task(task:BTTask, worker, db, bt_service):
    if task.status != STATUS_PENDING:
        print(f'Task {task.timestamp} is not pending')
        return
    print('Task accepted')
    print(task)

    task.status = STATUS_PROCESSING
    worker.poll()

    ok = False
    try:
        result, features, cam_image = await bt_service.predict(
            f'data/weights/bt/{task.weight}',
            # 'data/weights/bt_resnetrs50_f0.pt',
            os.path.join(config.UPLOAD_DIR, task.image),
            with_cam=task.cam,
            # with_cam=True,
        )
        if cam_image:
            cam_image_name = f'{task.timestamp}.png'
            print('CAM SAVED')
            cam_image.save(os.path.join(config.CAM_DIR, cam_image_name))
        else:
            print('NO CAM')
            cam_image_name = ''
        ok = True
    except torch.cuda.OutOfMemoryError as e:
        print(e)
        task.status = STATUS_TOO_LARGE
    except Exception as e:
        print(e)
        task.status = STATUS_ERROR

    await asyncio.sleep(1)
    worker.poll()

    if ok:
        db_item = BTResultDB(
            timestamp=task.timestamp,
            name=task.name,
            original_image=task.image,
            cam_image=cam_image_name,
            weight=task.weight,
            memo='',
            L=result[0],
            M=result[1],
            G=result[2],
            B=result[3],
        )
        db.add(db_item)
        db.commit()
        db.refresh(db_item)

        task.status = STATUS_DONE

    worker.poll()
    print('PRED DONE', task.timestamp)


tasks = []
bt_weights = [
    {
        'label': 'ConvNeXt V2 Nano',
        'weight': 'convnextv2_nano_all.pt',
    },
    # {
    #     'label': 'CAFormer S18',
    #     'weight': 'caformer_s18_all.pt',
    # },
    {
        'label': 'ResNet RS50',
        'weight': 'resnetrs50_all.pt',
    },
    {
        'label': 'ResNet RS50(fold0)',
        'weight': 'resnetrs50_f0.pt',
    },
]

async def get_status(db):
    bt_results = [
        BTResult.from_orm(r)
        for r in db.query(BTResultDB).all()
    ]

    status = {
        'tasks': [t.dict() for t in tasks],
        'bt_results': [r.dict() for r in  bt_results],
        'bt_weights': bt_weights,
    }

    return status


router = APIRouter(
    prefix='/api',
    tags=['api'],
)

@router.get('/weights')
def read_items():
    w = [
        {
            'label': 'ResNet-RS 50',
            'weight': 'bt_resnetrs50_f0.pt',
        },
        {
            'label': 'EfficientNet B0',
            'weight': 'bt_efficientnet_b0_f5.pt',
        }
    ]
    return JSONResponse(content=w)


async def send_status(state, worker, db):
    while True:
        status = await get_status(db)
        status_str = json.dumps(status)
        yield f'data: {status_str}\n\n'
        await asyncio.sleep(1)
        await worker.wait()


@router.get('/status_sse')
async def status_sse(
    request: Request,
    response: Response,
    worker: Worker = Depends(),
    db: Session = Depends(get_db),
):
    state = request.app.state
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Content-Type'] = 'text/event-stream'
    response.headers['Connection'] = 'keep-alive'
    return StreamingResponse(
        send_status(state, worker, db),
        media_type='text/event-stream',
    )

@router.get('/status')
async def status(response: Response, db: Session = Depends(get_db)):
    status = await get_status(db)
    return JSONResponse(content=status)

@router.delete('/tasks/{timestamp}')
async def read_results(
    timestamp: int,
    worker: Worker = Depends(),
    db:Session = Depends(get_db)
):
    needle = -1
    for i, t in enumerate(tasks):
        if t.timestamp == timestamp:
            needle = i
    if needle < 0:
        return JSONResponse(content={
            'message': 'No Task found'
        }, status_code=404)


    del tasks[needle]
    worker.poll()
    return JSONResponse(content={
        'message': 'Task deleted'
    })


@router.get('/results/bt')
async def read_result(db:Session = Depends(get_db)):
    return db.query(BTResultDB).all()


@router.delete('/results/bt/{id}')
async def read_results(
    id: int,
    worker: Worker = Depends(),
    db:Session = Depends(get_db)
):
    db_item = db.query(BTResultDB).filter(BTResultDB.id == id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail='Item not found')

    db.delete(db_item)
    db.commit()
    worker.poll()
    return JSONResponse(content={
        'message': 'Record deleted'
    })



class PatchResult(BaseModel):
    name: str | None = None
    memo: str | None = None

@router.patch('/results/bt/{id}')
async def patch_results(
    id: int,
    q: PatchResult,
    worker:Worker = Depends(),
    db:Session = Depends(get_db)
):
    db_item = db.query(BTResultDB).filter(BTResultDB.id == id).first()
    print('query', q)
    if q.name is not None:
        db_item.name = q.name
        print('update name', q.name)
    if q.memo is not None:
        db_item.memo = q.memo
        print('update memo', q.memo)
    db.commit()
    worker.poll()

    return JSONResponse(content={
        'message': 'Task updated'
    })



@router.post('/predict/bt')
async def predict(
    file: UploadFile = File(...),
    scale: float = Form(),
    cam: bool = Form(),
    weight: str = Form(),
    worker:Worker = Depends(),
    bt_service: BTService = Depends(),
    db:Session = Depends(get_db),
):
    timestamp = int(time.time())

    img = Image.open(io.BytesIO(await file.read()))
    img = img.resize((int(img.width*scale), int(img.height * scale)), Image.Resampling.LANCZOS)

    file_name = f'{timestamp}.png'
    image_path = os.path.join(config.UPLOAD_DIR, file_name)
    img.save(image_path)

    # # timestamp = int(datetime.now().timestamp())
    # with open(os.path.join(config.UPLOAD_DIR, file_name), 'wb') as buffer:
    #     shutil.copyfileobj(file.file, buffer)

    base = str(timestamp).encode('utf-8')
    hash = hashlib.sha256(base).hexdigest()

    task = BTTask(
        timestamp=timestamp,
        name=hash[:8],
        image=file_name,
        status=STATUS_PENDING,
        mode='bt',

        cam=cam,
        weight=weight,
    )
    tasks.append(task)
    worker.add_task(process_bt_task, task, worker, db, bt_service)
    worker.poll()

    return JSONResponse(content={
        **task.dict()
    })


@router.post('/predict/eosino')
async def predict(
    file: UploadFile = File(...),
    worker:Worker = Depends(),
    bt_service: BTService = Depends(),
    db:Session = Depends(get_db),
):
    return JSONResponse(content={
        'message': 'WIP'
    })
