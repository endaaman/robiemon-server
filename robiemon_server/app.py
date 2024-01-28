import os
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
from fastapi import FastAPI, HTTPException, Depends, Request, Header, File, UploadFile, Response, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import numpy as np
import torch

from .lib.config import config
from .lib.db import init_db

from .deps.bt import BTService
from .deps.db import get_db
from .deps.worker import Worker
from .models import BTResultDB, ItemDB


logger = logging.getLogger('uvicorn')

STATUS_PENDING = 'pending'
STATUS_PROCESSING = 'processing'
STATUS_DONE = 'done'
STATUS_TOO_LARGE = 'too_large'
STATUS_ERROR = 'error'


class Task(BaseModel):
    timestamp: int
    hash: str
    image: str
    status: str = Field(..., regex=r'^pending|processing|done$')

    mode: str = Field(..., regex=r'^bt$')
    with_cam: bool


async def process_task(task, worker, db, bt_service):
    if task.status != STATUS_PENDING:
        print(f'Task {task.timestamp} is not pending')
        return
    print('start', task.timestamp)
    task.status = STATUS_PROCESSING
    worker.poll()

    ok = False
    try:
        result, features, cam_image = await bt_service.predict(
            'data/weights/bt_efficientnet_b0_f5.pt',
            # 'data/weights/bt_resnetrs50_f0.pt',
            os.path.join(config.UPLOAD_DIR, task.image),
            with_cam=task.with_cam,
            # with_cam=True,
        )
        if cam_image:
            cam_image_name = f'{task.timestamp}.png'
            print('save cam', os.path.join(config.CAM_DIR, cam_image_name))
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
            original_image=task.image,
            cam_image=cam_image_name,
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
    print('PRED DONE POLL')


tasks = [
    # Task(
    #     timestamp=1705901087,
    #     tag='EXAMPLE1',
    #     image='1705901087.png',
    #     mode='bt',
    #     status=STATUS_DONE,
    # ),
    # Task(
    #     timestamp=1705903831,
    #     tag='EXAMPLE2',
    #     image='1705903831.png',
    #     mode='bt',
    #     status=STATUS_PROCESSING,
    # ),
    # Task(
    #     timestamp=1705904399,
    #     tag='EXAMPLE3',
    #     image='1705904399.png',
    #     mode='bt',
    #     status=STATUS_PENDING,
    # )
]

class BTResult(BaseModel):
    id: int
    timestamp: int
    original_image: str
    cam_image: str
    L: float
    M: float
    G: float
    B: float

    class Config:
        orm_mode = True


async def get_status(db):
    bt_results = [
        BTResult.from_orm(r)
        for r in db.query(BTResultDB).all()
    ]

    status = {
        'tasks': [t.dict() for t in tasks],
        'bt_results': [r.dict() for r in  bt_results],
    }
    return status

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[
        'http://localhost',
        'http://localhost:5173',
        '*',
    ],
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.on_event('startup')
async def on_startup():
    init_db()

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.CAM_DIR, exist_ok=True)
app.mount('/uploads', StaticFiles(directory=config.UPLOAD_DIR), name='uploads')
app.mount('/cams', StaticFiles(directory=config.CAM_DIR), name='cams')

async def send_status(worker, db):
    while True:
        status = await get_status(db)
        status_str = json.dumps(status)
        yield f'data: {status_str}\n\n'
        await asyncio.sleep(1)
        await worker.event()

@app.get('/status_sse')
async def status_sse(
    response: Response,
    worker: Worker = Depends(),
    db: Session = Depends(get_db),
):
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Content-Type'] = 'text/event-stream'
    response.headers['Connection'] = 'keep-alive'
    return StreamingResponse(
        send_status(worker, db),
        media_type='text/event-stream',
    )

@app.get('/status')
async def status(response: Response, db: Session = Depends(get_db)):
    status = await get_status(db)
    return JSONResponse(content=status)


@app.post('/predict')
async def predict(
    cam: bool = Form(),
    file: UploadFile = File(...),
    worker:Worker = Depends(),
    bt_service: BTService = Depends(),
    db:Session = Depends(get_db),
):
    # timestamp = int(datetime.now().timestamp())
    timestamp = int(time.time())
    filename = f'{timestamp}.png'
    with open(os.path.join(config.UPLOAD_DIR, filename), 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    base = str(timestamp).encode('utf-8')
    hash = hashlib.sha256(base).hexdigest()

    task = Task(
        timestamp=timestamp,
        hash=hash[:8],
        image=filename,
        status=STATUS_PENDING,
        mode='bt',
        with_cam=cam,
    )
    tasks.append(task)
    print('add')
    worker.add_task(process_task, task, worker, db, bt_service)
    print('add end')
    worker.poll()

    return JSONResponse(content={
        **task.dict()
    })


@app.get('/results/bt')
async def read_result(db:Session = Depends(get_db)):
    return db.query(BTResultDB).all()

@app.delete('/results/bt/{id}')
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


class RequestItemCreate(BaseModel):
    name: str
    desc: str

class ResponseItem(BaseModel):
    id: int
    name: str
    desc: str

    class Config:
        orm_mode = True


@app.post('/items', response_model=ResponseItem)
def create_item(item: RequestItemCreate, db: Session = Depends(get_db)):
    db_item = ItemDB(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.get('/items/{item_id}', response_model=ResponseItem)
def read_item(item_id: int, db: Session = Depends(get_db)):
    db_item = db.query(ItemDB).filter(ItemDB.id == item_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item

@app.get('/items', response_model=list[ResponseItem])
def read_items(db: Session = Depends(get_db)):
    db_items = db.query(ItemDB).all()
    return db_items
