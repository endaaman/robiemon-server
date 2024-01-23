import os
import re
import json
import threading
import logging
import asyncio
import shutil
import time
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, Header, File, UploadFile, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .lib.config import config
from .lib.db import init_db
from .lib.task import Task, SleepTask

from .deps.bt import BTService
from .deps.db import get_db
from .deps.worker import Worker
from .models import BTResult, ItemDB


logger = logging.getLogger('uvicorn')

STATUS_PENDING = 'pending'
STATUS_PROCESSING = 'processing'
STATUS_DONE = 'done'


class Task(BaseModel):
    timestamp: int
    image: str
    mode: str
    status: str


tasks = [
    Task(
        timestamp=1705901087,
        image='1705901087.png',
        mode='bt',
        status=STATUS_DONE,
    ),

    Task(
        timestamp=1705903831,
        image='1705903831.png',
        mode='bt',
        status=STATUS_PROCESSING,
    ),

    Task(
        timestamp=1705904399,
        image='1705904399.png',
        mode='bt',
        status=STATUS_PENDING,
    )
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.on_event('startup')
async def on_startup():
    os.makedirs('data/uploads', exist_ok=True)
    init_db()

app.mount('/uploads', StaticFiles(directory=config.UPLOAD_DIR), name='uploads')

@app.get('/')
async def root(worker: Worker = Depends()):
    return {
        'message': 'ok',
    }

@app.get('/results')
async def get_results_bt():
    return global_status

@app.get('/results/bt')
async def get_results_bt():
    return global_status['bt']


class Item(BaseModel):
    name: str

@app.post('/results/bt')
async def get_results_bt(item: Item):
    print(item)
    global_status['bt'].append(item.dict())
    return global_status['bt']


async def send_status():
    while True:
        s = json.dumps([t.dict() for t in tasks])
        yield f"data: {s}\n\n"
        await asyncio.sleep(1)

@app.get('/status')
async def events(response: Response):
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Content-Type'] = 'text/event-stream'
    response.headers['Connection'] = 'keep-alive'
    return StreamingResponse(
        send_status(),
        media_type='text/event-stream',
    )



@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # timestamp = int(datetime.now().timestamp())
    timestamp = int(time.time())
    filename = f'{timestamp}.png'
    with open(os.path.join(config.UPLOAD_DIR, filename), 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    task = Task(
        timestamp=timestamp,
        image=filename,
        mode='bt',
        status=STATUS_PENDING,
    )
    tasks.append(task)

    return JSONResponse(content={
        **task.dict()
    })


    # db_result = Result(filename=file.filename)
    # db.add(db_result)
    # db.commit()
    # db.refresh(db_result)

    # return {"info": f"file '{file.filename}' saved at '{file_location}'", "id": db_result.id}



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

class AddTaskRequest(BaseModel):
    s: int

@app.post('/tasks')
async def add_tasks(q: AddTaskRequest, worker: Worker = Depends()):
    t = SleepTask(q.s)
    worker.add_task(t)
    return q
