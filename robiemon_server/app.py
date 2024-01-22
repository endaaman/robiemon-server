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

from .middlewares import config
from .deps.bt import BTService
from .deps.db import get_db
from .models import BTResult


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
tasks_queue = asyncio.Queue()
async def worker():
    while True:
        task = await tasks_queue.get()
        await task()
        tasks_queue.task_done()

global_status = {
    'bt': [
        { 'name': 'initial', }
    ],
}



app = FastAPI(
    dependencies=[]
)
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
    asyncio.create_task(worker())  # ワーカータスクを起動


app.mount('/uploads', StaticFiles(directory=config.UPLOAD_DIR), name='uploads')

@app.get('/')
async def root():
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


class BaseTask(BaseModel):
    name: str
    status: str

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.status = 'pending'
        tasks.append(self)

    # def __dict__(self):
    #     return ''

    async def run(self):
        pass

    async def __call__(self):
        self.status = 'running'
        await self.run()
        self.status = 'done'

class SleepTask(BaseTask):
    async def run(self):
        await asyncio.sleep(5)



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


