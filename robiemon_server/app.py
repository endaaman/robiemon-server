import os
import re
import json
import threading
import logging
import asyncio
from functools import lru_cache
import uvicorn

from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Depends, Request, Header, File, UploadFile, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .middlewares import config
from .deps.bt import BTService
from .deps.db import get_db
from .models import BTResult


logger = logging.getLogger('uvicorn')



class Task(BaseModel):
    timestamp: int
    image: str
    status: str


tasks = []
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

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(worker())  # ワーカータスクを起動

@app.get('/')
async def root(
    # S:BTService=Depends()
):
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
        s = json.dumps(tasks)
        print('send', s)
        yield f"data: {s}\n\n"
        await asyncio.sleep(1)

# @app.get('/status')
# async def status() -> StreamingResponse:
#     return StreamingResponse(
#         str(send_status()),
#         media_type='text/event-stream',
#     )

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
async def predict():
# async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # file_location = f"uploads/{file.filename}"
    # with open(file_location, 'wb+') as file_object:
    #     file_object.write(file.file.read())

    print('pred')
    # task = SleepTask('sleep')
    print('pred ok')
    # tasks_queue.put(task)
    return []


    # db_result = Result(filename=file.filename)
    # db.add(db_result)
    # db.commit()
    # db.refresh(db_result)

    # return {"info": f"file '{file.filename}' saved at '{file_location}'", "id": db_result.id}


