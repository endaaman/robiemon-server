import os
import sys
import re
import logging
import asyncio
import signal

import uvicorn
from fastapi import FastAPI, APIRouter, Depends
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .lib.ml import using_gpu
from .lib.db import init_db
from .lib.worker import wait, unlock, poll
from .lib.config import config
from .api import router as api_router


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

async def auto_refresh():
    while True:
        await asyncio.sleep(10)
        poll()

@app.on_event('startup')
async def on_startup():
    asyncio.create_task(auto_refresh())
    init_db()

@app.on_event('shutdown')
def shutdown_event():
    unlock()

@app.on_event('shutdown')

@app.get('/')
def get_root():
    return JSONResponse(content={
        "message": "Welcome to ROBIEMON server."
    })


os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.THUMB_DIR, exist_ok=True)
os.makedirs(config.CAM_DIR, exist_ok=True)
app.mount('/static', StaticFiles(directory=config.STATIC_DIR), name='static')
app.include_router(api_router)


def dev():
    uvicorn.run('robiemon_server:app', host='0.0.0.0', port=3000, reload=True)

def prod():
    uvicorn.run('robiemon_server:app', host='0.0.0.0', port=3000)
