import os
import json
import sys
import re
import logging
import asyncio
import signal
import time
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, Header, File, UploadFile, Response, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .lib import get_hash
from .lib.df import init_dfs, save_dfs, start_watching_dfs, stop_watching_dfs
from .lib.worker import wait, unlock, poll, get_proc_count, add_proc2, add_coro2

from .api import router as api_router
from .schemas import BTResult, BTTask
# from .deps import BTService



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
app.mount('/static', StaticFiles(directory='data'), name='static')

async def auto_refresh():
    while True:
        await asyncio.sleep(10)
        poll()

@app.on_event('startup')
async def on_startup():
    asyncio.create_task(auto_refresh())
    init_dfs()
    start_watching_dfs()

@app.on_event('shutdown')
def shutdown_event():
    print('shutdown')
    unlock()
    print('done unlock')
    stop_watching_dfs()

@app.get('/')
def get_root():
    return JSONResponse(content={
        'message': 'Welcome to ROBIEMON server.'
    })

app.include_router(api_router)
