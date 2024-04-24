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
from .lib.df import reload_dfs, save_dfs, start_watching_dfs, stop_watching_dfs
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
    reload_dfs()
    start_watching_dfs()

@app.on_event('shutdown')
def shutdown_event():
    unlock()
    stop_watching_dfs()
    save_dfs()


class Foo1:
    def __init__(self, v):
        print('init foo1 v:', v)
        self.name = time.ctime()

@app.get('/')
def get_root(foo1=Depends(Foo1)):
    print(foo1.name)
    return JSONResponse(content={
        'message': 'Welcome to ROBIEMON server.'
    })


@app.post('/sleep')
async def post_sleep(t:int=Form()):
    async def sl():
        print('start sleep')
        await asyncio.sleep(t)
        print('center')
        time.sleep(t)
        print('end sleep')
    add_coro(sl)
    return JSONResponse(content={
        'count': get_proc_count()
    })

@app.post('/thread')
async def thread(t:int=Form()):
    def th():
        print('start sleep')
        time.sleep(t)
        print('end sleep')
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, th)
    return JSONResponse(content={'message': 'ok'})


app.include_router(api_router)
