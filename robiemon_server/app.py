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
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .lib import get_hash
from .lib.worker import wait, unlock, poll, poll1, get_proc_count, add_proc2, add_coro2


# from .api import router as api_router
import pandas as pd
from .schemas import BTResult, BTTask
# from .deps import BTService


# DF
global_dfs = {}
EXCEL_PATH = 'data/db.xlsx'
schemas = {
    'bt_results': BTResult,
}

def save_dfs():
    with pd.ExcelWriter(EXCEL_PATH, engine='xlsxwriter') as writer:
        for k, df in global_dfs.items():
            df.to_excel(writer, sheet_name=k)

def init_dfs():
    if not os.path.exists(EXCEL_PATH):
        for k, S in schemas.items():
            df = pd.DataFrame(columns=list(S.schema()['properties'].keys()))
            global_dfs[k] = df
        save_dfs()
        print(f'Created empty database: {EXCEL_PATH}')
    else:
        for k, v in schemas.items():
            df = pd.read_excel(EXCEL_PATH, sheet_name=k, index_col=0)
            global_dfs[k] = df
            print(df)
        print(f'Loaded database: {EXCEL_PATH}')


def get_dfs():
    return global_dfs

def get_df(name):
    return global_dfs[name]

def set_df(name, df):
    global_dfs[name] = df
    save_dfs()

def add_data(name, data):
    df = get_df(name)
    new_df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    set_df(name, new_df)

class BaseDFDriver:
    def get_name(self):
        raise NotImplementedError()

    def get_cls(self):
        raise NotImplementedError()

    def get_df(self):
        return get_df(self.get_name())

    def set_df(self, df):
        set_df(self.get_name(), df)

    def get_models(self):
        df = self.get_df()
        cls = self.get_cls()
        return [cls(**row) for i, row in df.iterrows()]

    def add(self, model):
        add_data(self.get_name(), model.dict())


class BTDFDriver(BaseDFDriver):
    def get_name(self):
        return 'bt_results'

    def get_cls(self):
        return BTResult


## TASKS
global_tasks = []
def get_last_timestamp():
    timestamp = int(time.time())
    if len(global_tasks) > 0:
        last_timestap = global_tasks[-1].timestamp
        if last_timestap >= timestamp:
            timestamp = last_timestap + 1
    return timestamp


class ProcessBTTaks:
    # def __init__(self,
    #              bt_service=Depends(BTService),
    #              ):
    #     self.bt_service = bt_service
    #     pass

    async def process(self, task:BTTask):
        if task.status != STATUS_PENDING:
            print(f'Task {task.timestamp} is not pending')
            return

        task.status = STATUS_PROCESSING
        poll1()

        memo = ''

        ok = False
        try:
            result, features, cam_image = await self.bt_service.predict(
                f'data/weights/bt/{task.weight}',
                # 'data/weights/bt_resnetrs50_f0.pt',
                os.path.join(config.UPLOAD_DIR, f'{task.timestamp}.png'),
                with_cam=task.cam,
                # with_cam=True,
            )
            if cam_image:
                cam_image.save(os.path.join(config.CAM_DIR, f'{task.timestamp}.png'))
            else:
                if task.cam:
                    memo = 'Too large to generate CAM.'
            ok = True
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            task.status = STATUS_TOO_LARGE
        except Exception as e:
            print(e)
            task.status = STATUS_ERROR

        if ok:
            retult = BTResult(
                timestamp=task.timestamp,
                name=task.name,
                with_cam=bool(cam_image),
                weight=task.weight,
                memo=memo,
                L=result[0],
                M=result[1],
                G=result[2],
                B=result[3],
            )
            self.bt_service.add_result(result)
            task.status = STATUS_DONE

        await asyncio.sleep(1)
        poll1()
        print('PRED DONE', task.timestamp)


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
app.mount('/data', StaticFiles(directory='data'), name='static')

async def auto_refresh():
    while True:
        await asyncio.sleep(10)
        poll()

@app.on_event('startup')
async def on_startup():
    asyncio.create_task(auto_refresh())
    init_dfs()

@app.on_event('shutdown')
def shutdown_event():
    unlock()


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

@app.post('/fake')
async def post_fake(dri=Depends(BTDFDriver)):
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
    dri.add(result)
    poll1()
    return JSONResponse(content={'message': 'ok'})

@app.delete('/fake')
async def post_fake(dri:BTDFDriver=Depends(BTDFDriver)):
    df = dri.get_df()
    df = df.iloc[:0]
    dri.set_df(df)
    poll1()
    return JSONResponse(content={'message': 'ok'})


## Status

class Status:
    def __init__(self, ):
        pass

    async def get(self):
        bt_results = [BTResult(**row).dict() for i, row in get_df('bt_results').iterrows()]
        return {
            'count': get_proc_count(),
            'bt_results': bt_results,
        }


## API

router = APIRouter(
    prefix='/api',
    tags=['api'],
)

@router.get('/status')
async def status(response: Response, status: Status = Depends()):
    data = await status.get()
    return JSONResponse(content=data)

async def send_status(status):
    print('Connected')
    while True:
        await wait()
        data = await status.get()
        data_str = json.dumps(data)
        yield f'data: {data_str}\n\n'
        await asyncio.sleep(1)


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



# @router.post('/predict/bt')
async def predict(
    file: UploadFile = File(...),
    scale: float = Form(),
    cam: bool = Form(),
    weight: str = Form(),
):
    if not scale:
        raise HTTPException(status_code=404, detail='Scale is required')

    timestamp = get_last_timestamp()

    hash = get_hash(timestamp)
    org_img = Image.open(io.BytesIO(await file.read()))

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
        scaled_img.save(f'data/results/bt/{timestamp}/original.png')
        thumb_img.convert('RGB').save(f'data/results/bt/{timestamp}/thumb.png')

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

    global_tasks.append(task)
    process_bt_task(task)

    return JSONResponse(content={
        **task.dict()
    })

app.include_router(router)
