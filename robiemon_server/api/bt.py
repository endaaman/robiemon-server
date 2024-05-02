import os
import io
import time
import json
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import joblib
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, Header, File, UploadFile, Response, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image

from ..deps.task import TaskService, get_last_timestamp
from ..deps.scale import ScaleService
from ..deps.bt import BTResultService, BTPredictService, BTModelService
from ..schemas import BTTask, BTResult

from ..lib import asdicts, get_hash
from ..lib.worker import poll, wait, add_coro2
from ..constants import *

## API

router = APIRouter(
    prefix='/bt',
    tags=['bt'],
)


@router.get('/models')
async def get_bt_models(
    bt_model_service:BTModelService=Depends(BTModelService),
):
    return bt_model_service.all()

@router.get('/results')
async def get_bt_result(
    bt_result_service:BTResultService=Depends(BTResultService),
):
    return bt_result_service.all()

@router.get('/results/{timestamp}')
async def get_bt_results(
    timestamp:int,
    bt_result_service:BTResultService=Depends(BTResultService),
):
    r = bt_result_service.find(timestamp=timestamp)
    if not r:
        raise HTTPException(status_code=404)
    return r

@router.delete('/results/{timestamp}')
async def delete_bt_result(
    timestamp: int,
    bt_result_service:BTResultService=Depends(BTResultService),
):
    ok = bt_result_service.remove(timestamp=timestamp)
    if not ok:
        raise HTTPException(status_code=404)
    poll()
    return JSONResponse(content={'message': f'BT result({timestamp}) deleted'})


class ParamPatchBTResult(BaseModel):
    name: str | None = None
    memo: str | None = None

@router.patch('/results/{timestamp}')
async def patch_bt_result(
    timestamp: int,
    q: ParamPatchBTResult,
    bt_result_service=Depends(BTResultService),
):
    r = bt_result_service.find(timestamp=timestamp)
    if not r:
        raise HTTPException(status_code=404)

    if q.name is not None:
        r.name = q.name
    if q.memo is not None:
        r.memo = q.memo
    ok = bt_result_service.update(r)
    if not ok:
        raise HTTPException(status_code=400)
    poll()
    return JSONResponse(content={
        'message': 'Editted BT result'
    })


## PREDICT

@router.post('/predict')
async def predict(
    file: UploadFile = File(...),
    scale: float = Form(),
    cam: bool = Form(),
    model: str = Form(),
    task_service = Depends(TaskService),
    bt_predict_service = Depends(BTPredictService),
):
    if not scale:
        raise HTTPException(status_code=404, detail='Scale is required')

    timestamp = get_last_timestamp()
    hash = get_hash(timestamp)

    task = BTTask(
        timestamp=timestamp,
        name=hash[:8],
        status=STATUS_PENDING,
        mode='bt',
        with_cam=cam,
        model=model,
    )
    print('added', task)
    task_service.add(task)

    org_img = Image.open(io.BytesIO(await file.read())).convert('RGB')

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
        scaled_img.save(f'{result_dir}/original.jpg')
        thumb_img.save(f'{result_dir}/thumb.png')

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, process_bt_images)
    poll()
    add_coro2(bt_predict_service.predict, task)
    return JSONResponse(content={
        **task.dict()
    })


## UMAP

@lru_cache
def prepare_embeddings(model):
    embeddings_path = f'data/weights/bt/{model}/umap.xlsx'
    if not os.path.exists(embeddings_path):
        return None
    df = pd.read_excel(embeddings_path, index_col=0)
    df['correct'] = df['diag'] == df['pred']

    df_origin = pd.read_excel('data/meta_origin.xlsx', index_col=0)
    df = pd.merge(df, df_origin, on='name', how='left')
    df['origin'] = df['origin'].fillna('')
    df.to_dict(orient='records')
    return df.to_dict(orient='records')


@router.get('/umap/embeddings/{model}')
async def get_umap_embeddings(
    model: str,
):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        embeddings = await loop.run_in_executor(executor, prepare_embeddings, model)
    if not embeddings:
        raise HTTPException(status_code=404)
    return JSONResponse(content=embeddings)

@lru_cache
def prepare_meta_origins():
    return pd.read_excel('data/meta_origin.xlsx', index_col=0)['origin'].to_dict()

@router.get('/umap/meta_origins')
async def get_meta_origins():
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        origins = await loop.run_in_executor(executor, prepare_meta_origins)
    return JSONResponse(content=origins)

class ParamUMAP(BaseModel):
    timestamp: int

@lru_cache
def get_umap_model(model):
    return joblib.load(f'data/weights/bt/{model}/umap_model.joblib')

def prepare_umap(r):
    feature = torch.load(f'data/results/bt/{r.timestamp}/feature.pt').flatten()
    umap_model = get_umap_model(r.model)
    x, y = umap_model.transform(feature[None, ...])[0].tolist()
    return x, y

@router.post('/umap/{timestamp}')
async def post_umap(
    timestamp: int,
    # q: ParamPatchBTResult,
    bt_result_service : BTPredictService = Depends(BTResultService),
):
    r = bt_result_service.find(timestamp=timestamp)
    if not r:
        raise HTTPException(status_code=404)

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        x, y = await loop.run_in_executor(executor, prepare_umap, r)

    return JSONResponse(content={
        'x': x,
        'y': y,
    })
