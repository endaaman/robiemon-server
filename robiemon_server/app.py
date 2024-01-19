import os
import re
import threading
import logging
import asyncio
from functools import lru_cache
import uvicorn

from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, Depends, Request, Header, File, UploadFile
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware

from .middlewares import config
from .deps.bt import BTService
from .deps.db import get_db
from .models import BTResult


logger = logging.getLogger('uvicorn')

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

@app.get('/')
async def root(
    # S:BTService=Depends()
):
    return {
        'result': 'ok',
    }


@app.post('/predict/')
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    # データベースに結果を保存
    db_result = Result(filename=file.filename)  # 他の情報もここに追加
    db.add(db_result)
    db.commit()
    db.refresh(db_result)

    return {"info": f"file '{file.filename}' saved at '{file_location}'", "id": db_result.id}
