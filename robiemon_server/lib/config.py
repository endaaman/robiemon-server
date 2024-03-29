import os
from datetime import timedelta

from functools import lru_cache
from pydantic import BaseSettings, validator


class Config(BaseSettings):
    # "sqlite:///./sql_app.db"
    DB_URL: str = 'sqlite:///./data/server.db'

    DATA_DIR: str = 'data'
    STATIC_DIR: str = 'data/static'
    UPLOAD_DIR: str = 'data/static/uploads'
    THUMB_DIR: str = 'data/static/thumbs'
    CAM_DIR: str = 'data/static/cams'
    WEIGHT_DIR: str = 'data'

    class Config:
        env_file = '.env'
        frozen = True

config = Config()
