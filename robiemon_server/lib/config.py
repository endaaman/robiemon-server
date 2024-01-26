import os
from datetime import timedelta

from functools import lru_cache
from pydantic import BaseSettings, validator


class Config(BaseSettings):
    # "sqlite:///./sql_app.db"
    DB_URL: str = 'sqlite:///./data/server.db'
    UPLOAD_DIR: str = 'data/uploads'
    CAM_DIR: str = 'data/cams'
    DATA_DIR: str = 'data'

    class Config:
        env_file = '.env'
        frozen = True

config = Config()
