import os
from datetime import timedelta

from functools import lru_cache
from pydantic import BaseSettings, validator


class Config(BaseSettings):
    DATA_DIR: str = 'data'
    RESULT_DIR: str = 'data/results/'

    WEIGHT_DIR: str = 'data'

    class Config:
        env_file = '.env'
        frozen = True

config = Config()
