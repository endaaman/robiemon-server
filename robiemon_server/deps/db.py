from tinydb import TinyDB
from functools import lru_cache
from .config import Config

@lru_cache
def get_db(config:Config=Depends()) -> TinyDB:
    db = TinyDB()
