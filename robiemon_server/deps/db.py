from fastapi import Request
from sqlalchemy.orm import sessionmaker, Session
from ..lib.db import SessionLocal

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
