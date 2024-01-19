import datetime
from pydantic import BaseModel, Field, validator
from sqlalchemy import (Boolean, Column, Integer, Float, String)

from .middlewares.db import Base


class BTResult(Base):
    __tablename__ = "bt_results"

    id = Column(Integer, primary_key=True, index=True)
    original_image = Column(String)
    cam_image = Column(String)

    L = Column(Float)
    M = Column(Float)
    G = Column(Float)
    B = Column(Float)


class BTResult(BaseModel):
    id: int
    original_image: str
    cam_image: str
    L: float
    M: float
    G: float
    B: float


# Base.metadata.create_all(bind=engine)
