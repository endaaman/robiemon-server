import datetime
from pydantic import BaseModel, Field, validator
from sqlalchemy import Boolean, Column, Integer, Float, String, ForeignKey

from .lib.db import Base


# class Task(Base):
#     __tablename__ = "tasks"
#     id = Column(Integer, primary_key=True, index=True)
#     image = Column(String)
#     status = Column(String)
#     bt_result_id = Column(Integer, ForeignKey('bt_results.id'))



class BTResult(Base):
    __tablename__ = 'bt_results'

    id = Column(Integer, primary_key=True, index=True)
    status = Column(String)
    original_image = Column(String)
    cam_image = Column(String)

    L = Column(Float)
    M = Column(Float)
    G = Column(Float)
    B = Column(Float)


class ItemDB(Base):
    __tablename__ = 'items'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    desc = Column(String)

class TaskDB(Base):
    __tablename__ = 'tasks'
    id = Column(Integer, primary_key=True, index=True)
    status = Column(String)
    image = Column(String)
