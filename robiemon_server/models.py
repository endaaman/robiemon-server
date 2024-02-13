import datetime
from pydantic import BaseModel, Field, validator
from sqlalchemy import Boolean, Column, Integer, Float, String, ForeignKey, Text

from .lib.db import Base


# class Task(Base):
#     __tablename__ = "tasks"
#     id = Column(Integer, primary_key=True, index=True)
#     image = Column(String)
#     status = Column(String)
#     bt_result_id = Column(Integer, ForeignKey('bt_results.id'))



class BTResultDB(Base):
    __tablename__ = 'bt_results'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Integer, index=True, nullable=False)
    name = Column(Text, nullable=False)
    cam = Column(Boolean, nullable=False)
    weight = Column(Text, nullable=False)
    memo = Column(Text, nullable=False)

    L = Column(Float)
    M = Column(Float)
    G = Column(Float)
    B = Column(Float)


class TaskDB(Base):
    __tablename__ = 'tasks'

    id = Column(Integer, primary_key=True, index=True)
    status = Column(Text)
    image = Column(Text)
