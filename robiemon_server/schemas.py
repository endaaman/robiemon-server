from pydantic import BaseModel, Field


class BaseTask(BaseModel):
    timestamp: int
    name: str
    status: str = Field(..., regex=r'^pending|processing|done$')
    mode: str = Field(..., regex=r'^bt$')


class BTTask(BaseTask):
    cam: bool
    weight: str


class BTResult(BaseModel):
    id: int
    timestamp: int
    name: str
    cam: bool
    weight: str
    memo: str
    L: float
    M: float
    G: float
    B: float

    class Config:
        orm_mode = True
