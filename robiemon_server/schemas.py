from pydantic import BaseModel, Field



class Weight(BaseModel):
    weight: str
    name: str


class BaseTask(BaseModel):
    timestamp: int
    name: str
    status: str = Field(..., regex=r'^pending|processing|done$')
    mode: str = Field(..., regex=r'^bt$')


# BT
class BTTask(BaseTask):
    cam: bool
    weight: str


class BTResult(BaseModel):
    id: int = -1
    timestamp: int
    name: str
    with_cam: bool
    weight: str
    memo: str
    L: float
    M: float
    G: float
    B: float
