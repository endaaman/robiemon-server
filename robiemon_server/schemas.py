from pydantic import BaseModel, Field


class BaseTask(BaseModel):
    timestamp: int
    name: str
    image: str
    status: str = Field(..., regex=r'^pending|processing|done$')
    mode: str = Field(..., regex=r'^bt$')


class BTTask(BaseTask):
    cam: bool
    weight: str


class BTResult(BaseModel):
    id: int
    timestamp: int
    name: str
    original_image: str
    cam_image: str
    weight: str
    L: float
    M: float
    G: float
    B: float

    class Config:
        orm_mode = True
