from pydantic import BaseModel, Field
import pandera as pa
# from pandera.engines.pandas_engine import pydanticModel


class BaseTask(BaseModel):
    timestamp: int
    name: str
    status: str = Field(..., regex=r'^pending|processing|done$')
    mode: str = Field(..., regex=r'^bt$')


class Model(BaseModel):
    model: str
    name: str

class Scale(BaseModel):
    label: str
    scale: float
    enabled: bool


class BTModel(BaseModel):
    name: str
    label: str
    enabled: bool


class BTTask(BaseTask):
    with_cam: bool
    model: str


class BTResult(BaseModel):
    timestamp: int
    name: str
    with_cam: bool
    model: str
    memo: str
    L: float
    M: float
    G: float
    B: float
