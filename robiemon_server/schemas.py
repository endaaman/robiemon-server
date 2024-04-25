from pydantic import BaseModel, Field
import pandera as pa
# from pandera.engines.pandas_engine import pydanticModel


class BaseTask(BaseModel):
    timestamp: int
    name: str
    status: str = Field(..., regex=r'^pending|processing|done$')
    mode: str = Field(..., regex=r'^bt$')


class Weight(BaseModel):
    weight: str
    name: str
class Scale(BaseModel):
    label: str
    scale: float
    enabled: bool



class BTTask(BaseTask):
    with_cam: bool
    weight: str
class BTResult(BaseModel):
    timestamp: int
    name: str
    with_cam: bool
    weight: str
    memo: str
    L: float
    M: float
    G: float
    B: float



# class WeightSchema(pa.SchemaModel):
#     class Config:
#         dtype = pydanticModel(Weight)

# class ScaleSchema(pa.SchemaModel):
#     class Config:
#         dtype = pydanticModel(Weight)

# class BTTaskSchema(pa.SchemaModel):
#     class Config:
#         dtype = pydanticModel(Weight)

# class BTTaskSchema(pa.SchemaModel):
#     class Config:
#         dtype = pydanticModel(Weight)
