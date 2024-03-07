from tinydb import Query, TinyDB
from fastapi import Depends
import numpy as np
from PIL import Image

from ..schemas import BTResult
from ..lib.ml import BTPredictor
from .db import get_db


@lru_cache
def get_predictor(checkpoint_path):
    return BTPredictor(checkpoint_path)

class BTService():
    def __init__(self):
        pass

    async def predict(self, checkpoint_path, image_path, with_cam) -> np.ndarray:
        predictor = get_predictor(checkpoint_path)
        result = await predictor(image_path, with_cam)
        return result


class BTResultService:
    def __init__(self, db:TinyDB=Depends(get_db)):
        self.db = db
        self.table = db.table('bt_results')

    def all(self) ->  list[BTResult]:
        return [BTResult(id=r.doc_id, **r) for r in self.table.all()]
