import numpy as np
from PIL import Image

from ..models import BTResultDB
from ..lib.ml import get_predictor

class BTService():
    def __init__(self):
        pass

    async def predict(self, checkpoint_path, image_path) -> np.ndarray:
        predictor = get_predictor(checkpoint_path)
        result = await predictor(image_path)
        return result
