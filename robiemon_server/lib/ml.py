import gc
import sys
import re
import asyncio
from functools import lru_cache
from typing import NamedTuple

from aioprocessing import AioEvent, AioProcess
from PIL import Image
from PIL.Image import Image as ImageType
import cv2
import numpy as np
import torch
import timm
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms, models

Image.MAX_IMAGE_PIXELS = None



class Checkpoint(NamedTuple):
    config: dict
    epoch: int
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict
    train_history: dict
    val_history: dict
    random_states: dict


class TimmModel(nn.Module):
    def __init__(self, name, num_classes, pretrained=True):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.base = timm.create_model(self.name, pretrained=pretrained, num_classes=num_classes)

    def get_cam_layers(self):
        return get_cam_layers(self.base, self.name)

    def forward(self, x, activate=False, with_feautres=False):
        features = self.base.forward_features(x)
        x = self.base.forward_head(features)

        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=1)
            else:
                x = torch.sigmoid(x)

        if with_feautres:
            features = self.base.global_pool(features)
            return x, features
        return x

global_predictor = {}


class BasePredictor:
    def inner(self, *args):
        raise NotImplementedError()

    async def __call__(self, *args):
        # event = AioEvent()
        # p = AioProcess(
        #     target=self.inner, args=(event, image)
        # )
        # p.start()
        # await event.coro_wait()
        # return self.result

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self.inner, *args
        )
        return result

class ClsPredictor(BasePredictor):
    def __init__(self, checkpoint_path):
        self.result = None
        self.checkpoint_path = checkpoint_path
        self.checkpoint: Checkpoint = torch.load(checkpoint_path)

        mean = self.checkpoint.config.get('mean', 0.7)
        std = self.checkpoint.config.get('std', 0.2)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def get_model(self):
        model = TimmModel(
            name=self.checkpoint.config['model_name'],
            num_classes=self.checkpoint.config['num_classes']
        )
        model.load_state_dict(self.checkpoint.model_state)
        model = model.eval().to('cuda')
        return model



class BTPredictor(ClsPredictor):
    def inner(self, image_path) -> np.ndarray:
        print('start pred', image_path)

        model = self.get_model()
        image = Image.open(image_path).convert('RGB')
        t = self.transform(image)[None, ...].to('cuda')
        try:
            with torch.no_grad():
                oo, features = model(t, activate=True, with_feautres=True)
        except torch.cuda.OutOfMemoryError as e:
            raise e
        finally:
            del t
            del model
            torch._C._cuda_clearCublasWorkspaces()
            gc.collect()
            torch.cuda.empty_cache()
        o = oo.detach().cpu().numpy()[0]
        print('done pred', image_path)
        return o


@lru_cache
def get_predictor(checkpoint_path):
    return BTPredictor(checkpoint_path)
