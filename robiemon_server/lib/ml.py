
import sys
import re
from functools import lru_cache
from collections import NamedTuple

import cv2
import numpy as np
import torch
import timm
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms, models

from PIL import Image
from PIL.Image import Image as ImageType



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
        self.base = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

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

global_models = {}

def get_model(checkpoint_path):
    key = checkpoint_path
    cached = global_models.get(key)
    if cached:
        return cached

    checkpoint: Checkpoint = torch.load(checkpoint)
    model = TimmModel(
        model_name=checkpoint.config['model_name'],
        num_classes=checkpoint.config['num_classes']
    )
    model.load_state_dict(checkpoint.model_state)
    model.to('cuda')

    global_models[key] = model
    return model



class Predictor:
    def __init__(self, checkpoint_path, mean=0.7, std=0.2):
        self.model = get_model(checkpoint_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image: ImageType):
        pass


class BTPredictor:
    def __call__(self, image: ImageType) -> np.ndarray:
        t = self.transform(image)
        ii = t[None, ...]
        oo = self.model(t)
        return oo.detach().cpu().numpy()[0]
