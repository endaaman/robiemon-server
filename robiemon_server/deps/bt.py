import gc
import sys
import re
import asyncio
from functools import lru_cache
from typing import NamedTuple

from fastapi import Depends

from PIL import Image
from PIL.Image import Image as ImageType
import cv2
import numpy as np
import pandas as pd
from matplotlib import cm as colormap
import torch
import timm
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms, models
import pytorch_grad_cam as CAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .df import BaseDFDriver
from ..lib.worker import poll, wait
from ..schemas import BTResult, BTTask, BTModel
from ..constants import *


Image.MAX_IMAGE_PIXELS = None


using_gpu = torch.cuda.is_available()
# print('GPU mode' if using_gpu else 'CPU mode')
device = torch.device('cuda' if using_gpu else 'cpu')


def get_cam_layers(m, name=None):
    name = name or m.default_cfg['architecture']
    if re.match(r'.*efficientnet.*', name):
        return [m.conv_head]
    if re.match(r'^resnetrs.*', name):
        return [m.layer4[-1]]
    if re.match(r'^resnetv2.*', name):
        return [m.stages[-1].blocks[-1]]
    if re.match(r'^resnet\d+', name):
        return [m.layer4[-1].act2]
    if re.match(r'^caformer_.*', name):
        return [m.stages[-1].blocks[-1].res_scale2]
    if re.match(r'^convnext.*', name):
        # return [m.stages[-1].blocks[-1].conv_dw]
        return [m.stages[-1].blocks[-1].norm]
    if name == 'swin_base_patch4_window7_224':
        return [m.layers[-1].blocks[-1].norm1]
    raise RuntimeError('CAM layers are not determined.')
    return []

def find_closest_pair(x, r):
    closest_diff = float('inf')
    closest_a = None
    closest_b = None
    for a in range(1, x + 1):
        if x % a != 0:
            continue
        b = x // a
        current_r = a / b
        diff = abs(current_r - r)
        if diff < closest_diff:
            closest_diff = diff
            closest_a = a
            closest_b = b
    return closest_a, closest_b


def get_reshaper(name, width, height):
    def reshape_transform(tensor):
        w, h = find_closest_pair(tensor.numel()//512, width/height)
        result = tensor.reshape(
            tensor.size(0),
            h,
            w,
            tensor.size(-1)
        )
        # ABCD -> ABDC -> ADBC
        result = result.transpose(2, 3).transpose(1, 2)
        # print(tensor[:, 1 :  , :].shape)
        # result = tensor[:, 1 :  , :].reshape(tensor.size(0), height, width, tensor.size(-1))
        # result = result.transpose(2, 3).transpose(1, 2)
        return result

    if re.match(r'^caformer_.*', name):
        return reshape_transform

    if re.match(r'^convnext_.*', name):
        return reshape_transform

    if name == 'swin_base_patch4_window7_224':
        return reshape_transform

    return None

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
            if hasattr(self.base, 'global_pool'):
                pool = self.base.global_pool
            else:
                pool = self.base.head.global_pool
            features = pool(features)
            return x, features
        return x

global_predictor = {}



class BTPredictor:
    def __init__(self, checkpoint_path):
        self.result = None
        self.checkpoint_path = checkpoint_path
        self.checkpoint = torch.load(checkpoint_path, map_location=device)

        mean = self.checkpoint['config'].get('mean', 0.7)
        std = self.checkpoint['config'].get('std', 0.2)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def get_model(self):
        model = TimmModel(
            name=self.checkpoint['config']['model_name'],
            num_classes=self.checkpoint['config']['num_classes']
        )
        model.load_state_dict(self.checkpoint['model_state'])
        model = model.eval().to(device)
        return model

    async def __call__(self, image_path, with_cam) -> np.ndarray:
        model = self.get_model()

        image = Image.open(image_path).convert('RGB')
        t = self.transform(image)[None, ...].to(device)
        try:
            with torch.no_grad():
                # TODO: imple with_feautres
                oo, features = model(t, activate=True, with_feautres=True)
            feature = features.detach().cpu().numpy()[0]
            o = oo.detach().cpu().numpy()[0]
            del oo
        except torch.cuda.OutOfMemoryError as e:
            del t
            del model
            torch._C._cuda_clearCublasWorkspaces()
            gc.collect()
            torch.cuda.empty_cache()
            raise e

        cam_mask = None
        if with_cam:
            try:
                gradcam = CAM.GradCAMPlusPlus(
                # gradcam = CAM.GradCAM(
                    model=model,
                    target_layers=get_cam_layers(model.base, model.name),
                    reshape_transform=get_reshaper(model.name, image.width, image.height)
                )
                pred_id = np.argmax(o)
                targets = [ClassifierOutputTarget(pred_id)]
                cam_mask = gradcam(input_tensor=t, targets=targets)[0]
                del gradcam
            except torch.cuda.OutOfMemoryError as e:
                del gradcam

        del t
        del model
        gc.collect()
        if using_gpu:
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
        return o, feature, cam_mask



class BTModelDriver(BaseDFDriver):
    def get_name(self):
        return 'bt_models'

    def get_cls(self):
        return BTModel


class BTModelService:
    def __init__(self, driver:BTModelDriver=Depends(BTModelDriver)):
        self.driver = driver

    def all(self):
        return self.driver.all()


@lru_cache
def get_predictor(checkpoint_path):
    return BTPredictor(checkpoint_path)


class BTDFDriver(BaseDFDriver):
    def get_name(self):
        return 'bt_results'

    def get_cls(self):
        return BTResult

class BaseService:
    @property
    def df(self):
        return self.driver.get_df()

    def add(self, model):
        data = model.dict()
        if len(self.df) == 0:
            df_new = pd.DataFrame([data])
        else:
            df_new = pd.concat([self.df, pd.DataFrame([data])], ignore_index=True)
        self.driver.replace(df_new)


class BTResultService(BaseService):
    def __init__(self, driver:BTDFDriver=Depends(BTDFDriver)):
        self.driver = driver

    def find_all(self, **kwargs):
        assert kwargs
        k, v = next(iter(kwargs.items()))
        df = self.df
        rows = df[df[k] == v]
        return [BTResult(**row) for i, row in rows.iterrows()]

    def find(self, **kwargs):
        rr = self.find_all(**kwargs)
        if len(rr) == 0:
            return None
        if len(rr) > 1:
            print('Items duplicated:')
            print(rr)
        return rr[0]

    def remove(self, **kwargs):
        assert kwargs
        k, v = next(iter(kwargs.items()))
        df = self.df
        df_new = df[df[k] != v]
        if len(df) - len(df_new) != 1:
            return False
        self.driver.replace(df_new)
        return True

    def edit(self, timestamp:int, patch:dict):
        df = self.df
        needle = df['timestamp'] == timestamp
        rows = df[needle]
        if len(rows) == 0:
            return True
        if len(rows) > 1:
            raise RuntimeError('Items duplicated:', rows)

        for k in patch.keys():
            if not k in df.columns:
                raise RuntimeError('Invalid key:', k)
        df.loc[needle, list(patch.keys())] = list(patch.values())
        self.driver.replace(df)
        return True

    def update(self, r:BTResult):
        return self.edit(r.timestamp, r.dict())

    def all(self):
        return self.driver.all()


class BTPredictService:
    def __init__(self, bt_result_service:BTResultService=Depends(BTResultService)):
        self.bt_result_service = bt_result_service
        pass

    async def predict_image(self, checkpoint_path, image_path, with_cam) -> np.ndarray:
        predictor = get_predictor(checkpoint_path)
        return await predictor(image_path, with_cam)

    async def predict(self, task:BTTask):
        if task.status != STATUS_PENDING:
            print(f'Task {task.timestamp} is not pending', task.status)
            print(task)
            return

        task.status = STATUS_PROCESSING
        poll()

        memo = ''

        print('Processing task:', task)

        ok = False
        try:
            pred, feature, cam_mask = await self.predict_image(
                f'data/weights/bt/{task.model}/checkpoint.pt',
                f'data/results/bt/{task.timestamp}/original.jpg',
                with_cam=task.with_cam,
            )
            if cam_mask is not None:
                cam_normal = Image.fromarray(np.uint8(cam_mask*255))
                cam_normal.save(f'data/results/bt/{task.timestamp}/cam.png')

                cam_jet = Image.fromarray(np.uint8(colormap.jet(cam_mask)*255))
                cam_jet.save(f'data/results/bt/{task.timestamp}/cam_jet.png')

                cam_inferno = Image.fromarray(np.uint8(colormap.inferno(cam_mask)*255))
                cam_inferno.save(f'data/results/bt/{task.timestamp}/cam_inferno.png')
            else:
                if task.with_cam:
                    memo = 'Too large to generate CAM.'
            ok = True
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            task.status = STATUS_TOO_LARGE
        except Exception as e:
            print(e)
            poll()
            task.status = STATUS_ERROR
            raise e

        if ok:
            result = BTResult(
                timestamp=task.timestamp,
                name=task.name,
                with_cam=cam_mask is not None,
                model=task.model,
                memo=memo,
                pred='LMGB'[np.argmax(pred)],
                L=pred[0],
                M=pred[1],
                G=pred[2],
                B=pred[3],
            )
            print(f'Predit success: cam:{result.with_cam}')
            self.bt_result_service.add(result)
            torch.save(feature, f'data/results/bt/{task.timestamp}/feature.pt')
            task.status = STATUS_DONE

        await asyncio.sleep(1)
        poll()
        print('PRED DONE', task.timestamp)
