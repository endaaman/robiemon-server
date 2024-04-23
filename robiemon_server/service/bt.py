import numpy as np
from PIL import Image
import gc
import sys
import re
import asyncio
from functools import lru_cache
from typing import NamedTuple

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
import pytorch_grad_cam as CAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from ..models import BTResult


Image.MAX_IMAGE_PIXELS = None


using_gpu = torch.cuda.is_available()
print('GPU mode' if using_gpu else 'CPU mode')
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



class BTPredictor(ClsPredictor):
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
        print('start pred', image_path)

        model = self.get_model()

        image = Image.open(image_path).convert('RGB')
        t = self.transform(image)[None, ...].to(device)
        try:
            with torch.no_grad():
                # TODO: imple with_feautres
                oo = model(t, activate=True)
            features = None
            # features = features.detach().cpu().numpy()[0]
            o = oo.detach().cpu().numpy()[0]
            del oo
        except torch.cuda.OutOfMemoryError as e:
            del t
            del model
            torch._C._cuda_clearCublasWorkspaces()
            gc.collect()
            torch.cuda.empty_cache()
            raise e

        cam_image = None
        if with_cam:
            try:
                print(image.width, image.height)
                gradcam = CAM.GradCAMPlusPlus(
                # gradcam = CAM.GradCAM(
                    model=model,
                    target_layers=get_cam_layers(model.base, model.name),
                    reshape_transform=get_reshaper(model.name, image.width, image.height)
                )
                pred_id = np.argmax(o)
                targets = [ClassifierOutputTarget(pred_id)]
                mask = gradcam(input_tensor=t, targets=targets)[0]
                cam_image = Image.fromarray((mask * 255).astype(np.uint8))
                del gradcam
            except torch.cuda.OutOfMemoryError as e:
                del gradcam

        del t
        del model
        gc.collect()
        if using_gpu:
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()

        print('done pred', image_path)
        return o, features, cam_image



@lru_cache
def get_predictor(checkpoint_path):
    return BTPredictor(checkpoint_path)


class DFDriver:
    def get_df(self, name):
        return get_df(name)

    def set_df(self, name, df):
        set_df(name, df)

    def add(self, name, data):
        df = self.get_df(name)
        new_df = pf.concat([df, pd.DataFrame([data])], ignore_index=True)
        self.set_df(name, new_df)


class BaseDFDriver:
    def get_name(self):
        raise NotImplementedError()

    def get_df(self):
        return get_df(self.get_name())

    def set_df(self, df):
        set_dfs(self.get_name(), df)

    def add(self, data):
        df = self.get_df()
        new_df = pf.concat([df, pd.DataFrame([data])], ignore_index=True)
        self.set_df(new_df)


class BTDFDriver(BaseDFDriver):
    def get_name(self):
        raise NotImplementedError()


class BTService:
    def __init__(self, driver=Depends(BTDFDriver)):
        self.driver = driver

    async def predict(self, checkpoint_path, image_path, with_cam) -> np.ndarray:
        predictor = get_predictor(checkpoint_path)
        result = await predictor(image_path, with_cam)
        return result

    def get_df(self):
        return self.driver.get_df()

    async def add_result(self, result:BTResult):
        self.driver.add(result.dict())


