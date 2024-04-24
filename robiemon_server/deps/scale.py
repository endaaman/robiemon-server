from fastapi import Depends
import pandas as pd

from .df import BaseDFDriver
from ..schemas import Scale


class ScaleDriver(BaseDFDriver):
    def get_name(self):
        return 'scales'

    def get_cls(self):
        return Scale


class ScaleService:
    def __init__(self, driver:ScaleDriver=Depends(ScaleDriver)):
        self.driver = driver

    def add(self, model:Scale):
        self.driver.add(model)

    def remove(self, i:int):
        df = self.driver.get()
        df_new = pd.concat([df.iloc[:i], df.iloc[i+1:]])
        if len(df) - len(df_new) != 1:
            return False
        self.driver.replace(df_new)
        return True

    def edit(self, i:int, patch:dict):
        df = self.driver.get()
        needle = df['timestamp'] == timestamp
        if not 0 < i < len(df):
            return False

        for k in patch.keys():
            if not k in df.columns:
                raise RuntimeError('Invalid key:', k)

        df.iloc[i, list(patch.keys())] = list(patch.values())
        return True

    def all(self):
        return self.driver.all()
