
from functools import lru_cache
from multiprocessing import Manager, Process
from typing import TypeVar, Sequence

from pydantic import BaseModel
import pandas as pd
from fastapi import Depends


EXCEL_PATH = 'data/db.xlsx'

class DFDriver:
    def __init__(self):
        manager = Manager()
        self.shared_df = manager.dict()

    def load_dfs(self,):
        names = ['result']
        for name in names:
            df = pd.read_excel(EXCEL_PATH, sheet_name=name, index_col=0)
            self.shared_df[name] = df

    def prepare(self):
        process = Process(target=load_data)
        process.start()
        process.join()

    def get_df(self, name):
        return self.shared_df[name]

    def inplace_df(self, name, df):
        self.shared_df[name] = df

    def commit(self, name):
        df = self.get_df(name)
        df.to_excel(EXCEL_PATH, sheet_name=name)


class BaseService:
    def __init__(self, name, driver:DFDriver=Depends()):
        pass

    def save(self, result: Result):
        pass

    def all(self, result: Result):
        pass


class ResultService:
    def __init__(self, driver:DFDriver=Depends()):
        self.name = name
        self.driver = driver

    @property
    def df(self):
        self.df = self.driver.get_df(self.name)

    @df.setter
    def df_setter(self, df):
        self.driver.inplace_df(self.name, df)



