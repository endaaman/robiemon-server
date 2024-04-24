from ..lib.df import get_df, set_df, add_data

from ..schemas import BTResult


class BaseDFDriver:
    def get_name(self):
        raise NotImplementedError()

    def get_cls(self):
        raise NotImplementedError()

    def get(self):
        return get_df(self.get_name())

    def replace(self, df):
        set_df(self.get_name(), df)

    def all(self):
        df = self.get()
        cls = self.get_cls()
        return [cls(**row) for i, row in df.iterrows()]

    def add(self, model):
        add_data(self.get_name(), model.dict())
