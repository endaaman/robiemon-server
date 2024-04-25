from ..lib.df import get_df, set_df

from ..schemas import BTResult


class BaseDFDriver:
    def get_name(self):
        raise NotImplementedError()

    def get_cls(self):
        raise NotImplementedError()

    def get_df(self):
        return get_df(self.get_name())

    def all(self):
        df = self.get_df()
        cls = self.get_cls()
        return [cls(**row) for i, row in df.iterrows()]

    def replace(self, df):
        set_df(self.get_name(), df)
