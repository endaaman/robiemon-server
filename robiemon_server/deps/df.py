from ..lib.df import get_df, set_df


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

