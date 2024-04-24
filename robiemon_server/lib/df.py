import os
import pandas as pd
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ..schemas import BTResult, Scale
from ..lib.worker import poll

default_scales = [{
    'label': 'VS x20 440nm/px',
    'scale': 1.0,
    'enabled': True,
}, {
    'label': 'JSC-HR5U x20',
    'scale': 0.941,
    'enabled': True,
}, {
    'label': 'JSC-HR5U x10',
    'scale': 1.8813,
    'enabled': True,
}, {
    'label': 'HY-2307 x40',
    'scale': 1.093,
    'enabled': True,
}, {
    'label': 'HY-2307 x20',
    'scale': 2.185,
    'enabled': True,
}, {
    'label': 'HY-2307 x10',
    'scale': 4.371,
    'enabled': True,
}]


dfs_lock = threading.Lock()
global_dfs = {}
EXCEL_PATH = 'data/db.xlsx'
schemas = {
    'bt_results': (BTResult, []),
    'scales': (Scale, default_scales),
}

def save_dfs():
    with dfs_lock:
        with pd.ExcelWriter(EXCEL_PATH, engine='xlsxwriter') as writer:
            for k, df in global_dfs.items():
                df.to_excel(writer, sheet_name=k, index=False)
        print('Save', EXCEL_PATH)

def empry_df_by_schema(S, values):
    return pd.DataFrame(
        columns=list(S.schema()['properties'].keys()),
        data=[S(**v).dict() for v in values],
    )

def reload_dfs():
    if not os.path.exists(EXCEL_PATH):
        for k, (S, values) in schemas.items():
            df = empry_df_by_schema(S, values)
            global_dfs[k] = df
        save_dfs()
        print(f'Created empty database: {EXCEL_PATH}')
    else:
        for k, (S, values) in schemas.items():
            try:
                df = pd.read_excel(EXCEL_PATH, sheet_name=k, index_col=None)
            except ValueError as e:
                df = empry_df_by_schema(S, values)
            global_dfs[k] = df
            print('Loaded', k, df)
            print()
        print(f'Loaded database: {EXCEL_PATH}')


class WatchedFileHandler(FileSystemEventHandler):
    def __init__(self, filename):
        self.filename = filename

    def on_modified(self, event):
        if not event.is_directory and event.src_path == self.filename:
            if dfs_lock.acquire(blocking=False):
                print('not locked -> reload')
                reload_dfs()
                poll()
                dfs_lock.release()
            else:
                print('locked -> skip')


global_observer = Observer()

def start_watching_dfs():
    handler = WatchedFileHandler(EXCEL_PATH)
    global_observer.schedule(handler, path=EXCEL_PATH, recursive=False)
    global_observer.start()

def stop_watching_dfs():
    global_observer.stop()


def get_dfs():
    return global_dfs

def get_df(name):
    return global_dfs[name]

def set_df(name, df):
    global_dfs[name] = df
    save_dfs()

def add_data(name, data):
    df = get_df(name)
    if len(df) == 0:
        df_new = pd.DataFrame([data])
    else:
        df_new = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    set_df(name, df_new)
