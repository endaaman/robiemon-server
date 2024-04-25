import os
import pandas as pd
import time
import asyncio
import threading

import numpy as np
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

from ..schemas import BTResult, Scale
from ..lib import debounce
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
main_loop = None
saving = False
global_dfs = {}
EXCEL_PATH = 'data/db.xlsx'
schemas = {
    'bt_results': (BTResult, []),
    'scales': (Scale, default_scales),
}

async def release_lock():
    global saving
    # watchdogの変更検知に捕まらないように3秒待ってからlockを開放する
    await asyncio.sleep(3)
    saving = False
    print('Lock released')
    dfs_lock.release()

def calc_len(x):
    if isinstance(x, (float, np.floating)):
        # 少数は固定幅
        return 8
    # 最大も制限
    return min(len(str(x)), 20)

def save_dfs(names=None):
    global saving
    print('Lock acquired')
    dfs_lock.acquire()
    saving = True
    if names:
        names = global_dfs.keys()
    try:
        with pd.ExcelWriter(EXCEL_PATH, engine='xlsxwriter') as writer:
            workbook = writer.book
            for name in names:
                df = global_dfs[name]
                df.to_excel(writer, sheet_name=name, index=False)
                worksheet = writer.sheets[name]
                num_format = workbook.add_format({'num_format': '0.00'})
                for col_idx, col_name in enumerate(df.columns):
                    col_width = max(*df[col_name].apply(lambda x: calc_len(x)), len(col_name)) + 1
                    if df[col_name].dtype in [np.float64]:
                        worksheet.set_column(
                            first_col=col_idx,
                            last_col=col_idx,
                            width=col_width,
                            cell_format=num_format,
                            options={'first_row': 2}
                        )
                    else:
                        worksheet.set_column(
                            first_col=col_idx,
                            last_col=col_idx,
                            width=col_width,
                        )
        print('Save', EXCEL_PATH)
        # 他スレッドで呼ばれた場合も安全に待つ
    except ValueError as e:
        raise e
    finally:
        main_loop.create_task(release_lock())
    # asyncio.create_task(release_lock())


def empry_df_by_schema(S, values):
    return pd.DataFrame(
        columns=list(S.schema()['properties'].keys()),
        data=[S(**v).dict() for v in values],
    )

def init_dfs():
    global main_loop
    main_loop = asyncio.get_running_loop()
    reload_dfs()

primitive2type = {
    'string': str,
    'number': float,
    'integer': int,
    'boolean': bool,
}

def reload_dfs():
    if not os.path.exists(EXCEL_PATH):
        for k, (S, values) in schemas.items():
            df = empry_df_by_schema(S, values)
            global_dfs[k] = df
        save_dfs()
        print(f'Created empty table: {EXCEL_PATH}')
    else:
        for name, (S, values) in schemas.items():
            dtype = {}
            loaded = False
            for k, prop in S.schema()['properties'].items():
                t = primitive2type.get(prop['type'])
                if t:
                    dtype[k] = t
            try:
                df = pd.read_excel(EXCEL_PATH, sheet_name=name, dtype=dtype, index_col=None)
                loaded = True
                print('Loaded table:', name)
            except ValueError as e:
                print('ERROR:', e)
                df = empry_df_by_schema(S, values)
                print('Create empty table:', name)
            if loaded:
                # fillna for string
                for k, prop in S.schema()['properties'].items():
                    if prop['type'] == 'string':
                        df[k] = df[k].fillna('')
            global_dfs[name] = df
            print(df)
            print()
        print(f'Loaded database: {EXCEL_PATH}')


class WatchedFileHandler(FileSystemEventHandler):
    def __init__(self, filename):
        self.filename = filename

    def on_modified(self, event):
        if not event.is_directory and event.src_path == self.filename:
            print(event.src_path, self.filename)
            self.reload()

    @debounce(1)
    def reload(self):
        if dfs_lock.acquire(blocking=False):
            print('Watchdog: reloading excel')
            reload_dfs()
            poll()
            dfs_lock.release()
        else:
            print(f'Skip reloading({EXCEL_PATH} is locked).')


# global_observer = Observer()
global_observer = PollingObserver()

def start_watching_dfs():
    handler = WatchedFileHandler(EXCEL_PATH)
    global_observer.schedule(handler, path=EXCEL_PATH, recursive=False)
    global_observer.start()

def stop_watching_dfs():
    if dfs_lock.acquire(blocking=False):
        dfs_lock.release()
    global_observer.stop()


def get_df(name):
    return global_dfs[name]

def set_df(name, df):
    global_dfs[name] = df
    save_dfs([name])
