import os
import shutil
import pandas as pd
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

from ..schemas import Scale, BTResult, BTModel
from ..lib import debounce
from ..lib.worker import poll


EXCEL_PATH = 'data/db.xlsx'
EXCEL_TMP_PATH = 'data/db_tmp.xlsx'


class WatchedFileHandler(FileSystemEventHandler):
    def __init__(self, filename):
        self.filename = filename
        self.stunned_until = -1

    def stun(self, duration=5):
        self.stunned_until = time.time() + duration

    def is_stunned(self):
        return time.time() < self.stunned_until

    def on_modified(self, event):
        if not event.is_directory and event.src_path == self.filename:
            self.reload()

    @debounce(1)
    def reload(self):
        if self.is_stunned():
            print(f'Skip reloading(Watchdog is stunned).')
            return
        print('Watchdog: reloading excel')
        reload_dfs()
        poll()

main_loop = None
global_lock = threading.Lock()
global_dfs = {}
global_observer = PollingObserver()
global_handler = WatchedFileHandler(EXCEL_PATH)


def start_watching_dfs():
    global_observer.schedule(global_handler, path=EXCEL_PATH, recursive=False)
    global_observer.start()

def stop_watching_dfs():
    if global_lock.acquire(blocking=False):
        global_lock.release()
    global_observer.stop()

def calc_len(x):
    if isinstance(x, (float, np.floating)):
        # 少数は固定幅
        return 8
    # 最大も制限
    return min(len(str(x)), 20)


default_scales = [
    {
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
    }
]

default_bt_models = [
    {
        'name': 'convnextv2_nano_v4',
        'label': 'ConvNeXt V2 Nano',
        'enabled': True,
    }, {
        'name': 'resnetrs50_v4',
        'label': 'ResNet RS50',
        'enabled': True,
    },
]

schemas = {
    'scales': (Scale, default_scales),
    'bt_results': (BTResult, []),
    'bt_models': (BTModel, default_bt_models),
}


def save_dfs(names=None):
    if names is None:
        names = list(global_dfs.keys())
    global_handler.stun()
    print('Lock acquired')
    global_lock.acquire(timeout=5)
    try:
        with pd.ExcelWriter(EXCEL_TMP_PATH, engine='xlsxwriter') as writer:
            workbook = writer.book
            for name in names:
                df = global_dfs[name]
                df.to_excel(writer, sheet_name=name, index=False)
                worksheet = writer.sheets[name]
                num_format = workbook.add_format({'num_format': '0.00'})
                for col_idx, col_name in enumerate(df.columns):
                    col_width = max(0, *df[col_name].apply(lambda x: calc_len(x)), len(col_name)) + 1
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
        os.remove(EXCEL_PATH)
        shutil.copy2(EXCEL_TMP_PATH, EXCEL_PATH)
        print('Saved', EXCEL_PATH)
    except ValueError as e:
        raise e
    finally:
        os.remove(EXCEL_TMP_PATH)
        print('Lock released')
        global_lock.release()


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
            print(k, df)
            global_dfs[k] = df
        save_dfs()
        print(f'Created empty table: {EXCEL_PATH}')
    else:
        table_created = False
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
            except ValueError as e:
                print('ERROR:', e)
                df = empry_df_by_schema(S, values)
                table_created = True
            if loaded:
                # fillna for string
                for k, prop in S.schema()['properties'].items():
                    if prop['type'] == 'string':
                        df[k] = df[k].fillna('')
            global_dfs[name] = df
            print('Created' if table_created else 'Loaded', 'table', name)
            print(df)
        print(f'Loaded database: {EXCEL_PATH}')
        if table_created:
            print('save tables for inital')
            save_dfs()



def get_df(name):
    return global_dfs[name]

def set_df(name, df):
    global_dfs[name] = df
    save_dfs([name])
