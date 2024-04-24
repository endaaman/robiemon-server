import time
import threading
import hashlib


def asdicts(items):
    return [i.dict() for i in items]

def debounce(wait):
    def decorator(fn):
        timer = None
        def debounced(*args, **kwargs):
            nonlocal timer
            def call_it():
                fn(*args, **kwargs)
            if timer is not None:
                timer.cancel()
            timer = threading.Timer(wait, call_it)
            timer.start()
        return debounced
    return decorator

def get_hash(t):
    return hashlib.sha256(str(t).encode('utf-8')).hexdigest()
