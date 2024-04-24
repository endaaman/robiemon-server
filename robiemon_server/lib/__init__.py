import hashlib


def asdicts(items):
    return [i.dict() for i in items]

def get_hash(t):
    return hashlib.sha256(str(t).encode('utf-8')).hexdigest()
