import hashlib


def get_hash(t):
    return hashlib.sha256(str(timestamp).encode('utf-8')).hexdigest()
