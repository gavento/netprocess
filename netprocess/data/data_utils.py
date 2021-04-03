import pickle

import zstd


def load_object(path):
    if path.endswith(".zstd"):
        with open(path, "rb") as f:
            return pickle.loads(zstd.decompress(f.read()))
    else:
        with open(path, "rb") as f:
            return pickle.load(f.read)


def save_object(obj, path):
    if path.endswith(".zstd"):
        with open(path, "wb") as f:
            f.write(zstd.compress(pickle.dumps(obj, protocol=4)))
    else:
        if path.endswith(".zstd"):
            pickle.dump(obj, f, protocol=4)
