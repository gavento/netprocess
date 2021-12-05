import encodings
import gzip
import lzma
import pickle
from pathlib import Path
from typing import Any

import pyzstd

COMPRESS_SUFFIXES = [".zstd", ".xz", ".gz"]


def file_basic_path(path: Path, suffix: str) -> Path:
    """
    Return the path stripped of a known compress suffix and then of `suffix`.

    Raises ValueError if path does not end in `suffix` after stripping compression
    suffix (if present).
    """
    if path.suffix in COMPRESS_SUFFIXES:
        path = path.with_suffix()
    if path.suffix != suffix:
        raise ValueError(
            f"{path!r} does not have suffix {suffix!r} (unknown compression?)"
        )
    return path.with_suffix("")


def open_file(path: Path, mode="r", level=None) -> Any:
    """
    Open the given path, transparently de/compressing according to last extension.

    Currently supported: .zstd, .xz, .gz
    """
    if "+" in mode:
        raise Exception("open_file does not support mixed reads and writes")
    elif "b" in mode:
        wrap = lambda x: x
    elif mode[0] == "r":
        wrap = encodings.utf_8.StreamReader
        mode = mode[0] + "b"
    elif mode[0] in "wax":
        wrap = encodings.utf_8.StreamWriter
        mode = mode[0] + "b"
    else:
        raise Exception(f"Unsupported mode: {mode!r}")

    if path.suffix == ".zstd":
        return wrap(pyzstd.ZstdFile(path, mode=mode, level_or_option=level))
    elif path.suffix == ".xz":
        return wrap(lzma.LZMAFile(path, mode=mode, preset=level))
    elif path.suffix == ".gz":
        if level is None:
            level = 9
        return wrap(gzip.GzipFile(path, mode=mode, compresslevel=level))
    else:
        return wrap(open(path, mode=mode))


def load_pickle(path):
    """Load pickled object from path, decompressing by extension.

    See `open_file` for details."""

    with open_file(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path, level=None):
    """Save pickled object to a path, compressing by extension.

    See `open_file` for details."""
    with open_file(path, "wb", level=level) as f:
        pickle.dump(obj, f, protocol=4)
