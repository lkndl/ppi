import gc
import subprocess as sp
import torch
from pathlib import Path
from typing import Set, Iterable, Union
import logging
import sys
import json
import hashlib


def wipe_memory():
    gc.collect()
    torch.cuda.empty_cache()
    return None


def getlogger(logging_path, name=''):
    logging_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    fileHandler = logging.FileHandler(logging_path)
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(logging.INFO)
    logger.addHandler(fileHandler)

    stream_log_handler = logging.StreamHandler(sys.stdout)
    stream_log_handler.setFormatter(formatter)
    stream_log_handler.setLevel(logging.INFO)
    logger.addHandler(stream_log_handler)

    return logger


def get_hash(hash_obj):
    def json_dumps():
        return json.dumps(
            hash_obj,
            ensure_ascii=False,
            sort_keys=True,
            indent=None,
            separators=(',', ':'),
        )
    return hashlib.md5(json_dumps().encode('utf-8')).digest().hex()


def gpu_mem(device):
    result = sp.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,nounits,noheader",
            "--id={}".format(device),
        ],
        encoding="utf-8",
    )
    gpu_memory = [int(x) for x in result.strip().split(",")]
    return gpu_memory[0], gpu_memory[1]


class DoesNotContain(Exception):
    pass


def glob_type(directory: Union[str, Path],
              suffix: str, relax: bool = False,
              recursive: bool = False) -> Set[Path]:
    """
    Glob inside a directory for files matching the given suffix.
    Throws an exception if no matches are found and not told to relax.
    Tailing whitespace is removed from the passed file suffix,
    but the left side is not modified.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError
    suffix = suffix.rstrip()
    func = directory.glob if not recursive else directory.rglob
    files = set(func(f'*{suffix}'))
    if not files:
        if relax:
            return set()
        raise DoesNotContain(f'No {suffix} files found in {directory}')
    return files


def glob_types(directory: Union[str, Path],
               suffixes: Iterable,
               recursive: bool = False) -> Set[Path]:
    files = set()
    for suffix in suffixes:
        try:
            files |= glob_type(directory, suffix, recursive=recursive)
        except DoesNotContain:
            pass
    if not files:
        raise DoesNotContain(
            f'No {"|".join(suffixes)} files found in {directory}')
    return files


def get_parent(path_set: Set[Path]) -> Path:
    """For a set ot filepaths from the same directory, get the directory path."""
    path = path_set.pop()
    path_set.add(path)
    assert all(p.parent == path.parent for p in path_set)
    return path.parent
