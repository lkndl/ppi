import gc
import subprocess as sp

import numpy as np
import torch
from pathlib import Path
from typing import Iterable, Union
import logging
import sys
import json
import hashlib

from sklearn import metrics as skl

from training.perresidue.train_t5_cval import device


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def wipe_memory():
    gc.collect()
    torch.cuda.empty_cache()
    return None


def flush_loggers(loggers):
    for logger in loggers:
        logger.flush()


def close_loggers(loggers):
    for logger in loggers:
        logger.close()


def save_config(config: dict, model_save_path: Path):
    model_save_path = Path(model_save_path)
    assert model_save_path.is_dir()
    with (model_save_path / 'config.json').open('w') as f:
        json.dump(config, f)


def save_model(model_save_path, model_name, model_text,
               model, optimizer, loss, epoch, eval_number=None):
    model.cpu()
    d = {
        'model': model,  # kann (evtl) auch .state_dict
        'optimizer': optimizer,
        'loss': loss,
        'epoch': epoch,
        'eval_number': eval_number
        # wo der dataloader gerade ist wsl dazu schreiben = iterationsstufe
    }
    if eval_number is not None:
        d['eval_number'] = eval_number
    torch.save(d, f'{model_save_path}/{model_name}_{model_text}.pth')
    model.to(device)
    wipe_memory()


def save_final(model_save_path, model_name):
    assert Path(model_save_path).is_dir()
    best_checkpoint = torch.load(f'{model_save_path}/{model_name}_best.pth')
    model_to_save = best_checkpoint['model']
    torch.save(model_to_save, f'{model_save_path}/{model_name}_final.pth')
    # einfacher zum finalen publishen mit model_to_save.state_dict()


def log_stats(eval_loss, labels, eval_counter,
              bin_predictions, predictions,
              tb_logger, logger,
              sfx=' Epoch Val', lower=False):
    metrics = list()
    metrics.append(('loss', eval_loss))
    metrics.append(('ACC', skl.accuracy_score(labels, bin_predictions)))
    metrics.append(('Pr', skl.precision_score(labels, bin_predictions)))
    metrics.append(('Re', skl.recall_score(labels, bin_predictions)))
    metrics.append(('F1', skl.f1_score(labels, bin_predictions)))
    metrics.append(('AUPR', skl.average_precision(labels, predictions)))
    metrics.append(('MCC', skl.matthews_corrcoef(labels, bin_predictions)))

    [tb_logger.add_scalar(f'{t[0].lower() if lower else t[0]}{sfx}', t[1],
                          eval_counter) for t in metrics]
    logger.info(', '.join([f'{t[0].lower()}: {t[1]}' for t in metrics]))


def getlogger(logging_path, name=''):

    logging_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)

    fh = logging.FileHandler(logging_path)

    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s\t%(name)s\t%(message)s',
                                      datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)

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


class PatienceExceededError(Exception):
    pass


class DoesNotContain(Exception):
    pass


def glob_type(directory: Union[str, Path],
              suffix: str, relax: bool = False,
              recursive: bool = False) -> set[Path]:
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
               recursive: bool = False) -> set[Path]:
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


def get_parent(path_set: set[Path]) -> Path:
    """For a set ot filepaths from the same directory, get the directory path."""
    path = path_set.pop()
    path_set.add(path)
    assert all(p.parent == path.parent for p in path_set)
    return path.parent
