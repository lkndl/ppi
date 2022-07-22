import gc
import hashlib
import json
import logging
import subprocess as sp
import sys
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import torch
from sklearn import metrics as skl
from torch import nn as nn
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def checkpoint(model: nn.Module, optim: Adam,
               path: Union[str, Path], **kwargs) -> None:
    model.cpu()
    chk = dict(model_state_dict=model.state_dict(),
               optim_state_dict=optim.state_dict())
    chk.update(dict(torch_rng_state=torch.get_rng_state(),
                    numpy_rng_state=np.random.get_state()))
    chk.update(kwargs)
    torch.save(chk, Path(path).with_suffix('.tar'))
    model.to(device)


def publish(checkpoint_file: Union[str, Path],
            cls: nn.Module, path: Union[str, Path]) -> None:
    model = cls()
    model.load_state_dict(torch.load(checkpoint_file)['model_state_dict'])
    torch.save(model, Path(path).with_suffix('.pth'))


def checkpoint_model(model_save_path, model_name, epoch, num_max_epoch,
                     model, optimizer, loss, eval_number=None, save_epoch=False) -> Path:
    digits = int(np.floor(np.log10(num_max_epoch))) + 1
    save_path = model_save_path / 'checkpoint'
    save_path.mkdir(parents=True, exist_ok=True)
    path = save_model(save_path, model_name,
                      f'epoch{str(epoch + 1).zfill(digits)}'
                      f'{"_checkpoint" if not save_epoch else ""}',
                      model, optimizer, loss, epoch, eval_number)
    return path


def save_model(model_save_path, model_name, model_text,
               model, optimizer, loss, epoch, eval_number: int = 0) -> Path:
    model.cpu()
    d = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'epoch': epoch,
        'eval_number': eval_number

    }
    path = Path(model_save_path) / f'{model_name}_{model_text}.tar'
    torch.save(d, path)  # TODO
    model.to(device)
    wipe_memory()
    return path


def save_final(model_save_path, model_name):
    assert Path(model_save_path).is_dir()
    best_checkpoint = torch.load(f'{model_save_path}/{model_name}_best.pth')
    model_to_save = best_checkpoint['model']
    torch.save(model_to_save, f'{model_save_path}/{model_name}_final.pth')  # TODO
    # einfacher zum finalen publishen mit model_to_save.state_dict()


def log_stats(eval_loss, labels, eval_counter,
              bin_predictions, predictions,
              tb_logger, logger,
              sfx=' Epoch Val', lower=False, desc=''):
    metrics = list()
    metrics.append(('loss', eval_loss))
    metrics.append(('ACC', skl.accuracy_score(labels, bin_predictions)))
    metrics.append(('Pr', skl.precision_score(labels, bin_predictions, zero_division=0)))
    metrics.append(('Re', skl.recall_score(labels, bin_predictions, zero_division=0)))
    metrics.append(('F1', skl.f1_score(labels, bin_predictions, zero_division=0)))
    metrics.append(('AUPR', skl.average_precision_score(labels, predictions)))
    metrics.append(('MCC', skl.matthews_corrcoef(labels, bin_predictions)))

    [tb_logger.add_scalar(f'{t[0].lower() if lower else t[0]}{sfx}', t[1],
                          eval_counter) for t in metrics]
    logger.info(desc + '\t'.join([f'{t[1]:.4f}' for t in metrics]))


def getlogger(logging_path, name=''):
    logging_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.DEBUG)

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
