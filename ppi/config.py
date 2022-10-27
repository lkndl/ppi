from __future__ import annotations

import hashlib
import importlib
import json
import torch
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
import typer
from dataclass_wizard import JSONWizard


def parse(ctx: typer.Context, write=True):
    # get a configuration
    path = ctx.params.get('config')
    if path is not None and (path := Path(path)).is_file():
        cfg = Config.from_file(path, **ctx.params)
    else:
        cfg = Config(**ctx.params)

    # process the additional args as options
    seen_underscore = False
    for i in range(0, len(ctx.args), 2):
        option = ctx.args[i]
        _, option = option[:2], option[2:]
        if _ != '--':
            raise IllegalOptionStart('additional options must start with "--"')
        # for consistency with the rest of the typer CLI, enforce - instead of _
        seen_underscore = seen_underscore or '_' in option
        option = option.replace('-', '_')
        if i + 1 >= len(ctx.args):
            raise MissingValueException(f'missing value for "--{option}"')
        value = ctx.args[i + 1]
        ctx.params[option] = value
    if seen_underscore:
        warnings.warn('"_" used in extra options. For consistency with typer, '
                      'please consider using "-" instead')

    # copy and convert type
    for option, value in ctx.params.items():
        for sub_config, type_config in zip(cfg, Config()):
            if hasattr(sub_config, option):
                setattr(sub_config, option, type(getattr(type_config, option))(value))

    # fix and write to file
    return cfg.process(write)


@dataclass
class SnakeConfig(JSONWizard):
    class _(JSONWizard.Meta):
        key_transform_with_dump = 'SNAKE'
        raise_on_unknown_json_key = True

    def __iter__(self):
        for k, v in vars(self).items():
            yield k, v

    def to_dict(self):
        d = dict()
        _vars = vars(self)
        _keys = sorted(_vars.keys())
        for k in _keys:
            v = _vars[k]
            if issubclass(type(v), Path):
                d[k] = str(v.resolve())
            elif issubclass(type(v), SnakeConfig):
                d[k] = v.to_dict()
            elif type(v) in {None, int, float}:
                d[k] = v
            elif issubclass(type(v), Enum):
                d[k] = v.value
            else:
                d[k] = str(v)
        return d

    def get(self, key: str, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def to_json(self, *, encoder=json.dumps, **kwargs):
        kwargs |= dict(indent=2, sort_keys=True)
        return encoder(self.to_dict(), **kwargs)

    def to_json_file(self, file, mode='w',
                     encoder=json.dump, **kwargs) -> None:
        kwargs |= dict(indent=2, sort_keys=True)
        with open(file, mode) as out_file:
            encoder(self.to_dict(), out_file, **kwargs)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<%s.%s object at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self))
        )

    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class Architecture(str, Enum):
    gelu = 'gelu'
    cnn = 'cnn'
    fft = 'fft'
    cmap = 'cmap'
    attn = 'attn'


@dataclass
class ModelParams(SnakeConfig):
    emb_projection_dim: int = 100
    dropout_p: float = .1
    map_hidden_dim: int = 50
    kernel_width: int = 7
    pool_size: int = 9
    activation: str = 'GELU'
    architecture: Architecture = Architecture.cnn

    def __post_init__(self):
        activations = importlib.import_module('torch.nn.modules.activation')
        self.activation = vars(activations)[self.activation.rstrip('()')]()
        self.architecture = Architecture(self.architecture)


@dataclass
class TrainParams(SnakeConfig):
    _: str = Path('/mnt/project/kaindl/ppi/ppi/smaller')

    train_tsv: str = _ / 'apid_train.tsv'
    val_tsv: str = _ / 'apid_validation.tsv'
    h5: str = Path('apid_huri.h5')

    lr: float = .001
    seed: int = 42  # np.random.default_rng().integers(0, 9999)
    epochs: int = 2
    batch_size: int = 11
    ppi_weight: float = 10.
    patience: int = 20
    augment: bool = True
    shuffle: bool = True

    start_epoch: float = 0.
    start_batch: int = 0

    eval_train_ratio: float = 999999.
    eval_time_interval: int = 60 * 60 * 2  # 2 hours
    eval_epoch_interval: float = .05

    def __post_init__(self):
        self.h5 = Path(self.h5)
        self.train_tsv = Path(self.train_tsv)
        self.val_tsv = Path(self.val_tsv)
        del self._


@dataclass
class TestParams(SnakeConfig):
    model: str = Path('../runs/eval/model.tar')
    test_tsv: str = Path('/mnt/project/kaindl/ppi/ppi/smaller/huri_test.tsv')
    out_path: str = Path('predictions.tsv')

    def __post_init__(self):
        self.model = Path(self.model)
        self.test_tsv = Path(self.test_tsv)
        self.out_path = Path(self.out_path)


class TrainMode(str, Enum):
    new = 'new'
    overwrite = 'overwrite'
    resume = 'resume'


@dataclass
class OutParams(SnakeConfig):
    name: str = ''
    wd: str = Path('.')
    mode: TrainMode = TrainMode.new

    def __post_init__(self):
        self.name = self.name.strip()
        self.wd = Path(self.wd)
        self.mode = TrainMode(self.mode)


class FlatConfig(SnakeConfig):
    def __init__(self, cfg: Config):
        for i in cfg:
            for k, v in vars(i).items():
                setattr(self, k, v)


@dataclass(init=False)
class Config(SnakeConfig):
    model_params: ModelParams
    train_params: TrainParams
    test_params: TestParams
    out_params = OutParams

    class _(JSONWizard.Meta):
        key_transform_with_dump = 'SNAKE'
        raise_on_unknown_json_key = True

    def __iter__(self):
        for cfg in [self.model_params, self.train_params,
                    self.test_params, self.out_params]:
            yield cfg

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs) -> Config:
        with open(path, 'r') as json_file:
            d = json.load(json_file)
        flat = dict()
        for k, v in d.items():
            if type(v) == dict:
                flat |= v
            else:
                flat[k] = v
        flat |= kwargs
        return Config(**flat)

    def __init__(self, **kwargs):
        self.model_params = ModelParams().update(kwargs)
        self.train_params = TrainParams().update(kwargs)
        self.test_params = TestParams().update(kwargs)
        self.out_params = OutParams().update(kwargs)

    def process(self, write: bool = True) -> FlatConfig:
        mode = self.out_params.mode
        name = self.out_params.name
        self.out_params.wd = Path(self.out_params.wd)
        if not name:
            # uuids are aaaaaawfully long, so I'm doing this shit:
            c2 = json.dumps({'model_params': self.model_params.to_dict(),
                             'train_params': self.train_params.to_dict()},
                            sort_keys=True, indent=2)
            name = hashlib.sha256(c2.encode()).hexdigest()
            name = 'm' + name[:3] + name[-2:]
            self.out_params.name = name
            print(f'--> {name}')

        if write:
            config = Path(self.out_params.wd) / name / f'config_{name}.json'
            config.parent.mkdir(
                parents=True, exist_ok=mode != TrainMode.new)
            if config.is_file() and mode == TrainMode.resume:
                config.rename(config.with_suffix('.json.bak'))
            self.to_json_file(str(config))

        seed = self.train_params.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        return FlatConfig(self)


class IllegalOptionStart(Exception):
    pass


class MissingValueException(Exception):
    pass
