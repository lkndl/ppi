from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path

from dataclass_wizard import JSONWizard, JSONFileWizard


@dataclass
class SnakeConfig(JSONWizard):
    class _(JSONWizard.Meta):
        key_transform_with_dump = 'SNAKE'
        raise_on_unknown_json_key = True

    def to_dict(self):
        d = dict()
        _vars = vars(self)
        _keys = sorted(_vars.keys())
        for k in _keys:
            v = _vars[k]
            if type(v) == Path:
                v = v.resolve()
            d[k] = str(v)
        return d


@dataclass
class ModelParams(SnakeConfig):
    emb_projection_dim: int = 100
    dropout_p: float = .1
    map_hidden_dim: int = 50
    kernel_width: int = 7
    pool_size: int = 9
    activation: str = 'GELU'

    def __post_init__(self):
        activations = importlib.import_module('torch.nn.modules.activation')
        self.activation = vars(activations)[self.activation.rstrip('()')]()


@dataclass
class TrainParams(SnakeConfig):
    _ = Path('/mnt/project/kaindl/ppi/ppi/smaller')

    train_tsv: str = _ / 'apid_train.tsv'
    val_tsv: str = _ / 'apid_validation.tsv'
    test_tsv: str = _ / 'huri_test.tsv'
    h5: str = Path('apid_huri.h5')

    lr: float = .005
    seed: int = 42
    epochs: int = 4
    patience: int = 200

    def __post_init__(self):
        self.h5 = Path(self.h5)


@dataclass
class TestParams(SnakeConfig):
    model: str = None
    test_tsv: str = Path('/mnt/project/kaindl/ppi/ppi/smaller/huri_test.tsv')
    out_path: str = Path('predictions.tsv')

    def __post_init__(self):
        self.test_tsv = Path(self.test_tsv)
        self.out_path = Path(self.out_path)


@dataclass
class OutParams(SnakeConfig):
    name: str = ''
    wd: str = Path('.')

    def __post_init__(self):
        self.wd = Path(self.wd)


@dataclass
class Config(SnakeConfig, JSONFileWizard):
    class _(JSONWizard.Meta):
        key_transform_with_dump = 'SNAKE'
        raise_on_unknown_json_key = True

    model: ModelParams = ModelParams()
    train: TrainParams = TrainParams()
    test: TestParams = TestParams()
    output: OutParams = OutParams()

    def __post_init__(self):
        if not self.output.name.strip():
            # uuids are aaaaaawfully long, so I'm doing this shit:
            c2 = {'model': self.model.to_dict(),
                  'train': self.train.to_dict()}
            _hash = hashlib.sha256(str(c2).encode()).hexdigest()
            _hash = 'm' + _hash[:3] + _hash[-2:]
            self.output.name = _hash
        self.output.wd = self.output.wd / self.output.name
        self.output.wd.mkdir(parents=True, exist_ok=False)
        self.to_json_file((self.output.wd /
                           f'config_{self.output.name}.json'), indent=2)


if __name__ == '__main__':
    c1 = Config()
    print(c1.output.name)

    c1.to_json_file('c.json', indent=2)
    print('hi')
    k = c1.to_dict()
    c2 = Config.from_dict(k)
    c1.to_json_file('c.json', indent=2)
    # c2 = Config.from_json_file('c.json')
    # c2
