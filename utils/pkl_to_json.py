#!/usr/bin/env python3

from pathlib import Path
from ppi.t5.utils.sequence_utils import save_identifiers_json


def pkl_to_json(folder):
    for pkl in Path(folder).resolve().parent.rglob('*.pkl'):
        save_identifiers_json(path=pkl)


if __name__ == '__main__':
    pkl_to_json(__file__)
