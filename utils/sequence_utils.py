import json
import pickle
from pathlib import Path
from typing import Union

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils.CheckSum import crc64


def get_seq_hash(seq: Union[str, Seq]) -> str:
    return crc64(seq)


def to_lines(_id: str, seq: Union[str, Seq], lw: int = 60) -> str:
    yield f'>{_id}\n'
    i = 0
    seq = str(seq)
    while i < len(seq):
        yield seq[i:i + lw] + '\n'
        i += lw


def parse_fasta(fasta_path: Union[str, Path], id_ref_path: Union[str, Path]) -> dict:
    print('Loading Sequences from: {}'.format(fasta_path))
    assert Path(fasta_path).is_file()

    seq_dict, identifiers = dict(), dict()

    with open(fasta_path, 'r') as fasta:
        for record in SeqIO.parse(fasta, 'fasta'):
            seq = record.seq.upper()
            seq_hash = get_seq_hash(seq)
            identifiers[record.id] = seq_hash
            if seq_hash not in seq_dict:
                seq_dict[seq_hash] = seq
            else:
                assert seq_dict[seq_hash] == seq, \
                    f'{fasta_path} contains a contradictory duplicate:\n' \
                    f'{record.format("fasta")}'

    save_identifiers_json(id_ref_path, identifiers)
    return seq_dict


def save_identifiers(path: Union[str, Path], identifiers: dict) -> None:
    print('Saving identifier references')
    with open(Path(path).with_suffix('.pkl'), 'wb') as f:
        pickle.dump(identifiers, f, pickle.HIGHEST_PROTOCOL)


def save_identifiers_json(path: Union[str, Path], identifiers: dict = None) -> None:
    p = Path(path)

    if identifiers is None:
        # translate an existing mapping from pickle to json
        assert p.is_file()
        identifiers = load_identifiers_json(p)

    # save identifiers as json
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix('.json'), 'w') as json_file:
        json.dump(identifiers, json_file)


def load_identifiers(path: Union[str, Path]) -> dict:
    with open(Path(path).with_suffix('.pkl'), 'rb') as f:
        return pickle.load(f)


def load_identifiers_json(path: Union[str, Path]) -> dict:
    with open(Path(path).with_suffix('.json'), 'r') as f:
        return json.load(f)


def check_ref_correspondence(raw_seqs, seq_dict, path):
    data = load_identifiers_json(path)
    for raw_id, seq in raw_seqs.items():
        ref_num = data[raw_id]
        if seq_dict[ref_num] != seq:
            print(f'Discrepancy for {raw_id}!')
    print('Reference Correspondence Check finished')
