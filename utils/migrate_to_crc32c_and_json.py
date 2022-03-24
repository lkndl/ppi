#!/usr/bin/env python3

"""
Migrate embeddings and the `TSV` files in the `ref_tsv`
folder based on the ID mappings from `PKL` files.

The args on the server will be:

python /mnt/project/ppi-t5/scripts/t5/utils/migrate_to_crc32c_and_json.py \
migrate \
--fasta_dir /mnt/project/ppi-t5/scripts/data/seqs \
--h5_dir /mnt/project/ppi-t5/embeddings/t5_dscript_data \
--pkl_dir /mnt/project/ppi-t5/embeddings/t5_dscript_data \
--tsv_dir /mnt/project/ppi-t5/scripts/data/pairs/ids_tsv/
"""

import sys
from pathlib import Path

if (ppi_path := str(Path(__file__).parents[2])) not in sys.path:
    sys.path.append(ppi_path)

import argparse
import shutil

import h5py
import json
import numpy as np
import pandas as pd
import tqdm

from general_utils import DoesNotContain, glob_type, glob_types, get_parent
from sequence_utils import parse_fasta, load_identifiers, load_identifiers_json


def migrate_main(args) -> None:
    if not args.pkl_dir:
        raise DoesNotContain('No .pkl files found')

    old_identifiers = {path.stem.replace('_ref', ''):
                           load_identifiers(path) for path in args.pkl_dir}

    # get the directory where the ID mappings are stored
    pkl_dir = get_parent(args.pkl_dir)

    # generate the new identifiers
    for path in args.fasta_dir:
        parse_fasta(path, pkl_dir / path.name)

    new_identifiers = {path.stem: load_identifiers_json(
        pkl_dir / path.name) for path in args.fasta_dir}

    if missing := old_identifiers.keys() ^ new_identifiers.keys():
        print(f'Missing a file for: {", ".join(missing)}!')

    lookup = dict()
    for species in old_identifiers.keys() & new_identifiers.keys():
        assert not old_identifiers[species].keys() ^ new_identifiers[species].keys()
        new_ids, old_ids = new_identifiers[species], old_identifiers[species]
        lookup[species] = {int_id: new_ids[old_id] for old_id, int_id in old_ids.items()}

    # save the lookup as well
    with open(pkl_dir / 'lookup.json', 'w') as json_file:
        json.dump(lookup, json_file)

    # translate H5s
    print('Translating H5s ...')
    h5s = {path.stem: path for path in args.h5_dir}
    if missing := h5s.keys() ^ lookup.keys():
        print(f'Missing an H5 file for: {", ".join(missing)}!')

    skip = False
    for species, sp_lookup in lookup.items():
        if species not in h5s:
            continue
        h5_file = h5s[species]

        # rename the existing H5 file
        h5_bak = h5_file.with_suffix(h5_file.suffix + '.bak')
        if not h5_bak.is_file():
            shutil.move(h5_file, h5_bak)
        else:
            # File already exists
            if skip:
                continue
            else:
                print('Looks like you\'ve run this script before. '
                      'Skip [y/yes] or overwrite [n/no] '
                      f'{h5_file} with content from {h5_bak}?')
                response = ''
                while response not in {'y', 'yes', 'n', 'no'}:
                    response = input().strip().lower()
                if response.startswith('y'):
                    skip = True
                    continue

        # then translate it to a new one
        with h5py.File(h5_file, 'w') as new_h5, h5py.File(h5_bak, 'r') as old_h5:
            for old_id, mbed in tqdm.tqdm(old_h5.items(), desc=f'iterate {species}'):
                new_h5[sp_lookup[old_id]] = np.array(mbed)

    # translate TSVs
    print('Translating TSVs ...')
    if missing := {path.stem for path in args.tsv_dir} \
                  ^ {'ecoli_test', 'human_test', 'mouse_test',
                     'yeast_test', 'fly_test', 'human_train', 'worm_test'}:
        print(f'Missing a TSV file for: {", ".join(missing)}!')

    for tsv in args.tsv_dir:

        df = pd.read_csv(tsv, sep='\t', header=None)
        sp_lookup = new_identifiers[tsv.stem.split('_')[0]]

        # map to hash IDs
        for i in [0, 1]:
            df[i] = df[i].apply(sp_lookup.get)

        # make the new directory
        new_tsv_dir = tsv.parent.with_name('ref_tsv')
        new_tsv_dir.mkdir(exist_ok=True)

        # save
        df.to_csv(new_tsv_dir / tsv.name, sep='\t',
                  header=None, index=False)


def clean_main(args) -> None:
    # Delete the .pkl files
    [pkl.unlink() for pkl in args.pkl_dir]

    # Delete the old .h5 files
    h5_baks = glob_types(str(get_parent(args.h5_dir)), ('.hdf5.bak', '.h5.bak'))

    strip_bak_extension = lambda f: f.replace('.h5.bak', '').replace('.hdf5.bak', '')

    if missing := {strip_bak_extension(path.name) for path in h5_baks} \
                  - {path.stem for path in args.h5_dir}:
        print(f'Missing a new H5 file for the following .bak files: '
              f'{", ".join(missing)}\nDelete anyway? [y/n]')
        response = ''
        while response not in {'y', 'yes', 'n', 'no'}:
            response = input().strip().lower()
        if response.startswith('y'):
            [path.unlink() for path in h5_baks if
             strip_bak_extension(path.name) in missing]

    if present := {strip_bak_extension(path.name) for path in h5_baks} \
                  & {path.stem for path in args.h5_dir}:
        print(f'Delete .bak files for {", ".join(present)}? [y/n]')
        response = ''
        while response not in {'y', 'yes', 'n', 'no'}:
            response = input().strip().lower()
        if response.startswith('y'):
            [path.unlink() for path in h5_baks if
             strip_bak_extension(path.name) in present]


def add_migrate_args():
    """Generate new CRC32C IDs and JSON mappings, migrate the embeddings and the TSVs."""
    pass


def add_clean_args():
    """Remove the old pickle ID mappings and the backup H5 files."""
    pass


def add_args(parser):
    """Actually both subparsers need and get these arguments"""
    parser.add_argument('--fasta_dir', required=True,
                        type=lambda arg: glob_type(arg, '.fasta'),
                        help='Folder containing FASTA files that were embedded')
    parser.add_argument('--h5_dir', required=True,
                        type=lambda arg: glob_types(arg, ('.hdf5', '.h5')),
                        help='Folder containing H5 embedding files', )
    parser.add_argument('--pkl_dir', required=True,
                        type=lambda arg: glob_type(arg, '.pkl', True),
                        help='Folder containing pickled ID mappings')
    parser.add_argument('--tsv_dir', required=True,
                        type=lambda arg: glob_type(arg, '.tsv'),
                        help='Folder containing original, unmapped interaction TSVs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    subparsers = parser.add_subparsers(title='migrate commands', dest='cmd')
    subparsers.required = True

    sp = subparsers.add_parser('migrate', description=add_migrate_args.__doc__)
    add_args(sp)
    sp.set_defaults(func=migrate_main)

    sp = subparsers.add_parser('clean', description=add_clean_args.__doc__)
    add_args(sp)
    sp.set_defaults(func=clean_main)

    args = parser.parse_args()
    args.func(args)
