import hashlib
import json
import subprocess as cmd
from pathlib import Path
from typing import Set, Iterable, Union, Dict, IO

from Bio.Seq import Seq
from Bio.SeqUtils.CheckSum import crc64


def cmd_run(args, verbose=False):
    stdout = cmd.DEVNULL
    if verbose:
        stdout = None
        print(f'\033[94m' + ' '.join(args) + '\033[0m')
    cmd.run(args, stdout=stdout, stderr=cmd.STDOUT)


def values_sorted_by_key(d: dict):
    return (d[k] for k in sorted(d.keys()))


def file_hash(file_path):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)

    with open(file_path, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


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


def run_uniqueprot(input_file: Union[str, Path],
                   output_file: Union[str, Path],
                   hval_config: Dict,
                   pretend: bool = False):
    hval_path = Path('hval_config.json')
    with hval_path.open('w') as json_file:
        json_file.write(json.dumps(hval_config, indent=4))

    args = ['rostclust', 'uniqueprot',
            '--work-dir', 'ppi_rostclust',
            '--hval-config-path', str(hval_path),
            str(input_file), str(output_file)]
    if pretend:
        print(' '.join(args))
    else:
        cmd_run(args)


def run_uniqueprot2D(input_file: Union[str, Path],
                     database_file: Union[str, Path],
                     output_file: Union[str, Path],
                     hval_config: Dict, pretend: bool = False):
    hval_path = Path('hval_config.json')
    with hval_path.open('w') as json_file:
        json.dump(hval_config, json_file)

    args = ['rostclust', 'uniqueprot2d',
            '--work-dir', 'ppi_rostclust',
            '--hval-config-path', str(hval_path),
            str(input_file), str(database_file), str(output_file)]
    if pretend:
        print(' '.join(args))
    else:
        cmd_run(args)


def get_seq_hash(seq: Union[str, Seq]) -> str:
    return crc64(seq)


def to_fasta(_id: Union[str, int], seq: str, file_handle: IO) -> None:
    _ = file_handle.write(''.join(to_lines(_id, seq)))


def to_lines(_id: str, seq: Union[str, Seq], lw: int = 60) -> str:
    yield f'>{_id}\n'
    i = 0
    seq = str(seq)
    while i < len(seq):
        yield seq[i:i + lw] + '\n'
        i += lw
