import json
import pickle
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Union

import crc32c
import pandas as pd
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm.auto import tqdm


def uniprot_api_fetch(uniprot_ids: Union[set, list],
                      out_file: Union[str, Path] =
                      Path.cwd() / 'apid_q1',
                      patience: int = 2,
                      ) -> None:
    """
    Download sequences from UniProt: First, post the given set of
    UniProt IDs to the ID mapping site and retrieve `tab` and `fasta`.
    Then, check UniParc for up-to-date replacement entries, finally
    use obsolete versions of the given IDs.
    :param uniprot_ids:
    :param out_file:
    :param patience:
    :return:
    """
    uniprot_ids = set(uniprot_ids)
    out_file = Path(out_file).resolve()
    if out_file.is_dir():
        raise IsADirectoryError
    out_file.parent.mkdir(parents=True, exist_ok=True)

    upload_url = 'https://www.uniprot.org/uploadlists/'
    entry_url = 'https://www.uniprot.org/uniprot/{}.fasta?'  # include=yes'
    uniparc_url = 'https://www.uniprot.org/uniparc/?query={}&format=tab&columns=id,kb'
    archive_url = 'https://www.uniprot.org/uniprot/{}.fasta?version=*'

    params = {
        'from': 'ACC+ID',
        'to': 'ACC+ID',
        'format': 'tab',
        'query': ' '.join(sorted(uniprot_ids))
    }
    b = '\b'
    errors = list()
    seen_hashes = set()

    print(f'{out_file.stem}: query + tab ...', end='')
    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(upload_url, data)
    with urllib.request.urlopen(req) as f:
        with open(out_file.with_suffix('.tab'), 'w') as g:
            response = f.read()
            g.write(response.decode('utf-8'))

    print(f'{b * 16} fasta ...', end='')
    req = urllib.request.Request(f.url[:-3] + 'fasta')
    with urllib.request.urlopen(req) as f:
        with open(out_file.with_suffix('.fasta'), 'w') as g:
            response = f.read()
            g.write(response.decode('utf-8'))

    # save intermediary results in a dict, not via pd.loc
    m_dict = dict()

    print(f'{b * 10}', end='')
    with open(out_file.with_name(out_file.stem + '_hash.fasta'), 'w') as hash_fasta:
        with open(out_file.with_suffix('.fasta'), 'r') as in_fasta:
            for r in tqdm(SeqIO.parse(in_fasta, 'fasta'), desc='hash FASTA', position=0, leave=True, ascii=True):
                _id = r.id.split('|')[1]
                _hash = get_seq_hash(r.seq)
                m_dict[_id] = [_hash, 'query', r.description]

                if _hash in seen_hashes:
                    continue
                seen_hashes.add(_hash)
                for line in to_lines(_hash, r.seq):
                    _ = hash_fasta.write(line)

    from_to = pd.read_csv(out_file.with_suffix('.tab'), sep='\t', header=0)

    forgot = uniprot_ids - set(from_to.From)
    from_to = from_to.append(pd.DataFrame(
        zip(forgot, forgot), columns=['From', 'To']))
    from_to['crc_hash'] = ''

    def _refresh_and_save_from_to_():
        from_to.loc[from_to.crc_hash == '', ['crc_hash', 'source', 'description']] = \
            from_to.loc[from_to.crc_hash == '', 'To'].apply(
                lambda t: m_dict.get(t, ['', '', ''])).to_list()

        from_to.sort_values(by='crc_hash').to_csv(out_file.with_suffix('.tsv'), index=False, sep='\t')

    _refresh_and_save_from_to_()
    if patience < 1:
        return

    obs = lambda s: s.endswith('(obsolete)')
    known = lambda _id: _id in m_dict.keys()

    def _save_(_txt_, _id_, _nid_, _source_, _hashf_, _seqf_):
        _desc, *_seq_ = _txt_.strip()[1:].split('>')[0].split('\n')
        _seq_ = ''.join(_seq_)
        _hash_ = get_seq_hash(_seq_)
        m_dict[_id_] = [_hash_, _source_, _desc]
        # from_to.loc[from_to.To == _id_, 'To'] = _nid_

        if _hash_ in seen_hashes:
            return
        seen_hashes.add(_hash_)
        for _line in to_lines(_hash_, _seq_):
            _ = _hashf_.write(_line)
        for _line in to_lines(_nid_, _seq_):
            _ = _seqf_.write(_line)

    # print(f'{b * 8}')
    with open(out_file.with_name(out_file.stem + '_hash.fasta'), 'a') \
            as hash_fasta, open(out_file.with_suffix('.fasta'), 'a') as seq_fasta:
        for _id in tqdm(set(from_to.loc[from_to.crc_hash == '', 'To'])
                        - m_dict.keys(), desc='fetch UniParc', position=0, leave=True, ascii=True):
            r = requests.get(uniparc_url.format(_id))
            if not r.ok:
                errors.append(f'uniparc: {_id}')
                continue

            new_id = ''
            _, *upi_lines = r.text.strip().split('\n')
            for line in upi_lines:
                upi, alt_ids = line.split('\t')
                alt_ids = {a.strip() for a in alt_ids.split(';')}
                # prefer already seen IDs
                alt_ids = sorted([a for a in alt_ids if a and not obs(a)],
                                 key=known, reverse=True)
                for new_id in alt_ids:
                    # fetch a record
                    r = requests.get(entry_url.format(new_id))
                    if not r.ok:
                        errors.append(f'uniparc 1: {_id}: {new_id}')
                        continue

                    _save_(r.text, _id, new_id, 'uniparc', hash_fasta, seq_fasta)
                    break
                if new_id:
                    break

    _refresh_and_save_from_to_()
    if patience < 2:
        return
    with open(out_file.with_name(out_file.stem + '_hash.fasta'), 'a') \
            as hash_fasta, open(out_file.with_suffix('.fasta'), 'a') as seq_fasta:
        for _id in tqdm(set(from_to.loc[from_to.crc_hash == '', 'To'])
                        - m_dict.keys(), desc='fetch isoforms/archive', position=0, leave=True, ascii=True):
            source = 'isoform' if '-' in _id else 'entry'
            r = requests.get(entry_url.format(_id))
            if not r.ok:
                errors.append(f'{source}: {_id}')
                continue
            if not r.text:
                r = requests.get(archive_url.format(_id))
                source += '-archive'
            _save_(r.text, _id, _id, source, hash_fasta, seq_fasta)

    if errors:
        print('\n'.join(errors))
        for e in errors:
            m_dict[e.split(':')[-1].strip()] = ['', '']

    _refresh_and_save_from_to_()

    with open(out_file.with_suffix('.json'), 'w') as json_file:
        json.dump(from_to.iloc[:, [0, 2]].copy()
                  .set_index('From').to_dict()['crc_hash'], json_file)
    missing = sorted(set(from_to.To) - m_dict.keys())
    if missing:
        print('missing:\n' + ' '.join(missing))


def ensembl_api_fetch(embl_ids: Union[set, list],
                      out_file: Union[str, Path] =
                      Path.cwd() / 'ensembl.json',
                      batch_size: int = 50,
                      ) -> dict:
    """

    :param embl_ids:
    :param out_file:
    :param batch_size:
    :return:
    """
    out_file = Path(out_file).resolve()
    if out_file.is_dir():
        raise IsADirectoryError
    out_file.parent.mkdir(parents=True, exist_ok=True)

    server = 'https://rest.ensembl.org'
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    embl_records = dict()
    for archive in [False, True]:
        # cut off version identifiers
        embl_ids = sorted({_id.split('.')[0] for _id in embl_ids} - embl_records.keys())
        if not embl_ids:
            continue
        ext = f'/{"archive" if archive else "sequence"}/id?type=protein;'
        _s = '"id"' if archive else '"ids"'

        for j in tqdm(range(0, len(embl_ids) + batch_size - 1, batch_size), position=0, leave=True, ascii=True):
            l = embl_ids[j:j + batch_size]
            if not l:
                continue
            r = requests.post(server + ext, headers=headers,
                              data=('{ ' + _s + ': ' + str(l) + ' }').replace('\'', "\""))
            if not r.ok:
                continue
            for rj in r.json():
                if archive and rj['peptide'] is not None:
                    embl_records[rj['id']] = rj
                elif rj['seq'] is not None:
                    embl_records[rj['query']] = rj
            j += batch_size

    with open(out_file.with_suffix('.json'), 'w') as json_file:
        json.dump(embl_records, json_file)
    return embl_records


def uniprot_api_fetch_ensembl(embl_ids: Union[list, set]) -> dict:
    """
    :param embl_ids:
    :return:
    """
    query_url = 'https://www.uniprot.org/uniprot/?query={}+organism%3AHuman&sort=score&format=tab&columns=id'
    entry_url = 'https://www.uniprot.org/uniprot/{}.fasta'

    embl_records = dict()
    not_found = list()
    # url_vs_fasta = list()

    for _id in tqdm(sorted({_id.split('.')[0] for _id in embl_ids}), position=0, leave=True, ascii=True):
        r = requests.get(query_url.format(_id))
        if not r.ok or not r.text:
            not_found.append(_id)
            continue

        found_id = r.text.strip().split('\n')[1]
        r = requests.get(entry_url.format(found_id))

        if not r.ok or not r.text:
            not_found.append(_id)
            continue

        seq = ''.join(r.text.strip().split('\n')[1:])
        embl_records[_id] = {'id': found_id, 'seq': seq, 'query': _id}
    if not_found:
        print(f'missing {len(not_found)}:')
        print(not_found)
    return embl_records


def to_hash_fasta(fasta_path: Union[str, Path]) -> pd.DataFrame:
    """
    For a FASTA-formatted file, create a sequence-unique copy
    with the filename suffix `_hash` and CRC32C sequence hashes
    instead of the original IDs.
    :param fasta_path:
    :return:
    """
    fasta_path = Path(fasta_path).resolve()
    meta = dict()
    seen_hashes = set()

    with open(fasta_path, 'r') as in_fasta, open(
            fasta_path.with_name(fasta_path.stem + '_hash.fasta'), 'w') as out_fasta:
        for r in SeqIO.parse(in_fasta, 'fasta'):
            _hash = get_seq_hash(r.seq)
            if r.id in meta:
                assert meta[r.id][0] == _hash, 'Found a FASTA entry with the repeated ' \
                                               f'ID {r.id}, but different sequences: ' \
                                               f'{meta[r.id][0]} <> {_hash}'
            meta[r.id] = [_hash, r.description]
            if _hash not in seen_hashes:
                for line in to_lines(_hash, r.seq):
                    out_fasta.write(line)
            seen_hashes.add(_hash)

    if n_dups := len(meta) - len(seen_hashes):
        print(f'{n_dups}/{len(meta)} sequences were duplicates')

    return (pd.DataFrame.from_dict(meta).T
        .reset_index().rename(columns={
        'index': 'original_id', 0: 'crc_hash', 1: 'description'}))


def write_filtered_tsv(ppis: pd.DataFrame,
                       out_file: Union[str, Path],
                       filtered_ids: set = None,
                       ) -> pd.DataFrame:
    """

    :param ppis:
    :param filtered_ids:
    :param out_file:
    :return:
    """
    out_file = Path(out_file).resolve()

    # don't filter by default
    if filtered_ids is None:
        filtered_ids = set(ppis[['A', 'B']].values.flatten())

    filtered_ppis = (pd.DataFrame(
        ppis.loc[(ppis['A'].isin(filtered_ids))
                 & (ppis['B'].isin(filtered_ids)), ['A', 'B']].copy()
            .apply(sorted, axis=1).tolist())
                     .drop_duplicates().sort_values(by=[0, 1]))

    filtered_ppis.to_csv(out_file.with_suffix('.tsv'), sep='\t', header=None, index=False)
    return filtered_ppis


def get_seq_hash(seq: Union[str, Seq]) -> str:
    if seq is None:
        return '0'
    seq = str(seq)
    return str(crc32c.crc32c(bytes(seq.upper(), 'UTF8')))


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
    get_seq_hash = lambda seq: str(crc32c.crc32c(bytes(seq)))

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
