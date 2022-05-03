import json
import urllib.parse
import urllib.request
from io import StringIO
from pathlib import Path
from typing import Union, Tuple, Dict

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

from data.utils.general import get_seq_hash, to_fasta, file_hash


def uniprot_api_fetch(uniprot_ids: Union[set, list],
                      out_file: Union[str, Path] = Path('uniprot'),
                      ) -> Dict[str, str]:
    """
    Download sequences from UniProt: First, post the given set of
    UniProt IDs to the ID mapping site and retrieve `tab` and `fasta`.
    Then, check UniParc for up-to-date replacement entries, finally
    use obsolete versions of the given IDs.
    :param uniprot_ids:
    :param out_file:
    :return:
    """
    uniprot_ids = set(uniprot_ids)
    out_file = Path(out_file)
    if out_file.is_dir():
        out_file = out_file / 'uniprot'
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
    with out_file.with_suffix('.hash.fasta').open('w') as hash_fasta:
        with out_file.with_suffix('.fasta').open('r') as in_fasta:
            for r in tqdm(SeqIO.parse(in_fasta, 'fasta'), desc='hash FASTA'):
                _id = r.id.split('|')[1]
                _hash = get_seq_hash(r.seq)
                m_dict[_id] = [_hash, 'query', r.description]

                if _hash in seen_hashes:
                    continue
                seen_hashes.add(_hash)
                to_fasta(_hash, r.seq, hash_fasta)

    from_to = pd.read_csv(out_file.with_suffix('.tab'), sep='\t', header=0)

    forgot = uniprot_ids - set(from_to.From)
    from_to = pd.concat([from_to, pd.DataFrame(
        zip(forgot, forgot), columns=['From', 'To'])])
    from_to['crc_hash'] = ''

    def _refresh_and_save_from_to_():
        from_to.loc[from_to.crc_hash == '', ['crc_hash', 'source', 'description']] = \
            from_to.loc[from_to.crc_hash == '', 'To'].apply(
                lambda t: m_dict.get(t, ['', '', ''])).to_list()

        from_to.sort_values(by='crc_hash').to_csv(out_file.with_suffix('.tsv'), index=False, sep='\t')

    _refresh_and_save_from_to_()

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
        to_fasta(_hash_, _seq_, _hashf_)
        to_fasta(_nid_, _seq_, _seqf_)

    # print(f'{b * 8}')
    with out_file.with_suffix('.hash.fasta').open('a') as hash_fasta, \
            out_file.with_suffix('.fasta').open('a') as seq_fasta:
        for _id in tqdm(set(from_to.loc[from_to.crc_hash == '', 'To'])
                        - m_dict.keys(), desc='fetch UniParc'):
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
    with out_file.with_suffix('.hash.fasta').open('a') as hash_fasta, \
            out_file.with_suffix('.fasta').open('a') as seq_fasta:
        for _id in tqdm(set(from_to.loc[from_to.crc_hash == '', 'To'])
                        - m_dict.keys(), desc='fetch isoforms/archive'):
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

    simple_from_to = from_to.iloc[:, [0, 2]].copy().set_index('From').to_dict()['crc_hash']
    with open(out_file.with_suffix('.json'), 'w') as json_file:
        json.dump(simple_from_to, json_file)
    missing = sorted(set(from_to.To) - m_dict.keys())
    if missing:
        print('missing:\n' + ' '.join(missing))

    return simple_from_to


def fetch_huri_seqs(huri_ids: Dict,
                    out_file: Union[str, Path] = Path('huri'),
                    ) -> Tuple[Dict[str, SeqRecord], Dict[str, str]]:
    if (fasta_file := out_file.with_suffix('.hash.fasta')).is_file():
        print(f'loading from {fasta_file} and {out_file.with_suffix(".json")}')
        fasta = SeqIO.to_dict(SeqIO.parse(fasta_file, 'fasta'))
        with out_file.with_suffix('.json').open('r') as json_file:
            from_to = json.load(json_file)
        return fasta, from_to

    fasta, from_to = ensembl_api_fetch(huri_ids['ensembl'],
                                       out_file.with_name('ensembl'))
    uniprot_from_to = uniprot_api_fetch(huri_ids['uniprotkb'],
                                        out_file.with_name('uniprot'))
    fasta.update(SeqIO.to_dict(SeqIO.parse(
        out_file.with_name('uniprot.hash.fasta'), 'fasta')))
    SeqIO.write(fasta.values(), out_file.with_suffix('.fasta'), 'fasta')
    from_to.update(uniprot_from_to)
    with out_file.with_suffix('.json').open('w') as json_file:
        json.dump(from_to, json_file)

    return fasta, from_to


def fetch_taxonomic_info(taxonomic_ids: pd.Series) -> pd.DataFrame:
    server = 'https://rest.ensembl.org'
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    common_names = dict()
    for tax_id in tqdm(taxonomic_ids.unique()):
        ext = f'/taxonomy/id/{tax_id}?'
        r = requests.get(server + ext, headers=headers)
        if not r.ok:
            r.raise_for_status()
            continue
        decoded = r.json()
        common_names[tax_id] = decoded['name']
    return pd.DataFrame.from_dict(common_names, orient='index', columns=['name']) \
        .rename_axis('species').reset_index() \
        .sort_values(by='species').join(
        taxonomic_ids.value_counts(), on='species', rsuffix='n_ppis'
    ).reset_index(drop=True).rename(columns=dict(speciesn_ppis='n_ppis'))


def ensembl_api_fetch(embl_ids: Union[set, list],
                      out_file: Union[str, Path] = Path('ensembl'),
                      batch_size: int = 50,
                      ) -> Tuple[Dict[str, SeqRecord], Dict[str, str]]:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    batch_size = min(batch_size, 50)
    server = 'https://rest.ensembl.org'
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    embl_records = dict()
    for archive in [False, True]:
        # cut off version identifiers because we can't use them for queries anyway
        embl_ids = sorted({_id.split('.')[0] for _id in embl_ids} - embl_records.keys())
        if not embl_ids:
            continue
        ext = f'/{"archive" if archive else "sequence"}/id?type=protein;'
        _s = '"id"' if archive else '"ids"'
        for j in tqdm(range(0, batch_size * int(
                np.ceil(len(embl_ids) / batch_size)), batch_size),
                      desc=('archive ' if archive else 'Ensembl ') + 'batch'):
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

    with out_file.with_suffix('.response.json').open('w') as json_file:
        json.dump(embl_records, json_file)

    embl_records = {_id: r.get('seq', r.get('peptide'))
                    for _id, r in embl_records.items()}

    hash_records = dict()
    from_to = dict()
    with out_file.with_suffix('.fasta').open('w') as clear_fasta, \
            out_file.with_suffix('.hash.fasta').open('w') as hash_fasta:
        for _id in sorted(embl_records.keys()):
            seq = embl_records[_id]
            to_fasta(_id, seq, clear_fasta)
            _hash = get_seq_hash(seq)
            hash_records[_hash] = SeqRecord(Seq(seq), id=_hash, description='')
            from_to[_id] = _hash
            to_fasta(_hash, seq, hash_fasta)
    with out_file.with_suffix('.json').open('w') as json_file:
        json.dump(from_to, json_file)

    missing = sorted(set(embl_ids) - embl_records.keys())
    if missing:
        print('missing:\n' + ' '.join(missing))

    return hash_records, from_to


def download_y2h_interactome(
        out_file: Union[str, Path] = Path('hi_union.psi'),
        target_hash: str = '25c9566fe06fe0682e4183b326fef7ff705'
                           'fbe4d52fdd276242b030ec746d87f') -> Path:
    # kudos to https://stackoverflow.com/a/37573701
    out_file = Path(out_file)
    if out_file.is_file():
        if file_hash(out_file) == target_hash:
            print('already downloaded and SHA256 checks out')
            return out_file
    url = 'http://www.interactome-atlas.org/data/HI-union.psi'
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 ^ 2  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes,
                        unit='iB', unit_scale=True,
                        position=0, leave=True)
    with out_file.open('wb') as file:
        for data in response.iter_content(block_size):
            _ = progress_bar.update(len(data))
            _ = file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print('ERROR, something went wrong')
    return out_file


def fetch_proteomes(species: set,
                    proteome_dir: Path = Path('proteomes')) -> None:
    url = 'https://www.uniprot.org/proteomes/?fil=reference:yes&format=tab&query=' \
          + '+OR+'.join(f'organism:{sp}' for sp in species)

    r = requests.get(url)
    tab = pd.read_csv(StringIO(r.text), sep='\t')
    assert len(set(tab['Proteome ID'])) == len(tab), \
        'Got multiple/no proteomes for some species'

    if missing := species - set(tab['Organism ID']):
        if missing == {37296}:
            s = ['UP000097197', 'Human herpesvirus 8 (HHV-8)', 37296, 72, '', 'Standard', '']
            tab = pd.concat([tab, pd.DataFrame(s, index=tab.columns).T])
        else:
            raise ValueError(f'Missing proteomes: {" ".join(missing)}')

    proteome_dir.mkdir(exist_ok=True, parents=True)
    proteome_url = 'https://www.uniprot.org/uniprot/?format=fasta&query=proteome:'
    with tqdm(tab.iterrows(), total=len(tab)) as pbar:
        for i, proteome in pbar:
            pbar.set_postfix(batch=proteome['Organism'].split('(')[0].strip())
            r = requests.get(proteome_url + proteome['Proteome ID'], stream=True)
            with (proteome_dir / f'{proteome["Organism ID"]}.fasta').open('wb') as fd:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    fd.write(chunk)
