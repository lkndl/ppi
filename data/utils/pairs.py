import concurrent.futures
from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Set, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from matplotlib.figure import Figure
from scipy import stats
from tqdm import tqdm

mpl.rcParams['figure.dpi'] = 200


class SamplingStrategy(Enum):
    RANDOM = 0
    BALANCED = 1


class CorrelationType(Enum):
    PEARSON = 0
    SPEARMAN = 1


def make_c_classes(test_tsv: Union[str, Path],
                   c3_fasta: Union[str, Path],
                   test_fasta: Union[str, Path]
                   ) -> pd.DataFrame:
    assert Path(c3_fasta).is_file(), f'C3 FASTA {c3_fasta.name} is missing!'
    c3_ids, test_fasta_ids = [{r.id for r in SeqIO.parse(
        f, 'fasta')} for f in [c3_fasta, test_fasta]]
    assert test_fasta_ids, 'WTF?'
    assert c3_ids, 'No non-redundant sequences left; cannot construct C3 test!'
    assert not c3_ids - test_fasta_ids, \
        f'Forgot to replace {test_fasta} after "shrink_files_both_ways", ' \
        f'before the rostclust uniqueprot2d run?'
    assert test_fasta_ids - c3_ids, 'Test set completely non-redundant, ' \
                                    'cannot construct C1-2 sets!'

    ppis = pd.read_csv(test_tsv, sep='\t', header=0)
    ppi_ids = set(np.unique(ppis.iloc[:, [0, 1]]))
    assert ppi_ids == test_fasta_ids, \
        f'The IDs in {test_fasta} and {test_tsv} should be the same! ' \
        f'Did you forget "shrink_files_both_ways" after the redundancy reduction?'

    _c123 = lambda df: 1 + df.iloc[:, 0].isin(c3_ids) + df.iloc[:, 1].isin(c3_ids)
    ppis['cclass'] = _c123(ppis)
    return ppis


def make_negatives(ppis: pd.DataFrame,
                   strategy: SamplingStrategy = SamplingStrategy.BALANCED,
                   ratio: float = 10.0, seed: int = 42,
                   accept_homodimers: bool = True,
                   proteome: Dict[int, Dict[str, str]] = None,
                   sus_ppis: pd.DataFrame = None,
                   ) -> Tuple[pd.DataFrame, pd.DataFrame,
                              pd.DataFrame, Union[Figure, None]]:
    negatives, bias, fig, _ = find_negative_pairs(
        ppis, sus_ppis, strategy, ratio, seed,
        accept_homodimers, proteome)
    ppis['label'] = 1
    ppis = ppis[[c for c in ppis.columns if c not in ('minlen', 'maxlen')]]
    sp = 9606 if 'species' not in ppis.columns else ppis.species.unique()[0]
    if 'species' not in negatives.columns:
        negatives['species'] = sp
    negatives['label'] = 0
    negatives.columns = ['hash_A', 'hash_B'] + list(negatives.columns[2:])
    if type(bias) == np.ndarray:
        bias = pd.DataFrame(bias.T, columns=['species', 'bias']
                            ).convert_dtypes()
    return ppis, negatives, bias, fig


def fetch_ratios(pairs: pd.DataFrame) -> pd.DataFrame:
    degrees = list()
    for l, df in pairs.groupby(list(pairs.columns[2:])):
        proteins, counts = np.unique(df.iloc[:, [0, 1]], return_counts=True)
        degrees.append([*l, np.vstack((np.arange(len(counts)),
                                       np.sort(counts)[::-1])).T])
    dt = pd.DataFrame(degrees, columns=list(
        pairs.columns[2:]) + ['degree'])
    dt = dt.explode('degree')
    dt[['x', 'degree']] = dt.degree.tolist()
    return dt


def count_homodimers(pairs: pd.DataFrame) -> Tuple[int, float, int]:
    homod = len(pairs.loc[pairs.hash_A == pairs.hash_B])
    return homod, np.round(homod / len(pairs), 4), len(pairs)


def make_validation_species(pairs: pd.DataFrame, species: Set[int]
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pairs.loc[pairs.species.isin(species)]
    pairs = pairs.loc[~pairs.species.isin(species)].copy()
    train_ids = set(np.unique(pairs.iloc[:, [0, 1]]))
    df = df.loc[(~df.hash_A.isin(train_ids)) & (~df.hash_B.isin(train_ids))]
    return pairs, df


def make_validation_split(pairs: pd.DataFrame,
                          val_set_size: float = .5, seed: int = 42,
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed=seed)
    i, j = 0, 0
    val_proteins = set()
    val_ppis = pd.DataFrame()
    # iteratively add more proteins until we fulfill the target size
    while len(val_ppis) < val_set_size * 3 * len(pairs):
        i += 1
        val_ppis = pairs.loc[(pairs.hash_A.isin(val_proteins)) |
                             (pairs.hash_B.isin(val_proteins))]
        if j == len(val_ppis):
            # no new PPIs added: only homodimers in last step, or species barrier
            idx = rng.choice(range(len(pairs)))
            val_proteins |= set(pairs.iloc[idx, [0, 1]])
        else:
            # add another random protein
            val_proteins |= set(np.unique(val_ppis.iloc[:, [0, 1]]))
            j = len(val_ppis)
    print(f'{i} loops')

    idcs = sorted(set(val_ppis.index))
    # idcs = list(val_ppis.index)[:int(val_set_size * len(pairs))]
    # idcs = np.sort(rng.choice(
    #     list(val_ppis.index), size=int(len(pairs) * val_set_size), replace=False))
    n_idcs = np.delete(np.arange(0, len(pairs)), idcs)

    _train, _val = pairs.iloc[n_idcs, :].copy(), pairs.iloc[idcs, :].copy()

    # drop proteins that occur in train from val
    train_proteins = set(np.unique(_train.iloc[:, [0, 1]]))
    _val = _val.loc[~(_val.hash_A.isin(train_proteins))
                    & ~(_val.hash_B.isin(train_proteins))]

    return _train, _val


def find_negative_pairs_tuple(args):
    return find_negative_pairs(*args)


def find_negative_pairs(true_ppis: pd.DataFrame,
                        sus_ppis: pd.DataFrame = None,
                        strategy: SamplingStrategy = SamplingStrategy.BALANCED,
                        ratio: float = 10.0,
                        seed: int = 42,
                        accept_homodimers: bool = True,
                        proteome: Dict[int, Dict[str, str]] = None,
                        quiet: bool = False,
                        ) -> Tuple[pd.DataFrame, Union[float, np.ndarray],
                                   Union[Figure, None], int]:
    # recursive call for multi-species case
    if 'species' in true_ppis.columns and len(set(true_ppis.species)) > 1:
        print(f'sampling negatives per species! aim for '
              f'{int(len(true_ppis) * ratio)}')
        negatives, biases = list(), dict()
        tuples = [(ppis, sus_ppis, strategy, ratio, seed + int(sp),
                   accept_homodimers, {int(sp): proteome[int(sp)]}, True) for sp, ppis in
                  true_ppis.groupby('species')]
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(tuples)) as executor:
            for n, b, f, s in executor.map(find_negative_pairs_tuple, tuples):
                n['species'] = s
                negatives.append(n)
                biases[s] = b
        # for n, b, f, s in process_map(find_negative_pairs_tuple, tuples,
        #                               max_workers=len(set(true_ppis.species))):
        #     n['species'] = s
        #     negatives.append(n)
        #     biases[s] = b
        negatives = pd.concat(negatives)
        bias = np.array(list(biases.items()), dtype=float).T
        print(f'{len(negatives)} negatives with overall '
              f'{estimate_bias(true_ppis, negatives)[0]:.3f} '
              f'and average per-species bias of '
              f'{np.nanmean(bias[1, :]):.3f}±{np.nanstd(bias[1, :]):.3f} (std)')
        return negatives, bias, None, 0

    if 'species' in true_ppis.columns:
        sp = set(true_ppis.species).pop()
    else:
        quiet = True
        sp = ''

    # map protein IDs to their sorting index
    uniq_true = {_id: idx for idx, _id in enumerate(
        np.unique(true_ppis.iloc[:, [0, 1]]))}
    uniq_neg = {idx: _id for _id, idx in uniq_true.items()}
    indexize = np.vectorize(uniq_true.get)
    unindex = np.vectorize(uniq_neg.get)
    idx_ppis = indexize(true_ppis.iloc[:, [0, 1]])
    if sus_ppis is not None:
        idx_ppis = np.vstack((idx_ppis, indexize(sus_ppis.iloc[:, [0, 1]])))

    # np.unique returns sorted values, so this works out
    proteins, counts = np.unique(true_ppis.iloc[:, [0, 1]], return_counts=True)
    n = len(proteins)
    vertices = np.array(range(n))
    indices = np.array(range(n + 1))

    rng = np.random.default_rng(seed=seed)

    wants = np.floor(counts * ratio).astype(int)
    if strategy != 1:
        wants = np.full_like(wants, np.floor(sum(counts) * ratio / n)).astype(int)

    # make sure that wants is an integer vector and its sum as close to the target as possible
    idcs = list(rng.choice(vertices, size=n, replace=True, p=counts / sum(counts)))
    while np.round(sum(counts) * ratio) > sum(wants):
        idx = idcs.pop(0)
        wants[idx] += 1

    wants = np.append(wants, 0)
    limit = sum(wants)

    if not quiet:
        tqdm.write(f'{sp}: {len(idx_ppis)} positives, aim for {limit // 2} negatives')

    # initialize the matrix
    mat = np.zeros((n + 1, n + 1), dtype=int)
    if not accept_homodimers:
        np.fill_diagonal(mat, 1)
        mat[-1, -1] = 0
    mat[idx_ppis[:, 0], idx_ppis[:, 1]] = 1
    mat[idx_ppis[:, 1], idx_ppis[:, 0]] = 1

    with tqdm(total=limit, position=0, desc=str(sp)) as pbar:
        while np.sum(wants[:n]):
            x = rng.choice(vertices, size=1, replace=False, p=wants[:n] / sum(wants[:n]))[0]
            wants[x] -= 1
            wants[-1] = max(0, 2 * wants[x] - (mat[x, :n] == 0) @ wants[:n])
            p_proteome = np.append((mat[x, :n] == 0) * wants[:n], wants[n])
            if not np.sum(p_proteome):
                p_proteome[-1] = 1
            y = rng.choice(indices, size=1, p=p_proteome / sum(p_proteome))[0]
            mat[y, x] -= 1  # tolerant against y=n
            if x != y:
                mat[x, y] -= 1
            wants[y] -= 1  # y=n will ignore this
            pbar.update(2)

    if not quiet:
        fig, ax = plt.subplots()
        cmap = mpl.colors.ListedColormap(-(np.min(mat) + 1) * ['#6B0E30']
                                         + ['#D81B60', '#FFFFFF', '#1E88E5'])
        heat = sns.heatmap(mat,  # annot=True, linewidth=.2,
                           ax=ax, cmap=cmap, cbar=False)
        ax.set(box_aspect=1, xticks=[], yticks=[])
    else:
        fig = None

    negs = np.vstack(np.nonzero(np.triu(mat, k=0) < 0)).T
    negs = negs[(negs[:, 0] < n) & (negs[:, 1] < n)]  # filter out the last col
    negatives = pd.DataFrame(unindex(negs)) if len(negs) else pd.DataFrame()
    if not quiet or not len(negatives):
        tqdm.write(f'{sp}: {len(negatives)}/{limit // 2} in-network negatives')

    idcs = np.flatnonzero(mat[:, n])
    extra = np.vstack((idcs, -mat[idcs, n])).T
    if len(extra) and proteome:
        min_extra = np.max(extra[:, 1])
        extra_interactions = np.sum(extra[:, 1])
        avg_extra = np.ceil(extra_interactions / ratio).astype(int)
        available = len(proteome[sp].keys() - uniq_true.keys())
        n_extra = min(available, max(min_extra, avg_extra))
        if not quiet:
            print(f'{sp}: need {min_extra} extra proteines; select {n_extra} from '
                  f'{available}/{len(proteome[sp])} (new/available) SwissProt proteins. '
                  f'Try to create {extra_interactions} interactions, '
                  f'ideally {ratio} per protein.')

        extra_proteins = list(rng.choice(list(proteome[sp].keys() - uniq_true.keys()),
                                         size=n_extra, replace=False))
        extra_pairs = list()
        for p_idx, n_partners in extra:
            p = uniq_neg[p_idx]
            partners = rng.choice(extra_proteins, size=min(
                n_partners, len(extra_proteins)), replace=False)
            extra_pairs.extend([(p, partner) for partner in partners])

        extras = pd.DataFrame(extra_pairs)
        # extra_crcs = set(extras.iloc[:, 1])
        negatives = pd.concat((negatives, extras))

    if not len(negatives):
        if not quiet:
            print(f'{sp}: got 0 negatives!')
        return negatives, np.NaN, fig, sp

    bias = estimate_bias(true_ppis, negatives)[0]
    if not quiet:
        print(f'{sp}: got {len(negatives)} negatives with bias {bias:.3f}')
    return negatives, bias, fig, sp


def estimate_bias(positives: pd.DataFrame,
                  negatives: pd.DataFrame = None,
                  corrtype: CorrelationType = CorrelationType.PEARSON,
                  ) -> Tuple[float, float]:
    """
    Calculate the similarity between two sets of protein pairs:
    the Spearman or Pearson correlation coefficient between their
    protein-appearance frequency vectors.
    """
    plus, minus = split_pos_neg_plus_minus(positives, negatives)

    for p, m in ((plus, minus), (minus, plus)):
        m.update({k: 0 for k in p.keys() - m.keys()})
    if corrtype.value == 0:
        return stats.pearsonr(*[[ar[k] for k in sorted(ar.keys())]
                                for ar in (plus, minus)])
    elif corrtype.value == 1:
        return stats.spearmanr(*[[ar[k] for k in sorted(ar.keys())]
                                 for ar in (plus, minus)], axis=1)
    else:
        assert False, 'illegal correlation type'


def split_pos_neg_plus_minus(positives: pd.DataFrame,
                             negatives: Union[pd.DataFrame, None]
                             ) -> Tuple[Dict, Dict]:
    if negatives is None:
        assert 'label' in positives.columns
        negatives = positives.loc[positives.label == 0].copy()
        positives = positives.loc[positives.label == 1].copy()
        assert len(positives), 'no positives in passed DataFrame'
        assert len(negatives), 'no negatives in passed DataFrame'

    plus, minus = [dict(zip(*np.unique(ar.iloc[:, [0, 1]], return_counts=True)))
                   for ar in (positives, negatives)]
    return plus, minus


def find_multi_species_proteins(ppis: pd.DataFrame) -> pd.DataFrame:
    dfs = list()
    for c in 'AB':
        pairs = ppis[[f'UniprotID_{c}', 'species']].drop_duplicates()
        dfs += [df for i, df in pairs.groupby(f'UniprotID_{c}') if len(df) > 1]
        dfs.append(pairs.loc[pairs.species == 'is there a marsupilami?'].values)
    return pd.DataFrame(np.vstack(dfs), columns=['UniprotID', 'species']) \
        .drop_duplicates().groupby('UniprotID') \
        .agg({'species': lambda l: sorted(l.astype(int).tolist())}).reset_index()
