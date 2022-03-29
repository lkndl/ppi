from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Set

import matplotlib as mpl
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import stats
from scipy.special import binom

mpl.rcParams['figure.dpi'] = 200


class SamplingStrategy(Enum):
    RANDOM = 0
    BALANCED = 1


class CorrelationType(Enum):
    PEARSON = 0
    SPEARMAN = 1


def make_test_checks(ppis: pd.DataFrame,
                     test_tsv: Union[str, Path],
                     c3_fasta: Union[str, Path],
                     test_fasta: Union[str, Path]
                     ) -> set:
    c3_fasta_ids, test_fasta_ids = [{r.id for r in SeqIO.parse(
        f, 'fasta')} for f in [c3_fasta, test_fasta]]
    print(test_fasta)
    assert test_fasta_ids, 'WTF?'
    assert c3_fasta_ids, 'No non-redundant sequences left; cannot construct C3 test!'
    assert not c3_fasta_ids - test_fasta_ids, \
        f'Forgot to replace {test_fasta} after "shrink_files_both_ways", ' \
        f'before the rostclust uniqueprot2d run?'
    assert test_fasta_ids - c3_fasta_ids, 'Test set completely non-redundant, ' \
                                          'cannot construct C1-2 sets!'
    ppi_ids = set(np.unique(ppis.iloc[:, [0, 1]]))
    assert ppi_ids == test_fasta_ids, \
        f'The IDs in {test_fasta} and {test_tsv} should be the same! ' \
        f'Did you forget "shrink_files_both_ways" after the redundancy reduction?'
    return c3_fasta_ids


def make_test_negatives(test_tsv: Union[str, Path],
                        c3_fasta: Union[str, Path],
                        test_fasta: Union[str, Path],
                        strategy: SamplingStrategy = SamplingStrategy.BALANCED,
                        ratio: float = 10.0,
                        seed: int = 42,
                        accept_homodimers: bool = True,
                        proteome_dir: Path = None,
                        ) -> Tuple[pd.DataFrame, pd.DataFrame,
                                   Union[float, pd.DataFrame]]:
    ppis = pd.read_csv(test_tsv, sep='\t', header=0)
    sp = 9606 if 'species' not in ppis.columns else ppis.species.unique()[0]
    # 9606 is the taxid for humans. everybody pays taxes!
    c3_ids = make_test_checks(ppis, test_tsv, c3_fasta, test_fasta)
    _c123 = lambda df: 1 + df.iloc[:, 0].isin(c3_ids) + df.iloc[:, 1].isin(c3_ids)
    ppis['cclass'] = _c123(ppis)

    negatives, bias = find_negative_pairs(ppis[['hash_A', 'hash_B', 'species']],
                                          strategy=strategy,
                                          ratio=ratio, seed=seed,
                                          accept_homodimers=accept_homodimers,
                                          proteome_dir=proteome_dir)
    if 'species' not in negatives.columns:
        negatives['species'] = sp
    negatives['cclass'] = _c123(negatives)
    print(len(ppis))
    return _tweak(ppis, negatives, bias)


def make_train_negatives(ppis: pd.DataFrame,
                         strategy: SamplingStrategy = SamplingStrategy.BALANCED,
                         ratio: float = 10.0, seed: int = 42,
                         accept_homodimers: bool = True,
                         sus_ppis: pd.DataFrame = None,
                         proteome_dir: Path = None,
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    negatives, bias = find_negative_pairs(ppis, sus_ppis, strategy, ratio, seed,
                                          accept_homodimers, proteome_dir)
    return _tweak(ppis, negatives, bias)


def _tweak(ppis: pd.DataFrame, negatives: pd.DataFrame, bias: np.ndarray,
           ) -> Tuple[pd.DataFrame, pd.DataFrame, Union[float, pd.DataFrame]]:
    negatives['label'] = 0
    ppis['label'] = 1
    negatives.columns = ['hash_A', 'hash_B'] + list(negatives.columns[2:])
    ppis = ppis[[c for c in ppis.columns if c not in ('minlen', 'maxlen')]]
    if type(bias) == np.ndarray:
        bias = pd.DataFrame(bias.T, columns=['species', 'bias']).convert_dtypes()
    return ppis, negatives, bias


def fetch_degrees(pairs: pd.DataFrame) -> pd.DataFrame:
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


def find_negative_pairs(true_ppis: pd.DataFrame,
                        sus_ppis: pd.DataFrame = None,
                        strategy: SamplingStrategy = SamplingStrategy.BALANCED,
                        ratio: float = 10.0,
                        seed: int = 42,
                        accept_homodimers: bool = True,
                        proteome_dir: Path = None,
                        quiet: bool = False,
                        ) -> Tuple[pd.DataFrame, Union[float, np.ndarray]]:
    if 'species' in true_ppis.columns and len(set(true_ppis.species)) > 1:
        print(f'sampling negatives per species! aim for '
              f'{int(len(true_ppis) * ratio)}')
        negatives, biases = list(), dict()
        for sp, ppis in true_ppis.groupby('species'):
            sp = int(sp)
            sp_negatives, biases[sp] = find_negative_pairs(
                ppis, sus_ppis, strategy, ratio, seed, accept_homodimers, proteome_dir, True)
            sp_negatives['species'] = sp
            negatives.append(sp_negatives)
        negatives = pd.concat(negatives)
        bias = np.array(list(biases.items()), dtype=float).T
        print(f'{len(negatives)} negatives with overall '
              f'{estimate_bias(true_ppis, negatives)[0]:.3f} '
              f'and average per-species bias of '
              f'{np.nanmean(bias[1, :]):.3f}±{np.nanstd(bias[1, :]):.3f} (std)')
        return negatives, bias

    # map protein IDs to their sorting index
    uniq_true = {_id: idx for idx, _id in enumerate(
        np.unique(true_ppis.iloc[:, [0, 1]]))}
    uniq_neg = {idx: _id for _id, idx in uniq_true.items()}
    indexize = np.vectorize(uniq_true.get)
    unindex = np.vectorize(uniq_neg.get)
    idx_ppis = indexize(true_ppis.iloc[:, [0, 1]])
    if sus_ppis is not None:
        idx_ppis = np.vstack((idx_ppis, indexize(sus_ppis.iloc[:, [0, 1]])))
    drop_ppis = pd.DataFrame(idx_ppis.copy())

    # np.unique returns sorted values, so this works out
    proteins, counts = np.unique(true_ppis.iloc[:, [0, 1]], return_counts=True)
    indices = np.array(range(len(proteins)))
    frequencies = counts / np.sum(counts)
    if strategy.value != 1:
        frequencies = np.full_like(frequencies, 1 / len(frequencies))

    rng = np.random.default_rng(seed=seed)
    target_len = int(len(idx_ppis) * ratio)
    limit = binom(len(proteins), 2) - len(drop_ppis) \
            + len(drop_ppis.loc[drop_ppis[0] == drop_ppis[1]])

    homodimer_share = len(drop_ppis.loc[drop_ppis[0] == drop_ppis[1]]) / len(drop_ppis)
    if accept_homodimers and homodimer_share > 0:
        limit = binom(len(proteins), 2) + len(proteins) - len(drop_ppis)
    if not quiet:
        print(f'aim for {target_len} negatives; upper limit is {limit:.0f}')

    def _fetch_pairs():
        cols = rng.choice(indices, size=(len(idx_ppis), 2),
                          replace=True, p=frequencies)
        cols = pd.DataFrame(np.unique(np.sort(cols, axis=1), axis=0))
        if not accept_homodimers or homodimer_share == 0:
            cols = cols.loc[cols[0] != cols[1]]
        cols = pd.concat((drop_ppis, drop_ppis, cols)) \
            .drop_duplicates(keep=False)  # using the ppis twice so they are all dropped!
        return cols

    df = pd.concat([_fetch_pairs() for _ in range(
        int(np.ceil(ratio)))]).drop_duplicates()
    while len(df) < target_len and len(df) < limit:
        df = pd.concat((df, _fetch_pairs())).drop_duplicates()

    if not len(df):
        if not quiet:
            print('got 0 negatives!')
        return df, np.NaN

    negatives = pd.DataFrame(unindex(df.iloc[:target_len, :]))
    bias = estimate_bias(true_ppis, negatives)[0]
    if not quiet:
        print(f'got {len(negatives)} negatives with bias {bias:.3f}')

    # TODO most of the nodes have too many interaction partners; and hubs have too few
    return negatives, bias


def estimate_bias(positives: pd.DataFrame,
                  negatives: pd.DataFrame = None,
                  corrtype: CorrelationType = CorrelationType.PEARSON,
                  ) -> Tuple[float, float]:
    """
    Calculate the similarity between two sets of protein pairs:
    the Spearman or Pearson correlation coefficient between their
    protein-appearance frequency vectors.
    """
    if negatives is None:
        assert 'label' in positives.columns
        negatives = positives.loc[positives.label == 0].copy()
        positives = positives.loc[positives.label == 1].copy()
        assert len(positives), 'no positives in passed DataFrame'
        assert len(negatives), 'no negatives in passed DataFrame'

    plus, minus = [dict(zip(*np.unique(ar.iloc[:, [0, 1]], return_counts=True)))
                   for ar in (positives, negatives)]
    minus.update({k: 0 for k in plus.keys() - minus.keys()})
    if corrtype.value == 0:
        return stats.pearsonr(*[[ar[k] for k in sorted(ar.keys())]
                                for ar in (plus, minus)])
    elif corrtype.value == 1:
        return stats.spearmanr(*[[ar[k] for k in sorted(ar.keys())]
                                 for ar in (plus, minus)], axis=1)
    else:
        assert False, 'illegal correlation type'


def find_multi_species_proteins(ppis: pd.DataFrame) -> pd.DataFrame:
    dfs = list()
    for c in 'AB':
        pairs = ppis[[f'UniprotID_{c}', 'species']].drop_duplicates()
        dfs += [df for i, df in pairs.groupby(f'UniprotID_{c}') if len(df) > 1]
        dfs.append(pairs.loc[pairs.species == 'is there a marsupilami?'].values)
    return pd.DataFrame(np.vstack(dfs), columns=['UniprotID', 'species']) \
        .drop_duplicates().groupby('UniprotID') \
        .agg({'species': lambda l: sorted(l.astype(int).tolist())}).reset_index()
