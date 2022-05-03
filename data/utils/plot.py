from typing import Dict, Tuple, Set, List, Union

import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from data.utils.pairs import fetch_ratios, count_homodimers, \
    estimate_bias, split_pos_neg_plus_minus

mpl.rcParams['figure.dpi'] = 200


@mpl.style.context('seaborn-poster')
def draw_toy_ppis(ppis: pd.DataFrame,
                  n_dict: Dict[str, pd.DataFrame],
                  seed: int = 42,
                  extra_nodes: Set = None
                  ) -> Tuple[Figure, Figure]:
    if not extra_nodes:
        extra_nodes = set()

    def _edges_from_pd(df):
        return ((a, b) for _, a, b in df.iloc[:, [0, 1]].itertuples())

    def _minmax(ar, _min=60, _max=320):
        ar = np.array(ar)
        mi = np.min(ar)
        ma = np.max(ar)
        return (ar - mi) / (ma - mi) * (_max - _min) + _min

    # some magic so that the order of the plots stays
    # the same, but the true PPIs still come first
    keys = list(n_dict.keys())[::-1] + ['positive']
    n_dict['positive'] = ppis
    nodes = set(np.unique(ppis.iloc[:, [0, 1]])) | extra_nodes
    densest = 0, None
    graphs = dict()

    for key in reversed(keys):
        negatives = n_dict[key]
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(_edges_from_pd(negatives), attr=dict(source=key))
        graphs[key] = G

        if (l := len(G.edges)) > densest[0]:
            densest = l, G

    # H = nx.Graph(densest[1])
    H = nx.Graph(graphs['positive'])
    H.add_edges_from(_edges_from_pd(ppis))
    pos = nx.spring_layout(H, iterations=100, seed=seed)
    pos = nx.kamada_kawai_layout(H)
    circ = nx.circular_layout(H)

    figs = list()
    for anno, lay in enumerate((pos, circ)):
        fig, axes = plt.subplots(1, len(n_dict))
        for ax, (key, G) in zip(axes, graphs.items()):
            nx.draw(H, pos=lay, ax=ax, alpha=.4, edge_color='#FF70A4',
                    with_labels=False, node_size=0)
            nx.draw(G, pos=lay, ax=ax, with_labels=True,
                    node_size=_minmax([G.degree[v] for v in G]),
                    node_color=[float(G.degree(v)) for v in G],
                    cmap=mpl.colors.ListedColormap(['#E53374', '#1E88E5', '#FFC107', '#00E0BB'][::-1]),
                    )
            if anno:
                ax.set(box_aspect=1)
            else:
                ax.set(box_aspect=1, ylim=(-1.2, None), xlim=(-1, None))
                bias = estimate_bias(ppis, n_dict[key])[0]
                ax.annotate(key + (f':\n{bias:.2f}' if key != 'positive' else ''),
                            xy=(-1, -1.2))
        fig.tight_layout()
        figs.append(fig)

    return figs[0], figs[1]


def plot_homodimer_share(pairs: pd.DataFrame) -> Figure:
    df = pairs.groupby('species').apply(count_homodimers).reset_index()
    df[['_count', '_share', '_total']] = df[0].to_list()

    fig, ax = plt.subplots(figsize=(8, 3), facecolor='white')
    scatter = sns.scatterplot(data=df,
                              x='_total',
                              y='_share',
                              s=40,
                              ax=ax,
                              )
    ax.set(xscale='log',  # ylim=(None, 1),
           xlabel='species interactome size',
           ylabel='homodimer share',
           )
    for i, point in df.iterrows():
        ax.annotate(point.species,
                    (point._total * 1.05,
                     point._share + .05),
                    rotation=50, size=6)
    sns.despine(left=True, bottom=True)
    fig.tight_layout()
    return fig


def plot_interactome_sizes(taxonomy: pd.DataFrame,
                           val_species: Set) -> Figure:
    tax = taxonomy.copy()
    tax['species'] = pd.Categorical(tax.species)
    order = list(tax.species)[::-1]
    tax['hue'] = tax.species.apply(lambda sp: sp in val_species)

    j = sns.catplot(
        data=tax,
        x='n_ppis',
        y='species',
        hue='hue',
        palette={0: '#D81B60', 1: '#1E88E5'},
        order=order,
        s=7,
        height=6,
        aspect=2,
        zorder=1000,
        legend=False
    )
    j.set(xscale='log', xlabel='interactome size',
          ylim=(None, -.5), xlim=(None, max(tax.n_ppis) * 1.4))
    j.ax.get_yaxis().set_visible(False)
    j.despine(left=True, bottom=True)
    j.ax.xaxis.tick_top()
    j.ax.xaxis.set_label_position('top')

    for i, (idx, row) in enumerate(tax.iterrows()):
        sp, name, x, hue = row
        y = len(tax) - i - 1
        j.ax.text(x * 1.2, y, name, ha='left', va='center', zorder=100)
        # j.ax.plot([x, x], [y, 0], marker=None, color='gray', lw=.5, alpha=.4)

    j.ax.xaxis.grid(True, 'minor', linewidth=.4, alpha=.6, zorder=0)
    j.ax.xaxis.grid(True, 'major', linewidth=1, zorder=0)

    j.ax.fill_between([.9] + list(tax.n_ppis) + [1e6],
                      [28] + [len(tax) - k - 1 for k in range(len(tax))] + [-2],
                      len(tax) + 2,
                      zorder=10, color='w')
    j.tight_layout()
    j.figure.set_facecolor('white')
    # for sfx in ['png', 'pdf', 'svg']:
    #     j.savefig(f'interactome_sizes.{sfx}', dpi=300, transparent=False)
    return j


def plot_bias(plus: pd.DataFrame, minus: pd.DataFrame,
              bias: pd.DataFrame, ratio: float = 10.0) -> Figure:
    s = 'species'
    bias = (bias.merge(plus.groupby(s)['label'].size(), on=s)
            .merge(minus.groupby(s)['label'].size(), on=s))
    bias.columns = ['species', 'bias', 'positives', 'negatives']
    bias['fulfill'] = bias.negatives / ratio / bias.positives
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    bias['bias'] = bias.bias.astype(float)

    fig, ax = plt.subplots(figsize=(6, 4.4), facecolor='white')
    scatter = sns.scatterplot(data=bias,
                              ax=ax,
                              x='positives', y='negatives',
                              hue='bias', palette=cmap,
                              # aspect=1, height=4,
                              s=40, alpha=.8,
                              zorder=99,
                              )

    for i, point in bias.iterrows():
        if point.negatives < 10000:
            ax.annotate(point.species,
                        (point.positives * 1.4,
                         point.negatives * .9),
                        rotation=0, size=6)

    ax.set(xscale='log', yscale='log')
    ax.xaxis.grid(True, 'minor', linewidth=.2, alpha=.6, zorder=0)
    ax.yaxis.grid(True, 'minor', linewidth=.2, alpha=.6, zorder=0)
    ax.xaxis.grid(True, 'major', linewidth=.5, zorder=0)
    ax.yaxis.grid(True, 'major', linewidth=.5, zorder=0)
    sns.despine(fig, left=True, bottom=True)
    ax.set(box_aspect=1)
    ax.axline((1, ratio), (100, 100 * ratio), lw=1,
              alpha=.5, zorder=1)
    ax.legend(frameon=False, bbox_to_anchor=(1, 1),
              title='set bias', loc='upper left')
    fig.tight_layout()
    return fig


def plot_ratio_degree(positives: pd.DataFrame,
                      negatives: pd.DataFrame = None,
                      ratio: float = 10,
                      taxonomy: pd.DataFrame = None,
                      ) -> Tuple[Figure, Union[None, Figure], pd.DataFrame]:
    """
    Calculate the similarity between two sets of protein pairs:
    the Spearman or Pearson correlation coefficient between their
    protein-appearance frequency vectors.
    """
    # TODO test with homodimers to make sure this does not return
    #  node *degrees*, but the ratio of negative to positive edges
    positives, negatives = positives.copy(), negatives.copy()
    plus, minus = split_pos_neg_plus_minus(positives, negatives)
    new_negatives = minus.keys() - plus.keys()
    minus.update({k: 0 for k in plus.keys() - minus.keys()})
    sp_lookup = positives[['hash_A', 'hash_B', 'species']].melt(
        id_vars='species')[['value', 'species']].drop_duplicates().set_index(
        'value').to_dict()['species']
    df = pd.DataFrame.from_records([(k, v, minus[k] / v, sp_lookup[k])
                                    for k, v in plus.items()],
                                   columns=['crc_hash', 'degree', 'ratio', 'species'])
    df['species'] = pd.Categorical(df.species)
    g = sns.JointGrid(data=df, x='ratio', y='degree', hue='species',
                      marginal_ticks=True, height=5, space=0, ratio=6,
                      palette='colorblind')
    g.plot_joint(sns.scatterplot, legend=False,
                 s=25, alpha=.8
                 )
    g.plot_marginals(sns.kdeplot, warn_singular=False)
    g.ax_joint.set(  # xlim=(ratio - 4, ratio + 4),
        yscale='log', xscale='linear',
        xlabel='ratio −:+ interactions',
        ylabel='node degree')
    # g.ax_joint.axvline(x=ratio, lw=1, alpha=.5, zorder=0)
    # g.ax_marg_x.axvline(x=ratio, lw=1, alpha=.5, zorder=0)
    g.refline(x=ratio, lw=1, alpha=.5, zorder=0)
    g.ax_marg_x.set(yticks=[])
    g.ax_marg_y.set(xticks=[])
    g.figure.set_facecolor('white')
    sns.despine(ax=g.ax_marg_x, left=True)
    sns.despine(ax=g.ax_marg_y, bottom=False)

    if not new_negatives:
        return g.figure, None, df

    nsp = (negatives[['hash_A', 'hash_B', 'species']]
           .melt(id_vars='species', value_name='crc_hash')
           [['crc_hash', 'species']].value_counts().reset_index()
           .rename(columns={0: 'degree'}))
    nsp['species'] = pd.Categorical(nsp.species)
    nsp['kind'] = nsp.crc_hash.apply(lambda crc: 'proteome'
    if crc in new_negatives else 'interactome')

    med_degrees = nsp.loc[nsp.kind == 'interactome'].groupby(
        'species')['degree'].median().to_dict()
    avg_degrees = nsp.loc[nsp.kind == 'interactome'].groupby(
        'species')['degree'].mean().to_dict()
    if taxonomy is None:
        order = sorted(set(nsp.species))
    else:
        order, names = [list(ar) for ar in taxonomy.loc[taxonomy.species.isin(
            set(nsp.species)), ['species', 'name']].values.T]

    h = sns.catplot(data=nsp,
                    x='degree',
                    y='species',
                    hue='kind',
                    dodge='True',
                    order=order,
                    orient='h',
                    jitter=.2,
                    height=5,
                    aspect=1,
                    s=1.4,
                    alpha=.3,
                    # palette='colorblind',
                    legend=False,
                    )
    h.set(xscale='log')
    for y, sp in enumerate(order):
        h.ax.text(avg_degrees[sp], y, '|', va='center', ha='center')
        h.ax.text(med_degrees[sp], y, '·', va='center', ha='center')

    h.ax.legend(frameon=False, title='', loc=(.15, 1), ncol=2, markerscale=.6)
    h.set(xlabel='number of negative interactions per protein')
    if taxonomy is not None:
        h.set(ylabel='', yticklabels=names)

    h.figure.set_facecolor('white')
    h.despine(left=True)
    h.despine(bottom=False)

    return g.figure, h, nsp


def plot_ratio_grids(df: pd.DataFrame, order: List = None, ratio: float = 10.0) -> Tuple[Figure, Figure]:
    g = sns.relplot(data=df,
                    x='ratio', y='degree',  # hue='species',
                    col='species', col_wrap=5,
                    col_order=order,
                    height=1.5,
                    alpha=.4,
                    )
    g.set(box_aspect=1, yscale='log', xlabel='', ylabel='')
    g.refline(x=ratio, lw=1, alpha=.5, zorder=0)
    g.set_titles(col_template='{col_name}')
    g.tight_layout()
    g.figure.set_facecolor('white')
    g.axes[-5].set(xlabel='ratio -:+ interactions', ylabel='degree')
    # g.savefig('train_ratio_degree_grid.png', dpi=300, transparent=False)

    h = sns.displot(data=df, kind='hist',
                    x='ratio', y='degree',  # hue='species',
                    col='species', col_wrap=5,
                    col_order=order,
                    height=1.5,
                    # alpha=.4,
                    # rug=True,
                    stat='density',
                    pmax=.07,
                    bins=40,
                    discrete=(True, False),
                    log_scale=(False, True)
                    )
    h.set(box_aspect=1,  # yscale='log',
          xlabel='', ylabel='')
    # h.refline(x=c.ratio, lw=1, alpha=.5, zorder=0)
    h.set_titles(col_template='{col_name}')
    h.tight_layout()
    h.figure.set_facecolor('white')
    h.axes[-5].set(xlabel='ratio -:+ interactions', ylabel='degree')
    # h.savefig('train_ratio_degree_grid_hist.png', dpi=300, transparent=False)

    return g, h


@mpl.style.context('seaborn')
def plot_test_ratios(test_all: pd.DataFrame, ratio: float = 10.0,
                     flip: bool = False) -> Figure:
    pairs = fetch_ratios(test_all)
    species = list(pairs.species.value_counts().index)[:5]

    shape = (len(species), 3)
    size = (8, 3 * len(species))
    if flip:
        shape, size = shape[::-1], size[::-1]

    fig, axes = plt.subplots(*shape, figsize=size,
                             sharey='row' if not flip else 'col',
                             sharex='row' if not flip else 'col')
    # if shape[0] == 1:
    #     axes = axes.reshape(axes.shape[0], 1)
    for row, sp in enumerate(species):
        # ymax = max(pairs.loc[(pairs.species == sp)
        #                      & (pairs.label == 1) & (pairs.x == 0), 'degree'])
        # xmax = max(pairs.x)
        # return pairs.loc[pairs.species == sp]
        axes_now = (axes[row, :] if not flip else axes[:, row]) \
            if len(species) > 1 else axes
        for cclass, ax in enumerate(axes_now):
            dat = pairs.loc[(pairs.cclass == cclass + 1)
                            & (pairs.species == sp)]
            for label, (color, name) in enumerate(zip(
                    ('#D81B60', '#1E88E5'), ('−', '+'))):
                dat_ = dat.loc[dat.label == label]
                ax.plot(dat_.x, dat_.degree, color=color, label=name)
            ax.plot(dat_.x, ratio * dat_.degree, color='.5',
                    alpha=.5, zorder=0, label=f'{ratio} × +')
            ax.set(yscale='linear', xscale='log', box_aspect=1)
            #        xlim=(1, xmax),
            #        ylim=(1, int(ratio * ymax + 1)))
            if row == 0 and not flip:
                ax.set(title=f'C{cclass + 1}')
            elif row == len(species) - 1 and flip:
                ax.set_title(f'C{cclass + 1}', y=.5, loc='right')

    ax1, ax2 = (axes[-1, 0], axes[0, -1]) if len(species) > 1 else (axes[0], axes[-1])
    ax1.set(xlabel='degree rank', ylabel='node degree')
    ax2.legend(frameon=False)
    fig.subplots_adjust(hspace=.2 if flip else .05, wspace=.2 if flip else .1)
    # fig.tight_layout()
    raise DeprecationWarning('Proteins by themselves do not have a C123 '
                             'class but are assigned one here! '
                             'PPIs belong to a class.')
    return fig


@mpl.style.context('seaborn-whitegrid')
def plot_c_classes(df: pd.DataFrame) -> Tuple[Figure, Dict[int, int]]:
    fig, ax = plt.subplots(figsize=(4, 3), facecolor='white')
    bg = sns.countplot(
        x=df.cclass,
        palette={1: '#FF669D', 2: '#71BDFF', 3: '#FFE083'}, ax=ax)
    fg = sns.countplot(
        x=df.loc[df.label == 1, 'cclass'],
        palette={1: '#D81B60', 2: '#1E88E5', 3: '#FFC107'}, ax=ax)
    vc = df.loc[df.label == 1, 'cclass'].value_counts().sort_index()
    ax.bar_label(container=ax.containers[1], labels=vc)
    ax.set(box_aspect=1)
    sns.despine(ax=ax, left=True, bottom=True, right=True)
    a2b = lambda y: y / len(df.cclass)
    b2a = lambda y: len(df.cclass) * y
    ax2 = ax.secondary_yaxis('right', functions=(a2b, b2a))
    ax2.set(ylabel='share')
    sns.despine(ax=ax2, left=True, right=True)
    return fig, dict(vc)
