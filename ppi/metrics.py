from __future__ import annotations

import sys
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torchmetrics as tm
import torchmetrics.classification as tmc
from torch.utils.tensorboard import SummaryWriter

from ppi.utils.general_utils import device


# TODO this is dumb. use a tm.MetricCollection

class Metrics:
    class _prc(tmc.BinaryPrecisionRecallCurve):
        """This returns a tuple, which we can't easily stack -> Tensorify"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, thresholds=101, **kwargs)

        def compute(self):
            pr, re, th = tmc.BinaryPrecisionRecallCurve.compute(self)
            return torch.stack((pr.float(), re.float(),
                                torch.cat((th.float(), torch.Tensor([1.]).to(device)))))

    class _mcc(tmc.BinaryMatthewsCorrCoef):
        """The values may not be floats for some reason."""

        def compute(self):
            return tmc.BinaryMatthewsCorrCoef.compute(self).float()

    class _conf(tmc.BinaryConfusionMatrix):
        """The values may not be floats for some reason."""

        def compute(self):
            return tmc.BinaryConfusionMatrix.compute(self).float()

    acc: tmc.BinaryAccuracy
    pr: tmc.BinaryPrecision
    re: tmc.BinaryRecall
    f1: tmc.BinaryF1Score
    aupr: tmc.BinaryAveragePrecision
    prc: _prc
    mcc: _mcc
    conf: _conf
    loss: tm.MeanMetric
    p0: tm.MeanMetric
    p1: tm.MeanMetric

    def __init__(self, bootstraps: int = 0):
        for name, cls in zip(
                ['acc', 'pr', 're', 'f1',
                 'aupr', 'prc', 'mcc', 'conf',
                 'loss', 'p0', 'p1'],
                [tmc.BinaryAccuracy, tmc.BinaryPrecision, tmc.BinaryRecall,
                 tmc.BinaryF1Score, tmc.BinaryAveragePrecision,
                 Metrics._prc, Metrics._mcc, Metrics._conf, *[tm.MeanMetric] * 3]):
            setattr(self, name, cls().to(device))
        for _, metric in self:
            metric.persistent(True)

    def __iter__(self):
        for name, metric in vars(self).items():
            yield name, metric

    def reset(self):
        for _, metric in self:
            metric.reset()

    def load_state(self, state: dict = None):
        if state is None:
            return
        for name, state_dict in state.items():
            try:
                getattr(self, name).load_state_dict(state_dict)
            except RuntimeError:
                print(f'Failed loading {name} state dict')

    def get_state(self) -> dict[str, dict]:
        return {name: metric.state_dict() for name, metric in self}

    def compute(self):
        d = dict()
        for name, metric in self:
            d[name] = metric.compute()
        d['p_hat'] = {'0': d.pop('p0'), '1': d.pop('p1')}
        return d

    def __call__(self, preds: torch.Tensor, labels: torch.Tensor,
                 loss: Union[None, torch.Tensor] = None,
                 keep_all: bool = False) -> dict[str, float]:
        d = dict()
        preds = preds.clone().detach()
        for name, metric in self:
            if type(metric) != tm.MeanMetric:
                d[name] = metric(preds.float(), labels.int())
        if loss is not None:
            d['loss'] = self.loss(loss.clone().detach())
        p_d = dict()
        for i, (metric, name) in enumerate(zip([self.p0, self.p1], ['neg', 'pos'])):
            t = preds[labels == i]
            if not t.numel():
                continue
            p_d[str(i)] = metric(t)
            d[name] = t
        if p_d:
            d['p_hat'] = p_d
        if 'aupr' in d and torch.isnan(d['aupr']) and not keep_all:
            d.pop('aupr')
        if not keep_all:
            d.pop('conf')
            d.pop('prc')
        return d

    @staticmethod
    def pivot(cclass_metrics: dict[str, dict]) -> dict[str, dict]:
        """Transform a {C1,C2,C3} -> {ACC, MCC, ...} dict of results to {ACC, ...} -> {C1: ..., C2: }"""
        m = dict()
        for c, c_dict in cclass_metrics.items():
            for metric, value in c_dict.items():
                m[metric] = m.get(metric, dict()) | {f'C{c}': value}
        m['p_hat'] = {f'{cclass}_{label}': v for cclass, d in m['p_hat'].items()
                      for label, v in d.items()}
        return m


class Writer(SummaryWriter):

    def __init__(self, *args, **kwargs):
        super(Writer, self).__init__(*args, **kwargs)

    def plot_cm(self, d: dict, interval: str, idx: int):
        plt.style.use('default')
        if (conf := d.pop('conf', None)) is not None:
            # for validation intervals, draw a confusion matrix and a PR curve
            if type(conf) == dict:
                fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.8),
                                         sharey=True, sharex=True)
                for i, ax in enumerate(axes):
                    c = f'C{i + 1}'
                    cm = conf[c].cpu().float()
                    cm /= cm.sum(dim=-1).view(-1, 1).clamp(min=1)
                    ax.imshow(cm, cmap='gray_r', vmin=0, vmax=1)
                    ax.set(title=c, xticks=[0, 1], yticks=[0, 1])
                    for y in [0, 1]:
                        for x in [0, 1]:
                            c = cm[y, x]
                            ax.text(x, y, f'{c:.2g}', ha='center', va='center', color=f'{c.round().item()}')
                    if not i:
                        ax.set(xlabel='prediction', ylabel='label')
            else:
                fig, ax = plt.subplots(1, 1, figsize=(1.8, 1.8))
                cm = conf.cpu().float()
                cm /= cm.sum(dim=-1).view(-1, 1).clamp(min=1)
                ax.imshow(cm, cmap='gray_r', vmin=0, vmax=1)
                ax.set(xticks=[0, 1], yticks=[0, 1], xlabel='prediction', ylabel='label')
                for y in [0, 1]:
                    for x in [0, 1]:
                        c = cm[y, x]
                        ax.text(x, y, f'{c:.2g}', ha='center', va='center', color=f'{c.round().item()}')
            fig.tight_layout()
            self.add_figure(f'conf/{interval}', fig, idx)
        d.pop('conf', None)

    def plot_prc(self, d: dict, interval: str, idx: int):
        plt.style.use('default')
        if (prc := d.pop('prc', None)) is not None:
            if type(prc) == dict:
                dfs = list()
                for cclass, ts in prc.items():
                    df = pd.DataFrame(t.cpu().squeeze().view(-1).numpy() for t in ts).T
                    df = df.rename(columns={0: 'precision', 1: 'recall', 2: 'thresholds'})
                    df['cclass'] = cclass
                    dfs.append(df)
                df = pd.concat(dfs).reset_index()
            else:
                df = pd.DataFrame(t.cpu().squeeze().view(-1).numpy() for t in prc).T
                df = df.rename(columns={0: 'precision', 1: 'recall', 2: 'thresholds'})
                df['cclass'] = 'train'

            plt.style.use('default')
            sns.set_theme()
            fig = sns.relplot(kind='line',
                              data=df,
                              x='recall', y='precision',
                              hue='cclass',
                              aspect=1, height=2.8,
                              errorbar=None, drawstyle='steps-post')
            t = [0, .25, .5, .75, 1]
            tl = ['0', '.25', '.5', '.75', '1']
            fig.set(xlabel='recall', ylabel='precision', box_aspect=1,
                    xlim=(-.1, 1.1), ylim=(-.1, 1.1),
                    xticks=t, yticks=t, xticklabels=tl, yticklabels=tl)
            fig.tight_layout()
            self.add_figure(f'prc/{interval}', fig.fig, idx)
        d.pop('prc', None)

    def add_interval(self, d: dict, interval: str, idx: int):
        if 'quirin' not in sys.executable:
            self.plot_cm(d, interval, idx)
            self.plot_prc(d, interval, idx)
        d.pop('conf', None)
        d.pop('prc', None)

        for k, v in d.items():
            if k in ['neg', 'pos']:
                if interval != 'batch' or not idx % 100:
                    self.add_histogram(f'p_hat_{k}/{interval}', v, idx)
            elif type(v) != dict:
                self.add_scalar(f'{k}/{interval}', v, idx)
            elif k != 'p_hat':
                self.add_scalars(f'{k}/{interval}', v, idx)
