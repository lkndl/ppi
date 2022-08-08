from __future__ import annotations

from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torchmetrics as tm
from torch.utils.tensorboard import SummaryWriter

from utils.general_utils import device


class Metrics:
    acc: tm.Accuracy
    pr: tm.Precision
    re: tm.Recall
    f1: tm.F1Score
    aupr: tm.AveragePrecision
    prc: tm.PrecisionRecallCurve
    loss: tm.MeanMetric
    p0: tm.MeanMetric
    p1: tm.MeanMetric
    mcc: tm.MatthewsCorrCoef
    conf: tm.ConfusionMatrix

    def __init__(self):
        for name, cls in zip(
                ['acc', 'pr', 're', 'f1',
                 'aupr', 'prc', 'loss', 'p0', 'p1', 'mcc'],
                [tm.Accuracy, tm.Precision, tm.Recall, tm.F1Score,
                 tm.AveragePrecision, tm.PrecisionRecallCurve, *[tm.MeanMetric] * 3]):
            setattr(self, name, cls().to(device))
        self.mcc = tm.MatthewsCorrCoef(2).to(device)
        self.conf = tm.ConfusionMatrix(2).to(device)
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
            getattr(self, name).load_state_dict(state_dict)

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
                d[name] = metric(preds, labels.int())
        if loss is not None:
            d['loss'] = self.loss(loss.clone().detach())
        pd = dict()
        for i, metric in enumerate([self.p0, self.p1]):
            t = preds[labels == i]
            if not t.numel():
                continue
            pd[str(i)] = metric(t)
        if pd:
            d['p_hat'] = pd
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

    def add_interval(self, d: dict, interval: str, idx: int):
        plt.style.use('default')
        if (conf := d.pop('conf', None)) is not None:
            if type(conf) == dict:
                fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.8),
                                         sharey=True, sharex=True)
                for i, ax in enumerate(axes):
                    c = f'C{i + 1}'
                    cm = conf[c].cpu().float()
                    cm /= cm.sum(dim=-1).view(-1, 1).clamp(min=1)
                    ax.imshow(cm, cmap='gray_r')
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
                ax.imshow(cm, cmap='gray_r')
                ax.set(xticks=[0, 1], yticks=[0, 1], xlabel='prediction', ylabel='label')
                for y in [0, 1]:
                    for x in [0, 1]:
                        c = cm[y, x]
                        ax.text(x, y, f'{c:.2g}', ha='center', va='center', color=f'{c.round().item()}')
            fig.tight_layout()
            self.add_figure(f'conf/{interval}', fig, idx)
        d.pop('conf', None)

        if (prc := d.pop('prc', None)) is not None:
            if type(prc) == dict:
                dfs = list()
                for cclass, ts in prc.items():
                    df = pd.DataFrame(t.cpu().numpy() for t in ts).T
                    df = df.rename(columns={0: 'precision', 1: 'recall', 2: 'thresholds'})
                    df['cclass'] = cclass
                    dfs.append(df)
                df = pd.concat(dfs).reset_index()
            else:
                df = pd.DataFrame(t.cpu().numpy() for t in prc).T
                df = df.rename(columns={0: 'precision', 1: 'recall', 2: 'thresholds'})
                df['cclass'] = 'train'

            plt.style.use('default')
            sns.set_theme()
            fig = sns.relplot(kind='line',
                              data=df,
                              x='recall', y='precision',
                              hue='cclass',
                              aspect=1, height=2.8,
                              ci=None, drawstyle='steps-post')
            t = [0, .25, .5, .75, 1]
            tl = ['0', '.25', '.5', '.75', '1']
            fig.set(xlabel='recall', ylabel='precision', box_aspect=1,
                    xlim=(-.1, 1.1), ylim=(-.1, 1.1),
                    xticks=t, yticks=t, xticklabels=tl, yticklabels=tl)
            fig.tight_layout()
            self.add_figure(f'prc/{interval}', fig.fig, idx)
        d.pop('prc', None)

        for k, v in d.items():
            if type(v) != dict:
                self.add_scalar(f'{k}/{interval}', v, idx)
            else:
                self.add_scalars(f'{k}/{interval}', v, idx)
