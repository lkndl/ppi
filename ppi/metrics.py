from __future__ import annotations

from typing import Union

import torch
import matplotlib.pyplot as plt
import torchmetrics as tm
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Metrics:
    acc: tm.Accuracy
    pr: tm.Precision
    re: tm.Recall
    f1: tm.F1Score
    aupr: tm.AveragePrecision
    loss: tm.MeanMetric
    p0: tm.MeanMetric
    p1: tm.MeanMetric
    mcc: tm.MatthewsCorrCoef
    conf: tm.ConfusionMatrix

    def __init__(self):
        for name, cls in zip(
                ['acc', 'pr', 're', 'f1',
                 'aupr', 'loss', 'p0', 'p1', 'mcc'],
                [tm.Accuracy, tm.Precision, tm.Recall, tm.F1Score,
                 tm.AveragePrecision, *[tm.MeanMetric] * 3]):
            setattr(self, name, cls().to(device))
        self.mcc = tm.MatthewsCorrCoef(2).to(device)
        self.conf = tm.ConfusionMatrix(2).to(device)

    def __iter__(self):
        for name, metric in vars(self).items():
            yield name, metric

    def reset(self):
        for _, metric in self:
            metric.reset()

    def compute(self):
        d = dict()
        for name, metric in self:
            d[name] = metric.compute()
        d['p_hat'] = {'0': d.pop('p0'), '1': d.pop('p1')}
        return d

    def __call__(self, preds: torch.Tensor, labels: torch.Tensor,
                 loss: Union[None, torch.Tensor] = None) -> dict[str, float]:
        d = dict()
        for name, metric in self:
            if type(metric) != tm.MeanMetric:
                d[name] = metric(preds, labels.int())
        if loss is not None:
            d['loss'] = self.loss(loss)
        pd = dict()
        for i, metric in enumerate([self.p0, self.p1]):
            t = preds[labels == i]
            if not t.numel():
                continue
            pd[str(i)] = metric(t)
        if pd:
            d['p_hat'] = pd
        if torch.isnan(d['aupr']):
            d.pop('aupr')
        d.pop('conf')
        return d


class Writer(SummaryWriter):

    def __init__(self, *args, **kwargs):
        super(Writer, self).__init__(*args, **kwargs)

    def add_interval(self, d: dict, interval: str, idx: int):
        if (conf := d.pop('conf', None)) is not None:
            if type(conf) == dict:
                fig, axes = plt.subplots(1, 3, figsize=(3, 1),
                                         sharey=True, sharex=True)
                for i, ax in enumerate(axes):
                    c = f'C{i + 1}'
                    ax.imshow(conf[c].cpu())
                    ax.set(title=c, xticks=[0, 1], yticks=[0, 1])
            else:
                fig, ax = plt.subplots(1, 1, figsize=(1, 1))
                ax.imshow(conf.cpu())
                ax.set(xticks=[0, 1], yticks=[0, 1])
            fig.tight_layout()
            self.add_figure(f'conf/{interval}', fig, idx)

        for k, v in d.items():
            if type(v) != dict:
                self.add_scalar(f'{k}/{interval}', v, idx)
            else:
                self.add_scalars(f'{k}/{interval}', v, idx)
