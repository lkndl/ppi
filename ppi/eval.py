from __future__ import annotations

from time import perf_counter

import torch
import torch.nn as nn
from rich.progress import Progress
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppi.metrics import Metrics, Writer
from train.interaction import InteractionMap
from utils import general_utils as utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Intervalometer:
    start_time: int
    start_epoch: float = 0.
    start_batch: int = 0

    train_batches: int = 1000

    time: int = 60 * 10  # 10 minutes
    epochs: float = 2.
    batches: int = 100  # evaluate once every x train batches

    def __init__(self, **kwargs):
        self.start_time = int(perf_counter())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def restart(self, **kwargs):
        self.start_time = int(perf_counter())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, batch: int = 0) -> bool:
        epoch = (batch - self.start_batch) / self.train_batches + self.start_epoch
        do_it_now = False
        if epoch and epoch - self.start_epoch > self.epochs:
            do_it_now = 'epoch'
        if int(perf_counter()) - self.start_time > self.time:
            do_it_now = 'time'
        if batch and batch - self.start_batch > self.batches:
            do_it_now = 'batch'
        if not do_it_now:
            return False
        self.restart(start_epoch=epoch, start_batch=batch - 1)
        return do_it_now


class Evaluator:
    impatience: int = 0
    patience: int = 20

    def __init__(self, cfg, model: InteractionMap,
                 optim: Adam, writer: Writer,
                 train_loader: DataLoader,
                 dataloaders: dict[str, DataLoader],
                 embeddings: dict[str, torch.Tensor],
                 progress: Progress):
        self.cfg = cfg
        self.model = model
        self.optim = optim
        self.writer = writer
        self.train_loader = train_loader
        self.dataloaders = dataloaders
        self.embeddings = embeddings
        self.progress = progress
        self.interval = Intervalometer(
            start_epoch=cfg.get('start_epoch', 0.),
            start_batch=cfg.get('start_batch', 0),
            time=cfg.eval_time_interval, epochs=cfg.eval_epoch_interval,
            train_batches=len(train_loader),
            batches=len(self) * cfg.eval_train_ratio)
        self.metrics = Metrics()
        self.results = dict()

    def __len__(self):
        return sum([len(loader) for loader in self.dataloaders.values()])

    def evaluate_model(self, train_batch: int = 0):
        model, embeddings, writer, metrics = \
            self.model, self.embeddings, self.writer, self.metrics
        results = dict()
        task_ids = list()

        model.eval()
        with torch.no_grad():
            for (cclass, dataloader), color in zip(
                    self.dataloaders.items(), ['blue', 'red', 'yellow']):
                if self.progress is not None:
                    task = self.progress.add_task(f'[{color}]C{cclass}', total=len(dataloader))
                    task_ids.append(task)
                    tracker, tracker_kwargs = self.progress.track, dict(task_id=task)
                else:
                    tracker, tracker_kwargs = tqdm, dict(desc=f'C{cclass}', colour=color, leave=False)

                for n0, n1, labels in tracker(dataloader, **tracker_kwargs):
                    preds = list()
                    for _n0, _n1 in zip(n0, n1):
                        z_a = embeddings[_n0].to(device)
                        z_b = embeddings[_n1].to(device)

                        p_hat = model.predict_func(z_a, z_b)
                        preds.append(p_hat.flatten(0))

                    preds = torch.cat(preds).to(device)
                    labels = labels.float().to(device)

                    w = 1 + labels * (self.cfg.ppi_weight - 1)
                    loss = nn.BCELoss(weight=w)(preds, labels)

                    # update the metrics with this batch
                    metrics(preds, labels, loss)

                results[cclass] = metrics.compute()
                metrics.reset()

            if self.progress is not None:
                [self.progress.update(task_id=task, visible=False) for task in task_ids]

            cclass_metrics = dict()
            for c, c_dict in results.items():
                for metric, value in c_dict.items():
                    cclass_metrics[metric] = cclass_metrics.get(metric, dict()) | {f'C{c}': value}
            cclass_metrics['p_hat'] = {f'{cclass}_{label}': v for cclass, d in
                                       cclass_metrics['p_hat'].items() for
                                       label, v in d.items()}
            writer.add_interval(cclass_metrics, f'val', train_batch)
            self.results[train_batch] = cclass_metrics

            # TODO keep track of best checkpoint, and keep that checkpoint up-to-date.
            #  do not checkpoint at every eval just for the fun of it
            # self.checkpoint(file_name='chk_best.tar', batch=train_batch, eval_results=results)
            self.checkpoint(file_name=f'chk_eval_{train_batch}.tar', batch=train_batch, eval_results=results)

        model.train()
        return False

    def checkpoint(self, file_name: str, **kwargs):
        # fetch the current dataloader states
        d = dict(train=self.train_loader.sampler.get_state()) | \
            {c: dl.sampler.get_state() for c, dl in self.dataloaders.items()}
        # TODO save current patience and evaluation loss for resuming

        # save the checkpoint
        utils.checkpoint(self.model, self.optim, self.cfg.wd / self.cfg.name / file_name,
                         dataloader_states=d, epochs=self.cfg.epochs, **kwargs)

    def __call__(self, batch: int = 0) -> bool:
        # check what'sin dataloaders

        # test all three classes

        # write to the tensorboard

        # if this is so far the best: save the state dict
        # (or even the checkpoint?)

        # if this is not the best: become impatient

        # if this is not the best and we're fresh out of patience:
        # publish previous best as state dict or checkpoint

        self.impatience += 1
        if self.impatience >= self.patience:
            return True

        return False
