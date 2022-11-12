from __future__ import annotations

from time import perf_counter

import torch
import torch.nn as nn
from rich.progress import Progress
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from ppi.metrics import Metrics, Writer, pivot
from ppi.train.interaction import InteractionMap
from ppi.utils import general_utils as utils
from ppi.utils.general_utils import device


class Intervalometer:
    start_time: int
    start_epoch: float = 0.
    start_batch: int = 0

    train_batches: int = 1000

    time: int = 60 * 10  # 10 minutes
    epochs: float = 2.
    batches: int = 100  # evaluate once every x train batches

    def __init__(self, cfg, train_loader: DataLoader,
                 dataloaders: dict[str, DataLoader], **kwargs):
        self.start_epoch = cfg.get('start_epoch', 0.)
        self.start_batch = cfg.get('start_batch', 0)
        self.time = cfg.eval_time_interval
        self.epochs = cfg.eval_epoch_interval
        self.train_batches = len(train_loader)
        self.batches = cfg.eval_train_ratio * sum([len(loader) for loader in dataloaders.values()])
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.start_time = int(perf_counter())

    def restart(self, **kwargs):
        self.start_time = int(perf_counter())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, batch: int) -> bool:
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

    def __init__(self, cfg, model: InteractionMap = None,
                 optim: Adam = None, writer: Writer = None,
                 train_loader: DataLoader = None,
                 dataloaders: dict[str, DataLoader] = None,
                 embeddings: dict[str, torch.Tensor] = None,
                 progress: Progress = None,
                 intervalometer: Intervalometer = None,
                 metrics: Metrics = None,
                 ):
        self.cfg = cfg
        self.model = model
        self.optim = optim
        self.writer = writer
        self.train_loader = train_loader
        self.dataloaders = dataloaders
        self.embeddings = embeddings
        self.progress = progress
        self.intervalometer = intervalometer
        self.metrics = metrics

    def __len__(self):
        return sum([len(loader) for loader in self.dataloaders.values()])

    def __call__(self, train_batch: int = 0):
        finished, results, times = evaluate_model(
            self.model, self.embeddings, self.dataloaders, progress=self.progress)
        self.writer.add_interval(pivot(results), 'val', train_batch)
        self.writer.flush()
        self.checkpoint(file_name=f'chk_eval_{train_batch}.tar',
                        batch=train_batch, eval_results=results)
        self.intervalometer.restart()
        self.model.train()
        return finished

    def checkpoint(self, file_name: str, **kwargs):
        # fetch the current dataloader states
        d = dict(train=self.train_loader.sampler.get_state()) | \
            {c: dl.sampler.get_state() for c, dl in self.dataloaders.items()}
        # save the checkpoint
        utils.checkpoint(self.model, self.optim, self.cfg.wd / self.cfg.name / file_name,
                         dataloader_states=d, metrics_state=self.metrics.get_state(),
                         epochs=self.cfg.epochs, ppi_weight=self.model.ppi_weight, **kwargs)


def evaluate_model(model, embeddings, dataloaders,
                   bootstraps: int = 0, progress: Progress = None
                   ) -> tuple[pd.DataFrame, dict, dict]:
    results = dict()
    task_ids = list()
    times = dict()

    model.eval()
    with torch.no_grad():
        all_preds, all_labels, all_cc, all_A, all_B = list(), list(), list(), list(), list()
        for (cclass, dataloader), color in zip(dataloaders.items(), ['blue', 'red', 'yellow']):
            # print(f'C{cclass}:{len(dataloader)}')

            if progress is not None:
                task = progress.add_task(f'[{color}]C{cclass}', total=len(dataloader))
                task_ids.append(task)
                tracker, tracker_kwargs = progress.track, dict(task_id=task)
            else:
                tracker, tracker_kwargs = tqdm, dict(desc=f'C{cclass}', colour=color, leave=False)

            metrics = Metrics(bootstraps)
            start_time = perf_counter()
            for n0, n1, labels in tracker(dataloader, **tracker_kwargs):
                preds = list()
                for _n0, _n1 in zip(n0, n1):
                    z_a = embeddings[_n0].to(device)
                    z_b = embeddings[_n1].to(device)
                    all_A.append(_n0)
                    all_B.append(_n1)

                    p_hat = model.predict_func(z_a, z_b)
                    preds.append(p_hat.flatten(0))

                preds = torch.cat(preds).to(device)
                labels = labels.float().to(device)
                all_preds.extend(list(preds.clone().cpu().numpy()))
                all_labels.extend(list(labels.clone().cpu().numpy().astype(int)))
                all_cc.extend([cclass] * len(labels))

                w = 1 + labels * (model.ppi_weight - 1)
                loss = nn.BCELoss(weight=w)(preds, labels)

                metrics(preds, labels, loss, keep_all=True)

            times[cclass] = (perf_counter() - start_time) / len(dataloader)
            results[cclass] = metrics.compute()
            del metrics

        if progress is not None:
            [progress.update(task_id=task, visible=False) for task in task_ids]
    df = pd.DataFrame((all_A, all_B, all_preds, all_labels, all_cc)).T
    df.columns = ['hash_A', 'hash_B', 'p_hat', 'label', 'cclass']

    # TODO keep track of best checkpoint, and keep that checkpoint up-to-date.
    #  do not checkpoint at every eval just for the fun of it
    return df, results, times
