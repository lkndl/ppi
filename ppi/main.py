from __future__ import annotations

import contextlib
import warnings
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import pkg_resources
import rich
import torch
import torch.nn as nn
import typer
from Bio import SeqIO
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TaskProgressColumn
)
from torch.optim import Adam
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=UserWarning)

from ppi.metrics import Writer, Metrics
from ppi.eval import Evaluator
from ppi.config import parse

from train.interaction import InteractionMap
from utils import general_utils as utils
from utils.general_utils import device
from utils.dataloader import (
    DataLoader,
    get_dataloaders_and_ids,
    get_embeddings
)

__version__ = pkg_resources.get_distribution('ppi').version

app = typer.Typer()


@app.command()
def hello(name: str):
    typer.echo(f'Hello {name}')


@app.command()
def goodbye(name: str, formal: bool = False):
    if formal:
        typer.echo(f'Goodbye Ms. {name}. Have a good day.')
    else:
        typer.echo(f'Bye {name}!')


class H5WriteMode(str, Enum):
    exit = 'exit'
    overwrite = 'w'
    append = 'a'


@app.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
def embed(ctx: typer.Context,
          fasta: Path = 'train.fasta',
          h5_out: Path = 'all.h5',
          h5_mode: H5WriteMode = H5WriteMode.exit) -> None:
    cfg = parse(ctx, write=False)

    seqs = SeqIO.to_dict(SeqIO.parse(fasta, 'fasta'))

    if not h5_out.is_file():
        h5_mode = H5WriteMode.overwrite
    elif h5_mode == H5WriteMode.exit:
        raise FileExistsError(f'H5 file {h5_out} exists. Choose different path, '
                              f'or define append/overwrite mode')
    with h5py.File(h5_out, h5_mode) as new_h5, \
            h5py.File('/mnt/project/kaindl/ppi/mem_leak/smaller.h5', 'r') as old_h5:
        for seq_id in rich.progress.track(seqs, total=len(seqs)):
            if seq_id in old_h5:
                if seq_id in new_h5:
                    if not np.allclose(new_h5[seq_id], old_h5[seq_id]):
                        warnings.warn(f'h5py Dataset {seq_id} already exists'
                                      ' and values differ!', RuntimeWarning)
                    continue
                new_h5[seq_id] = np.array(old_h5[seq_id])


class WriteMode(str, Enum):
    none = None
    exit = 'exit'
    overwrite = 'w'
    append = 'a'


def predict(model: InteractionMap, dataloader: DataLoader,
            embeddings: dict[str, torch.Tensor],
            mode: WriteMode = WriteMode.none, cclass: str = '',
            ) -> tuple[torch.Tensor, torch.Tensor]:
    tsv = Path('predict.tsv')
    if mode == WriteMode.exit and tsv.is_file():
        exit(f'{tsv} exists')
    elif mode.value in 'wa':
        tsv = tsv.open(mode)

    if cclass := cclass.strip():
        cclass = f'\t{cclass}\t'

    if mode.value == 'w':
        line = f'batch\thash_A\thash_B\t' + ('cclass\t' if cclass else '') + f'y\tphat\n'
        print(line, end='')
        tsv.write(line)

    model.eval()
    with torch.no_grad():
        preds, all_labels, i_maps = [], [], []
        for batch, (n0, n1, labels) in enumerate(dataloader):
            all_labels.append(labels)

            for _n0, _n1, _y in zip(n0, n1, labels):
                z_a = embeddings[_n0].to(device)
                z_b = embeddings[_n1].to(device)
                # i_map, p_hat, yhat = model.map_predict_modified(z_a, z_b)
                # i_maps.append(torch.mean(i_map))
                p_hat = model.predict_func(z_a, z_b)

                line = f'{batch}\t{_n0}\t{_n1}{cclass}{_y.item()}\t{p_hat.item():.4f}\n'
                # if phat > .5:
                print(line, end='')
                if mode.value in 'wa':
                    tsv.write(line)

                preds.append(p_hat.flatten(0))

    if mode.value in 'wa':
        tsv.close()
    return torch.cat(preds), torch.cat(all_labels)


@app.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
def evaluate(ctx: typer.Context,
             model: Path = 'model.tar',
             tsv: Path = 'test.tsv',
             h5: Path = 'all.h5',
             ):
    cfg = parse(ctx)
    batch_size = 5

    # load the test, separated by class
    dataloaders, seq_ids = get_dataloaders_and_ids(
        tsv, batch_size, augment=False, shuffle=False, split_column='cclass')
    embeddings = get_embeddings(h5, seq_ids)

    path = model
    model = InteractionMap(**vars(cfg))
    if path.suffix == '.tar':
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif path.suffix == '.pth':
        model.load_state_dict(torch.load(path))
    else:
        raise ValueError('Expected model as checkpoint.TAR or published_model.PTH')

    model.to(device)
    model.eval()
    print('model loaded')

    lines = list()
    metrics = Metrics()

    for (cclass, dataloader), m in zip(dataloaders.items(), list('waa')):
        preds, labels = predict(model, dataloader, embeddings,
                                mode=WriteMode(m), cclass=cclass)
        vals = metrics(preds, labels, nn.BCELoss()(preds, labels.float()), keep_all=True)
        vals.pop('p_hat')
        if not lines:
            lines.append(f'cclass\t' + '\t'.join([f'{k:>7}' for k in vals.keys()]))
        lines.append(f'C{cclass:<6}\t' + '\t'.join(f'{v: .4f}' for v in vals.values()))

    print('\n'.join(lines))


@app.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
def resume(ctx: typer.Context,
           config: Path = typer.Option(None),
           checkpoint: Path = typer.Option(None),
           epochs: int = typer.Option(2),
           patience: int = typer.Option(200),
           use_tqdm: bool = typer.Option(True),
           ):
    cfg = parse(ctx)
    print(f'using device: {device}')
    checkpoint = torch.load(checkpoint, map_location=device)
    train_loop(cfg, use_tqdm, checkpoint)


@app.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
def train(ctx: typer.Context,
          train_tsv: Path = typer.Option(None),
          h5: Path = typer.Option(None),
          epochs: int = typer.Option(2),
          use_tqdm: bool = typer.Option(True),
          ) -> None:
    cfg = parse(ctx)
    print(f'using device: {device}')
    train_loop(cfg, use_tqdm=use_tqdm)


def train_loop(cfg, use_tqdm: bool, checkpoint: dict = None):
    writer = Writer(log_dir=cfg.wd / 'tboard' / cfg.name, flush_secs=10)
    writer.add_text(cfg.name, str(cfg))
    # layout = {
    #     'p_hat': {'batch': ['Multiline', ['p_hat/batch']],
    #               'epoch': ['Multiline', ['p_hat/epoch']]},
    #     'AUPR': {'batch': ['Multiline', ['aupr/batch']],
    #              'epoch': ['Multiline', ['aupr/epoch']]},
    #     'Loss': {'batch': ['Multiline', ['loss/batch']],
    #              'epoch': ['Multiline', ['loss/epoch']]},
    # }
    # writer.add_custom_scalars(layout)
    writer.flush()

    # toggle between using rich or not
    if use_tqdm:
        _print = print
        epoch_tracker, batch_tracker, cclass_progress = tqdm, tqdm, None
        e_kwargs, b_kwargs = dict(desc='epoch', colour='green'), \
                             dict(desc='batch', position=0, leave=False)
        cm = contextlib.nullcontext()
    else:
        epoch_progress = Progress(
            TextColumn('[bold blue]epoch {task.completed}/{task.total}'),
            BarColumn(), TimeElapsedColumn())
        batch_progress = Progress(
            TextColumn('batch {task.completed}/{task.total}'),
            BarColumn(), TaskProgressColumn(justify='right', show_speed=True))
        cclass_progress = Progress(
            TextColumn('{task.description} {task.completed}/{task.total}'), BarColumn())
        progress_group = rich.console.Group(
            epoch_progress, batch_progress, cclass_progress)
        epoch_task_id = epoch_progress.add_task('train', total=cfg.epochs)

        _print = epoch_progress.console.print
        epoch_tracker, batch_tracker = epoch_progress.track, batch_progress.track
        e_kwargs, b_kwargs = dict(task_id=epoch_task_id), dict()
        cm = rich.live.Live(progress_group)

    with cm as live:
        batch, epoch, finished = 0, 0, False

        model = InteractionMap(**vars(cfg))
        params = [p for p in model.parameters() if p.requires_grad]
        optim = Adam(params, lr=cfg.lr, weight_decay=0)
        model = model.to(device)
        _print('model loaded')

        dataloader, seq_ids = get_dataloaders_and_ids(
            cfg.train_tsv, cfg.batch_size, augment=cfg.augment,
            shuffle=cfg.shuffle, seed=cfg.seed)
        val_loaders, val_seq_ids = get_dataloaders_and_ids(
            cfg.val_tsv, cfg.batch_size, augment=cfg.augment,
            shuffle=cfg.shuffle, seed=cfg.seed, split_column='cclass')
        embeddings = get_embeddings(cfg.h5, seq_ids | val_seq_ids)
        _print('data loaded')

        metrics = Metrics()
        evaluator = Evaluator(cfg, model, optim, writer, dataloader,
                              val_loaders, embeddings, cclass_progress)

        if checkpoint is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optim_state_dict'])
            torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
            np.random.set_state(checkpoint['numpy_rng_state'])
            dataloader_states = checkpoint['dataloader_states']
            dataloader.sampler.set_state(dataloader_states.pop('train'))
            for cclass in val_loaders:
                val_loaders[cclass].sampler.set_state(dataloader_states[cclass])
            batch = checkpoint['batch']
            epoch = int(batch / len(dataloader))
            metrics.load_state(checkpoint.get('metrics_state'))
            _print('checkpoint loaded')

        if not use_tqdm:
            epoch_progress.advance(epoch_task_id)
        if batch == 0:
            _print('eval untrained model')
            evaluator.evaluate_model(batch)
        for epoch in epoch_tracker(range(epoch, cfg.epochs), **e_kwargs):
            model.train()
            optim.zero_grad()

            if use_tqdm:
                b_kwargs |= dict(desc=f'epoch {epoch + 1}/{cfg.epochs}',
                                 total=len(dataloader), initial=batch % len(dataloader))
            else:
                batch_task_id = batch_progress.add_task('', total=len(dataloader) - batch % len(dataloader))
                b_kwargs |= dict(task_id=batch_task_id)

            # i = 0
            for n0, n1, labels in batch_tracker(dataloader, **b_kwargs):
                # if (i := i + 1) > 10:
                #     break

                # now replace: -> step_model -> process_batch
                preds = list()

                for _n0, _n1 in zip(n0, n1):
                    z_a = embeddings[_n0].to(device)
                    z_b = embeddings[_n1].to(device)

                    p_hat = model.predict_func(z_a, z_b)
                    preds.append(p_hat.flatten(0))

                preds = torch.cat(preds).to(device)
                labels = labels.float().to(device)

                w = 1 + labels * (cfg.ppi_weight - 1)
                loss = nn.BCELoss(weight=w)(preds, labels)

                batch_metrics = metrics(preds, labels, loss)
                writer.add_interval(batch_metrics, 'batch', batch)

                loss.backward()
                optim.step()
                optim.zero_grad()
                batch += 1
                del loss, batch_metrics, preds, labels

                if eval_reason := evaluator.interval(batch=batch):
                    if finished := evaluator.evaluate_model(batch):
                        break

            if not use_tqdm:
                batch_progress.update(batch_task_id, visible=False)
            if finished:
                break

            epoch_metrics = metrics.compute()
            writer.add_interval(epoch_metrics, 'epoch', epoch + 1)
            metrics.reset()
            evaluator.checkpoint(file_name=f'chk_{epoch}.tar', batch=batch,
                                 epoch=epoch, epoch_metrics=epoch_metrics)

        if finished:
            utils.publish(cfg.wd / cfg.name / 'chk_best.tar', InteractionMap(**vars(cfg)),
                          cfg.wd / cfg.name / f'chk_best_{batch}.pth')
            _print(f'stopped training after epoch {epoch + 1} batch {batch}')

        writer.close()


if __name__ == '__main__':
    app()
