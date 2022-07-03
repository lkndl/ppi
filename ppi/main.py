from __future__ import annotations

import warnings
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import pkg_resources
import torch
import torch.nn as nn
import typer
from tqdm import tqdm
from Bio import SeqIO
from sklearn import metrics as skl
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from train.interaction import InteractionMap
from utils.dataloader import DataLoader, \
    get_dataloaders_and_ids, get_embeddings
from .config import Config

from train import interaction

__version__ = pkg_resources.get_distribution('ppi').version

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


@app.command()
def embed(fasta: Path = 'train.fasta',
          h5_out: Path = 'all.h5',
          mode: H5WriteMode = H5WriteMode.exit) -> None:
    seqs = SeqIO.to_dict(SeqIO.parse(fasta, 'fasta'))

    if not h5_out.is_file():
        mode = H5WriteMode.overwrite
    elif mode == H5WriteMode.exit:
        raise FileExistsError(f'H5 file {h5_out} exists. Choose different path, '
                              f'or define append/overwrite mode')
    with h5py.File(h5_out, mode) as new_h5, \
            h5py.File('/mnt/project/kaindl/ppi/mem_leak/smaller.h5', 'r') as old_h5:
        for seq_id in seqs:
            if seq_id in old_h5:
                if seq_id in new_h5:
                    if not np.allclose(new_h5[seq_id], old_h5[seq_id]):
                        warnings.warn(f'h5py Dataset {seq_id} already exists'
                                      ' and values differ!', RuntimeWarning)
                    continue
                new_h5[seq_id] = np.array(old_h5[seq_id])


def predict(model: InteractionMap, dataloader: DataLoader,
            embeddings: dict[str, torch.Tensor],
            mode: str = 'w', cclass: str = '',
            ) -> tuple[torch.Tensor, torch.Tensor]:
    tsv = Path('predict.tsv')

    tsv = tsv.open(mode)

    if cclass := cclass.strip():
        cclass = f'\t{cclass}\t'

    line = f'batch\thash_A\thash_B\t' + ('cclass\t' if cclass else '') + f'y\tphat\n'
    if mode == 'w':
        print(line, end='')
        tsv.write(line)

    model.eval()
    with torch.no_grad():
        predictions, labels, i_maps = [], [], []
        for batch, (n0, n1, label) in tqdm(enumerate(dataloader), desc=cclass):
            batch_size = len(n0)
            for i in range(batch_size):
                z_a = embeddings[n0[i]].to(device)
                z_b = embeddings[n1[i]].to(device)
                # i_map, phat, yhat = model.map_predict_modified(z_a, z_b)
                # i_maps.append(torch.mean(i_map))
                phat = model.predict(z_a, z_b).item()
                phat = np.clip(phat, a_min=0, a_max=1)

                line = f'{batch}\t{n0[i]}\t{n1[i]}{cclass}{label.item()}\t{phat:.4f}\n'
                if phat > .5:
                    print(line, end='')
                tsv.write(line)

                predictions.append(phat)
                labels.append(label)

    tsv.close()
    return torch.Tensor(predictions), torch.Tensor(labels)


@app.command()
def evaluate(model: Path = 'model.pth',
             tsv: Path = 'test.tsv',
             h5: Path = 'all.h5',
             ):
    batch_size = 1

    # load the test, separated by class
    dataloaders, seq_ids = get_dataloaders_and_ids(
        tsv, batch_size, augment=False, shuffle=False, split_column='cclass')
    embeddings = get_embeddings(h5, seq_ids)

    # load the model
    model = torch.load(model)['model'].to(device)
    print('model loaded')
    lines = list()

    for (cclass, dataloader), m in zip(dataloaders.items(), list('waa')):
        predictions, labels = predict(model, dataloader, embeddings,
                                      mode=m, cclass=cclass)

        bin_predictions = (torch.full_like(predictions, fill_value=.5) < predictions).float()

        metrics = list()
        metrics.append(('loss', nn.BCELoss()(predictions, labels)))
        metrics.append(('ACC', skl.accuracy_score(labels, bin_predictions)))
        metrics.append(('bACC', skl.balanced_accuracy_score(labels, bin_predictions)))
        metrics.append(('Pr', skl.precision_score(labels, bin_predictions, zero_division=0)))
        metrics.append(('Re', skl.recall_score(labels, bin_predictions, zero_division=0)))
        metrics.append(('F1', skl.f1_score(labels, bin_predictions, zero_division=0)))
        metrics.append(('AUPR', skl.average_precision_score(labels, predictions)))
        metrics.append(('MCC', skl.matthews_corrcoef(labels, bin_predictions)))

        if not lines:
            lines.append(f'cclass\t' + '\t'.join(t[0] for t in metrics))
        lines.append(f'C{cclass}\t' + '\t'.join(f'{t[1]:.4f}' for t in metrics))
    print('\n'.join(lines))


@app.command()
def train(tsv: Path, h5: Path, epochs: int, step: int) -> None:
    batch_size = 2
    ppi_weight = 6  # >= 0 !

    dataloader, seq_ids = get_dataloaders_and_ids(
        tsv, batch_size, augment=False, shuffle=False)
    embeddings = get_embeddings(h5, seq_ids)
    model = InteractionMap(
        emb_projection_dim=100,
        dropout_p=.1,
        map_hidden_dim=50,
        kernel_width=7,
        pool_size=9,
        activation=nn.GELU()
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optim = Adam(params, lr=.001, weight_decay=0)

    model = model.to(device)
    print('model loaded')

    log = open('train.log', 'w')
    log2 = open('train_illegal_losses.log', 'w')
    line = 'epoch\tbatch\ty\tloss\tphat'
    log.write(line + '\n')
    print(line)
    writer = SummaryWriter(log_dir='tboard', flush_secs=10)
    idx = 0

    for epoch in range(epochs):
        model.train()
        optim.zero_grad()
        e_loss = 0
        for batch, (n0, n1, y) in enumerate(dataloader):
            # now replace: -> step_model -> process_batch
            batch_size = len(n0)
            predictions = list()
            scalars = dict()

            for i in range(batch_size):
                z_a = embeddings[n0[i]].to(device)
                z_b = embeddings[n1[i]].to(device)

                phat = model.predict(z_a, z_b)
                if phat < 0 or phat > 1:
                    line = f'{epoch}\t{batch}\t{n0[i]}\t{n1[i]}\t{phat}'
                    # print(line)
                    log2.write(line + '\n')
                    phat = torch.clamp(phat, min=0, max=1)
                predictions.append(phat)
                scalars[f'{y[i].item():g}'] = phat.item()

            predictions = torch.stack(predictions, 0).to(device)
            y = Variable(y).float().to(device)
            w = (torch.ones_like(predictions) + y * (ppi_weight - 1))
            loss = nn.BCELoss(weight=w)(predictions, y)
            # loss = nn.BCELoss()(predictions, y)

            writer.add_scalar('loss/batch', loss.item(), idx)
            writer.add_scalars('p_hat', scalars, idx)

            for _y, _l, _p in zip(y, [loss] * batch_size, predictions):
                line = f'{epoch}\t{batch}\t{_y:g}\t{_l:.4f}\t{_p:.4f}'
                log.write(line + '\n')
                if not epoch % step:
                    print(line)

            e_loss += loss.item()
            idx += 1

            loss.backward()
            optim.step()
            optim.zero_grad()

        writer.add_scalar('loss/epoch', e_loss / len(dataloader), epoch)

    writer.close()
    log.close()
    log2.close()


class Trainer:
    cfg: Config
    dataloader: DataLoader
    val_loaders: list[DataLoader]
    model: InteractionMap
    optim: Adam

    def __init__(self, path):
        if path is not None and Path(path).is_file():
            self.cfg = Config.from_json_file(path)
        else:
            self.cfg = Config()


@app.command()
def full(config: Path = None):
    trainer = Trainer(config)
    print(f'wrote config to {trainer.cfg.output.wd / trainer.cfg.output.name}.json')

    """
    Model name: user-defined, or a shortened hash of model and dataset parameters
    
    
    """
    # Trainer:

    # main
    # train
    # eval


if __name__ == '__main__':
    app()
