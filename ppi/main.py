from __future__ import annotations

import contextlib
import warnings
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pkg_resources
import rich
import sys
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

from ppi.config import parse, Config
from ppi.metrics import Writer, Metrics
from ppi.eval import Evaluator, Intervalometer, evaluate_model

from ppi.train.interaction import InteractionModel

from ppi.utils import general_utils as utils
from ppi.utils.general_utils import device
from ppi.utils.embed import PLM
from ppi.utils.dataloader import (
    DataLoader,
    get_dataloaders_and_ids,
    get_embeddings
)

__version__ = pkg_resources.get_distribution('ppi').version

app = typer.Typer()


class H5WriteMode(str, Enum):
    exit = 'exit'
    overwrite = 'w'
    append = 'a'


@app.command()
def embed(fasta: Path = 'train.fasta',
          h5_file: Path = 'mbeds.h5',
          h5_mode: H5WriteMode = H5WriteMode.exit,
          per_protein: bool = False,
          model: PLM = PLM.t5,
          half: bool = True,
          cache_dir: Path = None
          ) -> None:
    if h5_file.is_file() and h5_mode == H5WriteMode.exit:
        raise FileExistsError(f'H5 file {h5_file} exists. Choose different path, '
                              f'or define append/overwrite mode')
    from ppi.utils.embed import get_model, read_fasta, generate_embeddings

    model, vocab, device = get_model(model, half, cache_dir)
    seqs = read_fasta(fasta)
    generate_embeddings(model, vocab, seqs, device, h5_file, per_protein=per_protein)


@app.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
def slice(ctx: typer.Context,
          fasta: Path = 'train.fasta',
          h5_in: Path = '/mnt/project/kaindl/ppi/embed_data/apid_huri.h5',
          h5_out: Path = 'mbeds.h5',
          h5_mode: H5WriteMode = H5WriteMode.exit) -> None:
    cfg = parse(ctx, write=False)

    seqs = SeqIO.to_dict(SeqIO.parse(fasta, 'fasta'))
    j = 0

    if not h5_out.is_file():
        h5_mode = H5WriteMode.overwrite
    elif h5_mode == H5WriteMode.exit:
        raise FileExistsError(f'H5 file {h5_out} exists. Choose different path, '
                              f'or define append/overwrite mode')
    with h5py.File(h5_out, h5_mode) as new_h5, h5py.File(h5_in, 'r') as old_h5:
        for seq_id in tqdm(seqs, total=len(seqs)):
            if seq_id in old_h5:
                if seq_id in new_h5:
                    if not np.allclose(new_h5[seq_id], old_h5[seq_id]):
                        warnings.warn(f'h5py Dataset {seq_id} already exists'
                                      ' and values differ!', RuntimeWarning)
                    continue
                new_h5[seq_id] = np.array(old_h5[seq_id]).astype(np.float16)
                j += 1
    print(f'copied {j} mbeds for {len(seqs)} query seqs')


@app.command()
def scramble(tsv: Path, seed: int = 42,
             no_header: bool = typer.Option(False),
             out_file: Path = typer.Option(None)) -> None:
    """
    For a PPI dataset as TSV, randomly shuffle CRC hashes,
    i.e. exchange all occurrences of A with B and vice versa.
    :param tsv: PPI dataset
    :param seed: for reproducible shuffling
    :param no_header: for a TSV without a header line
    :param out_file: an optional path to write to, else STDOUT
    """
    pairs = pd.read_csv(tsv, sep='\t', header=[0, None][no_header])  # [['hash_A', 'hash_B', 'label']]
    pair_ids = sorted(set(np.unique(pairs.iloc[:, [0, 1]])))
    rng = np.random.default_rng(seed=seed)
    shuffled_ids = rng.choice(pair_ids, size=len(pair_ids),
                              replace=False, shuffle=True)
    lookup = dict(zip(pair_ids, shuffled_ids)).get
    pairs.iloc[:, 0] = pairs.iloc[:, 0].apply(lookup)
    pairs.iloc[:, 1] = pairs.iloc[:, 1].apply(lookup)
    if out_file is None:
        print(pairs.to_csv(index=False, header=not no_header, sep='\t').rstrip())
    else:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(out_file, index=False, header=not no_header, sep='\t')


@app.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
def tmvec(ctx: typer.Context,
          h5_in: Path = 'mbeds.h5',
          h5_out: Path = 'mbeds_tmvec.h5',
          h5_mode: H5WriteMode = H5WriteMode.exit) -> None:
    cfg = parse(ctx, write=False)

    if not h5_out.is_file():
        h5_mode = H5WriteMode.overwrite
    elif h5_mode == H5WriteMode.exit:
        raise FileExistsError(f'H5 file {h5_out} exists. Choose different path, '
                              f'or define append/overwrite mode')

    from tm_vec.embed_structure_model import trans_basic_block_Config, trans_basic_block
    from tm_vec.tm_vec_utils import embed_tm_vec

    tmvec_config_path = '/mnt/project/kaindl/tm-vec/models/tm_vec_swiss_model_params.json'
    tmvec_model_path = '/mnt/project/kaindl/tm-vec/models/tm_vec_swiss_model.ckpt'

    tm_vec_model_config = trans_basic_block_Config.from_json(tmvec_config_path)
    tmvec_model = trans_basic_block.load_from_checkpoint(tmvec_model_path, config=tm_vec_model_config)
    tmvec_model = tmvec_model.to(device)
    tmvec_model = tmvec_model.eval()

    j = 0
    with h5py.File(h5_out, h5_mode) as new_h5, h5py.File(h5_in, 'r') as old_h5, \
            h5py.File(h5_out.with_name(h5_out.stem + '_only.h5'), h5_mode) as new_h5_tmvec_only:
        seq_ids = set(old_h5.keys())
        for seq_id in tqdm(seq_ids, total=len(seq_ids)):
            t5_mbed = torch.Tensor(old_h5[seq_id]).unsqueeze(1).to(device)
            tmvec_mbed = embed_tm_vec(t5_mbed, tmvec_model, device).astype(np.float16)
            concat = np.concatenate((t5_mbed.cpu().squeeze().numpy()
                                     .astype(np.float16), tmvec_mbed), axis=1)
            if seq_id in new_h5:
                if not np.allclose(new_h5[seq_id], concat):
                    warnings.warn(f'h5py Dataset {seq_id} already exists'
                                  ' and values differ!', RuntimeWarning)
                continue
            new_h5[seq_id] = concat
            new_h5_tmvec_only[seq_id] = tmvec_mbed
            j += 1
    print(f'created {j} tmvec concat mbeds for {len(seq_ids)} query seqs')


class WriteMode(str, Enum):
    none = 'none'
    exit = 'exit'
    overwrite = 'w'
    append = 'a'


# @app.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
def predict(ctx: typer.Context,
            model: InteractionModel, dataloader: DataLoader,
            embeddings: dict[str, torch.Tensor],
            mode: WriteMode = WriteMode.none, cclass: str = '',
            ) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO bare bones, write a TSV, timed
    cfg = parse(ctx, write=False)
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
        preds, all_labels = [], []
        for batch, (n0, n1, labels) in enumerate(dataloader):
            all_labels.append(labels)

            for _n0, _n1, _y in zip(n0, n1, labels):
                z_a = embeddings[_n0].to(device)
                z_b = embeddings[_n1].to(device)
                _, p_hat = model.map_predict(z_a, z_b)

                if mode.value in 'wa':
                    line = f'{batch}\t{_n0}\t{_n1}{cclass}{_y.item()}\t{p_hat.item():.4f}\n'
                    tsv.write(line)

                preds.append(p_hat.flatten(0))

    if mode.value in 'wa':
        tsv.close()
    return torch.cat(preds).to(device), torch.cat(all_labels).float().to(device)


@app.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
def evaluate(ctx: typer.Context,
             model: Path = typer.Option('model.tar'),
             pattern: str = '.tar',
             test_tsv: Path = 'in/val_4k.tsv',
             h5: Path = '/mnt/project/kaindl/ppi/embed_data/apid_huri_val.h5',
             bst: int = 4,
             ):
    # TODO this would be so much easier from a TSV ...
    cfg = parse(ctx, write=False)
    models = [model] if model.is_file() else utils.glob_type(model, pattern)

    # load the test, separated by class
    dataloaders, seq_ids = get_dataloaders_and_ids(
        cfg.test_tsv, cfg.batch_size, augment=cfg.augment, shuffle=False,
        seed=42, split_column='cclass')
    embeddings = get_embeddings(h5, seq_ids)

    writer = Writer(log_dir=cfg.wd / 'tboard' / 'eval', flush_secs=10)
    eval_tsv = cfg.wd / f'eval_{cfg.name}.tsv'
    cols = ['name', 'idx', 'metric', 'measure', 'C1', 'C2', 'C3']
    prc_cols = ['name', 'idx', 'cclass', 'th', 'pr', 'pr_std', 're']
    pred_cols = ['name', 'idx', 'hash_A', 'hash_B', 'p_hat', 'label', 'cclass']
    exists = (cfg.wd / f'eval_{cfg.name}.tsv').is_file()
    with open(cfg.wd / f'eval_{cfg.name}.tsv', 'a') as tsv, \
            open(cfg.wd / f'eval_{cfg.name}_prc.tsv', 'a') as prc_tsv, \
            open(cfg.wd / f'eval_{cfg.name}_preds.tsv', 'a') as preds_tsv:
        if not exists:
            tsv.write('\t'.join(cols) + '\n')
            prc_tsv.write('\t'.join(prc_cols) + '\n')
            preds_tsv.write('\t'.join(pred_cols) + '\n')

        for path in tqdm(sorted(models)):
            name = path.resolve().parent.name
            # print(f'eval {path} ... ', end='')
            idx = path.stem.split('_')[-1]
            idx = int(idx) if idx.isnumeric() else 0
            arc = utils.get_architecture(path)

            model = InteractionModel(**vars(cfg) | dict(architecture=arc))
            if path.suffix == '.tar':
                checkpoint = torch.load(path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

                # load or guess the PPI weight for calculation of the test loss
                model.ppi_weight = utils.search_ppi_weight(checkpoint, path)
            # elif path.suffix == '.pth':
            #     model.load_state_dict(torch.load(path))
            #     # ppi weights won't be used when a model is final as no losses calculated
            else:
                raise ValueError('Expected model as checkpoint.TAR or published_model.PTH')

            model.to(device)
            preds, results, runtimes = evaluate_model(model, embeddings, dataloaders, bootstraps=bst)
            preds[['name', 'idx']] = [name, idx]
            preds = preds[pred_cols]
            preds_tsv.write(preds.to_csv(header=False, index=False, sep='\t'))

            res = Metrics.pivot(results)
            prcs = list()
            for cclass, d in res['prc'].items():
                df = pd.DataFrame(np.hstack((np.array(d['mean'].clone().cpu()).T,
                                             np.array(d['std'].clone().cpu()).T))[:, [0, 1, 2, 3]],
                                  columns=['pr', 're', 'th', 'pr_std'])
                df['cclass'] = cclass
                df[['idx', 'name']] = [idx, name]
                prcs.append(df[prc_cols].copy())
            prcs = pd.concat(prcs)
            res.pop('prc')
            prc_tsv.write(prcs.to_csv(header=False, index=False, sep='\t'))
            prc_tsv.flush()

            dfs = list()
            for metric in ['aupr', 'mcc', 'loss']:
                df = pd.DataFrame(res[metric]).astype(float)
                df[['idx', 'metric', 'name']] = [idx, metric, name]
                df = df.reset_index().rename(dict(index='measure'), axis='columns')[cols]
                dfs.append(df)
            dt = pd.concat(dfs)

            tsv.write(dt.to_csv(header=False, index=False, sep='\t'))
            tsv.flush()
    print(f'evaluated {len(models)} models, results in {eval_tsv}')


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


@app.command()
def publish(model: list[Path] = typer.Option([Path('model.tar')]),
            pattern: str = '*.tar'
            ) -> None:
    models = list()
    for m in model:
        m = Path(m)

        def search_dir(d: Path) -> None:
            for chk in sorted(d.glob(pattern)):
                models.append(chk)

        if m.is_dir():
            search_dir(m)
        elif m.is_file():
            models.append(m)
        else:
            print(f'No model found for {m}', file=sys.stderr)

    for m in tqdm(models):
        cfg = Config.from_file(m.parent / f'config_{m.parent.stem}.json').process(write=False)
        model = InteractionModel(**vars(cfg))
        utils.publish(m, model)


@app.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
def train(ctx: typer.Context,
          train_tsv: Path = typer.Option(None),
          h5: Path = typer.Option(None),
          epochs: int = typer.Option(2),
          use_tqdm: bool = typer.Option(True),
          ) -> None:
    cfg = parse(ctx)
    print(f'T5 PPI v{__version__}\nusing device: {device}')
    train_loop(cfg, use_tqdm=use_tqdm)


def train_loop(cfg, use_tqdm: bool, checkpoint: dict = None):
    writer = Writer(log_dir=cfg.wd / 'tboard' / cfg.name, flush_secs=10)
    writer.add_text(cfg.name, str(cfg))
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
        _print(f'write to: {cfg.wd / cfg.name}')
        batch, epoch, finished = 0, 0, False

        model = InteractionModel(**vars(cfg))
        params = [p for p in model.parameters() if p.requires_grad]
        optim = Adam(params, lr=cfg.lr, weight_decay=0)
        model.to(device)
        _print(f'model {"loaded" if checkpoint is not None else "initialized"}')

        dataloader, seq_ids = get_dataloaders_and_ids(
            cfg.train_tsv, cfg.batch_size, augment=cfg.augment,
            shuffle=cfg.shuffle, seed=cfg.seed)
        val_loaders, val_seq_ids = get_dataloaders_and_ids(
            cfg.val_tsv, cfg.batch_size, augment=cfg.augment,
            shuffle=cfg.shuffle, seed=cfg.seed, split_column='cclass')
        _print('data loaded')
        embeddings = get_embeddings(cfg.h5, seq_ids | val_seq_ids)

        metrics = Metrics()
        intervalometer = Intervalometer(cfg, dataloader, val_loaders)
        evaluator = Evaluator(cfg, model, optim, writer, dataloader, val_loaders,
                              embeddings, cclass_progress, intervalometer, metrics)

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
            evaluator(batch)
        for epoch in epoch_tracker(range(epoch, cfg.epochs), **e_kwargs):
            model.train()
            optim.zero_grad()
            print(f'ppi weight {model.ppi_weight} and acc {model.accuracy_weight}')

            if use_tqdm:
                b_kwargs |= dict(desc=f'epoch {epoch + 1}/{cfg.epochs}',
                                 total=len(dataloader), initial=batch % len(dataloader))
            else:
                batch_task_id = batch_progress.add_task('', total=len(dataloader) - batch % len(dataloader))
                b_kwargs |= dict(task_id=batch_task_id)

            for n0, n1, labels in batch_tracker(dataloader, **b_kwargs):
                preds, cmaps = list(), list()
                for _n0, _n1 in zip(n0, n1):
                    z_a = embeddings[_n0].to(device)
                    z_b = embeddings[_n1].to(device)

                    cm, p_hat = model.map_predict(z_a, z_b)
                    preds.append(p_hat.float().flatten(0))
                    cmaps.append(torch.mean(cm))

                preds = torch.cat(preds).to(device)
                labels = labels.float().to(device)

                w = 1 + labels * (model.ppi_weight - 1)
                bce_loss = nn.BCELoss(weight=w)(preds, labels)
                representation_loss = torch.mean(torch.stack(cmaps, 0))
                loss = (model.accuracy_weight * bce_loss) + (
                        (1 - model.accuracy_weight) * representation_loss)

                batch_metrics = metrics(preds, labels, loss)
                if not batch % 20:
                    writer.add_interval(batch_metrics, 'batch', batch)

                loss.backward()
                optim.step()
                optim.zero_grad()
                batch += 1

                if reason := intervalometer(batch):
                    if finished := evaluator(batch):
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
            utils.publish(cfg.wd / cfg.name / 'chk_best.tar', InteractionModel(**vars(cfg)),
                          cfg.wd / cfg.name / f'chk_best_{batch}.pth')
            _print(f'stopped training after epoch {epoch + 1} batch {batch}')

        writer.close()


if __name__ == '__main__':
    app()
