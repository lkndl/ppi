import argparse
import json
from copy import deepcopy
from logging import Logger
from pathlib import Path
from time import perf_counter

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from interaction import InteractionMap, InteractionMapDscript
from train_t5 import add_args, step_model, handle_config
from utils import general_utils as utils
from utils.dataloader import get_dataloaders_and_ids, get_embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_model(model: InteractionMap,
               eval_counter: int,
               validation_dataloaders: list[DataLoader],
               embeddings: dict[str, torch.Tensor],
               interaction_weight: float,
               logger: Logger,
               tensorboard_loggers: list[SummaryWriter]
               ) -> tuple[float, int]:
    logger.info(f' Evaluation {eval_counter} '.center(40, '+'))
    model.eval()
    with torch.no_grad():
        for cclass, (val_loader, tb_logger) in enumerate(
                zip(validation_dataloaders, tensorboard_loggers), 1):
            predictions, labels = [], []
            eval_loss, num_seqs = 0, 0
            for n0, n1, y in tqdm(val_loader, total=len(val_loader),
                                  desc=f'Evaluation {eval_counter}',
                                  position=0, leave=True, ascii=True):
                loss, batch_size, prediction = step_model(
                    model, n0, n1, y, embeddings,
                    weight=interaction_weight, evaluate=True)
                predictions.append(prediction.detach().cpu())
                labels.append(y.float().detach().cpu())
                eval_loss += loss.item()
                num_seqs += batch_size

            labels = torch.cat(labels, 0)
            predictions = torch.cat(predictions, 0)

            bin_predictions = ((.5 * torch.ones(num_seqs)) < predictions).float()
            utils.log_stats(eval_loss, labels, eval_counter,
                            bin_predictions, predictions,
                            tb_logger, logger, sfx='', lower=False)

    return eval_loss, num_seqs


def train_model(model: InteractionMap,
                optim: Adam,
                num_epochs: int,
                dataloader: DataLoader,
                validation_dataloaders: list[DataLoader],
                embeddings: dict[str, torch.Tensor],
                interaction_weight: float,
                logger: Logger,
                tb_loggers: list[SummaryWriter],
                model_save_path: Path,
                model_name: str,
                use_dscript: bool = False,
                evaluation_loss: float = None,
                start_epoch: int = 0,
                eval_counter: int = 0) -> None:
    # bisher nutzen wir vier Logger, damit die Farben im tensorboard konsistent bleiben
    tb_train_logger, *tb_val_loggers = tb_loggers

    def evaluate(old_loss, eval_counter, decline):
        return_loss, return_decline = old_loss, decline  # decline: wie oft wurde es nicht besser eh klar
        new_eval_loss, num_eval_seqs_now = eval_model(
            model, eval_counter, validation_dataloaders, embeddings,
            interaction_weight, logger, tb_val_loggers)
        if new_eval_loss <= old_loss:
            return_decline = 0
            return_loss = new_eval_loss
            utils.save_model(model_save_path, model_name, 'best', model, optim,
                             return_loss / num_eval_seqs_now, epoch, eval_counter)
            logger.info(f"Saving model to {model_save_path}")
        else:
            return_decline += 1
        model.train()
        return return_loss, eval_counter + 1, num_eval_seqs_now, return_decline

    iterations_counter = -1  # wir evaluieren schon mehrfach pro epoche, nach x batches
    eval_loss, patience, decline = evaluation_loss or float('inf'), 1000, 0
    value_rounder = lambda x, pow: round(x / 10 ** pow) * 10 ** pow

    num_train_seqs, num_eval_seqs = 0, 0
    train_total_timer, eval_total_timer = 0, 0
    total_time = perf_counter()
    try:
        for epoch in range(start_epoch, num_epochs):
            train_epoch_start = perf_counter()
            epoch_train_time = 0
            logger.info(f" Start Epoch {epoch} ".center(60, '#'))
            model.train()
            optim.zero_grad()
            num_seqs, epoch_loss = 0, 0
            for batch_idx, (z0, z1, y) in enumerate(
                    tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}",
                         total=len(dataloader), position=0, leave=True, ascii=True),
                    epoch * len(dataloader)):
                iterations_counter += 1
                if iterations_counter % (max(int(.05 * value_rounder(
                        len(dataloader), 3)), 1)) == 0:
                    # evaluation: whether we improved (we always do, anyway!!!!)
                    epoch_train_time += perf_counter() - train_epoch_start
                    eval_time_start = perf_counter()
                    eval_loss, eval_counter, num_eval_seqs, decline = evaluate(
                        eval_loss, eval_counter, decline)
                    eval_total_timer += perf_counter() - eval_time_start

                    if decline >= patience:
                        raise utils.PatienceExceededError('Termination due to patience exceedance! '.center(100, '!'))

                    train_epoch_start = perf_counter()
                    utils.wipe_memory()
                if iterations_counter % (max(int(.025 * value_rounder(
                        len(dataloader), 3)), 1)) == 0:
                    # checkpointing: to be able to resume
                    if eval_loss == float('inf') and num_seqs != 0:
                        considered_loss = epoch_loss / num_seqs
                    else:
                        considered_loss = float('inf')
                    utils.checkpoint_model(model_save_path, model_name, epoch,
                                           num_epochs, model, optim, considered_loss,
                                           eval_counter, save_epoch=False)  # savegame
                    logger.info(f"# Checkpoint model to {model_save_path}/checkpoint/")
                    utils.wipe_memory()

                loss, b, _ = step_model(model, z0, z1, y, embeddings,
                                        weight=interaction_weight, evaluate=False)
                num_seqs += b
                batch_loss = loss.item()
                epoch_loss += batch_loss

                loss.backward()
                optim.step()
                optim.zero_grad()

                if use_dscript:
                    model.clip()  # wie numpy aber parameter custom anpassen. funktion ist von denen selber implementiert

                tb_train_logger.add_scalar(
                    "Loss Batch Train", batch_loss / b, batch_idx)
                utils.flush_loggers(tb_loggers)
                utils.wipe_memory()

            tb_train_logger.add_scalar("Loss Epoch Train", epoch_loss / num_seqs, epoch + 1)
            utils.checkpoint_model(model_save_path, model_name, epoch,
                                   num_epochs, model, optim, epoch_loss / num_seqs,
                                   eval_counter, save_epoch=True)
            logger.info(f"# Epoch Checkpoint model to {model_save_path}/checkpoint/")
            num_train_seqs = num_seqs

            # nach der Epoche eh klar
            ####################### Copied from above ##################
            epoch_train_time += perf_counter() - train_epoch_start
            eval_time_start = perf_counter()
            eval_loss, eval_counter, num_eval_seqs, decline = evaluate(
                eval_loss, eval_counter, decline)
            eval_total_timer += perf_counter() - eval_time_start
            train_epoch_start = perf_counter()
            ############################################################

            for name, weight in model.named_parameters():
                # parameterverlauf anschauen
                tb_train_logger.add_histogram(name, weight, epoch + 1)
                tb_train_logger.add_histogram(f'{name}.grad', weight.grad, epoch + 1)
            epoch_train_time += perf_counter() - train_epoch_start
            train_total_timer += epoch_train_time
            utils.flush_loggers(tb_loggers)
            utils.wipe_memory()
    except utils.PatienceExceededError:
        logger.info(" Termination due to patience exceedance! ".center(100, '!'))
        utils.flush_loggers(tb_loggers)
    total_time = perf_counter() - total_time
    logger.info(' Time Elapsed '.center(100, '*'))
    logger.info(f'Time per Epoch in h: {total_time / num_epochs / 3600}')
    logger.info(f'Train Time per Epoch in h: {train_total_timer / num_epochs / 3600}')
    logger.info(f'Evaluation Time per Evaluation in h: {eval_total_timer / eval_counter / 3600}')
    logger.info('-----------')
    logger.info(f'Train Time per Epoch in s per Pair: {train_total_timer / num_epochs / num_train_seqs}')
    logger.info(f'Evaluation Time per Evaluation in s per Pair: {eval_total_timer / eval_counter / num_eval_seqs}')
    utils.save_final(model_save_path, model_name)


def main(args):
    utils.set_seed(42)

    if not args.config:
        config = handle_config(deepcopy(vars(args)))
    else:
        with open(args.config, 'r') as f:
            config = json.load(f)

    vars(args).update(config)
    model_name = args.model_name or utils.get_hash(config)

    out_dir = args.output_creation_dir  # this is usually not CWD!
    tb_path = args.tensorboard_path or out_dir / 'tensorboard' / model_name
    tb_path.mkdir(parents=True, exist_ok=True)
    tb_loggers = [SummaryWriter(tb_path / 'train'),
                  *[SummaryWriter(tb_path / 'val' / f'c{cclass}') for cclass in '123']]

    m_path = args.model_save_path or out_dir / 'models' / model_name
    m_path.mkdir(parents=True, exist_ok=True)
    utils.save_config(config, m_path)

    logger = utils.getlogger(args.logging_path or
                             out_dir / 'models' / model_name / f'{model_name}.log')
    logger.info(f'Using {device}')

    logger.info('Create Dataloaders and load embeddings ...')
    train_loader, train_ids = get_dataloaders_and_ids(
        args.train_file, args.batch_size, args.augment)
    val_loaders, val_ids = get_dataloaders_and_ids(
        args.val_file, args.batch_size, False, split_column='cclass')
    embeddings = get_embeddings(args.embedding, train_ids | val_ids)
    utils.wipe_memory()
    logger.info('... Dataloader and embeddings done!')

    if args.checkpoint is None:  # frisch initialisiert sozusagen
        model = [InteractionMap, InteractionMapDscript][args.use_dscript](
            args.projection_dim,
            args.dropout_p,
            args.hidden_dim,
            args.kernel_width,
            args.pool_width,
            use_w=args.use_w)
        logger.info(f'Initialize D-Script {"inspired " if not args.use_dscript else " "}Architecture')

        params = [p for p in model.parameters() if p.requires_grad]
        optim = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        evaluation_loss = None  # nur c3! die anderen stehen aber auch zB im tensorboard
        start_epoch = 0
        eval_number = 0
        # das sind die dinge die in den checkpoint geschrieben werden.
        # im Modell stehen schon Sachen aus einer Teilepoche, die anderen SAchen sind epochenstartbezogen
        # in den checkpoint soll final nur das state dict, nicht das komplette Modell.
        # für prototyping ok. beim laden vom state dict braucht man dann auch ein config

    else:
        logger.info(f"Loading model from checkpoint {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model = checkpoint['model']  # das geht nur weil wir das gesamte modell speichern.
        # sonst müssten wir das modell init, und model.load state dict
        optim = checkpoint['optimizer']
        evaluation_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        eval_number = checkpoint['eval_number']

    model = model.to(device)
    utils.wipe_memory()
    logger.info('Model loading done!')

    train_model(model, optim, args.num_epochs, train_loader, val_loaders,
                embeddings, args.interaction_weight, logger, tb_loggers,
                model_save_path=m_path,
                model_name=model_name,
                use_dscript=args.use_dscript,
                evaluation_loss=evaluation_loss,
                start_epoch=start_epoch,
                eval_counter=eval_number)

    utils.flush_loggers(tb_loggers)
    utils.close_loggers(tb_loggers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
