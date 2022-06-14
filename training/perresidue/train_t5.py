import argparse
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from logging import Logger

from interaction import InteractionMap, InteractionMapDscript
from utils.dataloader import get_training_dataloader, DataLoader
from utils import general_utils as utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_args(parser):
    data_grp = parser.add_argument_group("Data")
    proj_grp = parser.add_argument_group("Projection Module")
    contact_grp = parser.add_argument_group("Contact Module")
    inter_grp = parser.add_argument_group("Interaction Module")
    train_grp = parser.add_argument_group("Training")
    misc_grp = parser.add_argument_group("Output and Device")

    # Data
    data_grp.add_argument("--train_file", help="Training data")
    data_grp.add_argument("--val_file", help="Validation data")
    data_grp.add_argument("--embedding", help="h5 file with embedded sequences")
    data_grp.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)

    # Projection Module
    proj_grp.add_argument("--projection_dim", type=int, default=100,
                          help="Dimension of embedding projection layer (default: 100)")
    proj_grp.add_argument("--dropout_p", type=float, default=0.1,
                          help="Parameter p for embedding dropout layer (default: 0.5)")

    # Contact Module
    contact_grp.add_argument("--hidden_dim", type=int, default=50,
                             help="Number of hidden units for comparison layer in contact prediction (default: 50)")
    contact_grp.add_argument("--kernel_width", type=int, default=7,
                             help="Width of convolutional filter for contact prediction (default: 7)")

    # Interaction Module
    inter_grp.add_argument("--use_w", action=argparse.BooleanOptionalAction, default=True,
                           help="Don't use weight matrix in interaction prediction model")
    inter_grp.add_argument("--pool_width", type=int, default=9,
                           help="Size of max-pool in interaction model (default: 9)")

    # Training
    train_grp.add_argument("--num_epochs", type=int, default=10, help="Number of epochs (default: 10)")
    train_grp.add_argument("--batch_size", type=int, default=25, help="Minibatch size (default: 25)")
    train_grp.add_argument("--weight_decay", type=float, default=0, help="L2 regularization (default: 0)")
    train_grp.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    train_grp.add_argument("--interaction_weight", type=float, default=0.35,
                           help="Weight on the similarity objective (default: 0.35)")

    # Output
    misc_grp.add_argument("--checkpoint", help="Checkpoint model to start training from")
    misc_grp.add_argument("--config", help="Load config")
    misc_grp.add_argument('--output_creation_dir', type=Path, required=False, default='.')
    misc_grp.add_argument('--model_name', type=str, required=False)
    misc_grp.add_argument('--logging_path', type=Path, required=False)
    misc_grp.add_argument('--tensorboard_path', type=Path, required=False)
    misc_grp.add_argument('--model_save_path', type=Path, required=False)
    misc_grp.add_argument("--use_dscript", action=argparse.BooleanOptionalAction, default=False)

    return parser


def process_batch(model, n0, n1, embeddings, evaluate=False):
    batch_size = len(n0)

    predictions = []
    interaction_maps = []
    for i in range(batch_size):
        z_a = embeddings[n0[i]].to(device)
        z_b = embeddings[n1[i]].to(device)

        if evaluate:
            predict = model.predict(z_a, z_b)  # die beiden sind gleich aufwändig
        else:
            cm, predict = model.map_predict(z_a, z_b)  # predict ruft das eh auf
            interaction_maps.append(torch.mean(cm))
        predictions.append(predict)
    predictions = torch.stack(predictions, 0)
    if evaluate:
        return None, predictions
    interaction_maps = torch.stack(interaction_maps, 0)
    return interaction_maps, predictions


def step_model(model, n0, n1, y, embeddings, weight=0.35, evaluate=False):
    c_map_mag, predictions = process_batch(model, n0, n1, embeddings, evaluate)
    y = Variable(y).to(device)

    loss = nn.BCELoss()(predictions.float(), y.float())

    if not evaluate:
        cmap_loss = torch.mean(c_map_mag)
        loss = (weight * loss) + ((1 - weight) * cmap_loss)
    batch_size = len(predictions)

    return loss, batch_size, predictions


def eval_model(model: InteractionMap,
               eval_counter: int, val_loader,
               embeddings: dict[str, torch.Tensor],
               interaction_weight, logger, tb_logger):
    logger.info(f' Evaluation {eval_counter} '.center(40, '+'))
    model.eval()
    with torch.no_grad():
        predictions, labels = [], []
        eval_loss, num_seqs = 0, 0
        for n0, n1, y in tqdm(val_loader, total=len(val_loader),
                              desc=f'Evaluation {eval_counter}',
                              position=0, leave=True, ascii=True):
            loss, batch_size, prediction = step_model(
                model, n0, n1, y, embeddings, weight=interaction_weight, evaluate=True)
            predictions.append(prediction)
            labels.append(y)
            eval_loss += loss
            num_seqs += batch_size

        labels = torch.cat(labels, 0).to(device).float()
        predictions = torch.cat(predictions, 0).to(device).float()

        bin_predictions = ((.5 * torch.ones(num_seqs)) < predictions).float()
        utils.log_stats(eval_loss, labels, eval_counter,
                        bin_predictions, predictions,
                        tb_logger, logger, sfx=' Epoch Val')

    return eval_loss, num_seqs


def train_model(model: InteractionMap,
                optim: Adam,
                num_epochs: int,
                dataloader: DataLoader,
                val_dataloader: DataLoader,
                embeddings: dict[str, torch.Tensor],
                interaction_weight: float,
                logger: Logger,
                tb_logger: Logger,
                model_save_path: Path,
                model_name: str,
                use_dscript: bool = False,
                evaluation_loss=None,
                start_epoch: int = 0,
                eval_counter: int = 0) -> None:
    def evaluate(old_loss, eval_counter, decline):
        return_loss, return_decline = old_loss, decline
        new_eval_loss, num_eval_seqs_now = eval_model(model, eval_counter, val_dataloader, embeddings,
                                                      interaction_weight, logger, tb_logger)
        if new_eval_loss <= old_loss:
            return_decline = 0
            return_loss = new_eval_loss
            utils.save_model(model_save_path, model_name, 'best', model, optim, return_loss / num_seqs, epoch)
            logger.info(f"# Saving model to {model_save_path}")
        else:
            return_decline += 1
        model.train()
        return return_loss, eval_counter + 1, num_eval_seqs_now, return_decline

    iterations_counter = -1
    eval_loss, patience, decline = evaluation_loss or float('inf'), 15, 0
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
                    (epoch - 1) * len(dataloader)):
                iterations_counter += 1
                if iterations_counter % (max(int(.05 * value_rounder(len(dataloader), 3)), 1)) == 0:
                    epoch_train_time += perf_counter() - train_epoch_start
                    eval_time_start = perf_counter()
                    eval_loss, eval_counter, num_eval_seqs, decline = evaluate(eval_loss, eval_counter, decline)
                    eval_total_timer += perf_counter() - eval_time_start

                    if decline >= patience:
                        raise utils.PatienceExceededError('Termination due to patience exceedance! '.center(100, '!'))

                    train_epoch_start = perf_counter()
                if iterations_counter % (max(int(.025 * value_rounder(len(dataloader), 3)), 1)) == 0:
                    if eval_loss == float('inf') and num_seqs != 0:
                        considered_loss = epoch_loss / num_seqs
                    else:
                        considered_loss = float('inf')
                    checkpoint_model(model_save_path, model_name, epoch, num_epochs, model, optim, considered_loss,
                                     save_epoch=False)
                    logger.info(f"# Checkpoint model to {model_save_path}/checkpoint/")

                loss, b, _ = step_model(model, z0, z1, y, embeddings,
                                        weight=interaction_weight, evaluate=False)
                num_seqs += b
                epoch_loss += loss

                loss.backward()
                optim.step()
                optim.zero_grad()
                if use_dscript:
                    model.clip()

                tb_logger.add_scalar("Loss Batch Train", loss / b, batch_idx)

            tb_logger.add_scalar("Loss Epoch Train", epoch_loss / num_seqs, epoch + 1)
            checkpoint_model(model_save_path, model_name, epoch, num_epochs, model, optim, epoch_loss / num_seqs,
                             save_epoch=True)
            logger.info(f"# Epoch Checkpoint model to {model_save_path}/checkpoint/")
            num_train_seqs = num_seqs

            ####################### Copied from above ##################
            epoch_train_time += perf_counter() - train_epoch_start
            eval_time_start = perf_counter()
            eval_loss, eval_counter, num_eval_seqs, decline = evaluate(eval_loss, eval_counter, decline)
            eval_total_timer += perf_counter() - eval_time_start
            train_epoch_start = perf_counter()
            ############################################################

            for name, weight in model.named_parameters():
                tb_logger.add_histogram(name, weight, epoch + 1)
                tb_logger.add_histogram(f'{name}.grad', weight.grad, epoch + 1)
            epoch_train_time += perf_counter() - train_epoch_start
            train_total_timer += epoch_train_time
    except utils.PatienceExceededError:
        logger.info(" Termination due to patience exceedance! ".center(100, '!'))
    total_time = perf_counter() - total_time
    logger.info(' Time Elapsed '.center(100, '*'))
    logger.info(f'Time per Epoch in h: {total_time / num_epochs / 3600}')
    logger.info(f'Train Time per Epoch in h: {train_total_timer / num_epochs / 3600}')
    logger.info(f'Evaluation Time per Evaluation in h: {eval_total_timer / eval_counter / 3600}')
    logger.info('-----------')
    logger.info(f'Train Time per Epoch in s per Pair: {train_total_timer / num_epochs / num_train_seqs}')
    logger.info(f'Evaluation Time per Evaluation in s per Pair: {eval_total_timer / eval_counter / num_eval_seqs}')
    utils.save_final(model_save_path, model_name)


def checkpoint_model(model_save_path, model_name, epoch, num_max_epoch,
                     model, optimizer, loss, eval_number=None, save_epoch=False):
    digits = int(np.floor(np.log10(num_max_epoch))) + 1
    save_path = model_save_path / 'checkpoint'
    save_path.mkdir(parents=True, exist_ok=True)
    utils.save_model(save_path, model_name,
                     f'epoch{str(epoch + 1).zfill(digits)}'
                     f'{"_checkpoint" if not save_epoch else ""}',
                     model, optimizer, loss, epoch, eval_number)


def handle_config(config):
    for key in ['train', 'val', 'embedding', 'checkpoint',
                'output_creation_dir', 'logging_path', 'tensorboard_path',
                'model_save_path', 'config', 'model_name']:
        config.pop(key, None)
    return config


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
    tensorboard_logger = SummaryWriter(log_dir=tb_path)

    m_path = args.model_save_path or out_dir / 'models' / model_name
    m_path.mkdir(parents=True, exist_ok=True)
    utils.save_config(config, m_path)

    logger = utils.getlogger(args.logging_path or
                             out_dir / 'models' / model_name / f'{model_name}.log')
    logger.info(f'Using {device}')

    logger.info('Create Dataloaders and load embeddings ...')
    pairs_train_dataloader, pairs_val_dataloader, embeddings = get_training_dataloader(
        args.train_file, args.augment, args.batch_size, 2, args.val_file, args.batch_size, 2, args.embedding)
    logger.info('... Dataloader and embeddings done.')

    if args.checkpoint is None:
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
        evaluation_loss = None
        start_epoch = 0
    else:
        logger.info(f"Loading model from checkpoint {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model = checkpoint['model']
        optim = checkpoint['optimizer']
        evaluation_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    model = model.to(device)
    logger.info('Model loading done!')

    train_model(model, optim, args.num_epochs, pairs_train_dataloader, pairs_val_dataloader,
                embeddings, args.interaction_weight, logger, tensorboard_logger,
                model_save_path=m_path,
                model_name=model_name,
                use_dscript=args.use_dscript,
                evaluation_loss=evaluation_loss,
                start_epoch=start_epoch)

    tensorboard_logger.flush()
    tensorboard_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
