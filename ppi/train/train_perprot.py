import argparse
import json
from logging import Logger
from pathlib import Path
from time import perf_counter
from typing import Union

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from network import Attention, ProjectedAttention, MLP, Linear
from ppi.utils import general_utils as utils
from ppi.utils.dataloader import get_dataloaders_and_ids, get_embeddings

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def add_args(parser):
    data_grp = parser.add_argument_group("Data")
    train_grp = parser.add_argument_group("Training")
    misc_grp = parser.add_argument_group("Output and Device")

    # Data
    data_grp.add_argument("--train_file", help="Training data")
    data_grp.add_argument("--val_file", help="Validation data")
    data_grp.add_argument("--embedding", help="h5 file with embedded sequences")
    data_grp.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)

    ne = 4
    train_grp.add_argument("--num_epochs", type=int, default=ne, help=f'Number of epochs (default: {ne})')
    bs = 50
    train_grp.add_argument("--batch_size", type=int, default=bs, help=f'Minibatch size (default: {bs})')
    l2 = 0
    train_grp.add_argument("--weight_decay", type=float, default=l2, help=f'L2 regularization (default: {l2})')
    lr = .001
    train_grp.add_argument("--lr", type=float, default=lr, help=f'Learning rate (default: {lr})')

    # Output
    misc_grp.add_argument("--checkpoint_att", help="Checkpoint model to start training from")
    misc_grp.add_argument("--checkpoint_proj", help="Checkpoint model to start training from")
    misc_grp.add_argument("--checkpoint_mlp", help="Checkpoint model to start training from")
    misc_grp.add_argument("--checkpoint_linear", help="Checkpoint model to start training from")
    misc_grp.add_argument('--output_creation_dir', type=Path, required=False, default=Path('perprot'))
    misc_grp.add_argument('--logging_path', type=Path, required=False)
    misc_grp.add_argument('--tensorboard_path', type=Path, required=False)
    misc_grp.add_argument('--model_save_path', type=Path, required=False)

    attention_model_grp = parser.add_argument_group("Attention Model")
    projattention_model_grp = parser.add_argument_group("ProjectedAttention Model")
    mlp_model_grp = parser.add_argument_group("MLP Model")
    linear_model_grp = parser.add_argument_group("Linear Model")

    attention_model_grp.add_argument("--attention", default=True, action=argparse.BooleanOptionalAction)
    attention_model_grp.add_argument("--att_embed_dim", type=int, default=1024)
    attention_model_grp.add_argument("--att_num_heads", type=int, default=8)
    attention_model_grp.add_argument("--att_num_layers", type=int, default=2)
    attention_model_grp.add_argument("--att_dim_feedforward", type=int, default=1024)
    attention_model_grp.add_argument('--att_model_name', type=str, required=False)
    attention_model_grp.add_argument("--att_config", help="Load config")

    projattention_model_grp.add_argument("--projattention", default=True, action=argparse.BooleanOptionalAction)
    projattention_model_grp.add_argument("--proj_embed_dim", type=int, default=1024)
    projattention_model_grp.add_argument("--proj_projection_dim", type=int, default=128)
    projattention_model_grp.add_argument("--proj_num_heads", type=int, default=8)
    projattention_model_grp.add_argument("--proj_num_layers", type=int, default=2)
    projattention_model_grp.add_argument("--proj_dim_feedforward", type=int, default=128)
    projattention_model_grp.add_argument('--proj_model_name', type=str, required=False)
    projattention_model_grp.add_argument("--proj_config", help="Load config")

    mlp_model_grp.add_argument("--mlp", default=True, action=argparse.BooleanOptionalAction)
    mlp_model_grp.add_argument("--mlp_embed_dim", type=int, default=1024)
    mlp_model_grp.add_argument("--mlp_projection_dim", type=int, default=256)
    mlp_model_grp.add_argument("--mlp_hidden_dim", type=int, default=128)
    mlp_model_grp.add_argument('--mlp_model_name', type=str, required=False)
    mlp_model_grp.add_argument("--mlp_config", help="Load config")

    linear_model_grp.add_argument("--linear", default=True, action=argparse.BooleanOptionalAction)
    linear_model_grp.add_argument("--linear_embed_dim", type=int, default=1024)
    linear_model_grp.add_argument('--linear_model_name', type=str, required=False)
    linear_model_grp.add_argument("--linear_config", help="Load config")

    return parser


def train_step(model, n0, n1, y, embeddings, evaluate=False):
    get_emb = lambda ids: torch.cat([embeddings[_id] for _id in ids], dim=0)

    emb0, emb1 = get_emb(n0).to(device), get_emb(n1).to(device)
    predictions = nn.Sigmoid()(model(emb0, emb1))
    y = Variable(y).float().unsqueeze(-1).to(device)

    loss = nn.BCELoss()(predictions.float(), y)
    batch_size = len(predictions)

    if evaluate:
        return loss, batch_size, predictions, y
    return loss, batch_size


def eval_model_no_cval(model, eval_counter, pairs_val_dataloader, embeddings, logger, tensorboard_logger):
    logger.info(f' Evaluation {eval_counter} '.center(40, '+'))
    model.eval()
    with torch.no_grad():
        predictions, labels = [], []
        eval_loss, num_seqs = 0, 0
        for n0, n1, y in tqdm(pairs_val_dataloader, total=len(pairs_val_dataloader),
                              desc=f'Evaluation {eval_counter}',
                              position=0, leave=False, ascii=True):
            loss, batch_size, prediction, label = train_step(model, n0, n1, y, embeddings, evaluate=True)
            predictions.append(prediction)
            labels.append(label)
            eval_loss += loss
            num_seqs += batch_size

        labels = torch.cat(labels, 0).view(-1).to(device).float()
        predictions = torch.cat(predictions, 0).view(-1).to(device).float()

        bin_predictions = ((.5 * torch.ones(num_seqs)) < predictions).float()
        utils.log_stats(eval_loss, labels, eval_counter,
                        bin_predictions, predictions,
                        tensorboard_logger, logger, sfx=' Epoch Val')

    return eval_loss, num_seqs


def train_model_no_cval(model, optim, num_epochs, pairs_train_dataloader, pairs_val_dataloader,
                        embeddings, logger, tensorboard_logger, model_save_path, model_name,
                        evaluation_loss=None, start_epoch=0):
    logger.info(' Train new model '.center(50, '#'))
    logger.info(f'{model}')

    def evaluate(old_loss, eval_counter, decline):
        return_loss, return_decline = old_loss, decline
        new_eval_loss, num_eval_seqs_now = eval_model(
            model, eval_counter, pairs_val_dataloader, embeddings, logger, tensorboard_logger)
        if new_eval_loss <= old_loss:
            return_decline = 0
            return_loss = new_eval_loss
            utils.save_model(model_save_path, model_name, 'best', model, optim, return_loss / num_seqs, epoch)
            logger.info(f"# Saving model to {model_save_path}")
        else:
            return_decline += 1
        model.train()
        return return_loss, eval_counter + 1, num_eval_seqs_now, return_decline

    iterations_counter, eval_counter, = -1, 0
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
                    tqdm(pairs_train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}",
                         total=len(pairs_train_dataloader),
                         position=0, leave=False, ascii=True),
                    (epoch - 1) * len(pairs_train_dataloader)):
                iterations_counter += 1
                if iterations_counter % (max(int(.05 * value_rounder(len(pairs_train_dataloader), 3)), 1)) == 0:
                    epoch_train_time += perf_counter() - train_epoch_start
                    eval_time_start = perf_counter()
                    eval_loss, eval_counter, num_eval_seqs, decline = evaluate(eval_loss, eval_counter, decline)
                    eval_total_timer += perf_counter() - eval_time_start

                    if decline >= patience:
                        raise utils.PatienceExceededError('Termination due to patience exceedance! '.center(100, '!'))

                    train_epoch_start = perf_counter()
                if iterations_counter % (max(int(.025 * value_rounder(len(pairs_train_dataloader), 3)), 1)) == 0:
                    if eval_loss == float('inf') and num_seqs != 0:
                        considered_loss = epoch_loss / num_seqs
                    else:
                        considered_loss = float('inf')
                    utils.checkpoint_model(model_save_path, model_name, epoch, num_epochs,
                                           model, optim, considered_loss, save_epoch=False)
                    logger.info(f"# Checkpoint model to {model_save_path}/checkpoint/")

                loss, b = train_step(model, z0, z1, y, embeddings, evaluate=False)
                num_seqs += b
                epoch_loss += loss

                loss.backward()
                optim.step()
                optim.zero_grad()

                tensorboard_logger.add_scalar("Loss Batch Train", loss / b, batch_idx)

            tensorboard_logger.add_scalar("Loss Epoch Train", epoch_loss / num_seqs, epoch + 1)
            utils.checkpoint_model(model_save_path, model_name, epoch, num_epochs,
                                   model, optim, epoch_loss / num_seqs, save_epoch=True)
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
                tensorboard_logger.add_histogram(name, weight, epoch + 1)
                tensorboard_logger.add_histogram(f'{name}.grad', weight.grad, epoch + 1)
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


def get_optim(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    optim = Adam(params, lr=lr, weight_decay=weight_decay)
    return optim


def load_model_from_check(checkpoint):
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    optim = checkpoint['optimizer']
    evaluation_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
    return model, optim, evaluation_loss, start_epoch


def train_attention(args, train_dataloader, val_dataloader, embeddings, logger, tensorboard_logger,
                    model_save_path, model_name):
    """TODO Remember val_dataloader is a list of dataloaders differentiated via C1-3"""
    if args.checkpoint_att is None:
        att_embed_dim = args.att_embed_dim
        att_num_heads = args.att_num_heads
        att_num_layers = args.att_num_layers
        att_dim_feedforward = args.att_dim_feedforward

        model = Attention(embed_dim=att_embed_dim, num_heads=att_num_heads,
                          num_layers=att_num_layers, dim_feedforward=att_dim_feedforward).to(device)
        optim = get_optim(model, args.lr, args.weight_decay)
        evaluation_loss = None
        start_epoch = 0
    else:
        logger.info(f"# Loading model from checkpoint {args.checkpoint_att}")
        model, optim, evaluation_loss, start_epoch = load_model_from_check(args.checkpoint_att)

    model = model.to(device)
    logger.info('> Attention Model loading done!')

    train_model(model, optim, args.num_epochs, train_dataloader, val_dataloader,
                embeddings, logger, tensorboard_logger, model_save_path, model_name,
                evaluation_loss=evaluation_loss, start_epoch=start_epoch)


def train_proj_attention(args, train_dataloader, val_dataloader, embeddings, logger, tensorboard_logger,
                         model_save_path, model_name):
    if args.checkpoint_proj is None:
        proj_embed_dim = args.proj_embed_dim
        proj_projection_dim = args.proj_projection_dim
        proj_num_heads = args.proj_num_heads
        proj_num_layers = args.proj_num_layers
        proj_dim_feedforward = args.proj_dim_feedforward

        model = ProjectedAttention(embed_dim=proj_embed_dim, projection_dim=proj_projection_dim,
                                   num_heads=proj_num_heads, num_layers=proj_num_layers,
                                   dim_feedforward=proj_dim_feedforward).to(device)
        optim = get_optim(model, args.lr, args.weight_decay)
        evaluation_loss = None
        start_epoch = 0
    else:
        logger.info(f"# Loading model from checkpoint {args.checkpoint_proj}")
        model, optim, evaluation_loss, start_epoch = load_model_from_check(args.checkpoint_proj)

    model = model.to(device)
    logger.info('> Projection Attention Model loading done!')

    train_model(model, optim, args.num_epochs, train_dataloader, val_dataloader,
                embeddings, logger, tensorboard_logger, model_save_path, model_name,
                evaluation_loss=evaluation_loss, start_epoch=start_epoch)


def train_mlp(args, train_dataloader, val_dataloader, embeddings, logger, tensorboard_logger,
              model_save_path, model_name):
    if args.checkpoint_mlp is None:
        mlp_embed_dim = args.mlp_embed_dim
        mlp_projection_dim = args.mlp_projection_dim
        mlp_hidden_dim = args.mlp_hidden_dim

        model = MLP(embed_dim=mlp_embed_dim, projection_dim=mlp_projection_dim,
                    hidden=mlp_hidden_dim).to(device)
        optim = get_optim(model, args.lr, args.weight_decay)
        evaluation_loss = None
        start_epoch = 0
    else:
        logger.info(f"# Loading model from checkpoint {args.checkpoint_mlp}")
        model, optim, evaluation_loss, start_epoch = load_model_from_check(args.checkpoint_mlp)

    model = model.to(device)
    logger.info('> MLP Model loading done!')

    train_model(model, optim, args.num_epochs, train_dataloader, val_dataloader,
                embeddings, logger, tensorboard_logger, model_save_path, model_name,
                evaluation_loss=evaluation_loss, start_epoch=start_epoch)


def train_linear(args, train_dataloader, val_dataloader, embeddings, logger, tensorboard_logger,
                 model_save_path, model_name):
    if args.checkpoint_linear is None:
        linear_embed_dim = args.linear_embed_dim
        model = Linear(embed_dim=linear_embed_dim).to(device)
        optim = get_optim(model, args.lr, args.weight_decay)
        evaluation_loss = None
        start_epoch = 0
    else:
        logger.info(f"# Loading model from checkpoint {args.checkpoint_linear}")
        model, optim, evaluation_loss, start_epoch = load_model_from_check(args.checkpoint_linear)

    model = model.to(device)
    logger.info('> Linear Model loading done!')

    train_model(model, optim, args.num_epochs, train_dataloader, val_dataloader,
                embeddings, logger, tensorboard_logger, model_save_path, model_name,
                evaluation_loss=evaluation_loss, start_epoch=start_epoch)


def handle_config(config, model_group_prefix):
    return_config = {}
    for key in ['num_epochs', 'batch_size', 'weight_decay', 'lr']:
        return_config[key] = config[key]
    for key, value in config.items():
        if key.startswith(model_group_prefix) and ('model_name' not in key) and ('config' not in key):
            return_config[key] = value
    return return_config


def eval_model(model: Union[Attention, ProjectedAttention, MLP, Linear],
               eval_counter: int,
               validation_dataloaders: dict[str, DataLoader],
               embeddings: dict[str, torch.Tensor],
               logger: Logger,
               tb_logger: SummaryWriter
               ) -> tuple[float, int]:
    logger.info(f' Evaluation {eval_counter} '.center(40, '+'))
    model.eval()
    with torch.no_grad():
        for cclass, val_loader in validation_dataloaders.items():
            predictions, labels = [], []
            eval_loss, num_seqs = 0, 0
            for n0, n1, y in tqdm(val_loader, total=len(val_loader),
                                  desc=f'Evaluation {eval_counter} C{cclass}',
                                  position=0, leave=False, ascii=True):
                loss, batch_size, prediction, label = train_step(
                    model, n0, n1, y, embeddings, evaluate=True)
                predictions.append(prediction.detach().cpu())
                labels.append(y.float().detach().cpu())
                eval_loss += loss.item()
                num_seqs += batch_size

            labels = torch.cat(labels, 0).view(-1)
            predictions = torch.cat(predictions, 0).view(-1)

            bin_predictions = ((.5 * torch.ones(num_seqs)) < predictions).float()
            utils.log_stats(eval_loss, labels, eval_counter,
                            bin_predictions, predictions,
                            tb_logger, logger, sfx=f'/C{cclass}', lower=True)

    return eval_loss, num_seqs


def train_model(model: Union[Attention, ProjectedAttention, MLP, Linear],
                optim: Adam,
                num_epochs: int,
                dataloader: DataLoader,
                validation_dataloaders: dict[str, DataLoader],
                embeddings: dict[str, torch.Tensor],
                logger: Logger,
                tb_logger: SummaryWriter,
                model_save_path: Path,
                model_name: str,
                evaluation_loss: float = None,
                start_epoch: int = 0) -> None:
    logger.info(' Train new model '.center(50, '#'))
    logger.info(f'{model}')

    layout = {
        'Validation': {
            'Loss': ['Multiline', ['loss/C1', 'loss/C2', 'loss/C3']],
            'Accuracy': ['Multiline', ['acc/C1', 'acc/C2', 'acc/C3']],
            'Precision': ['Multiline', ['pr/C1', 'pr/C2', 'pr/C3']],
            'Recall': ['Multiline', ['re/C1', 're/C2', 're/C3']],
            'F1': ['Multiline', ['f1/C1', 'f1/C2', 'f1/C3']],
            'AUPR': ['Multiline', ['aupr/C1', 'aupr/C2', 'aupr/C3']],
            'MCC': ['Multiline', ['mcc/C1', 'mcc/C2', 'mcc/C3']],
        },
    }

    tb_logger.add_custom_scalars(layout)

    def evaluate(old_loss, eval_counter, decline):
        return_loss, return_decline = old_loss, decline
        new_eval_loss, num_eval_seqs_now = eval_model(
            model, eval_counter, validation_dataloaders, embeddings,
            logger, tb_logger)
        if new_eval_loss <= old_loss:
            return_decline = 0
            return_loss = new_eval_loss
            utils.save_model(model_save_path, model_name, 'best', model, optim,
                             return_loss / num_eval_seqs_now, epoch, eval_counter)
            logger.info(f"# Saving model to {model_save_path}")
        else:
            return_decline += 1
        model.train()
        return return_loss, eval_counter + 1, num_eval_seqs_now, return_decline

    iterations_counter, eval_counter, = -1, 0
    eval_loss, patience, decline = evaluation_loss or float('inf'), 100, 0
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
                         total=len(dataloader),
                         position=0, leave=False, ascii=True),
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
                    utils.checkpoint_model(model_save_path, model_name, epoch, num_epochs,
                                           model, optim, considered_loss, save_epoch=False)
                    logger.info(f"# Checkpoint model to {model_save_path}/checkpoint/")

                loss, b = train_step(model, z0, z1, y, embeddings, evaluate=False)
                num_seqs += b
                batch_loss = loss.item()
                epoch_loss += batch_loss

                loss.backward()
                optim.step()
                optim.zero_grad()

                tb_logger.add_scalar("Loss Batch Train", batch_loss / b, batch_idx)

            tb_logger.add_scalar("Loss Epoch Train", epoch_loss / num_seqs, epoch + 1)
            utils.checkpoint_model(model_save_path, model_name, epoch, num_epochs,
                                   model, optim, epoch_loss / num_seqs, save_epoch=True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    args = parser.parse_args()
    utils.set_seed(42)

    out_dir = args.output_creation_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = args.tensorboard_path or out_dir / 'tensorboard'
    tb_dir.mkdir(parents=True, exist_ok=True)

    big_list = list()

    for (model_type, model_prefix, func) in [('attention', 'att', train_attention),
                                             ('projattention', 'proj', train_proj_attention),
                                             ('mlp', 'mlp', train_mlp), ('linear', 'linear', train_linear)]:
        args_d = vars(args)
        if hasattr(args, model_type) and args_d[model_type]:

            config_path = args_d[f'{model_prefix}_config']
            if not config_path:
                config = handle_config(args_d, f'{model_prefix}_')
            else:
                with open(config_path, 'r') as f:
                    config = json.load(f)

            args_d.update(config)
            model_name = args_d[f'{model_prefix}_model_name'] or utils.get_hash(config)
            tb_path = tb_dir / model_name
            tb_path.mkdir(parents=True, exist_ok=True)
            tb_logger = SummaryWriter(log_dir=tb_path)
            m_path = args.model_save_path or out_dir / model_prefix / 'models' / model_name
            m_path.mkdir(parents=True, exist_ok=True)
            l_path = args.logging_path or out_dir / model_prefix / 'models' / model_name / f'{model_name}.log'
            logger = utils.getlogger(l_path, name=model_name)
            utils.save_config(config, m_path)
            logger.info(f'Using {device}')
            logger.info('Create Dataloaders and load embeddings ...')
            big_list.append((func, logger, tb_logger, m_path, model_name))

    train_loader, train_ids = get_dataloaders_and_ids(
        args.train_file, args.batch_size, args.augment)
    val_loaders, val_ids = get_dataloaders_and_ids(
        args.val_file, args.batch_size, False, split_column='cclass')
    embeddings = get_embeddings(args.embedding, train_ids | val_ids, per_protein=True)

    for (func, logger, tb_logger, m_path, model_name) in big_list:
        logger.info('Dataloader and embeddings done!')
        func(args, train_loader, val_loaders, embeddings, logger, tb_logger, m_path, model_name)


if __name__ == '__main__':
    main()

