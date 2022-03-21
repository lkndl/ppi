import sys
from pathlib import Path

ppi_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ppi_path))

import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from sklearn.metrics import average_precision_score as average_precision, precision_score, \
    recall_score, f1_score, accuracy_score, matthews_corrcoef as mcc
from utils.general_utils import wipe_memory, getlogger
from utils.dataloader import get_training_dataloader
from torch.utils.tensorboard import SummaryWriter
from training.perresidue.models.interaction import InteractionMap
from training.perresidue.models.interaction_dscript import InteractionMapDscript
from time import perf_counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    """
    Set the random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def add_args(parser):
    """
    Create parser for command line utility.

    :meta private:
    """

    data_grp = parser.add_argument_group("Data")
    proj_grp = parser.add_argument_group("Projection Module")
    contact_grp = parser.add_argument_group("Contact Module")
    inter_grp = parser.add_argument_group("Interaction Module")
    train_grp = parser.add_argument_group("Training")
    misc_grp = parser.add_argument_group("Output and Device")

    # Data
    data_grp.add_argument("--train", help="Training data")
    data_grp.add_argument("--val", help="Validation data")
    data_grp.add_argument("--embedding", help="h5 file with embedded sequences")
    data_grp.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)

    # Embedding model
    proj_grp.add_argument(
        "--projection-dim",
        type=int,
        default=100,
        help="Dimension of embedding projection layer (default: 100)",
    )
    proj_grp.add_argument(
        "--dropout-p",
        type=float,
        default=0.1,
        help="Parameter p for embedding dropout layer (default: 0.5)",
    )

    # Contact model
    contact_grp.add_argument(
        "--hidden-dim",
        type=int,
        default=50,
        help="Number of hidden units for comparison layer in contact prediction (default: 50)",
    )
    contact_grp.add_argument(
        "--kernel-width",
        type=int,
        default=7,
        help="Width of convolutional filter for contact prediction (default: 7)",
    )

    # Interaction Model
    inter_grp.add_argument(
        "--no-w",
        action="store_false",
        dest='use_w',
        help="Don't use weight matrix in interaction prediction model",
    )
    inter_grp.add_argument(
        "--pool-width",
        type=int,
        default=9,
        help="Size of max-pool in interaction model (default: 9)",
    )

    # Training
    train_grp.add_argument(
        "--epoch-scale",
        type=int,
        default=1,
        help="Report heldout performance every this many epochs (default: 1)",
    )
    train_grp.add_argument("--num_epochs", type=int, default=10, help="Number of epochs (default: 10)")
    train_grp.add_argument("--batch-size", type=int, default=25, help="Minibatch size (default: 25)")
    train_grp.add_argument("--weight-decay", type=float, default=0, help="L2 regularization (default: 0)")
    train_grp.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    train_grp.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=0.35,
        help="Weight on the similarity objective (default: 0.35)",
    )

    # Output
    misc_grp.add_argument("--checkpoint", help="Checkpoint model to start training from")
    misc_grp.add_argument('--output_creation_dir', type=Path, required=True)
    misc_grp.add_argument('--logging_path', type=Path, required=False)
    misc_grp.add_argument('--tensorboard_path', type=Path, required=False)
    misc_grp.add_argument('--model_save_path', type=Path, required=False)
    misc_grp.add_argument("--use_dscript", action=argparse.BooleanOptionalAction, default=False)

    return parser


def process_batch(model, n0, n1, embeddings, eval=False):
    batch_size = len(n0)

    predictions = []
    interaction_maps = []
    for i in range(batch_size):
        z_a = embeddings[n0[i]].to(device)
        z_b = embeddings[n1[i]].to(device)

        if eval:
            predict = model.predict(z_a, z_b)
        else:
            cm, predict = model.map_predict(z_a, z_b)
            interaction_maps.append(torch.mean(cm))
        predictions.append(predict)
    predictions = torch.stack(predictions, 0)
    if eval:
        return predictions
    interaction_maps = torch.stack(interaction_maps, 0)
    return interaction_maps, predictions


def step(model, n0, n1, y, embeddings, weight=0.35, eval=False):
    if eval:
        predictions = process_batch(model, n0, n1, embeddings, eval)
    else:
        c_map_mag, predictions = process_batch(model, n0, n1, embeddings, eval)
    y = Variable(y).to(device)

    loss = nn.BCELoss()(predictions.float(), y.float())

    if not eval:
        cmap_loss = torch.mean(c_map_mag)
        loss = (weight * loss) + ((1 - weight) * cmap_loss)
    batch_size = len(predictions)

    if eval:
        return loss, batch_size, predictions
    return loss, batch_size


def eval_model(model, eval_counter, pairs_val_dataloader, embeddings, interaction_weight, logger, tensorboard_logger):
    logger.info(' Evaluation '.center(40, '+'))
    model.eval()
    with torch.no_grad():
        predictions, labels = [], []
        eval_loss, num_seqs = 0, 0
        for n0, n1, y in tqdm(pairs_val_dataloader, total=len(pairs_val_dataloader), desc=f'Evaluation {eval_counter}',
                              position=0, leave=True, ascii=True):
            loss, batch_size, prediction = step(model, n0, n1, y, embeddings, weight=interaction_weight, eval=True)
            predictions.append(prediction)
            labels.append(y)
            eval_loss += loss
            num_seqs += batch_size

        labels = torch.cat(labels, 0).to(device).float()
        predictions = torch.cat(predictions, 0).to(device).float()

        threshold = .5
        bin_predictions = ((threshold * torch.ones(num_seqs)) < predictions).float()

        eval_acc = accuracy_score(labels, bin_predictions)
        eval_pr = precision_score(labels, bin_predictions)
        eval_re = recall_score(labels, bin_predictions)
        eval_f1 = f1_score(labels, bin_predictions)
        eval_aupr = average_precision(labels, bin_predictions)
        eval_mcc = mcc(labels, bin_predictions)

        tensorboard_logger.add_scalar("Loss Epoch Val", eval_loss, eval_counter)
        tensorboard_logger.add_scalar("Accuracy Epoch Val", eval_acc, eval_counter)
        tensorboard_logger.add_scalar("Precision Epoch Val", eval_pr, eval_counter)
        tensorboard_logger.add_scalar("Recall Epoch Val", eval_re, eval_counter)
        tensorboard_logger.add_scalar("F1 Epoch Val", eval_f1, eval_counter)
        tensorboard_logger.add_scalar("AUCPR Epoch Val", eval_aupr, eval_counter)
        tensorboard_logger.add_scalar("MCC Epoch Val", eval_mcc, eval_counter)
        logger.info(f'loss: {eval_loss}, acc: {eval_acc}, aupr: {eval_aupr}\npr: {eval_pr}, re: {eval_re}, f1: {eval_f1}, mcc: {eval_mcc}')

    return eval_loss, num_seqs


def train(model, optim, num_epochs, pairs_train_dataloader, pairs_val_dataloader, embeddings,
          interaction_weight, logger,
          tensorboard_logger, model_save_path, use_dscript=False, evaluation_loss=None, start_epoch=0):

    def evaluate(old_loss, eval_counter, decline):
        return_loss, return_decline = old_loss, decline
        new_eval_loss, num_eval_seqs_now = eval_model(model, eval_counter, pairs_val_dataloader, embeddings, interaction_weight, logger, tensorboard_logger)
        if new_eval_loss <= old_loss:
            return_decline = 0
            return_loss = new_eval_loss
            save_model(model_save_path, 'best', model, optim, return_loss / num_seqs, epoch)
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
                    tqdm(pairs_train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(pairs_train_dataloader),
                         position=0, leave=True, ascii=True),
                    (epoch - 1) * len(pairs_train_dataloader)):
                iterations_counter += 1
                if iterations_counter % (max(int(.05 * value_rounder(len(pairs_train_dataloader), 3)), 1)) == 0:
                    epoch_train_time += perf_counter() - train_epoch_start
                    eval_time_start = perf_counter()
                    eval_loss, eval_counter, num_eval_seqs, decline = evaluate(eval_loss, eval_counter, decline)
                    eval_total_timer += perf_counter() - eval_time_start

                    if decline >= patience:
                        raise NotImplementedError(" Termination due to patience exceedance! ".center(100, '!'))

                    train_epoch_start = perf_counter()
                if iterations_counter % (max(int(.025 * value_rounder(len(pairs_train_dataloader), 3)), 1)) == 0:
                    if eval_loss == float('inf') and num_seqs != 0:
                        considered_loss = epoch_loss / num_seqs
                    else:
                        considered_loss = float('inf')
                    checkpoint_model(model_save_path, epoch, num_epochs, model, optim, considered_loss, save_epoch=False)
                    logger.info(f"# Checkpoint model to {model_save_path}/checkpoint/")

                loss, b = step(model, z0, z1, y, embeddings, weight=interaction_weight, eval=False)
                num_seqs += b
                epoch_loss += loss

                loss.backward()
                optim.step()
                optim.zero_grad()
                if use_dscript:
                    model.clip()

                tensorboard_logger.add_scalar("Loss Batch Train", loss / b, batch_idx)

            tensorboard_logger.add_scalar("Loss Epoch Train", epoch_loss / num_seqs, epoch + 1)
            checkpoint_model(model_save_path, epoch, num_epochs, model, optim, epoch_loss / num_seqs, save_epoch=True)
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
    except NotImplementedError:
        logger.info(" Termination due to patience exceedance! ".center(100, '!'))
    total_time = perf_counter() - total_time
    logger.info(' Time Elapsed '.center(100, '*'))
    logger.info(f'Time per Epoch in h: {total_time / num_epochs / 3600}')
    logger.info(f'Train Time per Epoch in h: {train_total_timer / num_epochs / 3600}')
    logger.info(f'Evaluation Time per Evaluation in h: {eval_total_timer / eval_counter / 3600}')
    logger.info('-----------')
    logger.info(f'Train Time per Epoch in s per Pair: {train_total_timer / num_epochs / num_train_seqs}')
    logger.info(f'Evaluation Time per Evaluation in s per Pair: {eval_total_timer / eval_counter / num_eval_seqs}')
    save_final(model_save_path)


def save_model(model_save_path, model_text, model, optimizer, loss, epoch):
    model.cpu()
    torch.save({
        'model': model,
        'optimizer': optimizer,
        'loss': loss,
        'epoch': epoch
    }, f'{model_save_path}/model_{model_text}.pth')
    model.to(device)
    wipe_memory()


def checkpoint_model(model_save_path, epoch, num_max_epoch, model, optimizer, loss, save_epoch=False):
    digits = int(np.floor(np.log10(num_max_epoch))) + 1
    save_path = model_save_path.joinpath(f'checkpoint/')
    save_path.mkdir(parents=True, exist_ok=True)
    save_model(save_path, f'epoch{str(epoch + 1).zfill(digits)}{"_checkpoint" if not save_epoch else ""}', model, optimizer, loss, epoch)


def save_final(model_save_path):
    best_checkpoint = torch.load(f'{model_save_path}/model_best.pth')
    model_to_save = best_checkpoint['model']
    torch.save(model_to_save, f'{model_save_path}/model_final.pth')


def main(args):
    """
    Run training from arguments.

    :meta private:
    """
    set_seed(42)

    # TODO maybe use a more precise experiment_spec
    experiment_specs = f'residue_{"dscript_" if args.use_dscript else ""}lr{args.lr}_intactw{args.lambda_}{"_aug" if args.augment else ""}'

    output_creation_dir = args.output_creation_dir
    logging_path = args.logging_path or output_creation_dir.joinpath(f'logs/{experiment_specs}/runlog.log')
    logger = getlogger(logging_path)

    tensorboard_path = args.tensorboard_path or output_creation_dir.joinpath(f'tensorboard/{experiment_specs}')
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    tensorboard_logger = SummaryWriter(log_dir=tensorboard_path)

    model_save_path = args.model_save_path or output_creation_dir.joinpath(f'models/{experiment_specs}')
    model_save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f'Using {device}')

    batch_size = args.batch_size
    train_fi = args.train
    val_fi = args.val
    augment = args.augment
    embedding_h5 = args.embedding

    logger.info('# Create Dataloaders and load embeddings!')
    pairs_train_dataloader, pairs_val_dataloader, embeddings = get_training_dataloader(train_fi, augment, batch_size, 2,
                                                                                       val_fi, batch_size, 2,
                                                                                       embedding_h5)
    logger.info('> Dataloader and embeddings done!')

    if args.checkpoint is None:
        projection_dim = args.projection_dim
        dropout_p = args.dropout_p
        hidden_dim = args.hidden_dim
        kernel_width = args.kernel_width
        use_W = args.use_w
        pool_width = args.pool_width

        if args.use_dscript:
            logger.info('# Initialize D-Script Architecture')
            model = InteractionMapDscript(emb_projection_dim=projection_dim, dropout_p=dropout_p,
                                          map_hidden_dim=hidden_dim, kernel_width=kernel_width, use_W=use_W,
                                          pool_size=pool_width)
        else:
            logger.info('# Initialize D-Script inspired Architecture')
            model = InteractionMap(emb_projection_dim=projection_dim, dropout_p=dropout_p,
                                   map_hidden_dim=hidden_dim, kernel_width=kernel_width,
                                   pool_size=pool_width)

        lr = args.lr
        wd = args.weight_decay
        params = [p for p in model.parameters() if p.requires_grad]
        optim = Adam(params, lr=lr, weight_decay=wd)
        evaluation_loss = None
        start_epoch = 0
    else:
        logger.info(f"# Loading model from checkpoint {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model = checkpoint['model']
        optim = checkpoint['optimizer']
        evaluation_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    model = model.to(device)
    logger.info('> Model loading done!')

    num_epochs = args.num_epochs
    interaction_weight = args.lambda_
    cmap_weight = 1 - interaction_weight

    train(model, optim, num_epochs, pairs_train_dataloader, pairs_val_dataloader, embeddings,
          interaction_weight, logger, tensorboard_logger, model_save_path=model_save_path,
          use_dscript=args.use_dscript, evaluation_loss=evaluation_loss, start_epoch=start_epoch)

    tensorboard_logger.flush()
    tensorboard_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
