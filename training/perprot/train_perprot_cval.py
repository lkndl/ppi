import argparse
from time import perf_counter

import torch
from tqdm.auto import tqdm

from train_perprot import step_model, add_args, main
from training.perresidue.train_t5 import checkpoint_model
from utils import general_utils as utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def eval_model(model, eval_counter, pairs_val_dataloaders, embeddings, logger, tensorboard_logger):
    logger.info(f' Evaluation {eval_counter} '.center(40, '+'))
    model.eval()
    with torch.no_grad():
        # classes = {'c1': 0, 'c2': 0, 'c3': 0}
        # metrics = {'acc': classes.copy(), 'pr': classes.copy(), 're': classes.copy(), 'f1': classes.copy(),
        #            'aupr': classes.copy(), 'mcc': classes.copy()}
        for cclass, pairs_val_dataloader in enumerate(pairs_val_dataloaders, 1):
            predictions, labels = [], []
            eval_loss, num_seqs = 0, 0
            for n0, n1, y in tqdm(pairs_val_dataloader, total=len(pairs_val_dataloader),
                                  desc=f'Evaluation {eval_counter}',
                                  position=0, leave=True, ascii=True):
                loss, batch_size, prediction, label = step_model(model, n0, n1, y, embeddings, evaluate=True)
                predictions.append(prediction)
                labels.append(label)
                eval_loss += loss
                num_seqs += batch_size

            labels = torch.cat(labels, 0).view(-1).to(device).float().detach().cpu()
            predictions = torch.cat(predictions, 0).view(-1).to(device).float().detach().cpu()

            bin_predictions = ((.5 * torch.ones(num_seqs)) < predictions).float()
            utils.log_stats(eval_loss, labels, eval_counter,
                            bin_predictions, predictions,
                            tensorboard_logger, logger, sfx=f'/C{cclass}', lower=True)

    return eval_loss, num_seqs


def train_model(model, optim, num_epochs, pairs_train_dataloader, pairs_val_dataloaders,
                embeddings, logger, tensorboard_logger, model_save_path, model_name,
                evaluation_loss=None, start_epoch=0):
    logger.info(' Train new model '.center(50, '#'))
    logger.info(f'{model}')

    def_custom_scalar(tensorboard_logger)

    def evaluate(old_loss, eval_counter, decline):
        return_loss, return_decline = old_loss, decline
        new_eval_loss, num_eval_seqs_now = eval_model(model, eval_counter, pairs_val_dataloaders, embeddings,
                                                      logger, tensorboard_logger)
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
    eval_loss, patience, decline = evaluation_loss or float('inf'), 20, 0
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
                         position=0, leave=True, ascii=True),
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
                    checkpoint_model(model_save_path, model_name, epoch, num_epochs, model, optim, considered_loss,
                                     save_epoch=False)
                    logger.info(f"# Checkpoint model to {model_save_path}/checkpoint/")

                loss, b = step_model(model, z0, z1, y, embeddings, evaluate=False)
                num_seqs += b
                epoch_loss += loss

                loss.backward()
                optim.step()
                optim.zero_grad()

                tensorboard_logger.add_scalar("Loss Batch Train", loss / b, batch_idx)

            tensorboard_logger.add_scalar("Loss Epoch Train", epoch_loss / num_seqs, epoch + 1)
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


def def_custom_scalar(tensorboard_logger):
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

    tensorboard_logger.add_custom_scalars(layout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
