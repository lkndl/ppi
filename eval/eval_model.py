import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics as skl
from tqdm.auto import tqdm

from utils import general_utils as utils
from utils.dataloader import get_embeddings

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_RESIDUE, MODEL_PROT = 'perresidue', 'perprot'


def add_args(parser):
    parser.add_argument("--models", nargs='+', type=Path,
                        help="Trained prediction models, use multiple to evaluate more than one on one set"
                             "A mix of perprot and perresidue is not possible", required=True)
    parser.add_argument("--tests", nargs='+', type=Path, help="Test Data, use multiple to evaluate more than once")
    parser.add_argument("--embeddings", nargs='+', type=Path,
                        help="h5 file with embedded sequences, either contains all necessary embeddings or a list of embeddings is given"
                             "matching number of test files.")
    parser.add_argument("--model_type", type=str, choices=[MODEL_RESIDUE, MODEL_PROT], default=MODEL_RESIDUE)
    parser.add_argument('--output_creation_dir', type=Path, required=False)
    parser.add_argument('--logging_path', type=Path, required=False)
    parser.add_argument('--eval_path', type=Path, required=False)
    return parser


def load_data(eval_set_paths, emb_paths, model_type,
              id_columns: list[str] = ['hash_A', 'hash_B']):
    if len(emb_paths) == len(eval_set_paths):
        embeddings_paths = emb_paths
    elif len(emb_paths) == 1:
        embeddings_paths = [emb_paths[0]] * len(eval_set_paths)
    else:
        raise RuntimeError('Arbitrary Match between test files and embeddings files possible')

    test_dfs, embeddings = list(), dict()
    for eval_set_path, emb_path in zip(eval_set_paths, embeddings_paths):
        df = pd.read_csv(eval_set_path, sep='\t', header=0)
        test_dfs.append(df)
        embeddings.update(get_embeddings(emb_path, set(df[id_columns].values), per_protein=model_type == 'perprot'))
    return test_dfs, embeddings


def eval_model(model, test_df, embeddings, logger, save_path, test_name, model_name,
               id_columns: list[str] = ['hash_A', 'hash_B'],
               label_column: str = 'label',
               ):
    logger.info(f' Evaluation {test_name} for model {model_name} '.center(120, '+'))
    model.eval()

    start_inference = perf_counter()
    with torch.no_grad(), \
            open(save_path / f'predictions_{test_name}_{model_name}.tsv', 'w+') as out_file:
        predictions, labels = [], []
        eval_loss, num_pairs = 0, 0

        for _, (n0, n1, label) in tqdm(test_df[id_columns + [label_column]].iterrows(),
                                       total=len(test_df), desc="Predicting pairs",
                                       position=0, leave=False, ascii=True):
            num_pairs += 1
            p0 = embeddings[n0].to(device)
            p1 = embeddings[n1].to(device)

            prediction = model.predict(p0, p1).item()
            predictions.append(prediction)
            labels.append(label)
            out_file.write(f'{n0}\t{n1}\t{label}\t{prediction:.5}')
        end_inference = perf_counter()

        predictions = torch.Tensor(predictions)
        labels = torch.Tensor(labels)
        eval_loss = nn.BCELoss()(predictions, labels)

        bin_predictions = ((.5 * torch.ones(num_pairs)) < predictions).float()

        eval_acc = skl.accuracy_score(labels, bin_predictions)
        eval_pr = skl.precision_score(labels, bin_predictions, zero_division=0)
        eval_re = skl.recall_score(labels, bin_predictions, zero_division=0)
        eval_f1 = skl.f1_score(labels, bin_predictions, zero_division=0)
        eval_aupr = skl.average_precision_score(labels, predictions)
        eval_mcc = skl.matthews_corrcoef(labels, bin_predictions)

        logger.info(
            f'loss: {eval_loss}, acc: {eval_acc}, aupr: {eval_aupr}\npr: {eval_pr}, re: {eval_re}, f1: {eval_f1}, mcc: {eval_mcc}')
        logger.info(f'Elapsed Time in   s: {end_inference - start_inference:.3f}')
        logger.info(f'Elapsed Time in min: {(end_inference - start_inference) / 60:.4f}')
        logger.info(f'Elapsed Time in   h: {(end_inference - start_inference) / 3600:.6f}')
        logger.info(f'Time per pair in  s: {(end_inference - start_inference) / num_pairs:.6f}')
        logger.info(f'Number of pairs    : {num_pairs}')
    return eval_loss, num_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    args = parser.parse_args()

    model_paths = args.models

    emb_paths = args.embeddings
    test_files = args.tests
    test_sets, embeddings = load_data(test_files, emb_paths, args.model_type)

    for model_path in model_paths:
        model = torch.load(model_path).to(device)

        out_dir = args.output_creation_dir or model_path.parent
        model_name = model_path.stem

        for test_set, test_file in zip(test_sets, test_files):
            test_name = test_file.stem

            eval_path = args.eval_path or out_dir / 'eval' / test_name
            Path.mkdir(eval_path, parents=True, exist_ok=True)

            logging_path = args.logging_path or eval_path / f'{test_name}_{model_name}.log'
            logger = utils.getlogger(logging_path, name=f'{model_name}{test_name}')
            logger.info(
                f'Number parameters {model_name}: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
            eval_model(model, test_set, embeddings, logger, eval_path, test_name, model_name)


if __name__ == '__main__':
    main()
