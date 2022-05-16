from pathlib import Path
from typing import Union

import h5py
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm

#from .sequence_utils import load_identifiers_json


def get_embedding_from_ids(id_list: list, h5_path: Union[str, Path],
                           ref_path: Union[str, Path]) -> dict:
    """TODO this is unused so far!"""
    print('# Loading identifier references')
    identifiers = load_identifiers_json(ref_path)
    assert h5_path.is_file()
    print('# Load corresponding embeddings')
    emb_dict = dict()
    with h5py.File(h5_path, 'r') as f:
        for idx in id_list:
            emb_dict[idx] = f.get(identifiers[idx], default=None)
    return emb_dict


def get_training_dataloader(train_file_path, augment, train_batch_size, train_label_column,
                            val_file_path, val_batch_size, val_label_column, val_cclass_column, embedding_file_path, perprot=False):
    # TODO refactor dataloading functions to have less redundant code
    pairs_train_dataloader, train_embeddings = get_dataloader_and_embeddings(train_file_path, augment, train_batch_size, train_label_column, embedding_file_path, perprot)
    pairs_val_dataloaders, val_embeddings = get_c123_dataloader(val_file_path, val_batch_size, val_label_column, val_cclass_column, embedding_file_path, perprot)
    train_embeddings.update(val_embeddings)

    return pairs_train_dataloader, pairs_val_dataloaders, train_embeddings


def get_c123_dataloader(pairs_file_path, batch_size, label_column, cclass_column, embedding_file_path, perprot=False):
    pairs_df = pd.read_csv(pairs_file_path, sep="\t")
    pairs_df = pairs_df.astype({'hash_A': str, 'hash_B': str})

    val_cclass = [(pairs_df[pairs_df[cclass_column] == i]['hash_A'].reset_index(drop=True),
                   pairs_df[pairs_df[cclass_column] == i]['hash_B'].reset_index(drop=True),
                   torch.from_numpy(pairs_df[pairs_df[cclass_column] == i][label_column].values)) for i in range(1, 4)]
    cc_dataloaders = [torch.utils.data.DataLoader(
        PairedDataset(*val_set),
        batch_size=batch_size,
        collate_fn=collate_paired_sequences
    ) for val_set in val_cclass]

    with h5py.File(embedding_file_path, "r") as embedd_file:
        embeddings = {}
        all_proteins = set(pairs_df['hash_A']).union(set(pairs_df['hash_B']))
        for prot_name in tqdm(all_proteins, desc=f'Load embeddings for {pairs_file_path}', position=0, leave=True, ascii=True):
            if perprot:
                embeddings[prot_name] = torch.from_numpy(embedd_file[prot_name][:, :]).float().mean(dim=0).unsqueeze(
                    0).unsqueeze(0)
            else:
                embeddings[prot_name] = torch.from_numpy(embedd_file[prot_name][:, :]).float().unsqueeze(0)

    return cc_dataloaders, embeddings


def get_dataloader_and_embeddings(pairs_file_path, augment, batch_size, label_column, embedding_file_path, perprot=False):
    pairs_df = pd.read_csv(pairs_file_path, sep="\t")
    pairs_df = pairs_df.astype({'hash_A': str, 'hash_B': str})

    if augment:
        prot_n0 = pd.concat((pairs_df['hash_A'], pairs_df['hash_B']), axis=0).reset_index(drop=True)
        prot_n1 = pd.concat((pairs_df['hash_B'], pairs_df['hash_A']), axis=0).reset_index(drop=True)
        pair_y = torch.from_numpy(pd.concat((pairs_df[label_column], pairs_df[label_column])).values)
    else:
        prot_n0, prot_n1 = pairs_df['hash_A'], pairs_df['hash_B']
        pair_y = torch.from_numpy(pairs_df[label_column].values)

    interact_pairs = PairedDataset(prot_n0, prot_n1, pair_y)
    pairs_dataloader = torch.utils.data.DataLoader(
        interact_pairs,
        batch_size=batch_size,
        collate_fn=collate_paired_sequences,
        shuffle=True,
    )

    with h5py.File(embedding_file_path, "r") as embedd_file:
        embeddings = {}
        all_proteins = set(prot_n0).union(set(prot_n1))
        for prot_name in tqdm(all_proteins, desc=f'Load embeddings for {pairs_file_path}', position=0, leave=True, ascii=True):
            if perprot:
                embeddings[prot_name] = torch.from_numpy(embedd_file[prot_name][:, :]).float().mean(dim=0).unsqueeze(
                    0).unsqueeze(0)
            else:
                embeddings[prot_name] = torch.from_numpy(embedd_file[prot_name][:, :]).float().unsqueeze(0)

    return pairs_dataloader, embeddings


class PairedDataset(torch.utils.data.Dataset):
    """
    Dataset to be used by the PyTorch data loader for pairs of sequences and their labels.

    :param X0: List of first item in the pair
    :param X1: List of second item in the pair
    :param Y: List of labels
    """

    def __init__(self, X0, X1, Y):
        self.X0 = X0
        self.X1 = X1
        self.Y = Y
        assert len(X0) == len(X1), "X0: " + str(len(X0)) + " X1: " + str(len(X1)) + " Y: " + str(len(Y))
        assert len(X0) == len(Y), "X0: " + str(len(X0)) + " X1: " + str(len(X1)) + " Y: " + str(len(Y))

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, i):
        return self.X0[i], self.X1[i], self.Y[i]


def collate_paired_sequences(args):
    """
    Collate function for PyTorch data loader.
    """
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)
