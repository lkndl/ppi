import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataloader import get_embeddings, collate_paired_sequences, PairedDataset


def get_training_dataloader_cval(train_file_path, augment, train_batch_size, train_label_column,
                                 val_file_path, val_batch_size, val_label_column, val_cclass_column, embedding_file_path,
                                 perprot=False):
    # TODO refactor dataloading functions to have less redundant code
    pairs_train_dataloader, train_embeddings = get_dataloader_and_embeddings_cval(
        train_file_path, augment, train_batch_size,
        train_label_column, embedding_file_path, perprot)
    pairs_val_dataloaders, val_embeddings = get_c123_dataloader(
        val_file_path, val_batch_size,
        val_label_column, val_cclass_column,
        embedding_file_path, perprot)
    train_embeddings.update(val_embeddings)

    return pairs_train_dataloader, pairs_val_dataloaders, train_embeddings


def get_c123_dataloader(
        pairs_file_path, batch_size, label_column, cclass_column,
        embedding_file_path, perprot=False):
    pairs_df = pd.read_csv(pairs_file_path, sep="\t")

    val_cclass = [(d.hash_A, d.hash_B, torch.from_numpy(d[label_column].values))
                  for _, d in pairs_df.groupby(cclass_column, sort=True)]

    cc_dataloaders = [DataLoader(
        PairedDataset(*val_set),
        batch_size=batch_size,
        collate_fn=collate_paired_sequences
    ) for val_set in val_cclass]

    all_proteins = set(pairs_df['hash_A']) | set(pairs_df['hash_B'])
    embeddings = get_embeddings(embedding_file_path, all_proteins, perprot)

    return cc_dataloaders, embeddings


def get_dataloader_and_embeddings_cval(
        pairs_file_path, augment, batch_size, label_column,
        embedding_file_path, perprot=False):
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
    pairs_dataloader = DataLoader(
        interact_pairs,
        batch_size=batch_size,
        collate_fn=collate_paired_sequences,
        shuffle=True,
    )

    all_proteins = set(prot_n0) | set(prot_n1)
    embeddings = get_embeddings(embedding_file_path, all_proteins, perprot)

    return pairs_dataloader, embeddings
