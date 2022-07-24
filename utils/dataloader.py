from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd
import secrets
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from rich.progress import Progress


class ResumableRandomSampler(Sampler):
    """https://gist.github.com/usamec/1b3b4dcbafad2d58faa71a9633eea6a5"""

    def __init__(self, dataset: Dataset, seed: int, shuffle: bool = True):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.generator = torch.Generator().manual_seed(seed)

        self.shuffle = shuffle
        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)
        if not self.shuffle:
            self.perm = torch.arange(0, self.num_samples)

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)
            if not self.shuffle:
                self.perm = torch.arange(0, self.num_samples)

        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index - 1]

    def __len__(self):
        return self.num_samples

    def get_state(self):
        return {'perm': self.perm, 'perm_index': self.perm_index, 'shuffle': self.shuffle,
                'sampler_rng_state': self.generator.get_state()}

    def set_state(self, state):
        self.perm = state['perm']
        self.perm_index = state['perm_index']
        self.shuffle = state['shuffle']
        self.generator.set_state(state['sampler_rng_state'])


class PairedDataset(Dataset):
    def __init__(self, X0, X1, Y):
        self.X0 = X0
        self.X1 = X1
        self.Y = Y
        assert len(X0) == len(X1) == len(Y), f'X0: {len(X0)} X1: {len(X1)} Y: {len(Y)}'

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, i):
        return self.X0[i], self.X1[i], self.Y[i]


def collate_paired_sequences(args):
    """Collate function for PyTorch data loader."""
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)


def get_dataloaders_and_ids(tsv_path: Path,
                            batch_size: int = 50,
                            augment: bool = True,
                            id_columns: list[str] = ['hash_A', 'hash_B'],
                            label_column: str = 'label',
                            split_column: str = None,
                            shuffle: bool = True,
                            seed: int = torch.randint(0, int(1e6), (1,)).item()
                            ) -> tuple[Union[DataLoader, dict[str, DataLoader]], set[str]]:
    all_pairs = pd.read_csv(tsv_path, sep='\t', header=0)
    assert len(id_columns) == 2 and set(id_columns) < set(all_pairs.columns), \
        f'Only exactly two columns from {all_pairs.columns} can contain protein IDs, ' \
        f'but {id_columns} was passed.'

    if separate := (split_column is None):
        # add a dummy column
        split_column = secrets.token_urlsafe(8)
        all_pairs[split_column] = 0

    data_loader, loaders, prot_ids = None, dict(), set()
    for cclass, pairs in all_pairs.groupby(split_column, sort=True):
        prot_a, prot_b = pairs[id_columns].values.T
        prot_ids |= set(np.unique(prot_a)) | set(np.unique(prot_b))
        pair_label = torch.from_numpy(pairs[label_column].values)
        if augment:
            prot_a, prot_b = np.concatenate((prot_a, prot_b)), np.concatenate((prot_b, prot_a))
            pair_label = torch.concat((pair_label, pair_label))

        dataset = PairedDataset(prot_a, prot_b, pair_label)
        sampler = ResumableRandomSampler(dataset, seed, shuffle)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_paired_sequences,
            sampler=sampler,
        )
        loaders[str(cclass)] = data_loader

    return (data_loader if separate else loaders), prot_ids


def get_embeddings(h5_path: Path,
                   ids: set,
                   per_protein: bool = False,
                   progress: Progress = Progress()
                   ) -> dict[str, torch.Tensor]:
    with h5py.File(h5_path, 'r') as h5_file:
        embeddings = dict()
        for prot_id in progress.track(ids, description='read H5'):
            m_bed = torch.from_numpy(h5_file[prot_id][:, :]).float()
            if per_protein:
                embeddings[prot_id] = m_bed.mean(dim=0).unsqueeze(0).unsqueeze(0)
            else:
                embeddings[prot_id] = m_bed.unsqueeze(0)
        return embeddings


def get_training_dataloader(train_file_path, augment, train_batch_size, train_label_column,
                            val_file_path, val_batch_size, val_label_column, embedding_file_path, perprot=False):
    # TODO refactor dataloading functions to have less redundant code
    pairs_train_dataloader, train_embeddings = get_dataloader_and_embeddings(
        train_file_path, augment, train_batch_size,
        train_label_column, embedding_file_path, perprot)
    pairs_val_dataloader, val_embeddings = get_dataloader_and_embeddings(
        val_file_path, False, val_batch_size,
        val_label_column, embedding_file_path, perprot)
    train_embeddings.update(val_embeddings)

    return pairs_train_dataloader, pairs_val_dataloader, train_embeddings


def get_dataloader_and_embeddings(
        pairs_file_path, augment, batch_size, label_column,
        embedding_file_path, perprot=False):
    pairs_df = pd.read_csv(pairs_file_path, sep='\t', header=None)
    pairs_df = pairs_df.astype({0: str, 1: str})

    if augment:
        prot_n0 = pd.concat((pairs_df[0], pairs_df[1]), axis=0).reset_index(drop=True)
        prot_n1 = pd.concat((pairs_df[1], pairs_df[0]), axis=0).reset_index(drop=True)
        pair_y = torch.from_numpy(pd.concat((pairs_df[label_column], pairs_df[label_column])).values)
    else:
        prot_n0, prot_n1 = pairs_df[0], pairs_df[1]
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
