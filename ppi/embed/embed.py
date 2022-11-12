#!/usr/bin/env python3

import time
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch
import tqdm
from transformers import T5Tokenizer, T5EncoderModel


def read_fasta(fasta_path: Path, split_char='!', id_field=0):
    """
    Reads in fasta file containing multiple sequences.
    Split_char and id_field allow to control identifier extraction from header.
    E.g.: set split_char='|' and id_field=1 for SwissProt/UniProt Headers.
    Returns dictionary holding multiple sequences or only single
    sequence, depending on input file.
    """

    seqs = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace('/', '_').replace('.', '_')
                seqs[uniprot_id] = ''
            else:
                # repl. all white-space chars and join seqs spanning multiple lines,
                # drop gaps and cast to upper-case
                seq = ''.join(line.split()).upper().replace('-', '')
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                seqs[uniprot_id] += seq
    example_id = next(iter(seqs))
    print(f'Read {len(seqs)} sequences.')
    print(f'Example:\n{example_id}\n{seqs[example_id]}')

    return seqs


def save_embeddings(emb_dict: dict, out_path: Path, mode: str = 'w'):
    with h5py.File(str(out_path), mode) as hf:
        for sequence_id, embedding in emb_dict.items():
            hf.create_dataset(sequence_id, data=embedding)
    return None


def get_T5_model():
    """Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)"""
    model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # move model to GPU
    model = model.eval()  # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    return model, tokenizer, device


def get_embeddings(model, tokenizer, seqs, device,
                   max_residues=4000, max_seq_len=1500, max_batch=100):
    """
    Generate embeddings via batch-processing
    :param model:
    :param tokenizer:
    :param seqs:
    :param max_residues: the upper limit of residues within one batch
    :param max_seq_len: the upper sequence length for applying batch-processing
    :param max_batch: the upper number of sequences per batch
    :return:
    """
    emb_dict = dict()

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in tqdm.tqdm(
            enumerate(seq_dict, 1), total=len(seq_dict), ascii=' ▖▘▝▗▚▞█'):
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues \
                or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding='longest')
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print('RuntimeError during embedding for {} (L={})'.format(pdb_id, seq_len))
                continue

            for batch_idx, identifier in enumerate(pdb_ids):  # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim  
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]

                # store per-residue embeddings (Lx1024)
                emb_dict[identifier] = emb.detach().cpu().numpy().squeeze()

    passed_time = time.time() - start
    avg_time = passed_time / len(emb_dict)
    print(f'Total number of per-residue embeddings: {len(emb_dict)}')
    print(f'Time for generating embeddings: {passed_time / 60:.1f}[m] ({avg_time:.3f}[s/protein])')
    return emb_dict


if __name__ == '__main__':

    ppi_wd = Path('/mnt/project/kaindl/ppi/data/ppi_dataset')
    seqs = dict()
    for fasta in ppi_wd.glob('*.fasta'):
        seqs.update(read_fasta(fasta))
    all_needed_ids = set(seqs)
    print(f'Need {len(all_needed_ids)} embeddings overall')

    old_h5_files = [Path(p) for p in [
        '/mnt/project/ducanh.le/PPI/ppi_t5/embeddings/apid_huri_emb.h5',
        '/mnt/project/ducanh.le/PPI/ppi_t5/embeddings/old_apid_huri_emb.h5']]

    out_h5 = Path('/mnt/project/kaindl/ppi/embed_data/apid_huri.h5')

    # copy over previously generated embeddings
    with h5py.File(out_h5, 'w') as new_h5:
        for old_h5_file in old_h5_files[1:]:
            with h5py.File(old_h5_file, 'r') as old_h5:
                for key, mbed in tqdm.tqdm(
                        old_h5.items(), desc=f'iterate {old_h5_file.stem}'):
                    if key in all_needed_ids:
                        # write to the new H5
                        new_h5[key] = np.array(mbed)
                        # do not look for this ID again
                        all_needed_ids.remove(key)

    print(f'Generate {len(all_needed_ids)} new embeddings')
    seqs = {k: v for k, v in seqs.items() if k in all_needed_ids}
    print('Loading model')
    model, tokenizer, device = get_T5_model()
    print('Start embedding')
    results = get_embeddings(model, tokenizer, seqs, device)
    save_embeddings(results, out_h5, 'a')
