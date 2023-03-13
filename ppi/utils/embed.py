#!/usr/bin/env python3

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

import time
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm


class PLM(str, Enum):
    t5 = 't5'
    bert = 'bert'


def get_model(model_type: PLM = PLM.t5, half: bool = True, cache_dir: Path = None):
    if model_type == 't5':
        from transformers import T5EncoderModel, T5Tokenizer
        encoder, tokenizer, transformer_link = \
            T5EncoderModel, T5Tokenizer, 'Rostlab/prot_t5_xl_half_uniref50-enc'
    else:
        from transformers import BertModel, BertTokenizer
        encoder, tokenizer, transformer_link = \
            BertModel, BertTokenizer, 'Rostlab/prot_bert_bfd'

    cache_dir = cache_dir or ('t5_xl_weights' if model_type.value == 't5'
                              else 'prot_bert_bfd_weights')
    print(f'Loading {model_type.upper()} from: {cache_dir}')
    print(f'Using huggingface from: {transformer_link}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    kwargs = dict(torch_dtype=torch.float16) if half else dict()
    model = encoder.from_pretrained(transformer_link, cache_dir=cache_dir, **kwargs)
    model = model.to(device)
    model = model.eval()
    vocab = tokenizer.from_pretrained(transformer_link,
                                      do_lower_case=False,
                                      cache_dir=cache_dir)
    return model, vocab, device


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


def generate_embeddings(model, tokenizer, seqs, device, h5_file: Path = None,
                        per_protein: bool = False, keep_in_memory: bool = True,
                        max_residues=4000, max_seq_len=1500, max_batch=100):
    """
    Generate embeddings via batch-processing
    :param model:
    :param tokenizer:
    :param seqs:
    :param per_protein: average over the sequence length - or don't
    :param max_residues: the upper limit of residues within one batch
    :param max_seq_len: the upper sequence length for applying batch-processing
    :param max_batch: the upper number of sequences per batch
    :return:
    """
    emb_dict = dict()
    bunch = dict()
    counter = 0
    if not h5_file and not keep_in_memory:
        raise RuntimeError('Specified neither an output file nor to keep embeddings in RAM!')
    if h5_file and (h5 := Path(h5_file)).is_file():
        with h5py.File(h5, 'r') as open_h5:
            existing_keys = list(open_h5.keys())
        found, missing = set(seqs) & set(existing_keys), set(seqs) - set(existing_keys)
        if missing:
            print(f'Found {len(found)} keys already, generate {len(missing)} missing ones')
            seqs = {k: v for k, v in seqs.items() if k not in existing_keys}

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
            pdb_ids, batch_seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(
                batch_seqs, add_special_tokens=True, padding='longest')
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
                emb = emb.detach().cpu().numpy().squeeze()
                if per_protein:
                    # or a per-protein embedding (1x1024)
                    emb = emb.mean(0)
                bunch[identifier] = emb
                counter += 1

        if len(bunch) > 200:
            if h5_file:
                save_embeddings(bunch, h5_file)
            if keep_in_memory:
                emb_dict |= bunch
            bunch = dict()
    if keep_in_memory:
        emb_dict |= bunch
    if h5_file:
        save_embeddings(bunch, h5_file)

    passed_time = time.time() - start
    avg_time = passed_time / max(1, counter)
    print(f'Total number of per-residue embeddings: {counter}')
    print(f'Time for generating embeddings: {passed_time / 60:.1f}[m]'
          f' ({avg_time:.3f}[s/protein])')
    return emb_dict


def save_embeddings(emb_dict: dict, out_path: Path, mode: str = 'a'):
    with h5py.File(str(out_path), mode) as hf:
        for sequence_id, embedding in emb_dict.items():
            hf.create_dataset(sequence_id, data=embedding)
    return None


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
    model, tokenizer, device = get_model()
    print('Start embedding')
    _ = generate_embeddings(model, tokenizer, seqs, device, out_h5)
