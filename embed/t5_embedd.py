from pathlib import Path
from time import perf_counter

import h5py
import torch
from tqdm.auto import tqdm

from utils.sequence_utils import parse_fasta


def get_model(model_type, cache_dir, device):
    if model_type == 't5':
        from transformers import T5EncoderModel, T5Tokenizer
        encoder, tokenizer, transformer_link = \
            T5EncoderModel, T5Tokenizer, 'Rostlab/prot_t5_xl_half_uniref50-enc'
    else:
        from transformers import BertModel, BertTokenizer
        encoder, tokenizer, transformer_link = \
            BertModel, BertTokenizer, 'Rostlab/prot_bert_bfd'

    print(f'Loading {model_type.upper()} from: {cache_dir}')
    print(f'Using huggingface from: {transformer_link}')

    model = encoder.from_pretrained(transformer_link,
                                    cache_dir=cache_dir)  # TODO half precision torch_dtype=torch.float(16)
    model = model.to(device)
    model = model.eval()
    vocab = tokenizer.from_pretrained(transformer_link,
                                      do_lower_case=False,
                                      cache_dir=cache_dir)
    return model, vocab


def embed_from_fasta(fasta_path, identifier_ref_output_path, output_path, cache_dir,
                     device, model_type, per_protein, half_precision, verbose=True):
    if verbose:
        print('Loading Model ...')
    model, vocab = get_model(model_type, cache_dir, device)

    get_embeddings(fasta_path, identifier_ref_output_path, output_path, model, vocab,
                   device, per_protein, half_precision)


def embedding_init(fasta_path, identifier_ref_output_path, model, half_precision, max_seq_len=600, verbose=True):
    if verbose:
        print('Loading Sequences ...')
    seq_dict = parse_fasta(fasta_path, identifier_ref_output_path)

    if verbose:
        print('Starting Embedding...')

    if half_precision:
        model = model.half()
        if verbose:
            print('Using model in half-precision!')

    if verbose:
        print('#' * 40)
        print('Example sequence: {}\n{}'.format(next(iter(
            seq_dict.keys())), next(iter(seq_dict.values()))))
        print('-' * 40)
        print('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
    n_long = sum([1 for _, seq in seq_dict.items() if len(seq) > max_seq_len])
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(seq_dict[kv[0]]), reverse=True)

    if verbose:
        print('Average sequence length: {}'.format(avg_length))
        print('Number of sequences >{}: {}'.format(max_seq_len, n_long))
        print('#' * 40)

    return seq_dict, model, avg_length


def process_batch(batch, model, vocab, device, per_protein, emb_dict):
    pdb_ids, seqs, seq_lens = zip(*batch)

    token_encoding = vocab(seqs, add_special_tokens=True,
                           padding='longest', return_tensors='pt')
    input_ids = token_encoding['input_ids'].to(device)
    attention_mask = token_encoding['attention_mask'].to(device)

    try:
        # batch-size x seq_len x embedding_dim
        with torch.no_grad():
            embedding_repr = model(input_ids, attention_mask=attention_mask)
    except RuntimeError:
        print('RuntimeError for {} (L={})'.format(pdb_ids, seq_lens))
        return emb_dict

    new_emb_dict = dict()
    for batch_idx, identifier in enumerate(pdb_ids):
        s_len = seq_lens[batch_idx]
        emb = embedding_repr.last_hidden_state[batch_idx, :s_len]

        if per_protein:
            emb = emb.mean(dim=0)

            # if len(emb_dict) == 0:
            #    print('Embedded protein {} with length {} to emb. of shape: {}'.format(
            #        identifier, s_len, emb.shape))
        new_emb_dict[identifier] = emb.detach().cpu().numpy().squeeze()

    if new_emb_dict:
        emb_dict.update(new_emb_dict)
    return emb_dict


saving_pattern = 'w'


def save_embeddings(output_path, emb_dict):
    global saving_pattern
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(output_path), saving_pattern) as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)
    saving_pattern = 'a'


def get_embeddings(fasta_path, identifier_ref_output_path, output_path, model, vocab, device, per_protein,
                   half_precision, max_residues=8000, max_seq_len=600, max_batch=5, process_update=.1):
    seq_dict, model, avg_length = embedding_init(
        fasta_path, identifier_ref_output_path, model, half_precision, max_seq_len)

    emb_dict, emb_count = dict(), 0
    emb_start = perf_counter()
    batch, n_res_batch = [], 0

    process_slicer = (len(seq_dict) * process_update)
    num_seqs = len(seq_dict)

    for seq_idx, (pdb_id, seq) in tqdm(enumerate(seq_dict, 1), total=num_seqs, position=0, leave=True, ascii=True):
        original_seq = seq
        seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        seq_len = len(seq)
        seq = ' '.join(list(seq))

        if seq_len >= max_seq_len:
            emb_dict = process_batch([(pdb_id, seq, seq_len)], model, vocab, device, per_protein, emb_dict)
        else:
            if len(batch) >= max_batch or n_res_batch >= max_residues:
                emb_dict = process_batch(batch, model, vocab, device, per_protein, emb_dict)
                batch = []
                n_res_batch = 0

            batch.append((pdb_id, seq, seq_len))
            n_res_batch += seq_len

        if len(emb_dict) > 200:
            emb_count += len(emb_dict)
            save_embeddings(output_path, emb_dict)
            emb_dict = dict()

        if seq_idx % process_slicer == 0:
            print('{} / {}:\n\tLast Sequence: {} -> {}'.format(seq_idx, num_seqs, pdb_id, original_seq))

    if batch:
        emb_dict = process_batch(batch, model, vocab, device, per_protein, emb_dict)

    if emb_dict:
        emb_count += len(emb_dict)
        save_embeddings(output_path, emb_dict)
    emb_end = perf_counter()
    print('Embeddings stored to: {}'.format(output_path))

    print('\n############# STATS #############')
    print('Total number of embeddings: {}'.format(emb_count))
    print('Total time: {:.2f}[s]; time/prot: {:.2f}[s]; avg. len= {:.2f}'.format(
        emb_end - emb_start, (emb_end - emb_start) / emb_count, avg_length))
