import argparse
import sys
from pathlib import Path

import torch

ppi_path = str(Path(__file__).resolve().parents[2])
if ppi_path not in sys.path:
    sys.path.append(ppi_path)

from t5_embedd import embed_from_fasta


def add_args(parser):
    parser.add_argument('--seqs', help='Sequences to be embedded', required=True)
    parser.add_argument('-o', '--outfile', help='h5 file to write results', required=True)
    parser.add_argument('--fastaref', help='Output path to CRC64C ID mappings.')  # defaults to outfile + '_ref'
    parser.add_argument('--prot', help='Calculate embeddings on per protein level', default=False)
    parser.add_argument('--half', help='Use half-precision', default=True)
    parser.add_argument('--model', choices=['t5', 'bert'], default='t5',
                        help='Create either ProtT5 or ProtBert embeddings')
    parser.add_argument('--cache_dir', default=Path.cwd() / 'cache')
    return parser


def main(args):
    inPath = args.seqs  # in fasta-format
    outPath = args.outfile
    fastaRef = args.fastaref if args.fastaref else (p := Path(outPath)).with_name(p.stem + '_ref')
    per_protein = args.prot
    half_precision = args.half

    cache_dir = args.cache_dir or ('t5_xl_weights' if args.model == 't5' else 'prot_bert_bfd_weights')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))
    embed_from_fasta(args.seqs, fastaRef, outPath, cache_dir,
                     device, args.model, per_protein, half_precision, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
