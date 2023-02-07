#!/bin/bash -e
source /mnt/lsf-nas-1/os-shared/anaconda3/etc/profile.d/conda.sh
conda activate kaindl_e3

seth -i uniprot/huintaf.hash.fasta -o huintaf_seth_scores.json -f json -c /mnt/project/kaindl/ppi/embed_data/t5_xl_weights

