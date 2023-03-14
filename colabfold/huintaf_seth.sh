#!/bin/bash -e
source /mnt/lsf-nas-1/os-shared/anaconda3/etc/profile.d/conda.sh
conda activate kaindl_ppi

seth -i uniprot/huintaf2_fixed.fasta -o huintaf2_seth_scores.json -f json -c /mnt/project/kaindl/ppi/embed_data/t5_xl_weights

