#!/bin/bash -e
set -x
source /mnt/lsf-nas-1/os-shared/anaconda3/etc/profile.d/conda.sh
conda activate kaindl_ppi

ppi embed --fasta test.fasta --h5-file huintaf.h5 --h5-mode a --cache-dir "/mnt/project/kaindl/ppi/embed_data/t5_xl_weights"

rostclust uniqueprot2d --hval-config-path "/mnt/project/kaindl/ppi/ppi_data/hval_config.json" test.fasta "/mnt/project/kaindl/ppi/ppi_data/v2.1/1:1_small/apid_train.fasta" test_c3.fasta

ppi predict --model "/mnt/project/kaindl/ppi/runs/23/s23acc/chk_39.tar" --in-tsv test.tsv --out-tsv predict.tsv --no-header --h5 huintaf.h5 --mode w
