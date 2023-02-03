#!/bin/bash -e
# run bash file: ./run_colabfold.sh some/path/to/your_fasta.fasta directory/to/some_results/
source /mnt/lsf-nas-1/os-shared/anaconda3/etc/profile.d/conda.sh
conda activate collabFold

# Path to your input fasta file (can have multiple sequences)
QUERY=$(realpath "$1")
# Path to the directory where results get stored
RESULT=$(realpath "$2")

# Path to MMSeqs2 executable
MMSEQS=/mnt/home/mheinzinger/deepppi1tb/collabfold/mmseqs2/mmseqs/bin/mmseqs

# Path to databases for generating MSAs (UniRef30+environment_dbs); only available on LSF-1/LSF-2 currently! (DBs NEED to be on SSD due to I/O)
DBS=/home/mheinzinger/colabfold_dbs/

# Generate MSAs (use env. DBs but don't use templates; MMSeqs2 sensitivity set to 8 (high sens.))
colabfold_search --mmseqs $MMSEQS --db1 uniref30_2103_db --db3 colabfold_envdb_202108_db --use-env 1 --use-templates 0 -s 9 $QUERY $DBS $RESULT
# Split MSAs to generate one MSA-file per query
colabfold_split_msas $RESULT $RESULT/msas
# Predict structures (stop-at-score parameter triggers early-stopping if prediction is already of good/high quality)
colabfold_batch $RESULT $RESULT/predictions --stop-at-score 85
# Remove intermediate results
rm $RESULT/*.a3m
rm -d $RESULT/msas
