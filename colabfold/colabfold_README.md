# colabfold

Short description of how to run AlphaFold/ColabFold on our LSF-clusters.
The script "run_colabfold.sh" takes a fasta file as input (or a directory holding multiple fasta files) and will output a directory holding AlphaFold2 predictions. MSAs will be generated using a local MMSeqs2 version that first searches UniRef30 to generate profiles which are then used to search colabfold_env_db (for more information, please check the official ColabFold paper/github; all DBs are installed as explained there).

Download `run_colabfold.sh` to your project directory and run: 

`./run_colabfold.sh some/path/to/your_fasta.fasta directory/to/some_results/`

Important parameters: 
* Sensitivity/speed for generating MSAs can be adjusted with 'colabfold_search -s' parameter. By default, highest sensitivity is used (`-s 9`)
* Early stopping is currently activated (`colabfold_batch --stop-at-score 85`). AlphaFold2 consists of 5 models, each outputting a structure. With early stopping activated no additional models are exectued once one model achieves a pLDDT>x (with x being currently set to 85). However, single model runs are not affected by this and can reach pLDDT above x. 


IMPORTANT - MSA generation (irrespective of AlphaFold2!)

Once you activated the conda environment `collabFold` on the LSF, you can also access sub-modules of ColabFold.
Most importantly, you can access `colabfold_search` which allows you to generate MSAs that would also feed AlphaFold2 (see `run_colabfold.sh` for parameters/databases etc.). If you need a large MSA OR if you want to expand your search to metagenomic sequences, you should use this script. 
