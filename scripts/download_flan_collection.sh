#!/usr/bin/bash

#SBATCH --cpus-per-task 2
#SBATCH --mem-per-cpu=10GB
#SBATCH --job-name=dl
#SBATCH --output ./log/%J-%x.log
#SBATCH --time 2:00:00

if [ -n "$MAMBA_EXE" ]; then
  CONDA=micromamba
  eval "$(micromamba shell hook -s bash)"
else
  CONDA=conda
  eval "$(conda shell.bash hook)"
fi

$CONDA activate flan

dirpath_log=./log

python src/download_flan_collection.py \
    --dirpath_log $dirpath_log
