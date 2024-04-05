#!/usr/bin/bash

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu=48GB
#SBATCH --gres=gpu:2080Ti:1
#SBATCH --job-name=mmlu_eval_flan_64_5e-5
#SBATCH --output ./log/%J-%x.log
#SBATCH --partition=general
#SBATCH --time 2-00:00:00


export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online
export HF_HOME=/data/tir/projects/tir6/general/nkanduku/.cache/huggingface/
RUN_DIR="/data/tir/projects/tir6/general/nkanduku/t5finetune/run14"

if [ -n "$MAMBA_EXE" ]; then
  CONDA=micromamba
  eval "$(micromamba shell hook -s bash)"
else
  CONDA=conda
  eval "$(conda shell.bash hook)"
fi

$CONDA activate flan

# Check if the directory exists
if [ -d "$RUN_DIR" ]; then
    for dir in "$RUN_DIR"/*/; do
        if [ -d "$dir" ]; then
            echo "Running flan-eval.py for directory: $dir"
            # Run flan-eval.py for each directory
            python src/flan-eval.py --model "$dir"
            # python src/flan-eval.py
        fi
    done
else
    echo "Directory $RUN_DIR does not exist."
fi

