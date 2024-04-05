#!/usr/bin/bash

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu=48GB
#SBATCH --gres=gpu:A6000:2
#SBATCH --job-name=mmlu_eval_flan_280000
#SBATCH --output ./log/%J-%x.log
#SBATCH --partition=general
#SBATCH --time 2-00:00:00


export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=online
export HF_HOME=/data/tir/projects/tir6/general/nkanduku/.cache/huggingface/
RUN_DIR="/data/tir/projects/tir6/general/nkanduku/t5finetune/run11/checkpoint-180000"

if [ -n "$MAMBA_EXE" ]; then
  CONDA=micromamba
  eval "$(micromamba shell hook -s bash)"
else
  CONDA=conda
  eval "$(conda shell.bash hook)"
fi

$CONDA activate flan

python src/flan-eval.py --model $RUN_DIR
