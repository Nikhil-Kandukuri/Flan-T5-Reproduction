#!/usr/bin/bash

#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu=48GB
#SBATCH --gres=gpu:A6000:4
#SBATCH --job-name=t5_finetune_32_batchsize_5e-3
#SBATCH --output ./log/%J-%x.log
#SBATCH --partition=long
#SBATCH --time 6-00:00:00


export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=online
export HF_HOME=/data/tir/projects/tir6/general/nkanduku/.cache/huggingface/

# Set the paths and parameters
OUTPUT_DIR="/data/tir/projects/tir6/general/nkanduku/t5finetune/run14"
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=8
LEARNING_RATE=5e-3

if [ -n "$MAMBA_EXE" ]; then
  CONDA=micromamba
  eval "$(micromamba shell hook -s bash)"
else
  CONDA=conda
  eval "$(conda shell.bash hook)"
fi

$CONDA activate flan

python src/t5finetune.py --output_dir $OUTPUT_DIR --num_train_epochs $NUM_TRAIN_EPOCHS --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate $LEARNING_RATE