#!/bin/bash
#SBATCH --job-name=gdino_test
#SBATCH --output=logs/gdino_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# ---- LIMIT GPU USAGE HERE ----
# To use GPU 0:
export CUDA_VISIBLE_DEVICES=0

# To use GPUs 0 and 1:
# export CUDA_VISIBLE_DEVICES=0,1

# ------------------------------

source ~/.bashrc
conda activate label-refiner

python scripts/grounding_dino_detect_m60.py