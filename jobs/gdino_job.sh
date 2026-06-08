#!/bin/bash
#SBATCH --job-name=groundingdino_test
#SBATCH --output=logs/gdino_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source ~/.bashrc
conda activate label-refiner

python grounding_dino_detect.py