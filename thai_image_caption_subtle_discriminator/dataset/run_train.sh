#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH -t 48:00:00
#SBATCH -A lt200203
#SBATCH -J IMAGE_CAPTION_DISCRIMINATOR

module load Mamba/23.11.0-0
conda activate pytorch-2.0.0

python3 train.py