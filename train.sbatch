#!/bin/bash
#SBATCH --job-name=da3metric_decoder
#SBATCH --output=/home/ragerber/logs/%x-%j.out
#SBATCH --error=/home/ragerber/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gpus 5060ti:1
#SBATCH --account=cil_jobs
#SBATCH --time=24:00:00

set -e

cd /home/ragerber

mkdir -p logs
mkdir -p checkpoints
mkdir -p splits

source ~/.bashrc
conda activate /cluster/courses/cil/envs/envs/monocular-depth-estimation

python -m pip install --upgrade pip
python -m pip install xformers "torch>=2" torchvision
python -m pip install transformers

export PYTHONPATH="/home/ragerber:$PYTHONPATH"

python -u da_train.py
