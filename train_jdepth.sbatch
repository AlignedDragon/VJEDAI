#!/bin/bash
#SBATCH --job-name=jdepth_train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gpus 2080ti:1
#SBATCH --account=cil_jobs
#SBATCH --time=12:00:00

set -e

export SCRATCH="/work/scratch/$USER"
export HF_HOME="$SCRATCH/cache/hf"
export TORCH_HOME="$SCRATCH/cache/torch"

cd "$HOME/mono"

mkdir -p "external"

source ~/.bashrc
conda activate /cluster/courses/cil/envs/envs/monocular-depth-estimation

python -m pip install opencv-python timm einops

if [ ! -d "external/vjepa2" ]; then
    git clone https://github.com/facebookresearch/vjepa2.git external/vjepa2
fi

sed -i 's|VJEPA_BASE_URL = "http://localhost:8300"|VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"|' \
    external/vjepa2/src/hub/backbones.py

if [ ! -d "external/Depth-Anything-V2" ]; then
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git "external/Depth-Anything-V2"
fi

python -m pip install -r "external/Depth-Anything-V2/requirements.txt"

export PYTHONPATH="external/vjepa2/src:$PYTHONPATH"
export PYTHONPATH="external/vjepa2:$PYTHONPATH"
export PYTHONPATH="$HOME/mono/src:$PYTHONPATH"

python -u Full_Pipeline_JepaDepth.py
