#!/bin/bash
#SBATCH --job-name=depth_fusion
#SBATCH --output=/home/ragerber/logs/%x-%j.out
#SBATCH --error=/home/ragerber/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gpus 5060ti:1
#SBATCH --account=cil_jobs
#SBATCH --time=12:00:00

set -e

cd /home/ragerber

mkdir -p logs
mkdir -p checkpoints
mkdir -p external

source ~/.bashrc
conda activate /cluster/courses/cil/envs/envs/monocular-depth-estimation

python -m pip install opencv-python timm einops

if [ ! -d external/vjepa2 ]; then
    git clone https://github.com/facebookresearch/vjepa2.git external/vjepa2
fi

sed -i 's|VJEPA_BASE_URL = "http://localhost:8300"|VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"|' \
    external/vjepa2/src/hub/backbones.py

if [ ! -d external/Depth-Anything-V2 ]; then
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git external/Depth-Anything-V2
fi

python -m pip install -r external/Depth-Anything-V2/requirements.txt

if [ ! -f checkpoints/depth_anything_v2_vitb.pth ]; then
    curl -L \
      "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true" \
      -o checkpoints/depth_anything_v2_vitb.pth
fi

export PYTHONPATH="/home/ragerber/external/vjepa2/src:$PYTHONPATH"
export PYTHONPATH="/home/ragerber/external/vjepa2:$PYTHONPATH"
export PYTHONPATH="/home/ragerber:$PYTHONPATH"

python -u Full_Pipeline_For_Cluster.py
