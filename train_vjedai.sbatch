#!/bin/bash
#SBATCH --job-name=vjedai_train
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

# --- Run mode (video-native by default; image = T=1 special case) -----------
# Video mode trains on T-frame clips with the temporal-consistency loss and
# warm-starts the temporal head from Video-Depth-Anything-Large. Point
# VJEDAI_VIDEO_TRAIN_DIR at your per-clip dataset (see src/dataset.VideoTrainDataset).
export VJEDAI_DATA_MODE="${VJEDAI_DATA_MODE:-video}"
export VJEDAI_VIDEO_TRAIN_DIR="${VJEDAI_VIDEO_TRAIN_DIR:-}"
# export VJEDAI_VIDEO_TEST_DIR="..."          # optional per-frame prediction dump
export VJEDAI_CLIP_LEN="${VJEDAI_CLIP_LEN:-16}"
export VJEDAI_BATCH_SIZE="${VJEDAI_BATCH_SIZE:-1}"   # clips/batch (video is memory-heavy)
export VJEDAI_LOSS_MODE="${VJEDAI_LOSS_MODE:-nll}"
# encode_mode: "clip" = VJEPA video path (spatiotemporal attention) + learned
# temporal upsampler; "per_frame" = VJEPA per-frame image path (ablation).
export VJEDAI_ENCODE_MODE="${VJEDAI_ENCODE_MODE:-clip}"
# To reproduce the original single-image baseline instead, launch with:
#   VJEDAI_DATA_MODE=image sbatch train_vjedai.sbatch

python -u train_vjedai.py
