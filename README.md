<h1 align="center">VJEDAI</h1>
<p align="center"><b>V-JEPA Encoder for Depth Anything Inference</b></p>

<p align="center">
  <a href="https://huggingface.co/kalandarX/jdepth"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="Hugging Face Model"></a>
</p>

VJEDAI (V-JEPA Encoder for Depth Anything Inference) is a hybrid monocular depth estimation model using a V-JEPA 2.1 visual encoder coupled to a Depth Anything V2 depth decoder.

## VJEDAI pipeline

The core model pairs a **frozen V-JEPA 2.1 encoder** with a **trainable Depth Anything V2 DPT decoder** and a **per-pixel Gaussian uncertainty head**. Four intermediate V-JEPA token maps are reshaped to patch grids and fed through the DPT head; the model outputs a depth map and a log-variance map. Training is scale-invariant (per-image), with two separate objectives you choose between: `si_mse` (depth only) and `nll` (heteroscedastic Gaussian, default). See the report for details.

Key files:
- `train_vjedai.py` — end-to-end training + submission.
- `src/vjedai.py` — the model (`build_vjedai`).
- `src/{dataset,preprocessing,create_submission}.py` — data, V-JEPA input prep, Kaggle CSV.
- `utils/infer_vjedai.py` — standalone inference from a checkpoint.
- `utils/train_vjepa_linear_depth.py` — frozen-encoder linear-probe baseline.

### Environment setup

Runs on the CIL cluster conda env, plus a few pip packages and two external repos cloned under `external/`:

```bash
conda activate /cluster/courses/cil/envs/envs/monocular-depth-estimation
pip install opencv-python timm einops

git clone https://github.com/facebookresearch/vjepa2.git external/vjepa2
git clone https://github.com/DepthAnything/Depth-Anything-V2.git external/Depth-Anything-V2
pip install -r external/Depth-Anything-V2/requirements.txt

export PYTHONPATH="external/vjepa2/src:external/vjepa2:$PWD/src:$PYTHONPATH"
```

`train_vjedai.sbatch` performs all of the above automatically (it expects the project at `$HOME/mono`). Checkpoints are written to `$SCRATCH/checkpoints/vjedai_<variant>/`.

### Training

Pick one objective per run via `VJEDAI_LOSS_MODE` (the two are independent trainings, not stages):

```bash
sbatch train_vjedai.sbatch                          # nll (default): Gaussian uncertainty
VJEDAI_LOSS_MODE=si_mse sbatch train_vjedai.sbatch  # depth only, no uncertainty
```

Checkpoints are tagged by mode (`best_<mode>.pth`), so the two runs don't clobber each other. Model selection uses validation SI-RMSE; a `submission.csv` is written on each new best.

### Inference

```bash
# Default: downloads the published checkpoint from HuggingFace
python utils/infer_vjedai.py --out submission.csv

# Or use a local checkpoint / a different HF file
python utils/infer_vjedai.py --ckpt path/to/best_nll.pth
python utils/infer_vjedai.py --hf-repo kalandarX/jdepth --hf-file large/v1.2_nll_deliverable.pth
```


## Experiments

This repository contains two experiment notebooks for training and evaluating Depth Anything decoder heads:

- `experiments/da_train.ipynb`
- `experiments/da_evaluate.ipynb`

Both notebooks support experiments with **Depth Anything V2** and **Depth Anything 3**. The main goal is to compare the prediction heads and to test whether the depth prediction heads perform better when they are fine-tuned from pretrained weights or trained after random re-initialization, while keeping the backbone frozen.

### Data Format

The notebooks expect the training data to be located in `data/train/`. The training directory should contain RGB images (`.png`) and matching depth maps (`.npy`). To recreate our exact results, the data should come from the [ETHZ CIL Monocular Depth Estimation 2026 kaggle competition](https://www.kaggle.com/competitions/ethz-cil-monocular-depth-estimation-2026).

A deterministic train/validation split is created automatically and stored in `splits/`. This ensures that all experiments use the same validation samples and can be compared fairly.

### Training Notebook

The notebook `experiments/da_train.ipynb` is used to train the depth prediction head of either Depth Anything V2 or Depth Anything 3. In the configuration cell, choose the model family:

```python
MODEL_FAMILY = 'DA2' # or 'DA3'
```

and choose how the prediction head should be initialized:

```python
WEIGHTS_SOURCE = 'pretrained' # fine-tune pretrained head 

# or `reset` to train randomly reinitialized head
```

The notebook freezes the full pretrained model and re-enables gradients only for the prediction head. This allows a fair comparison between different decoder head initialization strategies.

The training objective combines:
1. an L1 depth reconstruction loss, and
2. a gradient-matching loss that encourages locally similar depth structure.

During training, the notebook evaluates the model on the validation split using scale-invariant RMSE. It saves one checkpoint after every epoch and also stores the best checkpoint according to validation si-RMSE. Checkpoints are written to `checkpoints/` with names such as
```
DA2_pretrained_epoch_001.pt
DA2_pretrained_best.pt
DA3_reset_epoch_001.pt
DA3_reset_best.pt
```

### Evaluation Notebook

The notebook `experiments/da_evaluate.ipynb` is used to evaluate either a pretrained model or a trained checkpoint on the same validation split used during training.

In the configuration cell, choose the model family:

```python
MODEL_FAMILY = 'DA2' # or 'DA3'
```

and choose whether to evaluate the original pretrained model or a trained head checkpoint:

```python
WEIGHTS_SOURCE = 'pretrained' # or checkpoint
```

When evaluating a checkpoint, uncomment the corresponding checkpoint file, for example:

```python
CHECKPOINT = 'DA2_finetuned.pt'
# CHECKPOINT = 'DA3_finetuned.pt'
```

The evaluation notebook loads the base pretrained model and then replaces the depth prediction heads with the checkpoint weights. By default, checkpoints are downloaded from the Hugging Face repository `ragerber13/depth-anything-custom-heads`. 

The notebook computes the following validation metrics:

- `mean_si_RMSE`: mean per-image scale-invariant RMSE
- `AbsRel`: absolute relative depth error
- `Delta1`: fraction of valid pixels where prediction and ground truth agree within a factor of 1.25

Evaluation results are saved as CSV files in `evaluation_results/` with names such as:

```
eval_DA2_finetuned.csv
eval_DA3_finetuned.csv
eval_DA2_pretrained.csv
```

### Typical Workflow
1. Place the training data in `data/train/`.
2. Open `experiments/da_train.ipynb`.
3. Select `MODEL_FAMILY` and `WEIGHTS_SOURCE`.
4. Run the notebook to train the prediction head.
5. Use the best checkpoint from `checkpoints/`.
6. Open `experiments/da_evaluate.ipynb`.
7. Select the same model family and the checkpoint to evaluate.
8. Run the notebook to compute validation metrics and save the results.

## AI Usage Declaration
1. 
Tool used: Claude Code

Files affected: `da_evaluated.ipynb`, `da_train.ipynb`

Purpose: Used to add comments inside of code cells.

2. 
Tool used: ChatGPT 5.5 Thinking 

Purpose: Used to get a better overview of how to use slurm and how to correctly set up the `.sbatch` files. 

3. 
Tool used: Claude Code

Files affected: all files

Purpose: Used to add comments to the code.

## Acknowledgements

VJEDAI builds on several open research efforts in monocular depth estimation and self-supervised visual representation learning.

The Depth Anything models provide the monocular depth estimation backbone and decoder components used in our experiments. V-JEPA 2.1 provides the visual representation learning foundation for the V-JEPA encoder part of VJEDAI.

We thank the authors for releasing their models, code, and pretrained checkpoints, which made this project possible.

If you use this repository, please also cite the original works listed below.

```bibtex
@article{depth_anything_v2,
  title   = {Depth Anything V2},
  author  = {Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal = {arXiv:2406.09414},
  year    = {2024}
}

@article{depthanything3,
  title   = {Depth Anything 3: Recovering the Visual Space from Any Views},
  author  = {Lin, Haotong and Chen, Sili and Liew, Jun Hao and Chen, Donny Y. and Li, Zhenyu and Shi, Guang and Feng, Jiashi and Kang, Bingyi},
  journal = {arXiv preprint arXiv:2511.10647},
  year    = {2025}
}

@article{murlabadia2026vjepa2_1,
  title   = {V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning},
  author  = {Mur-Labadia, Lorenzo and Muckley, Matthew and Bar, Amir and Assran, Mahmoud and Sinha, Koustuv and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas and Bardes, Adrien},
  journal = {arXiv preprint arXiv:2603.14482},
  year    = {2026}
}
```
