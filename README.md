# VJEDAI

VJEDAI (V-JEPA Encoder for Depth Anything Inference) is a hybrid monocular depth estimation model using a V-JEPA 2.1 visual encoder coupled to a Depth Anything V2 depth decoder. 

## Ali's part
e.g. short description, environment setup, experiments, training, inference


## Experiments

This repository contains two experiment notebooks for training and evaluating Depth Anything decoder heads:

- `experiments/da_train.ipynb`
- `experiments/da_evaluate.ipynb`

Both notebooks support experiments with **Depth Anything V2** and **Depth Anything 3**. The main goal is to compare the prediction heads and to test whether the depth prediction heads performs better when it is fine-tuned from pretrained weights or trained after random re-initialization, while keeping the backbone frozen.

### Data Format

The notebooks expect the training data to be located in `data/train/`. The training directory should contain RGB images (`.png`) and matching depth maps (`.npy`). To recreate our exact results, the data should come from the [ETHZ CIL Monocular Depth Estimation 2026 kaggle competition](https://www.kaggle.com/competitions/ethz-cil-monocular-depth-estimation-2026).

A deterministic train/validation split is created automatically and stored in `splits`. This ensures that all experiments use the same validation samples and can be compared fairly.

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

During training, the notebook evaluates the model on the validation split using scale invariant RMSE. It saves one checkpoint after every epoch and also stores the best checkpoint according to validation si-RMSE. Checkpoints are written to `checkpoints/` with names such as
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

and choose wheter to evaluate the original pretrained model or a trained head checkpoint:

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

Purpose: Used to get a better overview of how to use slurm and how to correctly setup the `.sbatch` files. 

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
