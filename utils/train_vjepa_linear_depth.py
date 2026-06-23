#!/usr/bin/env python
# coding: utf-8
"""Train a frozen VJEPA 2.1 linear probe for monocular depth.

This is a minimal baseline companion to ``train_vjedai.py``:
VJEPA 2.1 is frozen, the last VJEPA feature map is reshaped to a patch grid,
and a single 1x1 convolution predicts depth per patch before bilinear
upsampling to the target resolution.

Run from the ``mono`` directory:
    python -u utils/train_vjepa_linear_depth.py

Resume an interrupted run:
    LINEAR_RESUME_FROM=$SCRATCH/checkpoints/vjepa_linear_depth_large/last.pth \
      python -u utils/train_vjepa_linear_depth.py

Or auto-resume from ``last.pth`` when it exists:
    LINEAR_AUTO_RESUME=1 python -u utils/train_vjepa_linear_depth.py
"""

import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

TRAIN_DIR = "/cluster/courses/cil/monocular-depth-estimation/train/"
TEST_DIR = "/cluster/courses/cil/monocular-depth-estimation/test"

VARIANT = "large"       # "large" uses vjepa2_1_vit_large_384, last layer 23
BATCH_SIZE = 8
NUM_EPOCHS = 3
PATIENCE = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.1
NUM_WORKERS = 4
EPS = 1e-6

SCRATCH = Path(os.environ.get("SCRATCH", "."))
CHECKPOINT_DIR = SCRATCH / "checkpoints" / f"vjepa_linear_depth_{VARIANT}"
BEST_CKPT = CHECKPOINT_DIR / "best.pth"
LAST_CKPT = CHECKPOINT_DIR / "last.pth"
SUBMISSION_CSV = Path("./submission.csv")

RESUME_FROM = os.environ.get("LINEAR_RESUME_FROM", "").strip() or None
AUTO_RESUME = os.environ.get("LINEAR_AUTO_RESUME", "0").lower() in (
    "1",
    "true",
    "yes",
)

VARIANTS = {
    "large": {
        "vjepa_arch": "vjepa2_1_vit_large_384",
        "vjepa_out_layers": [23],
        "vjepa_dim": 1024,
        "vjepa_patch_size": 16,
    },
}


# -----------------------------------------------------------------------------
# Paths / sys.path setup
# -----------------------------------------------------------------------------

MONO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path.home().resolve()
SRC_DIR = MONO_ROOT / "src"


def first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


VJEPA_ROOT = (
    Path(os.environ["VJEPA_ROOT"])
    if "VJEPA_ROOT" in os.environ
    else first_existing(MONO_ROOT / "external" / "vjepa2", PROJECT_ROOT / "external" / "vjepa2")
)
VJEPA_SRC = VJEPA_ROOT / "src"

for p in [str(SRC_DIR), str(VJEPA_SRC), str(VJEPA_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

for module_name in list(sys.modules):
    if module_name == "app" or module_name.startswith("app."):
        del sys.modules[module_name]


# -----------------------------------------------------------------------------
# Imports that depend on sys.path
# -----------------------------------------------------------------------------

from dataset import TrainDataset, TestDataset       # noqa: E402
from preprocessing import vjepa_preprocessing    # noqa: E402
from create_submission import encode_depth, save_submission  # noqa: E402


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class VJEPALinearDepthProbe(nn.Module):
    """Frozen VJEPA encoder with a 1x1-conv linear depth probe."""

    def __init__(self, vjepa_encoder, vjepa_dim=1024, patch_size=16, eps=1e-6):
        super().__init__()
        self.vjepa_encoder = vjepa_encoder
        self.vjepa_encoder.eval()
        for p in self.vjepa_encoder.parameters():
            p.requires_grad = False

        self.probe = nn.Conv2d(vjepa_dim, 1, kernel_size=1)
        self.patch_size = patch_size
        self.eps = eps

    def train(self, mode=True):
        super().train(mode)
        self.vjepa_encoder.eval()
        return self

    def forward(self, vjepa_input, output_size):
        B = vjepa_input.shape[0]
        H_in, W_in = vjepa_input.shape[-2:]
        patch_h = H_in // self.patch_size
        patch_w = W_in // self.patch_size

        with torch.no_grad():
            feats = self.vjepa_encoder(vjepa_input)

        if isinstance(feats, (list, tuple)):
            tokens = feats[-1]
        else:
            tokens = feats

        expected_tokens = patch_h * patch_w
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"VJEPA returned {tokens.shape[1]} tokens, expected "
                f"{expected_tokens} ({patch_h}x{patch_w})."
            )

        # tokens -> patch grid -> 1x1 conv depth-per-patch -> upsample to full res
        x = tokens.permute(0, 2, 1).reshape(B, tokens.shape[-1], patch_h, patch_w)
        depth = self.probe(x)
        depth = F.interpolate(
            depth,
            size=tuple(output_size),
            mode="bilinear",
            align_corners=True,
        )
        depth = F.softplus(depth) + self.eps  # keep depth strictly positive
        return depth


# -----------------------------------------------------------------------------
# Loss / metric
# -----------------------------------------------------------------------------

def _flatten_to_bhw(*tensors):
    out = []
    for t in tensors:
        if t.ndim == 4:
            t = t.squeeze(1)
        out.append(t)
    return out


def scale_invariant_rmse(pred, target, eps=1e-6):
    """Scale-invariant RMSE on log depths."""
    pred, target = _flatten_to_bhw(pred, target)
    mask = target > eps
    if not mask.any():
        return pred.sum() * 0.0

    pred = torch.clamp(pred, min=eps)
    target = torch.clamp(target, min=eps)
    log_diff = (torch.log(pred) - torch.log(target))[mask]
    bias = -torch.mean(log_diff)
    return torch.sqrt(torch.mean((log_diff + bias) ** 2))


def trainable_state_dict(model):
    return {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("vjepa_encoder.")
    }


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

print("loading data...")
train_dataset = TrainDataset(TRAIN_DIR)
val_size = int(len(train_dataset) * VAL_FRACTION)
train_size = len(train_dataset) - val_size
train_subset, val_subset = random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(0),
)
train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_dataset = TestDataset(TEST_DIR)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
print(f"data loaded. train: {train_size} | val: {val_size} | test: {len(test_dataset)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# -----------------------------------------------------------------------------
# Build model
# -----------------------------------------------------------------------------

cfg = VARIANTS[VARIANT]
print(f"variant={VARIANT} | vjepa_arch={cfg['vjepa_arch']} | out_layers={cfg['vjepa_out_layers']}")
print(f"VJEPA root: {VJEPA_ROOT}")

vj_encoder, _ = torch.hub.load(
    str(VJEPA_ROOT),
    cfg["vjepa_arch"],
    source="local",
    out_layers=cfg["vjepa_out_layers"],
)
vj_encoder = vj_encoder.to(device).eval()
print(f"VJEPA ({cfg['vjepa_arch']}) loaded.")

model = VJEPALinearDepthProbe(
    vjepa_encoder=vj_encoder,
    vjepa_dim=cfg["vjepa_dim"],
    patch_size=cfg["vjepa_patch_size"],
    eps=EPS,
).to(device)

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Linear probe built. trainable: {n_trainable/1e6:.4f}M | frozen: {n_frozen/1e6:.2f}M")

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

best_val_rmse = float("inf")
epochs_without_improvement = 0
start_epoch = 0
last_epoch = -1
mean_train_loss = float("nan")
mean_val_loss = float("nan")

resume_path = Path(RESUME_FROM).expanduser() if RESUME_FROM is not None else None
if resume_path is None and AUTO_RESUME and LAST_CKPT.exists():
    resume_path = LAST_CKPT

if resume_path is not None:
    print(f"Resuming from {resume_path}...")
    ckpt = torch.load(resume_path, map_location=device)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    non_vjepa_missing = [k for k in missing if not k.startswith("vjepa_encoder.")]
    if non_vjepa_missing:
        sample = non_vjepa_missing[:5]
        suffix = "..." if len(non_vjepa_missing) > 5 else ""
        print(f"  WARN missing non-vjepa keys: {sample}{suffix}")
    if unexpected:
        sample = unexpected[:5]
        suffix = "..." if len(unexpected) > 5 else ""
        print(f"  WARN unexpected keys: {sample}{suffix}")

    if isinstance(ckpt, dict):
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        last_epoch = int(ckpt.get("epoch", -1))
        start_epoch = last_epoch + 1
        mean_train_loss = float(ckpt.get("train_loss", mean_train_loss))
        mean_val_loss = float(ckpt.get("val_rmse", mean_val_loss))
        best_val_rmse = float(
            ckpt.get("best_val_rmse", ckpt.get("val_rmse", best_val_rmse))
        )
        epochs_without_improvement = int(
            ckpt.get("epochs_without_improvement", epochs_without_improvement)
        )

    print(
        f"  resume loaded. next epoch={start_epoch + 1} | "
        f"best val si-rmse={best_val_rmse:.6f}"
    )


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Starting training at epoch {start_epoch + 1}/{NUM_EPOCHS}...")

for epoch in range(start_epoch, NUM_EPOCHS):
    last_epoch = epoch
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        targets = batch["depth"].to(device)

        optimizer.zero_grad()

        H, W = images.shape[-2:]
        pred_depth = model(vjepa_preprocessing(images), output_size=(H, W))

        loss = scale_invariant_rmse(pred_depth, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        print(f"batch {num_batches}: si-rmse = {loss.item():.6f}")

    mean_train_loss = total_loss / max(num_batches, 1)

    model.eval()
    val_loss_sum = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["depth"].to(device)

            H, W = images.shape[-2:]
            pred_depth = model(vjepa_preprocessing(images), output_size=(H, W))

            val_loss_sum += scale_invariant_rmse(pred_depth, targets).item()
            val_batches += 1

    mean_val_loss = val_loss_sum / max(val_batches, 1)

    print(
        f"Epoch {epoch + 1} | "
        f"train si-rmse: {mean_train_loss:.6f} | "
        f"val si-rmse: {mean_val_loss:.6f}"
    )

    if mean_val_loss < best_val_rmse:
        best_val_rmse = mean_val_loss
        epochs_without_improvement = 0
        torch.save(
            {
                "model_state_dict": trainable_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_rmse": mean_val_loss,
                "best_val_rmse": best_val_rmse,
                "config": {
                    "variant": VARIANT,
                    "vjepa_arch": cfg["vjepa_arch"],
                    "vjepa_out_layers": cfg["vjepa_out_layers"],
                    "probe": "conv1x1_last_layer",
                    "loss": "scale_invariant_rmse",
                },
            },
            BEST_CKPT,
        )
        print(f"Saved new best model: val si-rmse {mean_val_loss:.6f}")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement}/{PATIENCE} epochs")
        if epochs_without_improvement >= PATIENCE:
            print("Early stopping")
            break


# -----------------------------------------------------------------------------
# Save last
# -----------------------------------------------------------------------------

torch.save(
    {
        "model_state_dict": trainable_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": last_epoch,
        "train_loss": mean_train_loss,
        "val_rmse": mean_val_loss,
        "best_val_rmse": best_val_rmse,
        "epochs_without_improvement": epochs_without_improvement,
        "config": {
            "variant": VARIANT,
            "vjepa_arch": cfg["vjepa_arch"],
            "vjepa_out_layers": cfg["vjepa_out_layers"],
            "probe": "conv1x1_last_layer",
            "loss": "scale_invariant_rmse",
        },
    },
    LAST_CKPT,
)


# -----------------------------------------------------------------------------
# Inference for submission (uses the best checkpoint)
# -----------------------------------------------------------------------------

submission_ckpt = BEST_CKPT if BEST_CKPT.exists() else LAST_CKPT
print(f"Loading submission checkpoint from {submission_ckpt}...")
checkpoint = torch.load(submission_ckpt, map_location=device)
state_dict = (
    checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
    else checkpoint
)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
bad_missing = [k for k in missing if not k.startswith("vjepa_encoder.")]
if bad_missing or unexpected:
    raise RuntimeError(
        f"Unexpected/missing keys when restoring trained linear probe: "
        f"missing={bad_missing}, unexpected={unexpected}"
    )
model.eval()

rows = []
with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(device)
        image_ids = batch["id"]
        H, W = images.shape[-2:]

        pred_depths = model(vjepa_preprocessing(images), output_size=(H, W))
        if pred_depths.ndim == 4:
            pred_depths = pred_depths.squeeze(1)
        pred_depths = pred_depths.detach().cpu().numpy()

        for depth, image_id in zip(pred_depths, image_ids):
            rows.append({"id": f"{image_id}_depth", "Depths": encode_depth(depth)})

save_submission(rows, str(SUBMISSION_CSV))
print(f"Done. Best checkpoint: {BEST_CKPT}")
print(f"Last checkpoint: {LAST_CKPT}")
print(f"Submission written to {SUBMISSION_CSV} ({len(rows)} rows).")
