#!/usr/bin/env python
# coding: utf-8
"""Train the JepaDepthAnything architecture end-to-end.

Mirrors the structure of Full_Pipeline_For_Cluster.py but swaps the
DA + cross-attention fusion model for the unified JepaDepthAnything model
(VJEPA 2.1 encoder + DA DPT decoder + Gaussian uncertainty head).

Two training modes are supported via ``LOSS_MODE`` (or the
``JDEPTH_LOSS_MODE`` env var):

* ``"si_mse"``: scale-invariant MSE on log depths. Ignores the predicted
  log-variance (the uncertainty head still runs forward but receives zero
  gradients, so it stays at its zero-init). Use this as a deterministic
  Stage 1 to get a clean depth-only baseline.
* ``"nll"`` (default): scale-invariant Gaussian NLL using the predicted
  per-pixel log-variance. Use this for a Stage 2 fine-tune that adds the
  uncertainty head on top of an already-good depth model.

Two-stage workflow (set env vars in the sbatch, no script edits needed):
    JDEPTH_LOSS_MODE=si_mse sbatch train_jdepth.sbatch
    JDEPTH_LOSS_MODE=nll \\
      JDEPTH_INIT_FROM=$SCRATCH/checkpoints/jepa_depth_large/best_si_mse.pth \\
      sbatch train_jdepth.sbatch

Validation SI-RMSE (the submission metric) is used for checkpoint
selection in both modes, so Stage 1 and Stage 2 'best' checkpoints are
directly comparable. Checkpoint filenames are mode-tagged
(``best_{LOSS_MODE}.pth`` / ``last_{LOSS_MODE}.pth``) to avoid clobbering.

Switch ``VARIANT`` between ``"base"`` and ``"large"`` to pick the
VJEPA+DA pairing; everything else (out_layers, embed dim, checkpoint
filename, repo id) is derived automatically from
``src.jepa_depth_anything.VARIANTS``.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

TRAIN_DIR = "/cluster/courses/cil/monocular-depth-estimation/train/"
TEST_DIR = "/cluster/courses/cil/monocular-depth-estimation/test"

VARIANT = "large"       # "base" or "large"
BATCH_SIZE = 8          # tune per GPU; 8 is safe for vit-l on a single 5060 Ti
NUM_EPOCHS = 36
PATIENCE = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.1
NUM_WORKERS = 4

# "si_mse": deterministic Stage 1, ignores the uncertainty head.
# "nll": heteroscedastic Stage 2, uses predicted log-variance.
LOSS_MODE = os.environ.get("JDEPTH_LOSS_MODE", "nll").lower()
if LOSS_MODE not in ("nll", "si_mse"):
    raise ValueError(
        f"Unknown LOSS_MODE={LOSS_MODE!r}; expected 'nll' or 'si_mse'."
    )

# Optional path to a Stage 1 checkpoint to warm-start from. Loaded with
# strict=False, so missing vjepa_encoder.* keys are fine (the encoder is
# loaded fresh from torch.hub). Set via env var to avoid editing this
# file each time you flip stages.
INIT_FROM = os.environ.get("JDEPTH_INIT_FROM", "").strip() or None
RESUME_FROM = os.environ.get("JDEPTH_RESUME_FROM", "").strip() or None
AUTO_RESUME = os.environ.get("JDEPTH_AUTO_RESUME", "0").lower() in (
    "1",
    "true",
    "yes",
)

# Keep heavy checkpoints off $HOME: write them to $SCRATCH (the sbatch
# already points $HF_HOME/$TORCH_HOME under $SCRATCH/cache/, so we use a
# sibling $SCRATCH/checkpoints/ subtree to avoid colliding with caches).
# Falls back to a local ./checkpoints/ if $SCRATCH isn't set (e.g., laptop).
SCRATCH = Path(os.environ.get("SCRATCH", "."))
CHECKPOINT_DIR = SCRATCH / "checkpoints" / f"jepa_depth_{VARIANT}"
# Mode-tagged so Stage 1 (si_mse) and Stage 2 (nll) don't overwrite each
# other when run back-to-back from the same $SCRATCH.
BEST_CKPT = CHECKPOINT_DIR / f"best_{LOSS_MODE}.pth"
LAST_CKPT = CHECKPOINT_DIR / f"last_{LOSS_MODE}.pth"
SUBMISSION_CSV = Path("./submission.csv")


# -----------------------------------------------------------------------------
# Paths / sys.path setup
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path.home().resolve()
MONO_ROOT = Path(__file__).resolve().parent
SRC_DIR = MONO_ROOT / "src"
VJEPA_ROOT = PROJECT_ROOT / "external" / "vjepa2"
VJEPA_SRC = VJEPA_ROOT / "src"
DA_ROOT = PROJECT_ROOT / "external" / "Depth-Anything-V2"

for p in [str(DA_ROOT)]:
    while p in sys.path:
        sys.path.remove(p)

for p in [str(SRC_DIR), str(VJEPA_SRC), str(VJEPA_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

for module_name in list(sys.modules):
    if module_name == "app" or module_name.startswith("app."):
        del sys.modules[module_name]


# -----------------------------------------------------------------------------
# Imports that depend on sys.path
# -----------------------------------------------------------------------------

from dataset import TrainDataset, TestDataset           # noqa: E402
from preprocessing import vjepa_preprocessing           # noqa: E402
from create_submission import encode_depth, save_submission  # noqa: E402


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
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print(f"data loaded. train: {train_size} | val: {val_size} | test: {len(test_dataset)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# -----------------------------------------------------------------------------
# Build model
# -----------------------------------------------------------------------------

# IMPORTANT: DA_ROOT must NOT be on sys.path while VJEPA's hub.load runs.
# Inside the hub call, ``from app.vjepa_2_1.models import ...`` resolves
# ``app`` via sys.path. VJEPA ships ``vjepa2/app/`` WITHOUT __init__.py,
# making it only a PEP 420 namespace-package portion -- which never
# short-circuits the search. DA-V2 ships a regular ``app.py`` at its repo
# root, so Python prefers DA's app.py over VJEPA's namespace package
# regardless of sys.path order. The fix is to keep DA off sys.path until
# after the VJEPA hub.load returns.
for p in [str(DA_ROOT), str(MONO_ROOT / "external" / "Depth-Anything-V2")]:
    while p in sys.path:
        sys.path.remove(p)
for module_name in list(sys.modules):
    if module_name == "app" or module_name.startswith("app."):
        del sys.modules[module_name]

# jepa_depth_anything imports depth_anything_v2 lazily (inside
# JepaDepthAnything.__init__), so importing this module here does not yet
# require DA_ROOT on sys.path.
from jepa_depth_anything import VARIANTS, build_jepa_depth_anything  # noqa: E402

cfg = VARIANTS[VARIANT]
print(f"variant={VARIANT} | vjepa_arch={cfg['vjepa_arch']} | da_encoder={cfg['da_encoder']}")

vj_encoder, _ = torch.hub.load(
    str(VJEPA_ROOT),
    cfg["vjepa_arch"],
    source="local",
    out_layers=cfg["vjepa_out_layers"],
)
vj_encoder = vj_encoder.to(device).eval()
print(f"VJEPA ({cfg['vjepa_arch']}) loaded.")

# Now it's safe to expose DA so JepaDepthAnything can import its DPT head.
sys.path.append(str(DA_ROOT))

model = build_jepa_depth_anything(vj_encoder, variant=VARIANT, device=device)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"JepaDepthAnything built. trainable: {n_trainable/1e6:.2f}M | frozen: {n_frozen/1e6:.2f}M")
print(f"Loss mode: {LOSS_MODE}")

if INIT_FROM is not None:
    print(f"Warm-starting from {INIT_FROM}...")
    init_ckpt = torch.load(INIT_FROM, map_location=device)
    init_state = (
        init_ckpt["model_state_dict"]
        if isinstance(init_ckpt, dict) and "model_state_dict" in init_ckpt
        else init_ckpt
    )
    missing, unexpected = model.load_state_dict(init_state, strict=False)
    non_vjepa_missing = [k for k in missing if not k.startswith("vjepa_encoder.")]
    if non_vjepa_missing:
        sample = non_vjepa_missing[:5]
        suffix = "..." if len(non_vjepa_missing) > 5 else ""
        print(f"  WARN missing non-vjepa keys: {sample}{suffix}")
    if unexpected:
        sample = unexpected[:5]
        suffix = "..." if len(unexpected) > 5 else ""
        print(f"  WARN unexpected keys: {sample}{suffix}")
    print(
        f"  warm-start loaded. missing(non-vjepa)={len(non_vjepa_missing)} "
        f"unexpected={len(unexpected)}"
    )


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

def _flatten_to_BHW(*tensors):
    out = []
    for t in tensors:
        if t.ndim == 4:
            t = t.squeeze(1)
        out.append(t)
    return out


def scale_invariant_rmse(pred, target, eps=1e-6):
    """Per-image scale-invariant RMSE on log depths (the leaderboard siRMSE).

    Matches the competition metric exactly:
        siRMSE = sqrt( mean(d^2) - mean(d)^2 ),  d = log(pred) - log(target)
    computed independently per image (scale removed per image, not per batch),
    then averaged over the images in the batch. The previous version fitted a
    single bias across the whole flattened batch, which is not the metric and
    made the reported number diverge from the leaderboard.
    """
    pred, target = _flatten_to_BHW(pred, target)  # [B, H, W]
    scores = []
    for p, t in zip(pred, target):
        m = t > eps
        if not m.any():
            continue
        d = torch.log(torch.clamp(p[m], min=eps)) - torch.log(t[m])
        var = torch.mean(d ** 2) - torch.mean(d) ** 2
        scores.append(torch.sqrt(torch.clamp(var, min=0.0)))
    if not scores:
        return pred.new_tensor(0.0)
    return torch.stack(scores).mean()


def scale_invariant_gaussian_nll(pred, target, log_var, eps=1e-6):
    """Per-image scale-invariant Gaussian NLL using the predicted log-variance.

    The mean log-depth residual is removed independently per image (matching
    the per-image siRMSE objective), then the predicted log-variance
    calibrates the per-pixel noise. Aggregated as a mean over all valid pixels
    in the batch. The previous version removed a single batch-wide bias.
    """
    pred, target, log_var = _flatten_to_BHW(pred, target, log_var)  # [B, H, W]

    nll_sum = log_var.sum() * 0.0  # safe zero with grad
    n_valid = 0
    for p, t, lv in zip(pred, target, log_var):
        m = t > eps
        if not m.any():
            continue
        d = torch.log(torch.clamp(p[m], min=eps)) - torch.log(t[m])
        residual = d - torch.mean(d)  # per-image zero-mean

        # Tight clamp on log_var: a wide ceiling lets variance run away to an
        # "infinity wins" plateau where gradients die before depths are fit.
        # max=3 caps the collapsed plateau at 0.5 * 3 = 1.5, so the model
        # still has to fit depths to do better.
        lv_m = torch.clamp(lv[m], min=-7.0, max=3.0)
        inv_var = torch.exp(-lv_m)
        nll_sum = nll_sum + torch.sum(0.5 * (lv_m + residual ** 2 * inv_var))
        n_valid += int(m.sum())
    if n_valid == 0:
        return nll_sum  # zero with grad
    return nll_sum / n_valid


def scale_invariant_mse(pred, target, eps=1e-6):
    """Per-image scale-invariant MSE on log depths (= per-image siRMSE^2).

    Stage 1 / no-uncertainty loss. The mean log-depth residual is removed
    independently for each image (matching the leaderboard's per-image scale
    alignment), then the squared zero-mean residual is averaged. The previous
    version removed a single batch-wide bias, which penalised the per-image
    scale differences the metric ignores -- harmful here, since this dataset
    has no global distance unit. Transition to NLL via a warm-start once
    depths are good.
    """
    pred, target = _flatten_to_BHW(pred, target)  # [B, H, W]
    losses = []
    for p, t in zip(pred, target):
        m = t > eps
        if not m.any():
            continue
        d = torch.log(torch.clamp(p[m], min=eps)) - torch.log(t[m])
        losses.append(torch.mean(d ** 2) - torch.mean(d) ** 2)  # var(d) = siRMSE^2
    if not losses:
        return pred.sum() * 0.0  # safe zero with grad
    return torch.stack(losses).mean()


def compute_loss(out, targets):
    """Dispatch to the active training loss based on ``LOSS_MODE``."""
    if LOSS_MODE == "nll":
        return scale_invariant_gaussian_nll(out["depth"], targets, out["log_var"])
    if LOSS_MODE == "si_mse":
        return scale_invariant_mse(out["depth"], targets)
    raise ValueError(f"Unknown LOSS_MODE={LOSS_MODE!r}")


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


def trainable_state_dict(m):
    """State dict without the frozen VJEPA encoder (saves a lot of disk)."""
    return {
        k: v for k, v in m.state_dict().items()
        if not k.startswith("vjepa_encoder.")
    }


def save_training_checkpoint(path, epoch):
    torch.save(
        {
            "model_state_dict": trainable_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
            "val_rmse": mean_val_rmse,
            "best_val_rmse": best_val_rmse,
            "epochs_without_improvement": epochs_without_improvement,
            "config": {"variant": VARIANT, "loss_mode": LOSS_MODE},
        },
        path,
    )


def write_submission_from_model(submission_model, output_path):
    was_training = submission_model.training
    submission_model.eval()

    # siRMSE is scale-invariant and GT depths live in [DEPTH_MIN, DEPTH_MAX].
    # The model is trained scale-invariant, so its raw output scale is
    # arbitrary (it drifts into the millions). We normalise each prediction
    # by its own median (a per-image multiplicative constant -- exactly the
    # log-scale alignment siRMSE removes, so this is metric-neutral) and clamp
    # into the GT range. Every value then sits inside float16's well-behaved
    # band. This replaces the old global-rescale path, whose `max / raw_max`
    # factor collapsed to 0 when the raw max overflowed float16 to inf,
    # producing an all-zero submission.
    DEPTH_MIN, DEPTH_MAX = 0.001, 80.0

    n_clamped = 0
    n_pixels = 0
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            image_ids = batch["id"]
            H, W = images.shape[-2:]

            out = submission_model(vjepa_preprocessing(images), output_size=(H, W))
            pred_depths = out["depth"]
            if pred_depths.ndim == 4:
                pred_depths = pred_depths.squeeze(1)
            pred_depths = pred_depths.detach().float().cpu().numpy()

            for depth, image_id in zip(pred_depths, image_ids):
                valid = np.isfinite(depth) & (depth > 0)
                med = float(np.median(depth[valid])) if valid.any() else 1.0
                if not np.isfinite(med) or med <= 0:
                    med = 1.0
                depth = np.nan_to_num(
                    depth / med, nan=DEPTH_MIN, posinf=DEPTH_MAX, neginf=DEPTH_MIN
                )
                clamped = depth.clip(DEPTH_MIN, DEPTH_MAX)
                n_clamped += int((clamped != depth).sum())
                n_pixels += depth.size
                rows.append(
                    {"id": f"{image_id}_depth", "Depths": encode_depth(clamped)}
                )

    if n_clamped:
        print(
            f"Clamped {n_clamped}/{n_pixels} pixels "
            f"({100 * n_clamped / max(n_pixels, 1):.2f}%) into "
            f"[{DEPTH_MIN}, {DEPTH_MAX}]."
        )
    save_submission(rows, str(output_path))
    print(f"Submission written to {output_path} ({len(rows)} rows).")

    if was_training:
        submission_model.train()


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
print("Starting training...")

best_val_rmse = float("inf")
epochs_without_improvement = 0
start_epoch = 0
last_epoch = -1
mean_train_loss = float("nan")
mean_val_loss = float("nan")
mean_val_rmse = float("nan")

resume_path = Path(RESUME_FROM) if RESUME_FROM else (LAST_CKPT if AUTO_RESUME else None)
if resume_path is not None and resume_path.exists():
    print(f"Resuming training from {resume_path}...")
    resume_ckpt = torch.load(resume_path, map_location=device)
    missing, unexpected = model.load_state_dict(
        resume_ckpt["model_state_dict"], strict=False
    )
    bad_missing = [k for k in missing if not k.startswith("vjepa_encoder.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Unexpected/missing keys when resuming: "
            f"missing={bad_missing}, unexpected={unexpected}"
        )
    optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    start_epoch = resume_ckpt["epoch"] + 1
    last_epoch = resume_ckpt["epoch"]
    mean_train_loss = resume_ckpt.get("train_loss", mean_train_loss)
    mean_val_loss = resume_ckpt.get("val_loss", mean_val_loss)
    mean_val_rmse = resume_ckpt.get("val_rmse", mean_val_rmse)
    best_val_rmse = resume_ckpt.get(
        "best_val_rmse", resume_ckpt.get("val_rmse", float("inf"))
    )
    epochs_without_improvement = resume_ckpt.get("epochs_without_improvement", 0)
    print(
        f"  resume loaded. next epoch={start_epoch + 1} | "
        f"best val si-rmse={best_val_rmse:.6f}"
    )
elif resume_path is not None:
    print(f"Resume checkpoint not found at {resume_path}; starting fresh.")

for epoch in range(start_epoch, NUM_EPOCHS):
    last_epoch = epoch
    model.train()

    total_loss = 0.0
    total_rmse = 0.0
    num_batches = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        targets = batch["depth"].to(device)

        optimizer.zero_grad()

        H, W = images.shape[-2:]
        out = model(vjepa_preprocessing(images), output_size=(H, W))

        loss = compute_loss(out, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            total_rmse += scale_invariant_rmse(out["depth"], targets).item()
        num_batches += 1
        print(f"batch {num_batches}: {LOSS_MODE} = {loss.item():.6f}")

    mean_train_loss = total_loss / max(num_batches, 1)
    mean_train_rmse = total_rmse / max(num_batches, 1)

    model.eval()
    val_loss_sum = 0.0
    val_rmse_sum = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["depth"].to(device)

            H, W = images.shape[-2:]
            out = model(vjepa_preprocessing(images), output_size=(H, W))

            val_loss_sum += compute_loss(out, targets).item()
            val_rmse_sum += scale_invariant_rmse(out["depth"], targets).item()
            val_batches += 1

    mean_val_loss = val_loss_sum / max(val_batches, 1)
    mean_val_rmse = val_rmse_sum / max(val_batches, 1)

    print(
        f"Epoch {epoch + 1} | "
        f"train {LOSS_MODE}: {mean_train_loss:.6f} si-rmse: {mean_train_rmse:.6f} | "
        f"val {LOSS_MODE}: {mean_val_loss:.6f} si-rmse: {mean_val_rmse:.6f}"
    )

    # Select on val SI-RMSE (the submission metric) so Stage 1 (si_mse) and
    # Stage 2 (nll) checkpoints are picked by the same criterion and are
    # directly comparable.
    if mean_val_rmse < best_val_rmse:
        best_val_rmse = mean_val_rmse
        epochs_without_improvement = 0

        save_training_checkpoint(BEST_CKPT, epoch)
        print(f"Saved new best model: val si-rmse {mean_val_rmse:.6f}")
        write_submission_from_model(model, SUBMISSION_CSV)
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement}/{PATIENCE} epochs")

    save_training_checkpoint(LAST_CKPT, epoch)
    print(f"Saved last checkpoint: {LAST_CKPT}")

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping")
        break


# -----------------------------------------------------------------------------
# Save last
# -----------------------------------------------------------------------------

if last_epoch >= 0:
    save_training_checkpoint(LAST_CKPT, last_epoch)
    print(f"Saved final last checkpoint: {LAST_CKPT}")


# -----------------------------------------------------------------------------
# Inference for submission (uses the best checkpoint)
# -----------------------------------------------------------------------------

print(f"Loading best checkpoint from {BEST_CKPT}...")
checkpoint = torch.load(BEST_CKPT, map_location=device)
ckpt_variant = checkpoint["config"]["variant"]

# Reuse the in-memory VJEPA encoder; skip DA download since trained weights
# are about to overwrite depth_head anyway.
inference_model = build_jepa_depth_anything(
    vj_encoder,
    variant=ckpt_variant,
    device=device,
    load_da_pretrained=False,
)
missing, unexpected = inference_model.load_state_dict(
    checkpoint["model_state_dict"], strict=False,
)
bad_missing = [k for k in missing if not k.startswith("vjepa_encoder.")]
if bad_missing or unexpected:
    raise RuntimeError(
        f"Unexpected/missing keys when restoring trained model: "
        f"missing={bad_missing}, unexpected={unexpected}"
    )
write_submission_from_model(inference_model, SUBMISSION_CSV)
print("Done.")
