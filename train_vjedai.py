#!/usr/bin/env python
# coding: utf-8
"""Train the (video-native) vjedai architecture end-to-end.

VJEPA 2.1 encoder + Video-Depth-Anything temporal DPT head (DA DPT + four
TemporalModule blocks) + Gaussian uncertainty head. The model consumes a clip
of ``T`` frames and predicts temporally-consistent depth + per-pixel
log-variance per frame. A still image is the ``T == 1`` special case, so this
one script trains both regimes:

* ``VJEDAI_DATA_MODE=video`` (default): clips from a ``VideoTrainDataset`` (see
  src/dataset.py) with the temporal-consistency loss active.
* ``VJEDAI_DATA_MODE=image``: the original single-frame Kaggle CIL pipeline
  (``T == 1``); the temporal term vanishes and the objective reduces exactly to
  the per-image scale-invariant loss.

Two spatial objectives are supported via ``LOSS_MODE`` (``VJEDAI_LOSS_MODE``):

* ``"si_mse"``: scale-invariant MSE on log depths. The uncertainty head runs
  but receives zero gradient (stays at its zero-init). A clean depth-only
  Stage 1.
* ``"nll"`` (default): scale-invariant Gaussian NLL using the predicted
  per-pixel log-variance. Stage 2 fine-tune that adds the uncertainty head.

Both spatial objectives are combined with a temporal-consistency term (weight
``VJEDAI_LAMBDA_TEMPORAL``) when ``T > 1``. See src/video_depth_loss.py.

Temporal head init (``VJEDAI_INIT_SOURCE``):
* ``"vda"`` (default for the large variant): warm-start the whole temporal head
  -- DPT + motion modules -- from Video-Depth-Anything-Large.
* ``"da"``: load DA-V2 image depth_head, zero-init motion modules (use for the
  base variant, which has no released VDA video checkpoint).

Key env vars (set in the sbatch, no script edits needed):
    VJEDAI_DATA_MODE=video|image
    VJEDAI_CLIP_LEN=16                 # frames per clip (T), video mode
    VJEDAI_FRAME_STRIDE=1
    VJEDAI_VIDEO_TRAIN_DIR=/path/to/video/train
    VJEDAI_VIDEO_TEST_DIR=/path/to/video/test   # optional
    VJEDAI_LOSS_MODE=nll|si_mse
    VJEDAI_LAMBDA_TEMPORAL=1.0
    VJEDAI_INIT_SOURCE=vda|da|none
    VJEDAI_INIT_FROM=/path/to/stage1.pth        # optional warm-start (strict=False)

Validation SI-RMSE (the leaderboard metric, averaged over frames) drives
checkpoint selection in every mode, so checkpoints are directly comparable.
Filenames are tagged ``best_{DATA_MODE}_{LOSS_MODE}.pth`` to avoid clobbering.

Switch ``VARIANT`` between ``"base"`` and ``"large"`` to pick the VJEPA+DA
pairing; everything else (out_layers, embed dim, checkpoint filenames, repo ids)
is derived from ``src.vjedai.VARIANTS``.
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

# Image-mode (Kaggle CIL) data dirs.
TRAIN_DIR = "/cluster/courses/cil/monocular-depth-estimation/train/"
TEST_DIR = "/cluster/courses/cil/monocular-depth-estimation/test"

# Video-mode data dirs (per-clip layout; see src/dataset.VideoTrainDataset).
VIDEO_TRAIN_DIR = os.environ.get("VJEDAI_VIDEO_TRAIN_DIR", "").strip()
VIDEO_TEST_DIR = os.environ.get("VJEDAI_VIDEO_TEST_DIR", "").strip()

VARIANT = "large"       # "base" or "large"
BATCH_SIZE = int(os.environ.get("VJEDAI_BATCH_SIZE", "8"))  # video is heavy; lower via env
NUM_EPOCHS = int(os.environ.get("VJEDAI_NUM_EPOCHS", "36"))
PATIENCE = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.1
NUM_WORKERS = 4

# image | video (image = T=1 special case of the same code path)
DATA_MODE = os.environ.get("VJEDAI_DATA_MODE", "video").lower()
if DATA_MODE not in ("image", "video"):
    raise ValueError(f"Unknown VJEDAI_DATA_MODE={DATA_MODE!r}; expected 'image' or 'video'.")

CLIP_LEN = int(os.environ.get("VJEDAI_CLIP_LEN", "16"))      # T (video mode)
FRAME_STRIDE = int(os.environ.get("VJEDAI_FRAME_STRIDE", "1"))
# Optional fixed (H, W) to resize clips to, so clips from differently-sized
# sequences can share a batch. "" = keep native resolution.
_thw = os.environ.get("VJEDAI_TARGET_HW", "").strip()
TARGET_HW = tuple(int(x) for x in _thw.split(",")) if _thw else None

# "si_mse": deterministic Stage 1, ignores the uncertainty head.
# "nll": heteroscedastic Stage 2, uses predicted log-variance.
LOSS_MODE = os.environ.get("VJEDAI_LOSS_MODE", "nll").lower()
if LOSS_MODE not in ("nll", "si_mse"):
    raise ValueError(f"Unknown LOSS_MODE={LOSS_MODE!r}; expected 'nll' or 'si_mse'.")

LAMBDA_TEMPORAL = float(os.environ.get("VJEDAI_LAMBDA_TEMPORAL", "1.0"))

# Temporal-head init: default to VDA warm-start for large, DA for base.
INIT_SOURCE = os.environ.get(
    "VJEDAI_INIT_SOURCE", "vda" if VARIANT == "large" else "da"
).lower()

# How VJEPA consumes the clip: "clip" (video path, tubelet temporal patching +
# spatiotemporal attention, learned temporal upsampler) or "per_frame" (image
# path per frame; temporal modelling only in the motion modules).
ENCODE_MODE = os.environ.get("VJEDAI_ENCODE_MODE", "clip").lower()

# Optional path to a Stage 1 checkpoint to warm-start from. Loaded with
# strict=False, so missing vjepa_encoder.* keys are fine (the encoder is
# loaded fresh from torch.hub). Set via env var to avoid editing this file.
INIT_FROM = os.environ.get("VJEDAI_INIT_FROM", "").strip() or None
RESUME_FROM = os.environ.get("VJEDAI_RESUME_FROM", "").strip() or None
AUTO_RESUME = os.environ.get("VJEDAI_AUTO_RESUME", "0").lower() in ("1", "true", "yes")

# Keep heavy checkpoints off $HOME: write them to $SCRATCH. Falls back to a
# local ./checkpoints/ if $SCRATCH isn't set (e.g., laptop).
SCRATCH = Path(os.environ.get("SCRATCH", "."))
CHECKPOINT_DIR = SCRATCH / "checkpoints" / f"vjedai_{VARIANT}"
# Mode-tagged so image/video and si_mse/nll runs don't overwrite each other.
BEST_CKPT = CHECKPOINT_DIR / f"best_{DATA_MODE}_{LOSS_MODE}.pth"
LAST_CKPT = CHECKPOINT_DIR / f"last_{DATA_MODE}_{LOSS_MODE}.pth"
SUBMISSION_CSV = Path("./submission.csv")
VIDEO_PRED_DIR = SCRATCH / "predictions" / f"vjedai_{VARIANT}_{LOSS_MODE}"


# -----------------------------------------------------------------------------
# Paths / sys.path setup
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path.home().resolve()
MONO_ROOT = Path(__file__).resolve().parent
SRC_DIR = MONO_ROOT / "src"


def _first_existing(*paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return Path(paths[-1])


# Resolve the V-JEPA 2.1 checkout. Cluster sbatch clones into external/vjepa2;
# locally it may sit at the repo root (./vjepa2). $VJEPA_ROOT overrides.
VJEPA_ROOT = _first_existing(
    os.environ.get("VJEPA_ROOT"),
    MONO_ROOT / "external" / "vjepa2",
    MONO_ROOT / "vjepa2",
    PROJECT_ROOT / "external" / "vjepa2",
)
VJEPA_SRC = VJEPA_ROOT / "src"
DA_ROOT = _first_existing(
    os.environ.get("DA_ROOT"),
    MONO_ROOT / "external" / "Depth-Anything-V2",
    PROJECT_ROOT / "external" / "Depth-Anything-V2",
)

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

from dataset import (                                  # noqa: E402
    TrainDataset,
    TestDataset,
    VideoTrainDataset,
    VideoTestDataset,
)
from preprocessing import vjepa_preprocessing           # noqa: E402
from create_submission import encode_depth, save_submission  # noqa: E402
from video_depth_loss import vjedai_video_depth_loss, scale_invariant_rmse  # noqa: E402


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

print(f"loading data... (mode={DATA_MODE})")
if DATA_MODE == "video":
    if not VIDEO_TRAIN_DIR:
        raise ValueError(
            "VJEDAI_DATA_MODE=video requires VJEDAI_VIDEO_TRAIN_DIR to point at "
            "the per-clip video depth dataset (see src/dataset.VideoTrainDataset)."
        )
    train_dataset = VideoTrainDataset(
        VIDEO_TRAIN_DIR,
        clip_len=CLIP_LEN,
        frame_stride=FRAME_STRIDE,
        target_hw=TARGET_HW,
    )
else:
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
print(f"data loaded. train: {train_size} | val: {val_size}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


def batch_to_clip(batch):
    """Normalise an image- or video-mode batch to the (B, T, ...) convention.

    Returns ``(images_clip, targets_clip)`` with shapes
    ``[B, T, 3, H, W]`` and ``[B, T, 1, H, W]``. Image mode (``T == 1``) inserts
    the time axis; video mode passes the clip through unchanged.
    """
    images = batch["image"]
    targets = batch["depth"]
    if DATA_MODE == "image":
        images = images.unsqueeze(1)    # [B,3,H,W]  -> [B,1,3,H,W]
        targets = targets.unsqueeze(1)  # [B,1,H,W]  -> [B,1,1,H,W]
    return images, targets


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

# vjedai imports depth_anything_v2 lazily (inside
# vjedai.__init__), so importing this module here does not yet
# require DA_ROOT on sys.path.
from vjedai import VARIANTS, build_vjedai  # noqa: E402

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

# Now it's safe to expose DA so vjedai can import its DPT head.
sys.path.append(str(DA_ROOT))

model = build_vjedai(
    vj_encoder, variant=VARIANT, device=device,
    init_source=INIT_SOURCE, encode_mode=ENCODE_MODE,
)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"vjedai built. trainable: {n_trainable/1e6:.2f}M | frozen: {n_frozen/1e6:.2f}M")
print(
    f"init_source={INIT_SOURCE} | encode_mode={ENCODE_MODE} | "
    f"data_mode={DATA_MODE} | clip_len="
    f"{CLIP_LEN if DATA_MODE == 'video' else 1} | loss={LOSS_MODE} | "
    f"lambda_temporal={LAMBDA_TEMPORAL}"
)

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
# Loss
# -----------------------------------------------------------------------------

criterion = vjedai_video_depth_loss(mode=LOSS_MODE, lambda_temporal=LAMBDA_TEMPORAL)


def compute_loss(out, targets):
    """Combined spatial (si_mse|nll) + temporal loss; returns a scalar tensor."""
    return criterion(out["depth"], targets, out["log_var"])["total_loss"]


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
            "config": {
                "variant": VARIANT,
                "loss_mode": LOSS_MODE,
                "data_mode": DATA_MODE,
                "clip_len": CLIP_LEN if DATA_MODE == "video" else 1,
                "init_source": INIT_SOURCE,
            },
        },
        path,
    )


def _normalize_and_clamp(depth, depth_min=0.001, depth_max=80.0):
    """Per-frame median-normalise + clamp a depth map into a float16-safe band.

    siRMSE is scale-invariant and GT depths live in [depth_min, depth_max]. The
    model is trained scale-invariant, so its raw output scale is arbitrary; we
    divide each frame by its own median (a per-image multiplicative constant --
    exactly the log-scale alignment siRMSE removes, so this is metric-neutral)
    and clamp into the GT range. Returns (clamped_depth, n_clamped, n_pixels).
    """
    valid = np.isfinite(depth) & (depth > 0)
    med = float(np.median(depth[valid])) if valid.any() else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    depth = np.nan_to_num(
        depth / med, nan=depth_min, posinf=depth_max, neginf=depth_min
    )
    clamped = depth.clip(depth_min, depth_max)
    return clamped, int((clamped != depth).sum()), depth.size


def write_submission_from_model(submission_model, output_path):
    """Image-mode Kaggle CSV: one median-normalised depth row per test image."""
    test_dataset = TestDataset(TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    was_training = submission_model.training
    submission_model.eval()

    n_clamped = 0
    n_pixels = 0
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)  # [B,3,H,W]
            image_ids = batch["id"]
            H, W = images.shape[-2:]

            # image -> T=1 clip
            out = submission_model(
                vjepa_preprocessing(images.unsqueeze(1)), output_size=(H, W)
            )
            pred_depths = out["depth"][:, 0, 0]  # [B,H,W] (T=1, C=1)
            pred_depths = pred_depths.detach().float().cpu().numpy()

            for depth, image_id in zip(pred_depths, image_ids):
                clamped, nc, npx = _normalize_and_clamp(depth)
                n_clamped += nc
                n_pixels += npx
                rows.append(
                    {"id": f"{image_id}_depth", "Depths": encode_depth(clamped)}
                )

    if n_clamped:
        print(
            f"Clamped {n_clamped}/{n_pixels} pixels "
            f"({100 * n_clamped / max(n_pixels, 1):.2f}%) into [0.001, 80.0]."
        )
    save_submission(rows, str(output_path))
    print(f"Submission written to {output_path} ({len(rows)} rows).")

    if was_training:
        submission_model.train()


def write_video_predictions(pred_model, out_dir):
    """Video-mode inference: save one median-normalised depth .npy per frame.

    Runs the model over non-overlapping clips of the test video set and writes
    ``<out_dir>/<frame_id>.npy``. (The Kaggle CIL competition is image-only, so
    video mode produces per-frame arrays rather than a submission CSV.)
    """
    if not VIDEO_TEST_DIR:
        print("No VJEDAI_VIDEO_TEST_DIR set; skipping video prediction dump.")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = VideoTestDataset(
        VIDEO_TEST_DIR, clip_len=CLIP_LEN, frame_stride=FRAME_STRIDE, target_hw=TARGET_HW
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    was_training = pred_model.training
    pred_model.eval()
    # Short sequences are padded by repeating the last frame (see
    # _enumerate_clips), which yields duplicate frame ids in a clip. Track the
    # ids already written and keep the FIRST prediction for each (its temporal
    # window is the natural one; later duplicates sit at degraded padded
    # positions), so padding never silently overwrites a real prediction.
    seen = set()
    n_written = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)    # [B,T,3,H,W]
            ids = batch["id"]                     # list of T lists (one per frame)
            H, W = images.shape[-2:]
            out = pred_model(vjepa_preprocessing(images), output_size=(H, W))
            preds = out["depth"][:, :, 0].detach().float().cpu().numpy()  # [B,T,H,W]

            B, T = preds.shape[:2]
            for b in range(B):
                for t in range(T):
                    frame_id = ids[t][b]  # default collate transposes the T lists
                    if frame_id in seen:
                        continue
                    seen.add(frame_id)
                    clamped, _, _ = _normalize_and_clamp(preds[b, t])
                    np.save(out_dir / f"{frame_id}.npy", clamped.astype(np.float16))
                    n_written += 1
    print(f"Wrote {n_written} per-frame depth maps to {out_dir}.")
    if was_training:
        pred_model.train()


def write_predictions(pred_model):
    """Dispatch inference output by data mode."""
    if DATA_MODE == "image":
        write_submission_from_model(pred_model, SUBMISSION_CSV)
    else:
        write_video_predictions(pred_model, VIDEO_PRED_DIR)


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

    # --- train pass ---
    total_loss = 0.0
    total_rmse = 0.0
    num_batches = 0

    for batch in train_loader:
        images, targets = batch_to_clip(batch)
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        H, W = targets.shape[-2:]
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

    # --- validation pass (always tracks SI-RMSE for model selection) ---
    model.eval()
    val_loss_sum = 0.0
    val_rmse_sum = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            images, targets = batch_to_clip(batch)
            images = images.to(device)
            targets = targets.to(device)

            H, W = targets.shape[-2:]
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

    # Select on val SI-RMSE (the submission metric) so si_mse and nll
    # checkpoints are picked by the same criterion and are comparable.
    if mean_val_rmse < best_val_rmse:
        best_val_rmse = mean_val_rmse
        epochs_without_improvement = 0

        save_training_checkpoint(BEST_CKPT, epoch)
        print(f"Saved new best model: val si-rmse {mean_val_rmse:.6f}")
        write_predictions(model)
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
# Inference (uses the best checkpoint)
# -----------------------------------------------------------------------------

print(f"Loading best checkpoint from {BEST_CKPT}...")
checkpoint = torch.load(BEST_CKPT, map_location=device)
ckpt_variant = checkpoint["config"]["variant"]

# Reuse the in-memory VJEPA encoder; skip head init since trained weights are
# about to overwrite depth_head + motion_modules anyway.
inference_model = build_vjedai(
    vj_encoder,
    variant=ckpt_variant,
    device=device,
    init_source="none",
    encode_mode=ENCODE_MODE,
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
write_predictions(inference_model)
print("Done.")
