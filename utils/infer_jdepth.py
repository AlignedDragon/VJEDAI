#!/usr/bin/env python
# coding: utf-8
"""Standalone inference / submission writer for a saved JepaDepthAnything model.

Loads a trained checkpoint and writes a Kaggle submission.csv, without
touching the training pipeline. The model-build sequence (VJEPA hub.load
with DA kept off sys.path until afterwards) mirrors
``Full_Pipeline_JepaDepth.py`` exactly -- see the long comment there for why
the sys.path ordering matters.

Depth post-processing matches the fixed ``write_submission_from_model``:
siRMSE is scale-invariant and GT depths live in [0.001, 80], so each
prediction is normalised by its own median (a per-image multiplicative
constant -- metric-neutral) and clamped into the GT range. This keeps every
value inside float16's well-behaved band and avoids the old global-rescale
path that collapsed to an all-zero submission on float16 overflow.

Usage:
    python infer_jdepth.py [--ckpt PATH] [--variant large] \\
        [--out submission.csv] [--test-dir DIR] [--batch-size 8]

If --ckpt is omitted the checkpoint is downloaded from HuggingFace
(--hf-repo / --hf-file), defaulting to
    kalandarX/jdepth :: large/v1.2_nll_deliverable.pth
"""

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Defaults / paths
# -----------------------------------------------------------------------------

TEST_DIR = "/cluster/courses/cil/monocular-depth-estimation/test"
DEPTH_MIN, DEPTH_MAX = 0.001, 80.0  # GT range (scale-invariant metric)

# Default published checkpoint (downloaded via huggingface_hub when --ckpt omitted).
DEFAULT_HF_REPO = "kalandarX/jdepth"
DEFAULT_HF_FILE = "large/v1.2_nll_deliverable.pth"

PROJECT_ROOT = Path.home().resolve()
MONO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = MONO_ROOT / "src"


def first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


VJEPA_ROOT = first_existing(MONO_ROOT / "external" / "vjepa2", PROJECT_ROOT / "external" / "vjepa2")
VJEPA_SRC = VJEPA_ROOT / "src"
DA_ROOT = first_existing(
    MONO_ROOT / "external" / "Depth-Anything-V2",
    PROJECT_ROOT / "external" / "Depth-Anything-V2",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt",
        default=None,
        help="local checkpoint .pth (overrides HF download)",
    )
    p.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help="HuggingFace repo id")
    p.add_argument("--hf-file", default=DEFAULT_HF_FILE, help="checkpoint path within the HF repo")
    p.add_argument("--variant", default="large", choices=["base", "large"])
    p.add_argument("--out", default="./submission.csv", help="output csv path")
    p.add_argument("--test-dir", default=TEST_DIR)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--uncertainty-temperature",
        type=float,
        default=1.0,
        help="Scale predicted uncertainty as sigma *= T, equivalent to log_var += 2*log(T).",
    )
    p.add_argument(
        "--calibration-csv",
        default=None,
        help="Optional calibration_jdepth.csv; uses temperature_scale from it if present.",
    )
    return p.parse_args()


def load_calibration_temperature(path):
    if path is None:
        return None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("metric") == "temperature_scale":
                return float(row["value"])
    raise ValueError(f"temperature_scale not found in {path}")


def apply_uncertainty_temperature(log_var, temperature):
    if temperature <= 0:
        raise ValueError("uncertainty temperature must be positive")
    if temperature == 1.0:
        return log_var
    return log_var + 2.0 * math.log(temperature)


def resolve_ckpt(args):
    if args.ckpt:
        path = Path(args.ckpt)
        if not path.exists():
            sys.exit(f"Checkpoint not found: {path}")
        return path
    from huggingface_hub import hf_hub_download

    print(f"Downloading checkpoint from HF: {args.hf_repo} :: {args.hf_file}")
    path = Path(hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file))
    print(f"Checkpoint cached at {path}")
    return path


# -----------------------------------------------------------------------------
# sys.path setup (must run before the imports below)
# -----------------------------------------------------------------------------

def setup_syspath():
    # Keep DA off the path for now (see Full_Pipeline_JepaDepth.py).
    for p in [str(DA_ROOT)]:
        while p in sys.path:
            sys.path.remove(p)
    for p in [str(SRC_DIR), str(VJEPA_SRC), str(VJEPA_ROOT), str(PROJECT_ROOT)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    for module_name in list(sys.modules):
        if module_name == "app" or module_name.startswith("app."):
            del sys.modules[module_name]


def build_model(variant, device):
    """Replicates the pipeline's VJEPA-then-DA build ordering."""
    # DA must NOT be on sys.path while VJEPA's hub.load runs.
    for p in [str(DA_ROOT), str(MONO_ROOT / "external" / "Depth-Anything-V2")]:
        while p in sys.path:
            sys.path.remove(p)
    for module_name in list(sys.modules):
        if module_name == "app" or module_name.startswith("app."):
            del sys.modules[module_name]

    from jepa_depth_anything import VARIANTS, build_jepa_depth_anything  # noqa: E402

    cfg = VARIANTS[variant]
    print(
        f"variant={variant} | vjepa_arch={cfg['vjepa_arch']} | "
        f"da_encoder={cfg['da_encoder']}"
    )

    vj_encoder, _ = torch.hub.load(
        str(VJEPA_ROOT),
        cfg["vjepa_arch"],
        source="local",
        out_layers=cfg["vjepa_out_layers"],
    )
    vj_encoder = vj_encoder.to(device).eval()
    print(f"VJEPA ({cfg['vjepa_arch']}) loaded.")

    # Now safe to expose DA so the DPT head can import.
    sys.path.append(str(DA_ROOT))

    # Trained depth_head weights overwrite the DA init anyway, so skip the
    # DA download.
    model = build_jepa_depth_anything(
        vj_encoder, variant=variant, device=device, load_da_pretrained=False
    )
    return model


def main():
    args = parse_args()
    ckpt_path = resolve_ckpt(args)
    csv_temperature = load_calibration_temperature(args.calibration_csv)
    uncertainty_temperature = csv_temperature or args.uncertainty_temperature
    if uncertainty_temperature <= 0:
        sys.exit("--uncertainty-temperature must be positive")
    print(f"Uncertainty temperature: {uncertainty_temperature:.6g}")

    setup_syspath()
    from dataset import TestDataset                       # noqa: E402
    from preprocessing import vjepa_preprocessing         # noqa: E402
    from create_submission import encode_depth, save_submission  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    test_dataset = TestDataset(args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    print(f"test samples: {len(test_dataset)}")

    model = build_model(args.variant, device)

    print(f"Loading checkpoint {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if isinstance(ckpt, dict) and "config" in ckpt:
        ck_variant = ckpt["config"].get("variant")
        if ck_variant and ck_variant != args.variant:
            sys.exit(
                f"Checkpoint variant={ck_variant} != --variant={args.variant}; "
                f"rerun with --variant {ck_variant}."
            )
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith("vjepa_encoder.")]
    if bad_missing or unexpected:
        sys.exit(
            f"Unexpected/missing keys restoring model: "
            f"missing={bad_missing}, unexpected={unexpected}"
        )
    model.eval()

    n_clamped = 0
    n_pixels = 0
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            image_ids = batch["id"]
            H, W = images.shape[-2:]

            out = model(vjepa_preprocessing(images), output_size=(H, W))
            if "log_var" in out:
                out["log_var"] = apply_uncertainty_temperature(
                    out["log_var"], uncertainty_temperature
                )
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
                rows.append({"id": f"{image_id}_depth", "Depths": encode_depth(clamped)})

    if n_clamped:
        print(
            f"Clamped {n_clamped}/{n_pixels} pixels "
            f"({100 * n_clamped / max(n_pixels, 1):.2f}%) into [{DEPTH_MIN}, {DEPTH_MAX}]."
        )
    save_submission(rows, args.out)
    print(f"Submission written to {args.out} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
