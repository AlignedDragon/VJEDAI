#!/usr/bin/env python
# coding: utf-8

from torch.utils.data import DataLoader, random_split
import torch
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

from dataset import TrainDataset, TestDataset
from preprocessing import vjepa_preprocessing
from create_submission import encode_depth, save_submission
from fusion_transformer import DepthVJepaFusionTransformer

TRAIN_DIR = '/cluster/courses/cil/monocular-depth-estimation/train/'
TEST_DIR = '/cluster/courses/cil/monocular-depth-estimation/test'
DEPTH_ANYTHING_MODEL = 'vitb' # vits, vitb, vitl, vitg
BATCH_SIZE = 16

# Load data

print("loading data...")
train_dataset = TrainDataset(TRAIN_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_fraction = 0.1
val_size = int(len(train_dataset) * val_fraction)
train_size = len(train_dataset) - val_size
train_subset, val_subset = random_split(train_dataset,
                                        [train_size, val_size],
                                        generator=torch.Generator())
train_loader_fit = DataLoader(train_subset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
val_loader = DataLoader(val_subset,
                        batch_size=BATCH_SIZE,
                        num_workers=4,
                        pin_memory=True)
test_dataset = TestDataset(TEST_DIR)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print("data loaded.")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

PROJECT_ROOT = Path("/home/ragerber").resolve()
VJEPA_ROOT = PROJECT_ROOT / "external" / "vjepa2"
VJEPA_SRC = VJEPA_ROOT / "src"
DA_ROOT = PROJECT_ROOT / "external" / "Depth-Anything-V2"

for p in [str(DA_ROOT), str(PROJECT_ROOT / "external" / "Depth-Anything-V2")]:
    while p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(VJEPA_SRC))
sys.path.insert(0, str(VJEPA_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

for module_name in list(sys.modules):
    if module_name == "app" or module_name.startswith("app."):
        del sys.modules[module_name]
vj_encoder, _ = torch.hub.load(
    str(VJEPA_ROOT),
    "vjepa2_1_vit_large_384",
    source="local",
)
vj_encoder = vj_encoder.to(device).eval()
print("VJepa loaded.")

sys.path.append(str(DA_ROOT))
from depth_anything_v2.dpt import DepthAnythingV2
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

da_encoder = DEPTH_ANYTHING_MODEL

da_model = DepthAnythingV2(**model_configs[da_encoder])
da_model.load_state_dict(torch.load(f'./checkpoints/depth_anything_v2_{da_encoder}.pth', map_location='cpu'))
da_model = da_model.to(device).eval()
print("DepthAnything loaded.")


# Training

def scale_invariant_rmse(pred, target, eps=1e-6):
    if pred.ndim == 4:
        pred = pred.squeeze(1)
    if target.ndim == 4:
        target = target.squeeze(1)

    mask = target > eps

    pred = torch.clamp(pred, min=eps)
    target = torch.clamp(target, min=eps)

    log_diff = torch.log(pred) - torch.log(target)
    log_diff = log_diff[mask]

    bias = - torch.mean(log_diff)
    return torch.sqrt(torch.mean((log_diff + bias)**2))


# Hyperparameters
# AdamW Optimizer
LOSS_RATE = 1e-4
WEIGHT_DECAY = 1e-4
# Training Loop
NUM_EPOCHS = 50
PATIENCE = 5


fusion_model = DepthVJepaFusionTransformer().to(device)

optimizer = torch.optim.AdamW(fusion_model.parameters(),
                              lr=LOSS_RATE,
                              weight_decay=WEIGHT_DECAY)

da_model.eval()
vj_encoder.eval()


# Main Training Loop

print("Starting Training..")

best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):
    fusion_model.train()

    total_loss = 0.0
    num_batches = 0

    for batch in train_loader_fit:
        images = batch["image"].to(device)
        targets = batch["depth"].to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            da_depth = da_model(images / 255.0)
            vjepa_tokens = vj_encoder(vjepa_preprocessing(images))


        H, W = images.shape[-2:]

        refined_depth = fusion_model(
            da_depth=da_depth,
            vjepa_tokens=vjepa_tokens,
            output_size=(H, W),
        )

        loss = scale_invariant_rmse(refined_depth, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        print(f"batch {num_batches}: loss = {loss.item():.6f}")

    mean_train_loss = total_loss / num_batches

    fusion_model.eval()
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["depth"].to(device)

            da_depth = da_model(images / 255.0)
            vjepa_tokens = vj_encoder(vjepa_preprocessing(images))

            H, W = images.shape[-2:]

            refined_depth = fusion_model(da_depth=da_depth,
                                         vjepa_tokens=vjepa_tokens,
                                         output_size=(H,W))

            loss = scale_invariant_rmse(refined_depth, targets)

            val_loss += loss.item()
            val_batches += 1

        mean_val_loss = val_loss / val_batches

        print(
            f"Epoch {epoch + 1} | "
            f"train loss: {mean_train_loss:.6f} | "
            f"val loss: {mean_val_loss:.6f}"
        )

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            epochs_without_improvement = 0

            torch.save(
            {
                "model_state_dict": fusion_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": {
                    "vjepa_dim": 1024,
                    "d_model": 256,
                    "num_heads": 8,
                    "num_layers": 2,
                    "token_grid_size": 24,
                },
            },
            "./checkpoints/fusion_model_best.pth")

            print(f"Saved new best model with val loss {val_loss:.6f}")

        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{PATIENCE} epochs")

            if epochs_without_improvement >= PATIENCE:
                print("Early stopping")
                break



# Save Model


torch.save(
    {
        "model_state_dict": fusion_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": mean_train_loss,
        "val_loss": mean_val_loss,
        "config": {
            "vjepa_dim": 1024,
            "d_model": 256,
            "num_heads": 8,
            "num_layers": 2,
            "token_grid_size": 24,
        },
    },
    "./checkpoints/fusion_model_last.pth")


# Make Prediction
# load model
checkpoint = torch.load("./checkpoints/fusion_model_last.pth", map_location=device)
fusion_model = DepthVJepaFusionTransformer(**checkpoint["config"]).to(device)
fusion_model.load_state_dict(checkpoint["model_state_dict"])
fusion_model.eval()

rows = []
with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(device)
        image_ids = batch["id"]

        images_da = images / 255.0
        da_depth = da_model(images / 255.0)
        vjepa_tokens = vj_encoder(vjepa_preprocessing(images))
        H, W = images.shape[-2:]

        pred_depths = fusion_model(da_depth=da_depth,
                                         vjepa_tokens=vjepa_tokens,
                                         output_size=(H,W))
        if pred_depths.ndim == 4:
            pred_depths = pred_depths.squeeze(1)

        pred_depths = pred_depths.detach().cpu().numpy()

        for depth, image_id in zip(pred_depths, image_ids):
            rows.append({"id": f"{image_id}_depth","Depths": encode_depth(depth)})

df = save_submission(rows, "./submission.csv")

