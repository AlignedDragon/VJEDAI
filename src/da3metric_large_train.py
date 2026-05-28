
from pathlib import Path 
from torch.utils.data import DataLoader, Subset
from dataset import TrainDataset
import numpy as np 
import json 
import torch
import torch.nn.functional as F 
from depth_anything_3.api import DepthAnything3 


# Hyperparameters
VAL_FRAC = 0.10
SPLIT_SEED = 42

BATCH_SIZE = 8
NUM_WORKERS = 4

NUM_EPOCHS = 10


# Paths
PROJECT_ROOT = Path('/home/ragerber')
SPLIT_DIR = PROJECT_ROOT / 'splits'
SPLIT_PATH = SPLIT_DIR / f'train_val_split_seed{SPLIT_SEED}_val{int(VAL_FRAC * 100)}.json'
TRAIN_DIR = '/cluster/courses/cil/monocular-depth-estimation/train/'
CHECKPOINT_DIR = Path("checkpoints_da3metric_decoder")


# Load data
def make_or_load_split(train_dataset):
    n = len(train_dataset)

    if SPLIT_PATH.exists():
        with open(SPLIT_PATH, 'r') as f:
            split = json.load(f)

        train_indices = split['train_indices']
        val_indices = split['val_indices']

        print(f'Loaded split from {SPLIT_PATH}')

    else:
        rng = np.random.default_rng(SPLIT_SEED)
        all_indices = np.arange(n)
        rng.shuffle(all_indices)

        val_size = int(n * VAL_FRAC)
        val_indices = sorted(all_indices[:val_size].tolist())
        train_indices = sorted(all_indices[val_size:].tolist())

        split = {
            'dataset_size': n,
            'val_fraction': VAL_FRAC,
            'seed': SPLIT_SEED,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'train_indices': train_indices,
            'val_indices': val_indices
        }

        with open(SPLIT_PATH, 'w') as f:
            json.dump(split, f)

        print(f'Created and saved split to {SPLIT_PATH}')

    return train_indices, val_indices

train_dataset = TrainDataset(str(TRAIN_DIR))
train_indices, val_indices = make_or_load_split(train_dataset)

train_set = Subset(train_dataset, train_indices)
val_set = Subset(train_dataset, val_indices)


train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DepthAnything3.from_pretrained('depth-anything/da3metric-large')
model = model.to(device)

for p in model.parameters():
    p.requires_grad = False

for p in model.model.head.parameters():
    p.requires_grad = True 

def init_params(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

model.model.head.apply(init_params)

# AdamW optimizer
optimizer = torch.optim.AdamW(model.model.head.parameters(), lr=5e-5)

# Training Loss Function
def depth_loss(pred, target):
    mask = torch.isfinite(target) & (target > 1e-3) & (target < 80.0)

    l1 = F.l1_loss(pred[mask], target[mask])

    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]

    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]

    mask_x = mask[..., :, 1:] & mask[..., :, :-1]
    mask_y = mask[..., 1:, :] & mask[..., :-1, :]

    loss_grad_x = F.l1_loss(pred_dx[mask_x], target_dx[mask_x])
    loss_grad_y = F.l1_loss(pred_dy[mask_y], target_dy[mask_y])

    return l1 + (loss_grad_x + loss_grad_y)

# Training
def train_one_epoch(model, dataloader, optimizer, device):
    model.model.eval()
    model.model.head.train()

    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        target = batch["depth"].to(device)

        optimizer.zero_grad()

        images = images.unsqueeze(1)

        pred = model.model(images).depth

        loss = depth_loss(pred, target)
        loss.backward()
        
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if step % 20 == 0:
            print(f"step {step:04d} | loss {loss.item():.4f}")

    return total_loss / max(num_batches, 1)


def si_rmse(pred, target):
    mask = torch.isfinite(target) & (target > 1e-3) & (target < 80.0)

    pred_masked = pred[mask].clamp_min(1e-12)
    target_masked = target[mask].clamp_min(1e-12)

    log_pred = torch.log(pred_masked)
    log_target = torch.log(target_masked)

    delta = log_pred - log_target 

    alpha = torch.mean(log_target - log_pred)

    return torch.sqrt(torch.mean((delta + alpha) ** 2))


@torch.no_grad()
def validate(model, dataloader, device):
    model.model.eval()

    total_loss = 0.0
    num_batches = 0
    all_deltas = []

    for batch in dataloader:
        images = batch["image"].to(device)
        target = batch["depth"].to(device)

        images = images.unsqueeze(1)
        pred = model.model(images).depth

        loss = depth_loss(pred, target)

        total_loss += loss.item()
        num_batches += 1

        mask = (
            torch.isfinite(pred)
            & torch.isfinite(target)
            & (pred > 1e-12)
            & (target > 1e-3)
            & (target < 80.0)
        )

        log_pred = torch.log(pred[mask].clamp_min(1e-12))
        log_target = torch.log(target[mask].clamp_min(1e-12))

        delta = log_pred - log_target
        all_deltas.append(delta.detach().cpu())

    avg_loss = total_loss / max(num_batches, 1)

    all_deltas = torch.cat(all_deltas, dim=0)
    alpha = torch.mean(-all_deltas)
    val_si_rmse = torch.sqrt(torch.mean((all_deltas + alpha) ** 2))

    return avg_loss, val_si_rmse.item()


best_val_si_rmse = float("inf")

history = {
    "train_loss": [],
    "val_loss": [],
    "val_si_rmse": [],
}


# Actual training loop
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    train_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
    )

    val_loss, val_si_rmse = validate(
        model=model,
        dataloader=val_loader,
        device=device,
    )

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_si_rmse"].append(val_si_rmse)

    print(f"train loss:    {train_loss:.4f}")
    print(f"val loss:      {val_loss:.4f}")
    print(f"val si-RMSE:   {val_si_rmse:.4f}")

    checkpoint = {
        "epoch": epoch + 1,
        "model_head": model.model.head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_si_rmse": val_si_rmse,
        "history": history,
    }

    torch.save(
        checkpoint,
        CHECKPOINT_DIR / f"epoch_{epoch + 1:03d}.pt",
    )

    if val_si_rmse < best_val_si_rmse:
        best_val_si_rmse = val_si_rmse

        torch.save(
            checkpoint,
            CHECKPOINT_DIR / "best.pt",
        )

        print("saved new best checkpoint")


