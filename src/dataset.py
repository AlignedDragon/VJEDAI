from pathlib import Path
import numpy as np
import torch
from PIL import Image
import json
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(image):
    """
    Input:  PIL image
    Output: torch tensor [3, H, W], ImageNet-normalized
    """

    image = np.array(image)
    image = torch.tensor(image, dtype=torch.float32)

    # HWC -> CHW
    image = image.permute(2, 0, 1)

    # [0, 255] -> [0, 1]
    image = image / 255.0

    # ImageNet normalization
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image

def make_or_load_split(train_dataset, split_path, split_seed, val_frac, train_dir):
    n = len(train_dataset)

    if split_path.exists():
        with open(split_path, 'r') as f:
            split = json.load(f)

        train_indices = split['train_indices']
        val_indices = split['val_indices']

        print(f'Loaded split from {split_path}')

    else:
        rng = np.random.default_rng(split_seed)
        all_indices = np.arange(n)
        rng.shuffle(all_indices)

        val_size = int(n * val_frac)
        val_indices = sorted(all_indices[:val_size].tolist())
        train_indices = sorted(all_indices[val_size:].tolist())

        split = {
            'dataset_size': n,
            'val_fraction': val_frac,
            'seed': split_seed,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'train_indices': train_indices,
            'val_indices': val_indices
        }

        with open(split_path, 'w') as f:
            json.dump(split, f)

        print(f'Created and saved split to {split_path}')

    return train_indices, val_indices

class TrainDataset(Dataset):
    def __init__(self, train_dir):
        self.train_dir = Path(train_dir)

        self.image_paths = sorted(list(self.train_dir.glob("*.png")))
        self.depth_paths = sorted(list(self.train_dir.glob("*.npy")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]

        image = Image.open(image_path).convert("RGB")
        image = preprocess_image(image)

        depth = np.load(depth_path)
        depth = torch.tensor(depth, dtype=torch.float32)
        depth = depth.unsqueeze(0)

        return {
            "image": image,
            "depth": depth,
        }


class TestDataset(Dataset):
    def __init__(self, test_dir):
        self.test_dir = Path(test_dir)
        self.image_paths = sorted(list(self.test_dir.glob("*_rgb.png")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        image = preprocess_image(image)

        image_id = image_path.stem.replace("_rgb", "")

        return {
            "image": image,
            "id": image_id,
        }