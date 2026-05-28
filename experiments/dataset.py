from pathlib import Path
import numpy as np
import torch
from PIL import Image
import json
from torch.utils.data import Dataset

# ImageNet mean and standard deviation for RGB channels.
# These are used to normalize images in the same way as the Depth Anything models.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(image):
    """
    Convert a PIL image into a normalized PyTorch tensor.

    Input:
        image: PIL image in RGB format

    Output:
        image: torch tensor with shape [3, H, W],
            scaled to [0, 1] and normalized with ImageNet statistics
    """

    image = np.array(image) # [H, W, C]
    image = torch.tensor(image, dtype=torch.float32)

    # HWC -> CHW
    image = image.permute(2, 0, 1) # [C, H, W]

    # scale pixel values: [0, 255] -> [0, 1]
    image = image / 255.0

    # ImageNet normalization
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image

def make_or_load_split(train_dataset, split_path, split_seed, val_frac, train_dir):
    """
    Create or load a deterministic train/validation split.

    If a split file already exists, the same split is reused.
    Otherwise, a new split is created using the given random seed and saved to disk.

    This ensures that different training runs are evaluated on the same validation set.
    """

    # Total number of samples in the dataset.
    n = len(train_dataset)

    # If the split already exists, load it so that experiments stay comparable.
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

        # Number of samples assigned to the validation set.
        val_size = int(n * val_frac)

        val_indices = sorted(all_indices[:val_size].tolist())
        train_indices = sorted(all_indices[val_size:].tolist())

        # Store split metadata together with the actual indices.
        split = {
            'dataset_size': n,
            'val_fraction': val_frac,
            'seed': split_seed,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'train_indices': train_indices,
            'val_indices': val_indices
        }

        # Save the split so that future runs use exactly the same split.
        with open(split_path, 'w') as f:
            json.dump(split, f)

        print(f'Created and saved split to {split_path}')

    return train_indices, val_indices

class TrainDataset(Dataset):
    """
    Dataset for supervised depth training.

    Each sample consists of:
        - an RGB image loaded from a .png file
        - a depth map loaded from a .npy file

    The image and depth paths are sorted, so this assumes that the sorted image files
    and sorted depth files correspond to each other one-to-one.
    """
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
    """
    Dataset for test/inference images.

    Each sample contains:
        - a preprocessed RGB image
        - an image ID derived from the filename

    Unlike TrainDataset, this dataset does not load depth maps because test data
    does not include ground-truth depth.
    """
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