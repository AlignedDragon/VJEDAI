from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, train_dir):
        self.train_dir = Path(train_dir)

        self.image_paths = sorted(list(self.train_dir.glob('*.png')))
        self.depth_paths = sorted(list(self.train_dir.glob('*.npy')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = torch.tensor(image, dtype=torch.float32)

        # convert HWC to CHW
        image = image.permute(2, 0, 1)

        sample = {'image': image}

        depth_path = self.depth_paths[idx]
        depth = np.load(depth_path)
        depth = torch.tensor(depth, dtype=torch.float32)

        depth = depth.unsqueeze(0)

        sample['depth'] = depth

        return sample

class TestDataset(Dataset):
    def __init__(self, test_dir):
        self.test_dir = Path(test_dir)
        self.image_paths = sorted(list(self.test_dir.glob("*_rgb.png")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)

        image_id = image_path.stem.replace("_rgb", "")

        return {
            "image": image,
            "id": image_id,
        }