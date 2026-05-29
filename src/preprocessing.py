"""Input preprocessing for the V-JEPA encoder (see report Sec. 2.3)."""

import torch
import torch.nn.functional as F

def vjepa_preprocessing(images):
    """[B, 3, H, W] uint8-range RGB -> [B, 3, 1, 384, 384] V-JEPA input."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    images_vjepa = images / 255.0  # to [0, 1]

    # V-JEPA expects a fixed 384x384 input.
    images_vjepa = F.interpolate(
        images_vjepa,
        size=(384, 384),
        mode="bilinear"
    )

    # ImageNet-style normalization (matches DA / V-JEPA backbones)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images_vjepa = (images_vjepa - mean) / std

    # add singleton time dim so V-JEPA sees a 1-frame "video": [B,3,H,W] -> [B,3,T,H,W]
    images_vjepa = images_vjepa.unsqueeze(2)

    return images_vjepa