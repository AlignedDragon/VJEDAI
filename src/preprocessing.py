"""Input preprocessing for the V-JEPA encoder (see report Sec. 2.3).

Accepts a still-image batch ``[B, 3, H, W]`` (treated as T=1) or a clip
``[B, T, 3, H, W]``, returning ``[B, 3, T, 384, 384]``.
"""

import torch
import torch.nn.functional as F

def vjepa_preprocessing(images):
    """[B,3,H,W] (T=1) or [B,T,3,H,W] RGB -> [B, 3, T, 384, 384] V-JEPA input."""
    if images.dim() == 4:
        images = images.unsqueeze(1)  # [B,3,H,W] -> [B,1,3,H,W] (single-frame clip)
    B, T, C, H, W = images.shape
    assert C == 3, f"clips must be [B, T, 3, H, W]; got {tuple(images.shape)}"
    device = images.device  # keep on the caller's device

    images_vjepa = images.reshape(B * T, C, H, W) / 255.0  # to [0, 1]

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

    # restore time, channel-before-time: [B*T,3,384,384] -> [B,3,T,384,384]
    return images_vjepa.reshape(B, T, 3, 384, 384).permute(0, 2, 1, 3, 4).contiguous()
