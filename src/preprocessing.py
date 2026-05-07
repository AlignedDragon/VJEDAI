import torch
import torch.nn.functional as F

def vjepa_preprocessing(images):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    images_vjepa = images / 255.0


    images_vjepa = F.interpolate(
        images_vjepa,
        size=(384, 384),
        mode="bilinear"
    )

    # ImageNet-style normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images_vjepa = (images_vjepa - mean) / std

    # add fake time dimension: [B, 3, H, W] -> [B, 3, T, H, W]
    images_vjepa = images_vjepa.unsqueeze(2)

    return images_vjepa