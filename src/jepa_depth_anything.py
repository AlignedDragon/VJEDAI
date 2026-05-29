"""JepaDepthAnything: VJEPA 2.1 encoder + DepthAnythingV2 DPT decoder.

VJEPA 2.1 replaces DepthAnythingV2's DINOv2 encoder. Two paired variants are
supported via the ``variant`` argument of ``build_jepa_depth_anything``:

* ``"large"`` (default): VJEPA 2.1 ViT-Large 384 (embed_dim 1024) +
  DA-ViT-L DPT head (in_channels 1024, features 256,
  out_channels [256, 512, 1024, 1024], intermediate indices [5, 11, 17, 23]
  -- VJEPA 2.1's hardcoded hierarchical layers for depth=24).
* ``"base"``: VJEPA 2.1 ViT-Base 384 (embed_dim 768) + DA-ViT-B DPT head
  (in_channels 768, features 128, out_channels [96, 192, 384, 768],
  intermediate indices [2, 5, 8, 11]).

In both variants the VJEPA and DINOv2 embed dims line up exactly, so DA's
pretrained ``projects.*`` 1x1 convs see the channel count they were designed
for. The entire DPT head is loaded from the corresponding DA pretrained
checkpoint and is fully trainable; the VJEPA encoder is frozen. A parallel
Gaussian uncertainty head emits per-pixel log-variance alongside the depth
map.

Caller responsibility: load the VJEPA encoder yourself, passing the matching
``out_layers`` (``[5, 11, 17, 23]`` for large, ``[2, 5, 8, 11]`` for base) so
its ``forward`` returns the four intermediate feature maps the DPT head
consumes. These indices must lie inside VJEPA 2.1's hard-coded
``hierarchical_layers`` for the given depth, otherwise the encoder's
forward raises ``ValueError: <i> is not in list``. The ``Depth-Anything-V2`` repo must already be on ``sys.path``
before calling ``build_jepa_depth_anything`` (this mirrors the existing
pipeline setup).

Example (large):
    vj_encoder, _ = torch.hub.load(
        str(VJEPA_ROOT), "vjepa2_1_vit_large_384", source="local",
        out_layers=[5, 11, 17, 23],
    )
    model = build_jepa_depth_anything(vj_encoder, variant="large", device=device)

Example (base):
    vj_encoder, _ = torch.hub.load(
        str(VJEPA_ROOT), "vjepa2_1_vit_base_384", source="local",
        out_layers=[2, 5, 8, 11],
    )
    model = build_jepa_depth_anything(vj_encoder, variant="base", device=device)

    out = model(vjepa_preprocessing(images), output_size=(H, W))
    depth, log_var = out["depth"], out["log_var"]
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANTS = {
    "base": {
        "vjepa_arch": "vjepa2_1_vit_base_384",
        "vjepa_out_layers": [2, 5, 8, 11],
        "vjepa_dim": 768,
        "vjepa_patch_size": 16,
        "da_encoder": "vitb",
        "da_features": 128,
        "da_out_channels": [96, 192, 384, 768],
        "da_repo_id": "depth-anything/Depth-Anything-V2-Base",
        "da_filename": "depth_anything_v2_vitb.pth",
    },
    "large": {
        "vjepa_arch": "vjepa2_1_vit_large_384",
        # VJEPA 2.1 ViT-L hard-codes its hierarchical (intermediate) layer
        # indices to [5, 11, 17, 23] and only has LayerNorms (norms_block)
        # for those exact indices, so out_layers must be a subset. DA-ViT-L
        # was trained with DINOv2 features at [4, 11, 17, 23]; using
        # VJEPA's layer 5 instead of 4 is a one-block offset and gets
        # absorbed by full DPT fine-tuning.
        "vjepa_out_layers": [5, 11, 17, 23],
        "vjepa_dim": 1024,
        "vjepa_patch_size": 16,
        "da_encoder": "vitl",
        "da_features": 256,
        "da_out_channels": [256, 512, 1024, 1024],
        "da_repo_id": "depth-anything/Depth-Anything-V2-Large",
        "da_filename": "depth_anything_v2_vitl.pth",
    },
}


class JepaDepthAnything(nn.Module):
    """Depth model with a frozen VJEPA 2.1 encoder and a DA DPT decoder.

    Outputs a depth map (mean) and a pixel-wise Gaussian log-variance for
    uncertainty. The VJEPA encoder is held in eval mode with grads disabled.

    Most callers should use ``build_jepa_depth_anything(..., variant=...)``
    instead of constructing this class directly; the builder fills in the
    DA-specific args from a paired-variant config and loads pretrained
    weights.

    Args:
        vjepa_encoder: already-loaded VJEPA 2.1 encoder whose ``forward(x)``
            returns a list of four ``[B, N, vjepa_dim]`` tensors. Construct
            it with the matching ``out_layers`` (see module docstring).
        da_encoder: DepthAnythingV2 encoder name, used only to template the
            DPT head's input channel count. Must pair with ``vjepa_dim``:
            ``"vitb"`` <-> 768, ``"vitl"`` <-> 1024.
        features: DPT inner channel width.
        out_channels: per-level pyramid channel widths.
        vjepa_dim: VJEPA embed dim.
        vjepa_patch_size: VJEPA spatial patch size (default 16).
    """

    def __init__(
        self,
        vjepa_encoder: nn.Module,
        da_encoder: str = "vitl",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        vjepa_dim: int = 1024,
        vjepa_patch_size: int = 16,
    ) -> None:
        super().__init__()

        from depth_anything_v2.dpt import DepthAnythingV2

        da_template = DepthAnythingV2(
            encoder=da_encoder,
            features=features,
            out_channels=list(out_channels),
        )
        self.depth_head = da_template.depth_head
        del da_template

        encoder_embed = getattr(vjepa_encoder, "embed_dim", None)
        if encoder_embed is not None and encoder_embed != vjepa_dim:
            raise ValueError(
                f"vjepa_encoder.embed_dim={encoder_embed} does not match "
                f"vjepa_dim={vjepa_dim}. Variant mismatch between the loaded "
                f"VJEPA encoder and the DA DPT head."
            )

        self.vjepa_encoder = vjepa_encoder
        self.vjepa_encoder.eval()
        for p in self.vjepa_encoder.parameters():
            p.requires_grad = False

        self.da_encoder = da_encoder
        self.vjepa_dim = vjepa_dim
        self.vjepa_patch_size = vjepa_patch_size

        shared_channels = features // 2  # output of head.scratch.output_conv1
        head_features_2 = 32
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(shared_channels, head_features_2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, 1, kernel_size=1),
        )
        # Zero-init the final 1x1 conv so log_var == 0 (variance == 1) at
        # init. This prevents the well-known heteroscedastic-NLL collapse:
        # if log_var starts at a random non-zero value the optimizer races
        # to inflate it to the clamp ceiling because residual^2 * exp(-log_var)
        # shrinks much faster than +log_var grows, killing the depth-fitting
        # gradient before the model ever learns mu. With zero-init the NLL
        # at step 0 reduces to 0.5 * residual^2, giving mu a clean signal
        # before the uncertainty head opens up.
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.zeros_(self.uncertainty_head[-1].bias)

    def train(self, mode: bool = True) -> "JepaDepthAnything":
        super().train(mode)
        self.vjepa_encoder.eval()
        return self

    def _encode(self, vjepa_input: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            feats = self.vjepa_encoder(vjepa_input)
        if not isinstance(feats, (list, tuple)):
            raise RuntimeError(
                "VJEPA encoder must be constructed with the matching "
                "out_layers ([5, 11, 17, 23] for large, [2, 5, 8, 11] for "
                "base -- these must be VJEPA 2.1's hardcoded "
                "hierarchical_layers) so forward returns four intermediate "
                "feature tensors. Got a single tensor."
            )
        if len(feats) != 4:
            raise RuntimeError(
                f"Expected 4 intermediate VJEPA features, got {len(feats)}."
            )
        return list(feats)

    def forward(
        self,
        vjepa_input: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> dict:
        """Run the full encoder + DPT + dual-head pipeline.

        Args:
            vjepa_input: preprocessed VJEPA input from ``vjepa_preprocessing``,
                shape ``[B, 3, 1, H_in, W_in]``. ``H_in`` and ``W_in`` must be
                multiples of ``vjepa_patch_size`` (typically 384).
            output_size: ``(H, W)`` target spatial size of the depth and
                log-variance maps.

        Returns:
            ``{"depth": [B, 1, H, W], "log_var": [B, 1, H, W]}``.
            ``depth`` is non-negative (DA-style ReLU); ``log_var`` is
            unconstrained (Gaussian log-variance).
        """
        B = vjepa_input.shape[0]
        H_in, W_in = vjepa_input.shape[-2], vjepa_input.shape[-1]
        patch_h = H_in // self.vjepa_patch_size
        patch_w = W_in // self.vjepa_patch_size

        feats = self._encode(vjepa_input)

        head = self.depth_head
        pyramid = []
        for i, tokens in enumerate(feats):
            expected_tokens = patch_h * patch_w
            if tokens.shape[1] != expected_tokens:
                raise RuntimeError(
                    f"VJEPA feature {i} has {tokens.shape[1]} tokens but "
                    f"expected {expected_tokens} ({patch_h}x{patch_w}). "
                    f"Check vjepa_input size vs vjepa_patch_size."
                )
            x = tokens.permute(0, 2, 1).reshape(B, self.vjepa_dim, patch_h, patch_w)
            x = head.projects[i](x)
            x = head.resize_layers[i](x)
            pyramid.append(x)

        l1, l2, l3, l4 = pyramid
        l1_rn = head.scratch.layer1_rn(l1)
        l2_rn = head.scratch.layer2_rn(l2)
        l3_rn = head.scratch.layer3_rn(l3)
        l4_rn = head.scratch.layer4_rn(l4)

        p4 = head.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        p3 = head.scratch.refinenet3(p4, l3_rn, size=l2_rn.shape[2:])
        p2 = head.scratch.refinenet2(p3, l2_rn, size=l1_rn.shape[2:])
        p1 = head.scratch.refinenet1(p2, l1_rn)

        shared = head.scratch.output_conv1(p1)
        shared = F.interpolate(
            shared,
            size=tuple(output_size),
            mode="bilinear",
            align_corners=True,
        )

        mu = head.scratch.output_conv2(shared)
        log_var = self.uncertainty_head(shared)

        return {"depth": mu, "log_var": log_var}


def build_jepa_depth_anything(
    vjepa_encoder: nn.Module,
    variant: str = "large",
    device: Optional[torch.device] = None,
    load_da_pretrained: bool = True,
    da_checkpoint_path: Optional[str] = None,
) -> JepaDepthAnything:
    """Construct a ``JepaDepthAnything`` and load DA pretrained weights.

    The ``variant`` argument selects a matched VJEPA/DA pair (``"base"`` or
    ``"large"``); see ``VARIANTS`` for the exact configs. All ``depth_head.*``
    keys (including ``depth_head.projects.*``) are loaded from the
    corresponding DA checkpoint, and every parameter in ``depth_head``
    remains trainable. Only the VJEPA encoder is kept frozen.

    Args:
        vjepa_encoder: VJEPA 2.1 encoder loaded with the matching
            ``out_layers`` (see module docstring) for the chosen variant.
        variant: ``"base"`` or ``"large"``.
        device: optional device to move the assembled model to.
        load_da_pretrained: if ``True``, fetch and load the DA weights.
        da_checkpoint_path: optional path to a local copy of the matching
            ``depth_anything_v2_vit{b,l}.pth``; if ``None`` and
            ``load_da_pretrained`` is ``True``, download via
            ``huggingface_hub``.
    """
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}; must be one of {list(VARIANTS)}."
        )
    cfg = VARIANTS[variant]

    model = JepaDepthAnything(
        vjepa_encoder=vjepa_encoder,
        da_encoder=cfg["da_encoder"],
        features=cfg["da_features"],
        out_channels=cfg["da_out_channels"],
        vjepa_dim=cfg["vjepa_dim"],
        vjepa_patch_size=cfg["vjepa_patch_size"],
    )

    if load_da_pretrained:
        if da_checkpoint_path is None:
            from huggingface_hub import hf_hub_download

            da_checkpoint_path = hf_hub_download(
                repo_id=cfg["da_repo_id"],
                filename=cfg["da_filename"],
            )

        state = torch.load(da_checkpoint_path, map_location="cpu")
        prefix = "depth_head."
        filtered = {
            k[len(prefix):]: v
            for k, v in state.items()
            if k.startswith(prefix)
        }

        missing, unexpected = model.depth_head.load_state_dict(
            filtered, strict=False
        )
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys when loading DA depth_head weights: "
                f"{unexpected}"
            )
        if missing:
            raise RuntimeError(
                f"Missing keys when loading DA depth_head weights: {missing}"
            )

    if device is not None:
        model = model.to(device)

    return model
