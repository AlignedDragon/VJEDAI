"""Vjedai: VJEPA 2.1 encoder + Video-Depth-Anything temporal DPT head.

This is the **video-native** generalisation of the original image-only VJEDAI
model. VJEPA 2.1 (a video foundation model) replaces DepthAnythingV2's DINOv2
encoder, and the spatial DA DPT head is upgraded with Video-Depth-Anything's
four temporal ``TemporalModule`` blocks (cross-frame attention), so the model
produces temporally-consistent depth for a clip of ``T`` frames. A still image
is just the ``T == 1`` special case, so the original image pipeline keeps
working unchanged.

Two paired variants are supported via the ``variant`` argument of
``build_vjedai``:

* ``"large"`` (default): VJEPA 2.1 ViT-Large 384 (embed_dim 1024) +
  DA-ViT-L DPT head (in_channels 1024, features 256,
  out_channels [256, 512, 1024, 1024], intermediate indices [5, 11, 17, 23]
  -- VJEPA 2.1's hardcoded hierarchical layers for depth=24). The channel
  geometry is identical to Video-Depth-Anything-Large, so VDA's pretrained
  temporal head (``head.*`` incl. ``head.motion_modules.*``) loads key-for-key.
* ``"base"``: VJEPA 2.1 ViT-Base 384 (embed_dim 768) + DA-ViT-B DPT head
  (in_channels 768, features 128, out_channels [96, 192, 384, 768],
  intermediate indices [2, 5, 8, 11]). VDA never released a ViT-B video
  checkpoint, so ``"base"`` cannot use ``init_source="vda"``.

In both variants the VJEPA and DINOv2 embed dims line up exactly, so DA's
pretrained ``projects.*`` 1x1 convs see the channel count they were designed
for. The whole DPT head + the four motion modules are trainable; the VJEPA
encoder is frozen. A parallel Gaussian uncertainty head emits per-pixel
log-variance alongside the depth map (VJEDAI's distinguishing heteroscedastic
output), now sitting downstream of the temporal mixing.

Caller responsibility: load the VJEPA 2.1 encoder yourself, passing the matching
``out_layers`` (``[5, 11, 17, 23]`` for large, ``[2, 5, 8, 11]`` for base) so
its ``forward`` returns the four intermediate feature maps the DPT head
consumes. These indices must lie inside VJEPA 2.1's hard-coded
``hierarchical_layers`` for the given depth, otherwise the encoder's forward
raises ``ValueError: <i> is not in list``. The ``Depth-Anything-V2`` repo must
already be on ``sys.path`` before calling ``build_vjedai`` (this
mirrors the existing pipeline setup).

Video tokenisation (verified against vjepa2/ source by running the encoder):
VJEPA 2.1 uses a **temporal patch (tubelet) size of 2**, so a ``[B, 3, T_in, H, W]``
clip yields ``T_tok * (H/p) * (W/p)`` tokens where ``T_tok = max(1, T_in // 2)``
(measured: T_in 1->1, 2->1, 4->2, 8->4), laid out frame-major (all spatial
patches of temporal slice 0, then slice 1, ...). The encoder therefore
temporally downsamples by 2: a clip of ``T_in`` frames produces ``T_tok`` depth
slices. To keep a per-input-frame contract, the head runs at the native
``T_tok`` resolution (the motion modules mix over ``T_tok`` slices) and the final
depth/log-variance are **temporally interpolated back to** ``out_frames`` (default
``T_in``). The forward asserts ``T_tok == max(1, T_in // tubelet)`` so a change in
the encoder's tokeniser fails loudly rather than silently misaligning frames.

Example (large, video clip of T frames):
    vj_encoder, _ = torch.hub.load(
        str(VJEPA_ROOT), "vjepa2_1_vit_large_384", source="local",
        out_layers=[5, 11, 17, 23],
    )
    model = build_vjedai(vj_encoder, variant="large",
                                      init_source="vda", device=device)

    # clip: [B, 3, T, 384, 384] from vjepa_preprocessing(...)
    out = model(clip, output_size=(H, W))
    depth, log_var = out["depth"], out["log_var"]   # each [B, T, 1, H, W]
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vendored Video-Depth-Anything temporal block. Works both when this module is
# imported as ``vjedai`` (src/ on sys.path, the pipeline's setup)
# and as ``src.vjedai`` (package-relative).
try:
    from .motion_module import TemporalModule
except ImportError:  # pragma: no cover - import-style fallback
    from motion_module import TemporalModule


VARIANTS = {
    "base": {
        "vjepa_arch": "vjepa2_1_vit_base_384",
        "vjepa_out_layers": [2, 5, 8, 11],
        "vjepa_dim": 768,
        "vjepa_patch_size": 16,
        "vjepa_tubelet_size": 2,
        "da_encoder": "vitb",
        "da_features": 128,
        "da_out_channels": [96, 192, 384, 768],
        "da_repo_id": "depth-anything/Depth-Anything-V2-Base",
        "da_filename": "depth_anything_v2_vitb.pth",
        # VDA never released a ViT-B video checkpoint; init_source="vda" is
        # unavailable for "base" (use "da" + zero-init motion modules instead).
        "vda_repo_id": None,
        "vda_filename": None,
        "num_frames": 32,
        "pe": "ape",
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
        "vjepa_tubelet_size": 2,
        "da_encoder": "vitl",
        "da_features": 256,
        "da_out_channels": [256, 512, 1024, 1024],
        "da_repo_id": "depth-anything/Depth-Anything-V2-Large",
        "da_filename": "depth_anything_v2_vitl.pth",
        # Video-Depth-Anything-Large: identical features/out_channels, so its
        # pretrained DPTHeadTemporal weights (incl. motion modules) load 1:1.
        "vda_repo_id": "depth-anything/Video-Depth-Anything-Large",
        "vda_filename": "video_depth_anything_vitl.pth",
        "num_frames": 32,
        "pe": "ape",
    },
}

# Video-Depth-Anything's motion-module hyper-parameters (see
# dpt_temporal.DPTHeadTemporal.__init__). Kept identical so the VDA pretrained
# head.motion_modules.* weights -- including the temporal_max_len-sized
# positional-encoding buffer -- load without shape mismatches.
_MOTION_KWARGS = dict(
    num_attention_heads=8,
    num_transformer_block=1,
    num_attention_blocks=2,
    zero_initialize=True,
)


class Vjedai(nn.Module):
    """Depth model with a frozen VJEPA 2.1 encoder and a VDA temporal DPT head.

    Consumes a clip of ``T`` frames and outputs a depth map (mean) and a
    pixel-wise Gaussian log-variance for uncertainty, both per frame. The four
    temporal motion modules are spliced into the DPT fusion path at exactly the
    points Video-Depth-Anything uses them (post-projection ``layer_3`` and
    ``layer_4``, and fusion outputs ``path_4`` and ``path_3``). The VJEPA
    encoder is held in eval mode with grads disabled.

    Most callers should use ``build_vjedai(..., variant=...)``
    instead of constructing this class directly; the builder fills in the
    DA-specific args from a paired-variant config and loads pretrained weights.

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
        vjepa_tubelet_size: VJEPA temporal patch size (default 2). In clip mode a
            ``T_in``-frame clip resolves ``max(1, T_in // tubelet)`` depth states.
        num_frames: temporal window length the motion modules are sized for
            (``temporal_max_len``); must be >= the clip length used at runtime.
        pe: motion-module positional-encoding type, ``"ape"`` or ``"rope"``.
        encode_mode: how the encoder consumes the clip --
            ``"clip"`` (default) feeds the whole ``[B,3,T,H,W]`` clip through
            VJEPA's video path (tubelet temporal patching + spatiotemporal
            self-attention across frames), resolving ``T_tok = max(1, T//tubelet)``
            depth states which a **learned** temporal upsampler expands back to
            ``T``; ``"per_frame"`` feeds each frame through VJEPA's image path
            (tubelet 1) independently, giving ``T_tok == T`` and no temporal
            upsampling -- VJEPA acts as a per-frame backbone and all temporal
            modelling is left to the motion modules (the Video-Depth-Anything
            recipe). ``"clip"`` keeps VJEPA's cross-frame attention; ``"per_frame"``
            is the per-frame ablation.
    """

    def __init__(
        self,
        vjepa_encoder: nn.Module,
        da_encoder: str = "vitl",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        vjepa_dim: int = 1024,
        vjepa_patch_size: int = 16,
        vjepa_tubelet_size: int = 2,
        num_frames: int = 32,
        pe: str = "ape",
        encode_mode: str = "clip",
    ) -> None:
        super().__init__()

        from depth_anything_v2.dpt import DepthAnythingV2

        out_channels = list(out_channels)
        da_template = DepthAnythingV2(
            encoder=da_encoder,
            features=features,
            out_channels=out_channels,
        )
        self.depth_head = da_template.depth_head
        del da_template

        # Four temporal motion modules, mirroring DPTHeadTemporal: the first two
        # mix the two deepest projected feature maps (layer_3, layer_4), the
        # last two mix the two coarsest fusion outputs (path_4, path_3). Channel
        # widths must match those tensors for the VDA weights to load.
        self.motion_modules = nn.ModuleList([
            TemporalModule(in_channels=out_channels[2], temporal_max_len=num_frames,
                           pos_embedding_type=pe, **_MOTION_KWARGS),
            TemporalModule(in_channels=out_channels[3], temporal_max_len=num_frames,
                           pos_embedding_type=pe, **_MOTION_KWARGS),
            TemporalModule(in_channels=features, temporal_max_len=num_frames,
                           pos_embedding_type=pe, **_MOTION_KWARGS),
            TemporalModule(in_channels=features, temporal_max_len=num_frames,
                           pos_embedding_type=pe, **_MOTION_KWARGS),
        ])

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

        if encode_mode not in ("clip", "per_frame"):
            raise ValueError(
                f"Unknown encode_mode {encode_mode!r}; expected 'clip' or "
                f"'per_frame'."
            )
        self.da_encoder = da_encoder
        self.vjepa_dim = vjepa_dim
        self.vjepa_patch_size = vjepa_patch_size
        self.vjepa_tubelet_size = vjepa_tubelet_size
        self.num_frames = num_frames
        self.encode_mode = encode_mode

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

        # Learned temporal upsampler (clip mode only). VJEPA's video path
        # tubelets time, so the head produces T_tok = T_in // tubelet depth
        # states; this depthwise transposed conv along time *synthesises* the
        # T_in output frames from those states -- a learned replacement for the
        # earlier classical linear interpolation, so the whole temporal path is
        # the network. It runs on the shared feature (shared_channels wide) at
        # the head's native resolution, just before the two cheap output convs,
        # so the expensive decoder still runs only T_tok times. It is
        # bilinear-initialised (≈ linear interp at step 0) so the VDA-warm-started
        # head is undisturbed at the start of training, and is a no-op identity
        # in per_frame mode (T_tok == T_in) and for still images (T == 1).
        self.temporal_upsampler = nn.ConvTranspose1d(
            shared_channels,
            shared_channels,
            kernel_size=2 * vjepa_tubelet_size,
            stride=vjepa_tubelet_size,
            padding=vjepa_tubelet_size // 2,
            groups=shared_channels,
            bias=False,
        )
        self._init_temporal_upsampler_bilinear()

    def _init_temporal_upsampler_bilinear(self) -> None:
        """Init the temporal upsampler to a 1D bilinear (linear-interp) kernel.

        Each depthwise filter starts as the standard bilinear upsampling kernel
        for the tubelet factor, so at step 0 the learned upsampler reproduces
        (approximately) the classical linear interpolation it replaces -- then
        learns to improve on it without perturbing the warm-started head.
        """
        f = self.vjepa_tubelet_size
        k = 2 * f
        center = (k - 1) / 2.0
        og = torch.arange(k, dtype=torch.float32)
        kernel = 1.0 - torch.abs(og - center) / f  # triangular (bilinear) window
        with torch.no_grad():
            self.temporal_upsampler.weight.zero_()
            # weight shape (in=C, out_per_group=1, k) for groups=C depthwise
            self.temporal_upsampler.weight[:, 0, :] = kernel

    def train(self, mode: bool = True) -> "Vjedai":
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

    def _apply_motion(
        self, idx: int, x: torch.Tensor, B: int, T: int
    ) -> torch.Tensor:
        """Run motion module ``idx`` on a ``[B*T, C, h, w]`` feature stack.

        Reshapes the flattened frame batch to the ``[B, C, T, h, w]`` layout the
        TemporalModule expects (cross-attention over the ``T`` axis), then flattens
        back. The cached-hidden-state path (streaming inference) is unused here.
        """
        C, h, w = x.shape[1], x.shape[2], x.shape[3]
        x = x.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, h, w]
        x, _ = self.motion_modules[idx](x, None, None, None)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, h, w)
        return x

    def _temporal_upsample(
        self, x: torch.Tensor, B: int, t_tok: int, t_out: int
    ) -> torch.Tensor:
        """Learned temporal upsample of ``[B*t_tok, C, h, w]`` to ``[B*t_out, C, h, w]``.

        Runs the depthwise transposed-conv temporal upsampler per spatial
        location (folding ``h, w`` into the batch). The conv expands the time
        axis by ``tubelet`` (so ``t_tok -> tubelet*t_tok``); for the standard
        even clip length this is exactly ``t_out``. Any residual mismatch (odd
        clips) is closed with a final 1D linear resample.
        """
        if t_tok == t_out:
            return x
        C, h, w = x.shape[1], x.shape[2], x.shape[3]
        # [B*t_tok, C, h, w] -> [B, t_tok, C, h, w] -> [B*h*w, C, t_tok]
        x = x.unflatten(0, (B, t_tok)).permute(0, 3, 4, 2, 1).reshape(B * h * w, C, t_tok)
        x = self.temporal_upsampler(x)  # -> [B*h*w, C, tubelet*t_tok]
        if x.shape[-1] != t_out:
            x = F.interpolate(x, size=t_out, mode="linear", align_corners=True)
        # [B*h*w, C, t_out] -> [B, h, w, C, t_out] -> [B*t_out, C, h, w]
        x = x.reshape(B, h, w, C, t_out).permute(0, 4, 3, 1, 2).reshape(B * t_out, C, h, w)
        return x

    def forward(
        self,
        vjepa_input: torch.Tensor,
        output_size: Tuple[int, int],
        out_frames: Optional[int] = None,
    ) -> dict:
        """Run the full encoder + temporal DPT + dual-head pipeline.

        Args:
            vjepa_input: preprocessed VJEPA clip from ``vjepa_preprocessing``,
                shape ``[B, 3, T_in, H_in, W_in]`` (``T_in == 1`` for a still
                image). ``H_in`` and ``W_in`` must be multiples of
                ``vjepa_patch_size`` (typically 384).
            output_size: ``(H, W)`` target spatial size of the depth and
                log-variance maps.
            out_frames: number of output time steps (default ``T_in``). In clip
                mode the encoder resolves ``T_tok = max(1, T_in // tubelet)``
                states and the **learned** temporal upsampler synthesises the
                rest; in per_frame mode ``T_tok == T_in`` already.

        Returns:
            ``{"depth": [B, out_frames, 1, H, W], "log_var": [B, out_frames, 1, H, W]}``.
            ``depth`` is non-negative (DA-style ReLU); ``log_var`` is
            unconstrained (Gaussian log-variance).
        """
        if vjepa_input.dim() != 5:
            raise RuntimeError(
                f"Expected vjepa_input of shape [B, 3, T, H, W], got "
                f"{tuple(vjepa_input.shape)}. Use vjepa_preprocessing(...)."
            )
        B = vjepa_input.shape[0]
        T_in = vjepa_input.shape[2]
        H_in, W_in = vjepa_input.shape[-2], vjepa_input.shape[-1]
        patch_h = H_in // self.vjepa_patch_size
        patch_w = W_in // self.vjepa_patch_size
        n_spatial = patch_h * patch_w

        if self.encode_mode == "per_frame":
            # Encode each frame independently via VJEPA's image path (tubelet 1):
            # [B,3,T_in,H,W] -> [B*T_in,3,1,H,W]. Each single frame yields
            # n_spatial tokens; fold the frames back into a frame-major token
            # axis so the shared pyramid/motion code below is identical to clip
            # mode with T = T_in (no temporal downsampling, no upsampling).
            enc_in = vjepa_input.permute(0, 2, 1, 3, 4).reshape(
                B * T_in, 3, 1, H_in, W_in
            )
            raw = self._encode(enc_in)  # 4 x [B*T_in, n_spatial, dim]
            feats = [
                f.reshape(B, T_in, n_spatial, self.vjepa_dim).reshape(
                    B, T_in * n_spatial, self.vjepa_dim
                )
                for f in raw
            ]
            T = T_in
        else:
            # Clip mode: feed the whole clip through VJEPA's video path. VJEPA
            # 2.1 tubelets time by ``vjepa_tubelet_size``, so a T_in-frame clip
            # yields T_tok = max(1, T_in // tubelet) temporal slices (verified by
            # running the encoder). Validate the token count so a tokeniser
            # change fails loudly instead of misaligning frames.
            feats = self._encode(vjepa_input)
            n_tokens = feats[0].shape[1]
            if n_spatial == 0 or n_tokens % n_spatial != 0:
                raise RuntimeError(
                    f"VJEPA returned {n_tokens} tokens, not a multiple of the "
                    f"spatial grid {n_spatial} ({patch_h}x{patch_w}). Check the "
                    f"VJEPA 2.1 tokeniser / input spatial size vs vjepa_patch_size."
                )
            T = n_tokens // n_spatial  # T_tok: native temporal feature resolution
            expected_t_tok = max(1, T_in // self.vjepa_tubelet_size)
            if T != expected_t_tok:
                raise RuntimeError(
                    f"VJEPA returned {T} temporal token slices for a {T_in}-frame "
                    f"clip, but tubelet_size={self.vjepa_tubelet_size} predicts "
                    f"{expected_t_tok}. Reconcile vjepa_tubelet_size with the real "
                    f"VJEPA 2.1 tokeniser."
                )

        head = self.depth_head
        # Reshape each token sequence to per-frame spatial maps and project/
        # resize into the DPT feature pyramid (report Sec. 2.2). Frames are
        # flattened into the batch dim as [B*T, C, patch_h, patch_w].
        pyramid = []
        for i, tokens in enumerate(feats):
            if tokens.shape[1] != T * n_spatial:
                raise RuntimeError(
                    f"VJEPA feature {i} has {tokens.shape[1]} tokens but "
                    f"expected {T * n_spatial} (T={T} x {patch_h}x{patch_w})."
                )
            # [B, T*n_spatial, dim] -> [B, T, n_spatial, dim] -> [B*T, dim, ph, pw]
            x = tokens.reshape(B, T, n_spatial, self.vjepa_dim)
            x = x.permute(0, 1, 3, 2).reshape(B * T, self.vjepa_dim, patch_h, patch_w)
            x = head.projects[i](x)
            x = head.resize_layers[i](x)
            pyramid.append(x)

        layer_1, layer_2, layer_3, layer_4 = pyramid

        # Temporal mixing on the two deepest projected maps, before RefineNet.
        layer_3 = self._apply_motion(0, layer_3, B, T)
        layer_4 = self._apply_motion(1, layer_4, B, T)

        l1_rn = head.scratch.layer1_rn(layer_1)
        l2_rn = head.scratch.layer2_rn(layer_2)
        l3_rn = head.scratch.layer3_rn(layer_3)
        l4_rn = head.scratch.layer4_rn(layer_4)

        # Coarse-to-fine RefineNet fusion, with temporal mixing on path_4/path_3.
        p4 = head.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        p4 = self._apply_motion(2, p4, B, T)
        p3 = head.scratch.refinenet3(p4, l3_rn, size=l2_rn.shape[2:])
        p3 = self._apply_motion(3, p3, B, T)
        p2 = head.scratch.refinenet2(p3, l2_rn, size=l1_rn.shape[2:])
        p1 = head.scratch.refinenet1(p2, l1_rn)

        shared = head.scratch.output_conv1(p1)  # [B*T_tok, C, h, w]

        # Learned temporal upsampling: expand the native T_tok depth states to
        # the requested number of output frames (default T_in) on the small
        # head-resolution feature, BEFORE the spatial upsample + output convs --
        # so the expensive decoder ran only T_tok times while the two cheap
        # output convs run at t_out. A no-op when t_out == T_tok (per_frame mode
        # and still images).
        t_out = T_in if out_frames is None else out_frames
        shared = self._temporal_upsample(shared, B, T, t_out)

        shared = F.interpolate(
            shared,
            size=tuple(output_size),
            mode="bilinear",
            align_corners=True,
        )

        # Dual heads on the shared feature map: depth mean (mu) and the parallel
        # Gaussian log-variance (uncertainty), per frame ([B*t_out, 1, H, W]),
        # then unflattened back to [B, t_out, 1, H, W].
        mu = head.scratch.output_conv2(shared)
        log_var = self.uncertainty_head(shared)

        H_out, W_out = mu.shape[-2], mu.shape[-1]
        mu = mu.reshape(B, t_out, 1, H_out, W_out)
        log_var = log_var.reshape(B, t_out, 1, H_out, W_out)

        return {"depth": mu, "log_var": log_var}


def _load_da_depth_head(model: Vjedai, da_checkpoint_path: str) -> None:
    """Load DA-V2 image ``depth_head.*`` weights; leave motion modules zero-init."""
    state = torch.load(da_checkpoint_path, map_location="cpu")
    prefix = "depth_head."
    filtered = {
        k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
    }
    missing, unexpected = model.depth_head.load_state_dict(filtered, strict=False)
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading DA depth_head weights: {unexpected}"
        )
    if missing:
        raise RuntimeError(
            f"Missing keys when loading DA depth_head weights: {missing}"
        )


def _load_vda_temporal_head(
    model: Vjedai, vda_checkpoint_path: str
) -> None:
    """Warm-start the full temporal head from a Video-Depth-Anything checkpoint.

    The VDA checkpoint stores the encoder under ``pretrained.*`` (dropped -- we
    use VJEPA) and the temporal head under ``head.*``. The head keys split into
    ``head.motion_modules.*`` (-> ``motion_modules.*``) and everything else
    (``head.projects/resize_layers/scratch.* -> depth_head.*``). Both targets
    must load with no missing/unexpected keys, since VDA-Large's geometry is
    identical to VJEDAI-large.
    """
    state = torch.load(vda_checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    depth_head_sd, motion_sd = {}, {}
    for k, v in state.items():
        if k.startswith("head.motion_modules."):
            motion_sd[k[len("head.motion_modules."):]] = v
        elif k.startswith("head."):
            depth_head_sd[k[len("head."):]] = v
        # pretrained.* (DINOv2) intentionally ignored.

    if not motion_sd or not depth_head_sd:
        raise RuntimeError(
            "VDA checkpoint did not contain the expected 'head.*' / "
            "'head.motion_modules.*' keys; cannot warm-start the temporal "
            f"head from {vda_checkpoint_path}."
        )

    dh_missing, dh_unexpected = model.depth_head.load_state_dict(
        depth_head_sd, strict=False
    )
    mm_missing, mm_unexpected = model.motion_modules.load_state_dict(
        motion_sd, strict=False
    )
    problems = []
    if dh_missing:
        problems.append(f"depth_head missing={dh_missing}")
    if dh_unexpected:
        problems.append(f"depth_head unexpected={dh_unexpected}")
    if mm_missing:
        problems.append(f"motion_modules missing={mm_missing}")
    if mm_unexpected:
        problems.append(f"motion_modules unexpected={mm_unexpected}")
    if problems:
        raise RuntimeError(
            "VDA temporal-head warm-start key mismatch: " + "; ".join(problems)
        )


def build_vjedai(
    vjepa_encoder: nn.Module,
    variant: str = "large",
    device: Optional[torch.device] = None,
    init_source: str = "vda",
    num_frames: Optional[int] = None,
    pe: Optional[str] = None,
    encode_mode: str = "clip",
    da_checkpoint_path: Optional[str] = None,
    vda_checkpoint_path: Optional[str] = None,
    load_da_pretrained: Optional[bool] = None,
) -> Vjedai:
    """Construct a ``Vjedai`` and initialise its temporal head.

    The ``variant`` argument selects a matched VJEPA/DA pair (``"base"`` or
    ``"large"``); see ``VARIANTS`` for the exact configs. Every parameter in
    ``depth_head`` and ``motion_modules`` remains trainable; only the VJEPA
    encoder is kept frozen.

    Args:
        vjepa_encoder: VJEPA 2.1 encoder loaded with the matching
            ``out_layers`` (see module docstring) for the chosen variant.
        variant: ``"base"`` or ``"large"``.
        device: optional device to move the assembled model to.
        init_source: how to initialise the temporal head --
            ``"vda"`` warm-start the whole head (DPT + motion modules) from
            the Video-Depth-Anything checkpoint (large only);
            ``"da"`` load the DA-V2 image ``depth_head`` and zero-init the
            motion modules (VDA design: identity at start, learn temporal from
            scratch -- useful for the ``"base"`` variant);
            ``"none"`` leave everything at module init (used when trained
            weights are about to be loaded over the top).
        num_frames: motion-module temporal window (``temporal_max_len``);
            defaults to the variant's value (32). Must be >= runtime clip
            length, and == 32 to load VDA's positional-encoding buffer.
        pe: motion-module positional encoding (``"ape"``/``"rope"``); defaults
            to the variant's value (``"ape"``, required for VDA warm-start).
        encode_mode: ``"clip"`` (default, VJEPA video path + learned temporal
            upsampler) or ``"per_frame"`` (VJEPA image path per frame, no
            upsampling). See ``Vjedai`` for details.
        da_checkpoint_path: optional local path to ``depth_anything_v2_vit{b,l}.pth``
            (used when ``init_source="da"``); downloaded from HF if ``None``.
        vda_checkpoint_path: optional local path to
            ``video_depth_anything_vitl.pth`` (used when ``init_source="vda"``);
            downloaded from HF if ``None``.
        load_da_pretrained: deprecated legacy alias. ``True`` -> ``init_source=
            "da"``, ``False`` -> ``init_source="none"``. Overrides
            ``init_source`` when not ``None`` so old callers keep working.
    """
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}; must be one of {list(VARIANTS)}."
        )
    cfg = VARIANTS[variant]

    # Back-compat: the old boolean kwarg maps onto the new init_source enum.
    if load_da_pretrained is not None:
        init_source = "da" if load_da_pretrained else "none"
    if init_source not in ("vda", "da", "none"):
        raise ValueError(
            f"Unknown init_source {init_source!r}; expected 'vda', 'da' or 'none'."
        )

    num_frames = cfg["num_frames"] if num_frames is None else num_frames
    pe = cfg["pe"] if pe is None else pe

    model = Vjedai(
        vjepa_encoder=vjepa_encoder,
        da_encoder=cfg["da_encoder"],
        features=cfg["da_features"],
        out_channels=cfg["da_out_channels"],
        vjepa_dim=cfg["vjepa_dim"],
        vjepa_patch_size=cfg["vjepa_patch_size"],
        vjepa_tubelet_size=cfg["vjepa_tubelet_size"],
        num_frames=num_frames,
        pe=pe,
        encode_mode=encode_mode,
    )

    if init_source == "vda":
        if cfg["vda_filename"] is None:
            raise ValueError(
                f"init_source='vda' is not available for variant={variant!r} "
                f"(no released VDA video checkpoint). Use init_source='da'."
            )
        if pe != "ape" or num_frames != 32:
            raise ValueError(
                "VDA warm-start requires pe='ape' and num_frames=32 to match "
                f"the pretrained positional-encoding buffer; got pe={pe!r}, "
                f"num_frames={num_frames}."
            )
        if vda_checkpoint_path is None:
            from huggingface_hub import hf_hub_download

            vda_checkpoint_path = hf_hub_download(
                repo_id=cfg["vda_repo_id"], filename=cfg["vda_filename"]
            )
        _load_vda_temporal_head(model, vda_checkpoint_path)
    elif init_source == "da":
        if da_checkpoint_path is None:
            from huggingface_hub import hf_hub_download

            da_checkpoint_path = hf_hub_download(
                repo_id=cfg["da_repo_id"], filename=cfg["da_filename"]
            )
        _load_da_depth_head(model, da_checkpoint_path)
    # init_source == "none": leave depth_head + zero-init motion modules as-is.

    if device is not None:
        model = model.to(device)

    return model
