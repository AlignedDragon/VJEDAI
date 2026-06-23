"""Combined video depth loss for the video-native VJEDAI model.

This is the default training objective for ``vjedai`` on clips. It
stays entirely in **log-depth, per-image (per-frame) scale-invariant** space --
the same world as the leaderboard siRMSE metric and the original VJEDAI losses
in ``train_vjedai.py`` -- and adds a **temporal-consistency** term in
the spirit of Video-Depth-Anything's temporal gradient matching loss. Keeping a
single normalization lets the spatial term, the Gaussian-NLL uncertainty term,
and the temporal term compose cleanly. (For VDA's literal affine-invariant
recipe, see ``src/vda_loss.py``.)

For a clip of ``T`` frames the loss is::

    total = spatial(mode) + lambda_temporal * temporal

* ``spatial``  -- per-frame scale-invariant objective:
    - ``mode="si_mse"``: var of the zero-mean log residual (= siRMSE^2),
      averaged over frames. The uncertainty head receives no gradient.
    - ``mode="nll"``:    heteroscedastic Gaussian NLL on the zero-mean log
      residual using the predicted ``log_var`` (clamped), averaged over valid
      pixels. This is the term that trains the uncertainty head.
* ``temporal`` -- multi-temporal-scale matching of the predicted vs. ground-truth
  per-pixel change in log-depth across consecutive frames, masked to valid and
  temporally-stable pixels. Zero when ``T == 1`` (still images), so the loss
  degrades exactly to the original per-image objective.

All inputs are ``[B, T, 1, H, W]`` or ``[B, T, H, W]``. With ``T == 1`` the
numbers match the original ``scale_invariant_mse`` / ``scale_invariant_gaussian_nll``
applied per frame.
"""

import torch
import torch.nn as nn


def _to_bthw(t):
    """Accept [B,T,1,H,W] or [B,T,H,W] -> [B,T,H,W]."""
    if t.dim() == 5:
        if t.shape[2] != 1:
            raise ValueError(
                f"Expected a singleton channel at dim 2, got {tuple(t.shape)}."
            )
        return t[:, :, 0]
    if t.dim() == 4:
        return t
    raise ValueError(f"Expected [B,T,1,H,W] or [B,T,H,W], got {tuple(t.shape)}.")


def scale_invariant_rmse(pred, target, eps=1e-6):
    """Per-frame scale-invariant RMSE on log depths, averaged over all frames.

    The video generalisation of the leaderboard metric: scale is removed
    independently per frame (``b, t``), then siRMSE is averaged over the
    ``B * T`` frames. Accepts clip- or image-shaped tensors.
    """
    pred, target = _to_bthw(pred), _to_bthw(target)
    B, T = pred.shape[:2]
    scores = []
    for b in range(B):
        for t in range(T):
            p, g = pred[b, t], target[b, t]
            m = g > eps
            if not m.any():
                continue
            d = torch.log(torch.clamp(p[m], min=eps)) - torch.log(g[m])
            var = torch.mean(d ** 2) - torch.mean(d) ** 2
            scores.append(torch.sqrt(torch.clamp(var, min=0.0)))
    if not scores:
        return pred.new_tensor(0.0)
    return torch.stack(scores).mean()


class vjedai_video_depth_loss(nn.Module):
    """Spatial scale-invariant (si_mse | nll) + temporal-consistency loss.

    Args:
        mode: ``"nll"`` (default, trains the uncertainty head) or ``"si_mse"``.
        lambda_temporal: weight on the temporal-consistency term.
        temp_scales: number of temporal strides (1, 2, 4, ...) to match over.
        temp_decay: per-scale weight decay (coarser strides count less).
        diff_depth_th: only match temporal gradients where the *ground-truth*
            log-depth change is below this (focuses the term on temporally
            stable pixels, mirroring VDA's ``diff_depth_th`` gating).
        lv_min, lv_max: clamp range for the predicted log-variance (same
            rationale as ``train_vjedai.scale_invariant_gaussian_nll``:
            a tight ceiling stops the NLL collapsing to an "infinity wins"
            plateau before depths are fit).
        eps: positivity floor for logs / valid-depth threshold.
    """

    def __init__(
        self,
        mode="nll",
        lambda_temporal=1.0,
        temp_scales=4,
        temp_decay=0.5,
        diff_depth_th=0.05,
        lv_min=-7.0,
        lv_max=3.0,
        eps=1e-6,
    ):
        super().__init__()
        if mode not in ("nll", "si_mse"):
            raise ValueError(f"Unknown mode={mode!r}; expected 'nll' or 'si_mse'.")
        self.mode = mode
        self.lambda_temporal = lambda_temporal
        self.temp_scales = temp_scales
        self.temp_decay = temp_decay
        self.diff_depth_th = diff_depth_th
        self.lv_min = lv_min
        self.lv_max = lv_max
        self.eps = eps

    # -- spatial -------------------------------------------------------------

    def _spatial(self, logp, logt, log_var, valid):
        """Per-frame scale-invariant si_mse / NLL on zero-mean log residuals."""
        B, T = logp.shape[:2]
        eps = self.eps

        if self.mode == "si_mse":
            losses = []
            for b in range(B):
                for t in range(T):
                    m = valid[b, t]
                    if not m.any():
                        continue
                    d = logp[b, t][m] - logt[b, t][m]
                    losses.append(torch.mean(d ** 2) - torch.mean(d) ** 2)
            if not losses:
                return logp.sum() * 0.0
            return torch.stack(losses).mean()

        # mode == "nll": per-pixel mean over the whole clip batch.
        nll_sum = logp.sum() * 0.0
        n_valid = 0
        for b in range(B):
            for t in range(T):
                m = valid[b, t]
                if not m.any():
                    continue
                d = logp[b, t][m] - logt[b, t][m]
                residual = d - torch.mean(d)  # per-frame zero-mean
                lv = torch.clamp(log_var[b, t][m], min=self.lv_min, max=self.lv_max)
                inv_var = torch.exp(-lv)
                nll_sum = nll_sum + torch.sum(0.5 * (lv + residual ** 2 * inv_var))
                n_valid += int(m.sum())
        if n_valid == 0:
            return nll_sum
        return nll_sum / n_valid

    # -- temporal ------------------------------------------------------------

    def _temporal(self, logp, logt, valid):
        """Multi-scale matching of predicted vs GT temporal log-depth gradients."""
        T = logp.shape[1]
        total = logp.sum() * 0.0
        cnt = 0
        for scale in range(self.temp_scales):
            stride = 2 ** scale
            if stride >= T:
                break
            p = logp[:, ::stride]
            g = logt[:, ::stride]
            v = valid[:, ::stride]
            # temporal gradients between adjacent (strided) frames
            dp = p[:, 1:] - p[:, :-1]
            dg = g[:, 1:] - g[:, :-1]
            vmask = v[:, 1:] & v[:, :-1]
            # focus on temporally-stable pixels (small GT change)
            vmask = vmask & (dg.abs() < self.diff_depth_th)
            denom = vmask.sum()
            if denom > 0:
                term = (torch.abs(dp - dg) * vmask).sum() / denom
                total = total + term * (self.temp_decay ** scale)
                cnt += 1
        if cnt == 0:
            return logp.sum() * 0.0
        return total / cnt

    # -- forward -------------------------------------------------------------

    def forward(self, pred, target, log_var=None, mask=None):
        """Compute the combined loss.

        Args:
            pred: predicted depth ``[B,T,1,H,W]`` / ``[B,T,H,W]`` (non-negative).
            target: ground-truth depth, same shape.
            log_var: predicted Gaussian log-variance, same shape; required for
                ``mode="nll"``.
            mask: optional bool/float validity mask, same shape; defaults to
                ``target > eps``.

        Returns:
            ``{"total_loss", "spatial_loss", "temporal_loss"}`` scalar tensors.
        """
        pred = _to_bthw(pred)
        target = _to_bthw(target)
        if log_var is not None:
            log_var = _to_bthw(log_var)
        elif self.mode == "nll":
            raise ValueError("mode='nll' requires log_var.")

        if mask is None:
            valid = target > self.eps
        else:
            valid = _to_bthw(mask).bool()

        logp = torch.log(torch.clamp(pred, min=self.eps))
        logt = torch.log(torch.clamp(target, min=self.eps))

        spatial = self._spatial(logp, logt, log_var, valid)

        T = pred.shape[1]
        if T > 1 and self.lambda_temporal > 0:
            temporal = self._temporal(logp, logt, valid)
        else:
            temporal = spatial.new_tensor(0.0)

        total = spatial + self.lambda_temporal * temporal
        return {
            "total_loss": total,
            "spatial_loss": spatial,
            "temporal_loss": temporal,
        }
