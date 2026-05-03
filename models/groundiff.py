"""GrounDiff top-level wrapper.

Bundles UNet + diffusion + loss into one class with `training_step`
and `infer` methods. Keeps state minimal so checkpoints are simple.
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn

from .unet import GrounDiffUNet
from .diffusion import GrounDiffDiffusion, gating
from .losses import groundiff_loss


class GrounDiff(nn.Module):
    """Wrapper combining UNet + diffusion. The diffusion process holds
    the noise schedule but no parameters; only the UNet is trained."""

    def __init__(self, *, unet_kwargs: dict | None = None,
                 diffusion_kwargs: dict | None = None,
                 loss_kwargs: dict | None = None):
        super().__init__()
        unet_kwargs = unet_kwargs or {}
        diffusion_kwargs = diffusion_kwargs or {}
        # `alpha_metres` is a dataset-side config (used in __getitem__
        # to compute M_α in metres). Strip it out before passing to the
        # loss — the loss only knows about precomputed m_alpha.
        loss_kwargs = dict(loss_kwargs or {})
        loss_kwargs.pop('alpha_metres', None)
        self.loss_kwargs = loss_kwargs

        self.unet = GrounDiffUNet(**unet_kwargs)
        self.diffusion = GrounDiffDiffusion(**diffusion_kwargs)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.unet.parameters())

    def training_step(self, dsm: torch.Tensor, dtm: torch.Tensor,
                      valid: torch.Tensor,
                      m_alpha: torch.Tensor | None = None,
                      dsm_min: torch.Tensor | None = None):
        """One training pass.

        Args:
            dsm:     [B, 1, H, W] DSM (normalised to [-1, 1])
            dtm:     [B, 1, H, W] target DTM (normalised)
            valid:   [B, H, W]    valid pixel mask in {0, 1}
            m_alpha: [B, 1, H, W] precomputed M_α mask
            dsm_min: [B, 1, H, W] optional min-z DSM channel. When
                     supplied AND the UNet was constructed with
                     in_channel=3, it's prepended as 2nd condition.

        Returns:
            (total_loss, metrics_dict)
        """
        device = dsm.device
        bs = dsm.shape[0]

        # Sample γ_t per example, then forward-diffuse the GT DTM
        gammas = self.diffusion.sample_gammas(bs, device)
        noise = torch.randn_like(dtm)
        g_t = self.diffusion.q_sample(dtm, gammas, noise=noise)

        # Predict (r̂, ℓ) and gate. Channel order matches paper §3.2:
        # "The inputs are concatenated channel-wise as [g_t, s]".
        # When DSM_min is enabled (our addition), it follows DSM:
        # [g_t, s, s_min].
        if dsm_min is not None:
            x = torch.cat([g_t, dsm, dsm_min], dim=1)   # [B, 3, H, W]
        else:
            x = torch.cat([g_t, dsm], dim=1)            # [B, 2, H, W]
        out = self.unet(x, gammas)                      # [B, 2, H, W]
        r_hat, logit = out[:, 0:1], out[:, 1:2]
        g_hat = gating(r_hat, logit, dsm)

        loss, metrics = groundiff_loss(
            g_hat, dtm, logit, dsm, valid=valid, m_alpha=m_alpha,
            **self.loss_kwargs)
        return loss, metrics

    @torch.no_grad()
    def infer(self, dsm: torch.Tensor,
              init: str = 'noisy_dsm',
              prior_dtm: torch.Tensor | None = None,
              dsm_min: torch.Tensor | None = None,
              return_logit: bool = False) -> torch.Tensor:
        """Reverse diffusion to produce the predicted DTM.

        Args:
            dsm: [B, 1, H, W] normalised DSM
            init: 'noisy_dsm' (paper default) | 'gaussian' | 'priostitch'
            prior_dtm: required when init='priostitch'
            dsm_min: optional [B, 1, H, W] min-z DSM channel
            return_logit: also return final-step logit ℓ

        Returns:
            [B, 1, H, W] predicted DTM in normalised [-1, 1] frame.
            If return_logit, returns (dtm, logit) tuple.
        """
        if init == 'priostitch':
            assert prior_dtm is not None, \
                "init='priostitch' requires prior_dtm"
            return self.diffusion.sample_from_prior(
                self.unet, dsm, prior_dtm,
                return_logit=return_logit, dsm_min=dsm_min)
        return self.diffusion.sample(self.unet, dsm, init=init,
                                      return_logit=return_logit,
                                      dsm_min=dsm_min)

    # ---- Checkpoint I/O ----------------------------------------------

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(unet=self.unet.state_dict()), path)

    def load(self, path, *, strict: bool = True, map_location=None):
        d = torch.load(path, map_location=map_location, weights_only=True)
        if 'unet' in d:
            self.unet.load_state_dict(d['unet'], strict=strict)
        else:
            # Tolerate raw state dicts saved by older runs
            self.unet.load_state_dict(d, strict=strict)
        return self
