"""GrounDiff diffusion process (paper §3.2 Eq.1-10).

Forward (Eq.2):
    g_t = √ᾱ_t · g_0 + √(1-ᾱ_t) · ε,   ε ~ N(0, I)

Denoiser (Eq.4):
    (r̂, ℓ) = D_θ(g_t, s, t),   in_channel=2 [g_t, s], out_channel=2 [r̂, ℓ]

Gating (Eq.5):
    G(r̂, ℓ, s) = σ(ℓ) ⊙ s + (1 - σ(ℓ)) ⊙ (s - r̂)

Reverse (Eq.7-10):
    ĝ_t   = G(D_θ(g_t, s, t), s)               -- predicted clean DTM
    µ_θ   = (β_t · √ᾱ_{t-1} · ĝ_t + (1-ᾱ_{t-1}) · √α_t · g_t) / (1-ᾱ_t)
    σ_t²  = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)
    g_{t-1} = µ_θ + σ_t · ε

Inference start (paper §3.2):
    g_T ~ N(s, I)   -- noisy DSM, NOT pure Gaussian noise.

We use Palette's γ-encoding convention (γ_t = ᾱ_t) since the UNet
embeds γ rather than t directly. This is mathematically equivalent
to t-encoding for fixed schedule, but lets the same denoiser be
used at any noise level smoothly.

Schedule (paper §7.3):
    T = 10 by default, cosine schedule β ∈ [1e-4, 2e-2]
"""
from __future__ import annotations
import math
import numpy as np
import torch


def _make_betas(schedule: str, T: int,
                beta_start: float = 1e-4, beta_end: float = 2e-2,
                cosine_s: float = 8e-3):
    """Construct β schedule of length T.

    'linear': linear interpolation [beta_start, beta_end].
    'cosine': improved-DDPM cosine (Nichol & Dhariwal 2021), parametrised
              by cosine_s. Endpoints β_start/β_end ignored.

    Note paper §7.3 says 'cosine noise scheduler ranging from 0.0001 to
    0.02'. Two readings of that sentence are common; we ship both
    options. The 'cosine' default below uses Nichol & Dhariwal's exact
    formula, which is what Palette uses.
    """
    if schedule == 'linear':
        return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    if schedule == 'cosine':
        ts = torch.arange(T + 1, dtype=torch.float64) / T + cosine_s
        alphas = torch.cos(ts / (1 + cosine_s) * math.pi / 2).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        return betas.clamp(max=0.999)
    raise ValueError(f"Unknown schedule {schedule!r}")


def gating(r_hat: torch.Tensor, logit: torch.Tensor,
           dsm: torch.Tensor) -> torch.Tensor:
    """Paper Eq.5: G(r̂, ℓ, s) = σ(ℓ)⊙s + (1-σ(ℓ))⊙(s - r̂)."""
    sig = torch.sigmoid(logit)
    return sig * dsm + (1.0 - sig) * (dsm - r_hat)


class GrounDiffDiffusion:
    """Holds the noise schedule and provides forward / reverse helpers.

    Buffers (all length T, indexed 0..T-1 ↔ paper t=1..T):
        betas, alphas, alphas_bar, sqrt_alphas_bar, sqrt_one_minus_alphas_bar
        posterior_log_variance, posterior_mean_coef1, posterior_mean_coef2
    """

    def __init__(self, T: int = 10, schedule: str = 'cosine',
                 beta_start: float = 1e-4, beta_end: float = 2e-2,
                 cosine_s: float = 8e-3):
        betas = _make_betas(schedule, T, beta_start, beta_end, cosine_s)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat(
            [torch.ones(1, dtype=alphas_bar.dtype), alphas_bar[:-1]])

        self.T = int(T)
        self.schedule = schedule

        # Promote to fp32 once and stash; .to(device) on first batch use.
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alphas_bar = alphas_bar.float()
        self.alphas_bar_prev = alphas_bar_prev.float()
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)

        # Posterior variance is 0 at t=0; clamp the log for numerical safety.
        post_var = (self.betas
                    * (1.0 - self.alphas_bar_prev)
                    / (1.0 - self.alphas_bar).clamp(min=1e-20))
        self.posterior_log_variance = torch.log(post_var.clamp(min=1e-20))
        # Posterior mean coefs for q(g_{t-1} | g_t, g_0):
        #   µ = c1 · g_0 + c2 · g_t
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_bar_prev)
            / (1.0 - self.alphas_bar).clamp(min=1e-20)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_bar_prev) * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_bar).clamp(min=1e-20)
        )
        self._device = None

    def to(self, device):
        if self._device == device:
            return self
        for name in ('betas', 'alphas', 'alphas_bar', 'alphas_bar_prev',
                     'sqrt_alphas_bar', 'sqrt_one_minus_alphas_bar',
                     'posterior_log_variance',
                     'posterior_mean_coef1', 'posterior_mean_coef2'):
            setattr(self, name, getattr(self, name).to(device))
        self._device = device
        return self

    @staticmethod
    def _gather(coef: torch.Tensor, t: torch.Tensor,
                shape) -> torch.Tensor:
        """Gather schedule values at integer steps t and broadcast to
        the given shape (e.g. [B,1,1,1] for an image batch)."""
        out = coef.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(shape) - 1)))

    # ------- Training-time helpers -------------------------------------

    def sample_gammas(self, batch: int, device) -> torch.Tensor:
        """Palette-style continuous γ sampling.

        For each example, sample a step t uniformly in {1..T}, then
        sample γ uniformly in [ᾱ_t, ᾱ_{t-1}]. This gives the denoiser
        smooth coverage of γ ∈ (0, 1] rather than only T discrete points.
        """
        self.to(device)
        t = torch.randint(1, self.T + 1, (batch,), device=device)  # 1..T
        idx_t = (t - 1).long()
        idx_prev = (t - 2).clamp(min=0).long()
        gamma_lo = self.alphas_bar.gather(0, idx_t)         # ᾱ_t
        gamma_hi = torch.where(
            t > 1, self.alphas_bar.gather(0, idx_prev),
            torch.ones_like(gamma_lo))                       # ᾱ_{t-1}, ᾱ_0=1
        u = torch.rand(batch, device=device)
        return gamma_lo + (gamma_hi - gamma_lo) * u           # [B]

    def q_sample(self, g_0: torch.Tensor, gammas: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward Eq.2 with γ_t = ᾱ_t.

        g_t = √γ_t · g_0 + √(1-γ_t) · ε
        """
        if noise is None:
            noise = torch.randn_like(g_0)
        gammas = gammas.view(-1, 1, 1, 1).to(g_0)
        return gammas.sqrt() * g_0 + (1.0 - gammas).sqrt() * noise

    # ------- Inference -------------------------------------------------

    @torch.no_grad()
    def p_sample(self, model, g_t: torch.Tensor, dsm: torch.Tensor,
                 t: int, return_logit: bool = False,
                 dsm_min: torch.Tensor | None = None) -> torch.Tensor:
        """One reverse step from g_t to g_{t-1}.

        model(x, gamma) -> (r̂, ℓ) stacked along channel axis where
        x = cat([g_t, dsm, dsm_min?], dim=1) per paper §3.2.
        """
        device = g_t.device
        self.to(device)
        bs = g_t.shape[0]

        idx = torch.full((bs,), t - 1, device=device, dtype=torch.long)
        gamma = self.alphas_bar.gather(0, idx).view(bs, 1, 1, 1)

        if dsm_min is not None:
            x = torch.cat([g_t, dsm, dsm_min], dim=1)
        else:
            x = torch.cat([g_t, dsm], dim=1)
        out = model(x, gamma.view(bs))                    # [B, 2, H, W]
        r_hat, logit = out[:, 0:1], out[:, 1:2]
        g0_hat = gating(r_hat, logit, dsm).clamp(-1.0, 1.0)

        if t == 1:
            # Posterior variance = 0 at the final step; just return g_0
            return (g0_hat, logit) if return_logit else g0_hat

        c1 = self._gather(self.posterior_mean_coef1, idx, g_t.shape)
        c2 = self._gather(self.posterior_mean_coef2, idx, g_t.shape)
        mean = c1 * g0_hat + c2 * g_t
        log_var = self._gather(self.posterior_log_variance, idx, g_t.shape)
        noise = torch.randn_like(g_t)
        next_g = mean + (0.5 * log_var).exp() * noise
        return (next_g, logit) if return_logit else next_g

    @torch.no_grad()
    def sample(self, model, dsm: torch.Tensor,
               init: str = 'noisy_dsm',
               return_logit: bool = False,
               dsm_min: torch.Tensor | None = None) -> torch.Tensor:
        """Run full reverse process g_T → g_0, returning the final DTM."""
        device = dsm.device
        self.to(device)
        if init == 'noisy_dsm':
            gamma_T = self.alphas_bar[-1].view(1).repeat(dsm.shape[0])
            g_t = self.q_sample(dsm, gamma_T)
        elif init == 'gaussian':
            g_t = torch.randn_like(dsm)
        else:
            raise ValueError(f"unknown init {init!r}")

        last_logit = None
        for t in reversed(range(1, self.T + 1)):
            g_t, logit = self.p_sample(model, g_t, dsm, t,
                                        return_logit=True,
                                        dsm_min=dsm_min)
            last_logit = logit
        if return_logit:
            return g_t, last_logit
        return g_t

    @torch.no_grad()
    def sample_from_prior(self, model, dsm: torch.Tensor,
                          prior_dtm: torch.Tensor,
                          return_logit: bool = False,
                          dsm_min: torch.Tensor | None = None
                          ) -> torch.Tensor:
        """PrioStitch: start the reverse from `q_sample(prior_dtm, γ_T)`
        instead of N(s, I). `prior_dtm` is the upsampled global prior."""
        device = dsm.device
        self.to(device)
        gamma_T = self.alphas_bar[-1].view(1).repeat(dsm.shape[0])
        g_t = self.q_sample(prior_dtm, gamma_T)
        last_logit = None
        for t in reversed(range(1, self.T + 1)):
            g_t, logit = self.p_sample(model, g_t, dsm, t,
                                        return_logit=True,
                                        dsm_min=dsm_min)
            last_logit = logit
        if return_logit:
            return g_t, last_logit
        return g_t
