"""GrounDiff diffusion process (paper §3.2, Eq. 1-10).

We port the reference implementation from the original Stage-1 image
GrounDiff (`stage1/models/diffusion.py`) one-to-one — same schedule,
same sampling, same γ-encoding. The only thing this module changes
relative to Stage 1 is the *backbone* it calls (PPD-style DiT instead
of the OpenAI U-Net), and the multi-channel conditioning input.

Forward (Eq. 2):
    g_t = √γ_t · g_0 + √(1 − γ_t) · ε,   ε ~ N(0, I)

Denoiser (Eq. 4):
    (r̂, ℓ) = D_θ(x_t, γ_t),   where
    x_t = concat([g_t, dsm_max, dsm_min], dim=1)  # 3 channels

Gating (Eq. 5, see `gating.py`):
    G(r̂, ℓ, s) = σ(ℓ) ⊙ s + (1 − σ(ℓ)) ⊙ (s − r̂)
    where s = dsm_max (the principal DSM channel).

Reverse step (Eq. 7-10): predict g_0 via gating, then sample
g_{t-1} ~ q(g_{t-1} | g_t, g_0_pred).

Inference init (paper §3.2):
    g_T ~ q_sample(s, γ_T)   "noisy DSM" init — much better than pure
                              Gaussian init since s is already close to g.

Schedule (paper §7.3): T=10 cosine β.
"""
from __future__ import annotations

import math

import torch


def _make_betas(schedule: str, T: int,
                beta_start: float = 1e-4, beta_end: float = 2e-2,
                cosine_s: float = 8e-3) -> torch.Tensor:
    """Build the β schedule. 'cosine' is the Nichol & Dhariwal 2021
    parameterisation (used by Palette, recommended in GrounDiff §7.3)."""
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    if schedule == "cosine":
        ts = torch.arange(T + 1, dtype=torch.float64) / T + cosine_s
        alphas = torch.cos(ts / (1 + cosine_s) * math.pi / 2).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        return betas.clamp(max=0.999)
    raise ValueError(f"unknown schedule {schedule!r}")


class GrounDiffDiffusion:
    """Holds the discrete noise schedule (T steps) and helpers.

    All schedule tensors are 1-D of length T, indexed 0..T-1 ↔ paper
    timesteps t = 1..T. We use Palette's γ_t = ᾱ_t convention so the
    DiT can be γ-conditioned (continuous γ) at inference for smooth
    sampling at any t — even though training samples γ from a small
    discrete set this gives better behaviour at boundaries.
    """

    def __init__(self, T: int = 10, schedule: str = "cosine",
                 beta_start: float = 1e-4, beta_end: float = 2e-2,
                 cosine_s: float = 8e-3):
        betas = _make_betas(schedule, T, beta_start, beta_end, cosine_s)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat(
            [torch.ones(1, dtype=alphas_bar.dtype), alphas_bar[:-1]])

        self.T = int(T)
        self.schedule = schedule

        # Promote to fp32 and stash. Materialised on the device on first
        # use via `.to(device)`.
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alphas_bar = alphas_bar.float()
        self.alphas_bar_prev = alphas_bar_prev.float()
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)

        # Posterior variance (paper Eq. 8): β_t · (1 − ᾱ_{t-1}) / (1 − ᾱ_t).
        post_var = (self.betas * (1.0 - self.alphas_bar_prev)
                     / (1.0 - self.alphas_bar).clamp(min=1e-20))
        self.posterior_log_variance = torch.log(post_var.clamp(min=1e-20))

        # Posterior mean coefficients for q(g_{t-1} | g_t, g_0):
        #   µ = c1 · g_0 + c2 · g_t
        # (paper Eq. 10 split form).
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_bar_prev)
            / (1.0 - self.alphas_bar).clamp(min=1e-20)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_bar_prev) * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_bar).clamp(min=1e-20)
        )
        self._device: torch.device | None = None

    def to(self, device):
        if self._device == device:
            return self
        for name in ("betas", "alphas", "alphas_bar", "alphas_bar_prev",
                     "sqrt_alphas_bar", "sqrt_one_minus_alphas_bar",
                     "posterior_log_variance",
                     "posterior_mean_coef1", "posterior_mean_coef2"):
            setattr(self, name, getattr(self, name).to(device))
        self._device = device
        return self

    @staticmethod
    def _gather(coef: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        """Gather schedule values at int steps t and broadcast to `shape`."""
        out = coef.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(shape) - 1)))

    # ------------------------------------------------------------------ #
    #  Training-time
    # ------------------------------------------------------------------ #

    def sample_gammas(self, batch: int, device) -> torch.Tensor:
        """Palette-style continuous γ sampling.

        Sample t ~ Uniform{1..T}, then γ ~ Uniform[ᾱ_t, ᾱ_{t-1}]. This
        gives the denoiser smooth coverage of γ ∈ (0, 1] rather than
        only T discrete points.
        """
        self.to(device)
        t = torch.randint(1, self.T + 1, (batch,), device=device)
        idx_t = (t - 1).long()
        idx_prev = (t - 2).clamp(min=0).long()
        gamma_lo = self.alphas_bar.gather(0, idx_t)
        gamma_hi = torch.where(
            t > 1, self.alphas_bar.gather(0, idx_prev),
            torch.ones_like(gamma_lo))
        u = torch.rand(batch, device=device)
        return gamma_lo + (gamma_hi - gamma_lo) * u

    def q_sample(self, g_0: torch.Tensor, gammas: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward Eq. 2: g_t = √γ · g_0 + √(1−γ) · ε."""
        if noise is None:
            noise = torch.randn_like(g_0)
        gammas = gammas.view(-1, 1, 1, 1).to(g_0)
        return gammas.sqrt() * g_0 + (1.0 - gammas).sqrt() * noise

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def p_sample(self, model, g_t: torch.Tensor,
                 dsm_max: torch.Tensor, dsm_min: torch.Tensor,
                 dsm_mask: torch.Tensor,
                 t: int, *, return_logit: bool = False):
        """One reverse step g_t → g_{t-1}.

        `model(x, gamma)` must return [B, 2, H, W] (r̂, ℓ).
        """
        from .gating import gating  # local import to avoid cycles

        device = g_t.device
        self.to(device)
        bs = g_t.shape[0]

        idx = torch.full((bs,), t - 1, device=device, dtype=torch.long)
        gamma = self.alphas_bar.gather(0, idx).view(bs, 1, 1, 1)

        # [g_t, dsm_max, dsm_min] — 3 channels matching official
        # GrounDiff defra.json with `use_min_dsm: true`. dsm_mask is
        # kept on the signature for backward compatibility but is no
        # longer used as a network input.
        x = torch.cat([g_t, dsm_max, dsm_min], dim=1)
        out = model(x, gamma.view(bs))                # [B, 2, H, W]
        r_hat, logit = out[:, 0:1], out[:, 1:2]
        # Use the principal DSM channel (max-z) as `s` in the gating fn.
        g0_hat = gating(r_hat, logit, dsm_max).clamp(-1.0, 1.0)

        if t == 1:
            return (g0_hat, logit) if return_logit else g0_hat

        c1 = self._gather(self.posterior_mean_coef1, idx, g_t.shape)
        c2 = self._gather(self.posterior_mean_coef2, idx, g_t.shape)
        mean = c1 * g0_hat + c2 * g_t
        log_var = self._gather(self.posterior_log_variance, idx, g_t.shape)
        noise = torch.randn_like(g_t)
        next_g = mean + (0.5 * log_var).exp() * noise
        return (next_g, logit) if return_logit else next_g

    @torch.no_grad()
    def sample(self, model, dsm_max: torch.Tensor, dsm_min: torch.Tensor,
               dsm_mask: torch.Tensor,
               *, init: str = "noisy_dsm", return_logit: bool = False):
        """Run g_T → g_0 and return the predicted DTM (and final logit)."""
        device = dsm_max.device
        self.to(device)
        if init == "noisy_dsm":
            gamma_T = self.alphas_bar[-1].view(1).repeat(dsm_max.shape[0])
            g_t = self.q_sample(dsm_max, gamma_T)
        elif init == "gaussian":
            g_t = torch.randn_like(dsm_max)
        else:
            raise ValueError(f"unknown init {init!r}")

        last_logit = None
        for t in reversed(range(1, self.T + 1)):
            g_t, logit = self.p_sample(
                model, g_t, dsm_max, dsm_min, dsm_mask, t,
                return_logit=True)
            last_logit = logit
        if return_logit:
            return g_t, last_logit
        return g_t

    @torch.no_grad()
    def sample_from_prior(self, model, dsm_max: torch.Tensor,
                          dsm_min: torch.Tensor,
                          dsm_mask: torch.Tensor, prior_dtm: torch.Tensor,
                          *, return_logit: bool = False):
        """PrioStitch: start the reverse from q_sample(prior_dtm, γ_T)
        instead of from N(s, I). `prior_dtm` is the upsampled global prior.
        """
        device = dsm_max.device
        self.to(device)
        gamma_T = self.alphas_bar[-1].view(1).repeat(dsm_max.shape[0])
        g_t = self.q_sample(prior_dtm, gamma_T)
        last_logit = None
        for t in reversed(range(1, self.T + 1)):
            g_t, logit = self.p_sample(
                model, g_t, dsm_max, dsm_min, dsm_mask, t,
                return_logit=True)
            last_logit = logit
        if return_logit:
            return g_t, last_logit
        return g_t
