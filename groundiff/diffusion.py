"""GrounDiff diffusion process (paper §3.2-3.3), torch side.

Notation follows the paper: s = DSM (the gate surface), g = DTM, r = s - g.
All tensors are in the per-tile normalised space ([-1, 1] for the reference
channels, see `normalise.py`). Shapes are [B, C, H, W].

Initialisation of the reverse process (`init`):
  dsm_noise    g_T ~ N(s, I)            paper §3.2, default
  noise        g_T ~ N(0, I)            paper Table 6 "Init: Noise"
  dsm          g_T = s                  paper Table 6 "Init: DSM"
  prior        g_T = prior              PrioStitch, paper §3.3 ("we directly
                                        provide this prior DTM as the initial
                                        state for the denoiser")
  prior_noise  g_T ~ N(prior, I)        PrioStitch analogue of dsm_noise
  dsm_q        g_T ~ q(g_T | s) = N(sqrt(abar_T) s, (1 - abar_T) I)
  prior_q      the same from the prior. Extensions, NOT in the paper: with
               schedules whose abar_T is far from 0 (cosine_range, linear)
               the paper's N(s, I) is off the training distribution for the
               first steps; these match it. Identical to *_noise in practice
               for Palette's cosine (abar_T = 2.4e-5).

`t_start` (extension, NOT in the paper): begin the reverse chain at an
intermediate step t_start < T from q(g_{t_start} | init surface). With
Palette's cosine schedule the last step is almost pure noise
(sqrt(abar_T) = 0.005), so a surface injected at t = T barely reaches the
network; starting lower keeps it. Exposed for ablation only.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .schedule import bridge_schedule, bridge_step, bridge_times, build_schedule

INITS = ("dsm_noise", "noise", "dsm", "prior", "prior_noise", "dsm_q", "prior_q")


def _randn(shape, device, generator: torch.Generator | None = None) -> torch.Tensor:
    """Normal noise on `device`; a generator on another device (e.g. a CPU
    generator for reproducible validation on CUDA/MPS) draws there first."""
    if generator is not None and generator.device.type != torch.device(device).type:
        return torch.randn(shape, generator=generator, device=generator.device).to(device)
    return torch.randn(shape, device=device, generator=generator)


def gating(r_hat: torch.Tensor, logit: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Eq. 5: G = sigmoid(l) * s + (1 - sigmoid(l)) * (s - r_hat)."""
    p = torch.sigmoid(logit)
    return p * s + (1.0 - p) * (s - r_hat)


@dataclass
class DiffusionConfig:
    T: int = 10
    schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    cosine_s: float = 8e-3
    # "continuous" follows Palette: t ~ U{1..T-1}, gamma ~ U(abar_{t+1}, abar_t),
    # i.e. gamma in [abar_T, abar_1] (the levels the sampler visits).
    # "discrete": gamma = abar_t exactly, t ~ U{1..T}.
    gamma_sampling: str = "continuous"
    # Optional clamp of the predicted clean DTM during sampling, in normalised
    # units (Palette clamps to [-1, 1]). None disables it.
    clip_x0: float | None = None
    # Single-step end-to-end model (Garcia et al., "Fine-Tuning Image-Conditional Diffusion
    # Models is Easier than You Think", WACV 2025): the timestep is fixed to T and the noise
    # replaced by its mean (zero), so the denoiser maps the conditioning straight to the DTM in
    # one pass; fine-tuned from a diffusion checkpoint with the same task losses. Deterministic.
    one_step: bool = False
    # "gaussian": GrounDiff's DDPM (above). "rdbm": Residual Diffusion Bridge Model (Wang et al., CVPR
    # 2026, github.com/MiliLab/RDBM): a bridge from the gate surface (DSM) to the DTM,
    # x_t = s + Theta_t (g - s) + Sigma_t eps, sampled from x_T = s in bridge_steps x_0-prediction
    # steps; the denoiser's noise level input is Theta_t (1 = clean, as abar for "gaussian").
    process: str = "gaussian"
    bridge_T: int = 100
    bridge_steps: int = 10
    bridge_lambda: float = 1e-4
    # the paper's residual-modulated noise: eps scaled by |g - s|, so no noise where the surface is
    # already ground (the released code has this line commented out; off by default to match it)
    bridge_residual_noise: bool = False


class GrounDiff(nn.Module):
    """Wraps a denoiser D(x, gamma) -> [r_hat, logit].

    The denoiser input is concat([g_t, cond]) where `cond` holds the
    conditioning channels in `cond_channels` order, and the gate surface s is
    `cond[:, gate_index]`.
    """

    def __init__(self, denoiser: nn.Module, cfg: DiffusionConfig, cond_channels: list[str],
                 gate_channel: str = "dsm_max"):
        super().__init__()
        if gate_channel not in cond_channels:
            raise ValueError(f"gate channel {gate_channel!r} not in cond channels {cond_channels}")
        self.denoiser = denoiser
        self.cfg = cfg
        self.cond_channels = list(cond_channels)
        self.gate_channel = gate_channel
        self.gate_index = self.cond_channels.index(gate_channel)
        sch = build_schedule(cfg.schedule, cfg.T, cfg.beta_start, cfg.beta_end, cfg.cosine_s)
        self.schedule = sch
        for name in ("alphas_bar", "alphas_bar_prev", "posterior_var", "coef_x0", "coef_xt"):
            self.register_buffer(name, torch.tensor(getattr(sch, name), dtype=torch.float32),
                                 persistent=False)
        if cfg.process not in ("gaussian", "rdbm"):
            raise ValueError(f"diffusion.process must be gaussian or rdbm, got {cfg.process!r}")
        self.bridge = cfg.process == "rdbm"
        if self.bridge:
            th, sg = bridge_schedule(cfg.bridge_T, cfg.bridge_lambda)
            self.bridge_np = (th, sg)
            self.register_buffer("bridge_theta", torch.tensor(th, dtype=torch.float32), persistent=False)
            self.register_buffer("bridge_sigma", torch.tensor(sg, dtype=torch.float32), persistent=False)

    @property
    def T(self) -> int:
        return self.cfg.T

    def gate_surface(self, cond: torch.Tensor) -> torch.Tensor:
        return cond[:, self.gate_index:self.gate_index + 1]

    def denoise(self, g_t: torch.Tensor, cond: torch.Tensor, gamma: torch.Tensor):
        """Eq. 3-5. Returns (g0_hat, r_hat, logit)."""
        out = self.denoiser(torch.cat([g_t, cond], dim=1), gamma)
        r_hat, logit = out[:, 0:1].float(), out[:, 1:2].float()
        # log Laplace scale of the label noise (model.aleatoric), in normalised units
        self.last_log_b = out[:, 2:3].float() if out.shape[1] > 2 else None
        return gating(r_hat, logit, self.gate_surface(cond).float()), r_hat, logit

    # ------------------------------------------------------------------ training

    def sample_gammas(self, batch: int, device, generator: torch.Generator | None = None):
        if self.cfg.gamma_sampling == "discrete" or self.T < 2:
            t = torch.randint(1, self.T + 1, (batch,), device=device, generator=generator)
            return self.alphas_bar[t - 1], t
        t = torch.randint(1, self.T, (batch,), device=device, generator=generator)
        lo, hi = self.alphas_bar[t], self.alphas_bar[t - 1]          # abar_{t+1}, abar_t
        u = torch.rand(batch, device=device, generator=generator)
        return lo + (hi - lo) * u, t + 1

    def q_sample(self, g0: torch.Tensor, gamma: torch.Tensor, noise: torch.Tensor | None = None):
        """Eq. 2."""
        if noise is None:
            noise = torch.randn_like(g0)
        gamma = gamma.view(-1, 1, 1, 1).to(g0.dtype)
        return gamma.sqrt() * g0 + (1.0 - gamma).sqrt() * noise

    def training_forward(self, g0: torch.Tensor, cond: torch.Tensor):
        if self.bridge:
            return self._bridge_training_forward(g0, cond)
        if self.cfg.one_step:
            b = g0.shape[0]
            gamma = self.alphas_bar[self.T - 1].expand(b)
            t = torch.full((b,), self.T, device=g0.device, dtype=torch.long)
            g_t = torch.zeros_like(g0)                  # the mean of q(g_T | g0) for abar_T -> 0
            g0_hat, r_hat, logit = self.denoise(g_t, cond, gamma)
            return {"g0_hat": g0_hat, "r_hat": r_hat, "logit": logit, "gamma": gamma, "t": t, "log_b": self.last_log_b}
        gamma, t = self.sample_gammas(g0.shape[0], g0.device)
        g_t = self.q_sample(g0, gamma)
        g0_hat, r_hat, logit = self.denoise(g_t, cond, gamma)
        return {"g0_hat": g0_hat, "r_hat": r_hat, "logit": logit, "gamma": gamma, "t": t, "log_b": self.last_log_b}

    def _bridge_training_forward(self, g0: torch.Tensor, cond: torch.Tensor):
        b, T = g0.shape[0], self.cfg.bridge_T
        s = self.gate_surface(cond).float()
        if self.cfg.one_step:
            t = torch.full((b,), T - 1, device=g0.device, dtype=torch.long)
            g_t = s                                           # x_{T-1} = mu exactly (Sigma = 0)
        else:
            t = torch.randint(0, T, (b,), device=g0.device)
            eps = torch.randn_like(g0)
            if self.cfg.bridge_residual_noise:
                eps = eps * (g0 - s).abs()
            th = self.bridge_theta[t].view(-1, 1, 1, 1)
            g_t = s + th * (g0 - s) + self.bridge_sigma[t].view(-1, 1, 1, 1) * eps
        gamma = self.bridge_theta[t]
        g0_hat, r_hat, logit = self.denoise(g_t, cond, gamma)
        return {"g0_hat": g0_hat, "r_hat": r_hat, "logit": logit, "gamma": gamma, "t": t, "log_b": self.last_log_b}

    @torch.no_grad()
    def _bridge_sample(self, cond: torch.Tensor, one_step: bool):
        """RDBM's deterministic x_0-prediction sampler from x_{T-1} = s."""
        s = self.gate_surface(cond).float()
        th, sg = self.bridge_np
        T, bsz = self.cfg.bridge_T, cond.shape[0]
        pairs = [(T - 1, -1)] if one_step else bridge_times(T, self.cfg.bridge_steps)
        g_t, g0_hat, logit = s, None, None
        for t, t_next in pairs:
            g0_hat, _, logit = self.denoise(g_t, cond, self.bridge_theta[t].expand(bsz))
            if self.cfg.clip_x0 is not None:
                g0_hat = g0_hat.clamp(-self.cfg.clip_x0, self.cfg.clip_x0)
            g_t = bridge_step(g_t, s, g0_hat, th, sg, t, t_next)
        return g0_hat, logit

    # ----------------------------------------------------------------- sampling

    def initial_state(self, cond: torch.Tensor, init: str, prior: torch.Tensor | None,
                      t_start: int, generator: torch.Generator | None = None,
                      add_noise: bool = True) -> torch.Tensor:
        if init not in INITS:
            raise ValueError(f"init must be one of {INITS}")
        s = self.gate_surface(cond).float()
        if init.startswith("prior"):
            if prior is None:
                raise ValueError(f"init={init!r} needs a prior")
            base = prior.float()
        elif init == "noise":
            base = torch.zeros_like(s)
        else:
            base = s
        noise = _randn(s.shape, s.device, generator)
        if not add_noise:
            noise = torch.zeros_like(noise)
        if t_start < self.T or init.endswith("_q"):
            # Extension: q(g_{t_start} | base) instead of the paper's g_T.
            return self.q_sample(base, self.alphas_bar[t_start - 1].expand(s.shape[0]), noise)
        if init in ("dsm", "prior"):
            return base
        return base + noise           # N(base, I); for init="noise" this is N(0, I)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, init: str = "dsm_noise", prior: torch.Tensor | None = None,
               t_start: int | None = None, generator: torch.Generator | None = None,
               add_noise: bool = True, one_step: bool | None = None):
        """Reverse process Eq. 6-10. Returns (g0, logit) of the final step.
        add_noise=False gives the deterministic mean path (for tests/analysis).
        one_step (default: the model's cfg.one_step): a single pass at t = T from zeros.
        Bridge models (process="rdbm") always start from the gate surface; init, prior and
        t_start do not apply and sampling is deterministic."""
        if self.bridge:
            return self._bridge_sample(cond, bool(self.cfg.one_step if one_step is None else one_step))
        if self.cfg.one_step if one_step is None else one_step:
            s = self.gate_surface(cond).float()
            g0_hat, _, logit = self.denoise(torch.zeros_like(s), cond, self.alphas_bar[self.T - 1].expand(s.shape[0]))
            if self.cfg.clip_x0 is not None:
                g0_hat = g0_hat.clamp(-self.cfg.clip_x0, self.cfg.clip_x0)
            return g0_hat, logit
        t_start = self.T if t_start is None else int(t_start)
        if not 1 <= t_start <= self.T:
            raise ValueError(f"t_start must be in [1, {self.T}]")
        g_t = self.initial_state(cond, init, prior, t_start, generator, add_noise)
        bsz = cond.shape[0]
        g0_hat = logit = None
        for t in range(t_start, 0, -1):
            gamma = self.alphas_bar[t - 1].expand(bsz)
            g0_hat, _, logit = self.denoise(g_t, cond, gamma)
            if self.cfg.clip_x0 is not None:
                g0_hat = g0_hat.clamp(-self.cfg.clip_x0, self.cfg.clip_x0)
            if t > 1:
                mean = self.coef_x0[t - 1] * g0_hat + self.coef_xt[t - 1] * g_t
                noise = _randn(g_t.shape, g_t.device, generator)
                if not add_noise:
                    noise = torch.zeros_like(noise)
                g_t = mean + self.posterior_var[t - 1].sqrt() * noise
        return g0_hat, logit
