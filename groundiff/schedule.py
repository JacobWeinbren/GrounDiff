"""Noise schedules and posterior coefficients (pure numpy).

Kept free of torch so the same numbers drive training (torch), inference
(torch) and the ONNX runtime used by the QGIS plugin (numpy).

Schedules
---------
cosine        Nichol & Dhariwal (2021) as implemented in Palette's
              `make_beta_schedule(..., 'cosine')`. Palette ignores
              beta_start/beta_end for this schedule, and so do we. With T=10,
              abar_T = 2.4e-5, i.e. the last step is almost pure noise.
cosine_range  An alternative reading of GrounDiff §7.3 ("cosine noise
              scheduler ranging from 0.0001 to 0.02"): betas follow a
              half-cosine ramp from beta_start to beta_end. With T=10,
              abar_T = 0.90, i.e. every step keeps most of the signal.
linear        betas evenly spaced from beta_start to beta_end.

The paper does not say which it used; `cosine` matches Palette, which the
first author recommended building on. Treat the choice as an ablation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def make_betas(schedule: str, T: int, beta_start: float = 1e-4, beta_end: float = 2e-2,
               cosine_s: float = 8e-3) -> np.ndarray:
    if T < 1:
        raise ValueError("T must be >= 1")
    if schedule == "cosine":
        ts = np.arange(T + 1, dtype=np.float64) / T + cosine_s
        alphas = np.cos(ts / (1 + cosine_s) * math.pi / 2) ** 2
        alphas = alphas / alphas[0]
        return np.minimum(1 - alphas[1:] / alphas[:-1], 0.999)
    if schedule == "cosine_range":
        if T == 1:
            return np.array([beta_start], dtype=np.float64)
        ramp = (1 - np.cos(np.linspace(0.0, math.pi, T))) / 2
        return beta_start + (beta_end - beta_start) * ramp
    if schedule == "linear":
        return np.linspace(beta_start, beta_end, T, dtype=np.float64)
    raise ValueError(f"unknown schedule {schedule!r}")


@dataclass(frozen=True)
class Schedule:
    """Arrays indexed 0..T-1 for paper timesteps t = 1..T."""
    betas: np.ndarray
    alphas: np.ndarray
    alphas_bar: np.ndarray
    alphas_bar_prev: np.ndarray
    posterior_var: np.ndarray       # beta_t (1 - abar_{t-1}) / (1 - abar_t)   (Eq. 9)
    coef_x0: np.ndarray             # beta_t sqrt(abar_{t-1}) / (1 - abar_t)   (Eq. 8)
    coef_xt: np.ndarray             # (1 - abar_{t-1}) sqrt(alpha_t) / (1 - abar_t)

    @property
    def T(self) -> int:
        return len(self.betas)

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in
                ("betas", "alphas", "alphas_bar", "alphas_bar_prev",
                 "posterior_var", "coef_x0", "coef_xt")}


def build_schedule(schedule: str = "cosine", T: int = 10, beta_start: float = 1e-4,
                   beta_end: float = 2e-2, cosine_s: float = 8e-3) -> Schedule:
    betas = make_betas(schedule, T, beta_start, beta_end, cosine_s)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    abar_prev = np.concatenate([[1.0], abar[:-1]])
    denom = np.maximum(1.0 - abar, 1e-20)
    return Schedule(
        betas=betas, alphas=alphas, alphas_bar=abar, alphas_bar_prev=abar_prev,
        posterior_var=betas * (1.0 - abar_prev) / denom,
        coef_x0=betas * np.sqrt(abar_prev) / denom,
        coef_xt=(1.0 - abar_prev) * np.sqrt(alphas) / denom,
    )
