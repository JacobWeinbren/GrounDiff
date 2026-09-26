"""Scene inference with no torch dependency (numpy + scipy only).

The network is supplied as a callable, so the same code runs with PyTorch
(training machine, CUDA/MPS/CPU) or ONNX Runtime (QGIS plugin):

    GrounDiff denoiser:  denoise(x [B, 1+C, T, T] float32, gamma [B] float32) -> [B, 2, T, T]
    ResDepth network:    forward(x [B, 1+C', T, T] float32) -> [B, 1, T, T]

`RuntimeSpec` carries everything else (channels, normalisation, schedule) and
is stored as JSON next to exported models.

PrioStitch (GrounDiff §3.3): `prior="global"` downsamples the whole scene to
one network tile, runs the model once to get a coarse DTM, upsamples it, and
starts every full-resolution tile's reverse process from it. Overlapping
tiles are blended with "min" (best RMSE in the paper's Table 7), "linear"
(best balance) or "mean". `prior="channel"` uses the lasground_new DTM
(before -> after mode) instead of the self-generated coarse prior.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .normalise import FILL_CHANNELS, NEAREST_CHANNELS, channel_transform, fill_nearest, tile_range


@dataclass
class RuntimeSpec:
    kind: str
    cond_channels: list
    gate_channel: str
    norm_channels: list
    prior_channel: str | None
    norm_mode: str
    norm_std: float | None
    min_range: float
    tile: int
    alpha: float
    T: int = 10
    alphas_bar: list = field(default_factory=list)
    coef_x0: list = field(default_factory=list)
    coef_xt: list = field(default_factory=list)
    posterior_var: list = field(default_factory=list)
    clip_x0: float | None = None
    fill_empty: str = "zero"

    def to_json(self, path: str | Path):
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> "RuntimeSpec":
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def from_config(cls, cfg) -> "RuntimeSpec":
        from .schedule import build_schedule
        d, dc = cfg.data, cfg.diffusion
        sch = build_schedule(dc.schedule, dc.T, dc.beta_start, dc.beta_end, dc.cosine_s)
        return cls(kind=cfg.model.kind, cond_channels=list(d.cond_channels), gate_channel=d.gate_channel,
                   norm_channels=list(d.norm_channels), prior_channel=d.prior_channel,
                   norm_mode=d.norm_mode, norm_std=d.norm_std, min_range=d.min_range, tile=d.tile,
                   alpha=d.alpha, T=dc.T, alphas_bar=sch.alphas_bar.tolist(), coef_x0=sch.coef_x0.tolist(),
                   coef_xt=sch.coef_xt.tolist(), posterior_var=sch.posterior_var.tolist(),
                   clip_x0=dc.clip_x0, fill_empty=d.fill_empty)

    @property
    def needed_channels(self) -> list:
        extra = [self.prior_channel] if self.prior_channel else []
        return sorted(set(self.cond_channels) | set(self.norm_channels) | {self.gate_channel} | set(extra))


# ----------------------------------------------------------------------------- tiles

def tile_norm(arrs: dict, spec: RuntimeSpec) -> tuple[float, float]:
    if spec.norm_mode == "mean_std":
        base = arrs[spec.norm_channels[0]]
        ok = np.isfinite(base)
        mean = float(base[ok].mean()) if ok.any() else 0.0
        std = float(spec.norm_std or 1.0)
        return mean - std, 2.0 * std
    ref = np.stack([arrs[n] for n in spec.norm_channels])
    return tile_range(ref, np.isfinite(ref).all(0), spec.min_range)


def prepare(arrs: dict, spec: RuntimeSpec, lo: float, scale: float) -> np.ndarray:
    def prep(name):
        a = arrs[name].astype(np.float64)
        if spec.fill_empty == "nearest" and name in FILL_CHANNELS:
            a = fill_nearest(a)
        x = channel_transform(name, a, lo, scale)
        return np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    return np.stack([prep(n) for n in spec.cond_channels])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def sample(denoise: Callable, cond: np.ndarray, spec: RuntimeSpec, init: str = "dsm_noise",
           prior: np.ndarray | None = None, t_start: int | None = None,
           rng: np.random.Generator | None = None, add_noise: bool = True):
    """numpy mirror of GrounDiff.sample (diffusion.py). cond [B, C, T, T]."""
    rng = rng or np.random.default_rng()
    ab = np.asarray(spec.alphas_bar, np.float64)
    T = spec.T
    t_start = T if t_start is None else int(t_start)
    gi = spec.cond_channels.index(spec.gate_channel)
    s = cond[:, gi:gi + 1].astype(np.float64)
    noise = rng.standard_normal(s.shape) if add_noise else np.zeros(s.shape)
    if init.startswith("prior"):
        if prior is None:
            raise ValueError(f"init={init!r} needs a prior")
        base = prior.astype(np.float64)
    elif init == "noise":
        base = np.zeros_like(s)
    else:
        base = s
    if t_start < T:
        g = math.sqrt(ab[t_start - 1]) * base + math.sqrt(1 - ab[t_start - 1]) * noise
    elif init in ("dsm", "prior"):
        g = base
    else:
        g = base + noise
    bsz = cond.shape[0]
    g0 = logit = None
    for t in range(t_start, 0, -1):
        x = np.concatenate([g, cond], axis=1).astype(np.float32)
        out = np.asarray(denoise(x, np.full(bsz, ab[t - 1], np.float32)), np.float64)
        r_hat, logit = out[:, 0:1], out[:, 1:2]
        p = _sigmoid(logit)
        g0 = p * s + (1 - p) * (s - r_hat)                          # Eq. 5
        if spec.clip_x0 is not None:
            g0 = np.clip(g0, -spec.clip_x0, spec.clip_x0)
        if t > 1:
            mean = spec.coef_x0[t - 1] * g0 + spec.coef_xt[t - 1] * g
            eps = rng.standard_normal(g.shape) if add_noise else np.zeros(g.shape)
            g = mean + math.sqrt(spec.posterior_var[t - 1]) * eps
    return g0.astype(np.float32), logit.astype(np.float32)


# ----------------------------------------------------------------------------- resampling

def _resize(a: np.ndarray, out_h: int, out_w: int, nearest: bool = False) -> np.ndarray:
    """No-data-aware resize (box-filtered when shrinking)."""
    from scipy.ndimage import map_coordinates, uniform_filter

    h, w = a.shape
    ok = np.isfinite(a)
    filled = np.where(ok, a, 0.0).astype(np.float64)
    wts = ok.astype(np.float64)
    fy, fx = h / out_h, w / out_w
    if not nearest and (fy > 1 or fx > 1):
        size = (max(1, int(round(fy))), max(1, int(round(fx))))
        filled = uniform_filter(filled, size=size, mode="nearest")
        wts = uniform_filter(wts, size=size, mode="nearest")
    rr = (np.arange(out_h) + 0.5) * fy - 0.5
    cc = (np.arange(out_w) + 0.5) * fx - 0.5
    R, C = np.meshgrid(rr, cc, indexing="ij")
    order = 0 if nearest else 1
    v = map_coordinates(filled, [R, C], order=order, mode="nearest")
    wv = map_coordinates(wts, [R, C], order=order, mode="nearest")
    out = np.full((out_h, out_w), np.nan)
    good = wv > (0.5 if nearest else 1e-6)
    out[good] = v[good] / wv[good]
    return out


def _d4(x: np.ndarray, k: int, flip: bool, inverse: bool = False) -> np.ndarray:
    """Dihedral transform of the last two axes."""
    if not inverse:
        x = np.rot90(x, k, axes=(-2, -1))
        return x[..., ::-1] if flip else x
    x = x[..., ::-1] if flip else x
    return np.rot90(x, -k, axes=(-2, -1))


# ----------------------------------------------------------------------------- scenes

def _starts(n: int, tile: int, stride: int) -> list:
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, stride))
    if s[-1] != n - tile:
        s.append(n - tile)
    return s


def _ramp_weights(tile: int, stride: int) -> np.ndarray:
    """Linear ramps over the overlap (as in ResDepth's predict_linear_blend)."""
    ov = tile - stride
    w1 = np.ones(tile)
    if ov > 0:
        ramp = np.linspace(0, 1, ov + 2)[1:-1]
        w1[:ov] = ramp
        w1[-ov:] = ramp[::-1]
    return np.outer(w1, w1)


def predict_scene(arrs: dict, spec: RuntimeSpec, net: Callable, *, stride: int | None = None,
                  blend: str = "min", prior: str = "auto", init: str | None = None,
                  t_start: int | None = None, n_samples: int = 1, tta: bool = False,
                  batch_size: int = 8, seed: int = 0, progress: Callable | None = None,
                  add_noise: bool = True) -> dict:
    """arrs: full-scene rasters in metres (NaN = no data), at least
    spec.needed_channels. Returns metre-space rasters:
        dtm, p_ground (GrounDiff: sigmoid(l), probability that the gate
        surface is already right), p_edit (1 - p_ground, when the gate is the
        lasground_new DTM), std (if n_samples > 1 or tta), dz_before
        (dtm - prior channel, if present), coverage."""
    missing = [c for c in spec.needed_channels if c not in arrs]
    if missing:
        raise KeyError(f"missing input rasters: {missing}")
    rng = np.random.default_rng(seed)
    H, W = arrs[spec.gate_channel].shape
    t = spec.tile
    stride = stride or t // 2
    if prior == "auto":
        prior = "channel" if spec.prior_channel else "global"
    is_diff = spec.kind == "groundiff"

    prior_full = None
    if is_diff and prior == "global":
        small = {n: _resize(arrs[n], t, t, nearest=n in NEAREST_CHANNELS) for n in spec.needed_channels}
        lo, sc = tile_norm(small, spec)
        g0, _ = sample(net, prepare(small, spec, lo, sc)[None], spec, init="dsm_noise", rng=rng,
                       add_noise=add_noise)
        coarse = (g0[0, 0].astype(np.float64) + 1) * 0.5 * sc + lo
        prior_full = _resize(coarse, H, W)
    elif prior == "channel" or not is_diff:
        if not spec.prior_channel:
            raise ValueError("prior='channel' needs a prior channel in the spec")
        prior_full = arrs[spec.prior_channel].astype(np.float64)
    if init is None:
        init = "prior" if (is_diff and prior_full is not None) else "dsm_noise"

    rows, cols = _starts(H, t, stride), _starts(W, t, stride)
    acc = np.full((H, W), np.inf) if blend == "min" else np.zeros((H, W))
    wsum = np.zeros((H, W))
    pg_acc, sd_acc, cnt = np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
    wt = _ramp_weights(t, stride) if blend == "linear" else np.ones((t, t))
    views = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]

    def window(a, r, c):
        out = np.full((t, t), np.nan)
        out[: min(t, H - r), : min(t, W - c)] = a[r:r + t, c:c + t]
        return out

    jobs = [(r, c) for r in rows for c in cols]
    for j0 in range(0, len(jobs), batch_size):
        chunk = jobs[j0:j0 + batch_size]
        tiles, los, scs, priors = [], [], [], []
        for r, c in chunk:
            ta = {n: window(arrs[n], r, c) for n in spec.needed_channels}
            lo, sc = tile_norm(ta, spec)
            tiles.append(prepare(ta, spec, lo, sc))
            los.append(lo)
            scs.append(sc)
            if prior_full is not None:
                pn = channel_transform("dtm_before", window(prior_full, r, c), lo, sc)
                priors.append(np.where(np.isfinite(pn), pn, 0.0).astype(np.float32)[None])
        cond = np.stack(tiles)
        pri = np.stack(priors) if priors else None
        preds, probs = [], []
        for k, f in views:
            c_v = np.ascontiguousarray(_d4(cond, k, f))
            p_v = np.ascontiguousarray(_d4(pri, k, f)) if pri is not None else None
            for _ in range(n_samples if is_diff else 1):
                if is_diff:
                    g0, logit = sample(net, c_v, spec, init=init, prior=p_v, t_start=t_start, rng=rng,
                                       add_noise=add_noise)
                    probs.append(_d4(_sigmoid(logit), k, f, inverse=True))
                else:
                    keep = [i for i, n in enumerate(spec.cond_channels) if n != spec.prior_channel]
                    g0 = np.asarray(net(np.concatenate([p_v, c_v[:, keep]], axis=1).astype(np.float32)))
                preds.append(_d4(g0, k, f, inverse=True))
        P = np.stack(preds)                                   # [S, B, 1, t, t]
        mean, std = P.mean(0), (P.std(0) if P.shape[0] > 1 else np.zeros_like(P[0]))
        pg = np.stack(probs).mean(0) if probs else None
        for i, (r, c) in enumerate(chunk):
            hh, ww = min(t, H - r), min(t, W - c)
            m = (mean[i, 0].astype(np.float64) + 1) * 0.5 * scs[i] + los[i]
            sd = std[i, 0].astype(np.float64) * 0.5 * scs[i]
            sl = (slice(r, r + hh), slice(c, c + ww))
            if blend == "min":
                acc[sl] = np.minimum(acc[sl], m[:hh, :ww])
            else:
                acc[sl] += m[:hh, :ww] * wt[:hh, :ww]
                wsum[sl] += wt[:hh, :ww]
            sd_acc[sl] += sd[:hh, :ww]
            cnt[sl] += 1
            if pg is not None:
                pg_acc[sl] += pg[i, 0][:hh, :ww]
        if progress:
            progress(min(j0 + batch_size, len(jobs)) / len(jobs))

    dtm = acc if blend == "min" else acc / np.maximum(wsum, 1e-12)
    has_data = np.zeros((H, W), bool)
    for n in spec.cond_channels:
        has_data |= np.isfinite(arrs[n]) if n not in NEAREST_CHANNELS else arrs[n] > 0
    out = {"dtm": np.where(has_data, dtm, np.nan).astype(np.float32),
           "coverage": cnt.astype(np.float32)}
    if is_diff:
        out["p_ground"] = np.where(has_data, pg_acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
        if spec.prior_channel and spec.gate_channel == spec.prior_channel:
            # gate on the lasground_new DTM: sigmoid(l) = "keep lasground_new here"
            out["p_edit"] = (1.0 - out["p_ground"]).astype(np.float32)
    if n_samples > 1 or tta:
        out["std"] = np.where(has_data, sd_acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
    if spec.prior_channel and spec.prior_channel in arrs:
        out["dz_before"] = (out["dtm"] - arrs[spec.prior_channel]).astype(np.float32)
    if prior_full is not None and prior == "global":
        out["prior"] = prior_full.astype(np.float32)
    return out
