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

from .normalise import (FILL_CHANNELS, HEIGHT_CHANNELS, NEAREST_CHANNELS, channel_transform, coverage_mask,
                        fill_nearest, tile_range)

INITS = ("dsm_noise", "noise", "dsm", "prior", "prior_noise", "dsm_q", "prior_q")   # = diffusion.INITS


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
    norm_quantile: float = 0.0
    coverage_close_m: float = 30.0
    init: str | None = None               # sampler init used in validation (None: prior if any, else dsm_noise)
    # how the training rasters were made (from the scenes' meta.json); batch and
    # the QGIS plugin use these as defaults so inference matches training
    gsd: float | None = None
    ground_classes: list | None = None
    before_ground_classes: list | None = None
    read_opts: dict | None = None

    def to_json(self, path: str | Path):
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> "RuntimeSpec":
        d = json.loads(Path(path).read_text())
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_config(cls, cfg, data_meta: dict | None = None) -> "RuntimeSpec":
        from .schedule import build_schedule
        d, dc = cfg.data, cfg.diffusion
        sch = build_schedule(dc.schedule, dc.T, dc.beta_start, dc.beta_end, dc.cosine_s)
        m = data_meta or {}
        return cls(kind=cfg.model.kind, cond_channels=list(d.cond_channels), gate_channel=d.gate_channel,
                   norm_channels=list(d.norm_channels), prior_channel=d.prior_channel,
                   norm_mode=d.norm_mode, norm_std=d.norm_std, min_range=d.min_range, tile=d.tile,
                   alpha=d.alpha, T=dc.T, alphas_bar=sch.alphas_bar.tolist(), coef_x0=sch.coef_x0.tolist(),
                   coef_xt=sch.coef_xt.tolist(), posterior_var=sch.posterior_var.tolist(),
                   clip_x0=dc.clip_x0, fill_empty=d.fill_empty, norm_quantile=d.norm_quantile,
                   coverage_close_m=d.coverage_close_m,
                   init=getattr(cfg.train, "val_init", None) if cfg.model.kind == "groundiff" else None,
                   gsd=m.get("gsd"), ground_classes=m.get("ground_classes"),
                   before_ground_classes=m.get("before_ground_classes"), read_opts=m.get("read_opts"))

    @property
    def needs_before(self) -> bool:
        """Needs the lasground_new classification (dtm_before / sem_* channels)."""
        return bool(self.prior_channel) or any(c.startswith("sem_") or c in ("dtm_before", "before_valid")
                                               for c in self.cond_channels)

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
    return tile_range(ref, None, spec.min_range, spec.norm_quantile)


def _prep(arrs: dict, spec: RuntimeSpec, name: str, lo: float, scale: float) -> np.ndarray:
    """Same rule as TileDataset._finish."""
    a = arrs[name].astype(np.float64)
    if spec.fill_empty == "nearest" and name in FILL_CHANNELS:
        a = fill_nearest(a)
    x = channel_transform(name, a, lo, scale)
    return np.where(np.isfinite(x), x, 0.0).astype(np.float32)


def prepare(arrs: dict, spec: RuntimeSpec, lo: float, scale: float) -> np.ndarray:
    return np.stack([_prep(arrs, spec, n, lo, scale) for n in spec.cond_channels])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _normal(rng, shape) -> np.ndarray:
    """rng: one Generator, or one per batch element (so a tile's noise does not
    depend on which batch or job it is processed in)."""
    if isinstance(rng, (list, tuple)):
        return np.stack([r.standard_normal(shape[1:]) for r in rng])
    return rng.standard_normal(shape)


def sample(denoise: Callable, cond: np.ndarray, spec: RuntimeSpec, init: str = "dsm_noise",
           prior: np.ndarray | None = None, t_start: int | None = None,
           rng=None, add_noise: bool = True):
    """numpy mirror of GrounDiff.sample (diffusion.py). cond [B, C, T, T]."""
    rng = rng if rng is not None else np.random.default_rng()
    if init not in INITS:
        raise ValueError(f"init must be one of {INITS}, got {init!r}")
    ab = np.asarray(spec.alphas_bar, np.float64)
    T = spec.T
    t_start = T if t_start is None else int(t_start)
    if not 1 <= t_start <= T:
        raise ValueError(f"t_start must be in [1, {T}], got {t_start}")
    gi = spec.cond_channels.index(spec.gate_channel)
    s = cond[:, gi:gi + 1].astype(np.float64)
    noise = _normal(rng, s.shape) if add_noise else np.zeros(s.shape)
    if init.startswith("prior"):
        if prior is None:
            raise ValueError(f"init={init!r} needs a prior")
        base = prior.astype(np.float64)
    elif init == "noise":
        base = np.zeros_like(s)
    else:
        base = s
    if t_start < T or init.endswith("_q"):
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
            eps = _normal(rng, g.shape) if add_noise else np.zeros(g.shape)
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


LATTICE_TOP = 1_300_000.0      # northing of the tile-lattice origin (top of the British National Grid)


def lattice_anchor(xmin: float, ymax: float, gsd: float) -> tuple[int, int]:
    """Global (row, col) of a grid's top-left cell on the lattice anchored at
    (0, LATTICE_TOP): pass as predict_scene(anchor=...) so rasters processed
    separately (batch jobs, raster mode) use the same network tiles and noise."""
    return int(round((LATTICE_TOP - ymax) / gsd)), int(round(xmin / gsd))


def _tile_starts(n: int, t: int, stride: int, anchor: int | None, lo: int, hi: int) -> list:
    """Tile starts along one axis. anchor=None: cover [0, n) with the last tile
    flush with the edge. Otherwise tiles sit on a global lattice (multiples of
    stride in global pixel coordinates; `anchor` is the global index of local
    pixel 0) and only those touching [lo, hi) are returned, so neighbouring
    batch jobs place identical tiles on the cells they share."""
    if anchor is None:
        return _starts(n, t, stride)
    k0 = -((-(anchor + lo - t + 1)) // stride)          # ceil
    k1 = (anchor + hi - 1) // stride
    return [k * stride - anchor for k in range(k0, k1 + 1)]


def _window(a: np.ndarray, r: int, c: int, t: int) -> np.ndarray:
    """t x t window at (r, c), NaN outside the array (r, c may be negative)."""
    H, W = a.shape
    out = np.full((t, t), np.nan)
    r0, c0, r1, c1 = max(r, 0), max(c, 0), min(r + t, H), min(c + t, W)
    if r1 > r0 and c1 > c0:
        out[r0 - r:r1 - r, c0 - c:c1 - c] = a[r0:r1, c0:c1]
    return out


def survey_mask(arrs: dict, spec: RuntimeSpec, gsd: float | None = None) -> np.ndarray:
    """Cells inside the LiDAR coverage, where outputs are written."""
    if "in_survey" in arrs:
        return np.nan_to_num(arrs["in_survey"]) > 0.5
    if "has_return" in arrs:
        base = np.nan_to_num(arrs["has_return"]) > 0
    else:                                  # raster inputs: any finite LiDAR surface
        names = [n for n in spec.needed_channels if n in HEIGHT_CHANNELS and n != "dtm_before"] or \
                [n for n in spec.needed_channels if n in HEIGHT_CHANNELS]
        base = np.zeros(arrs[spec.gate_channel].shape, bool)
        for n in names:
            base |= np.isfinite(arrs[n])
    return coverage_mask(base, gsd or spec.gsd or 1.0, spec.coverage_close_m)


def predict_scene(arrs: dict, spec: RuntimeSpec, net: Callable, *, stride: int | None = None,
                  blend: str = "linear", prior: str = "auto", init: str | None = None,
                  t_start: int | None = None, n_samples: int = 1, tta: bool = False,
                  batch_size: int = 8, seed: int = 0, progress: Callable | None = None,
                  add_noise: bool = True, anchor: tuple | None = None, region: tuple | None = None,
                  gsd: float | None = None) -> dict:
    """arrs: full-scene rasters in metres (NaN = no data), at least
    spec.needed_channels. Returns metre-space rasters:
        dtm, p_ground (GrounDiff: sigmoid(l), probability that the gate
        surface is already right), p_edit (1 - p_ground, when the gate is the
        lasground_new DTM), std (if n_samples > 1 or tta), dz_before
        (dtm - prior channel, if present), coverage.
    anchor=(row, col): global pixel index of arrs[0, 0]; tiles then lie on a
    global lattice and each tile's noise is seeded by its global position, so
    separately processed neighbouring blocks agree where they overlap.
    region=(r0, c0, h, w): only tiles touching this local window are run
    (with anchor; the rest of the output is NaN).
    progress(fraction) may raise to cancel."""
    missing = [c for c in spec.needed_channels if c not in arrs]
    if missing:
        raise KeyError(f"missing input rasters: {missing}")
    H, W = arrs[spec.gate_channel].shape
    t = spec.tile
    stride = stride or t // 2
    if not 1 <= stride <= t:
        raise ValueError(f"stride must be in [1, tile={t}], got {stride}")
    if blend not in ("min", "linear", "mean"):
        raise ValueError(f"blend must be min, linear or mean, got {blend!r}")
    if prior not in ("auto", "global", "channel", "none"):
        raise ValueError(f"prior must be auto, global, channel or none, got {prior!r}")
    rng = np.random.default_rng(seed)
    if prior == "auto":
        # PrioStitch's global prior depends on the extent processed, so it cannot be seamless
        # across separately processed blocks (anchor): there, models without a prior channel
        # start from the DSM as in the paper's default (with Palette's cosine schedule the
        # prior barely reaches the network anyway)
        prior = "channel" if spec.prior_channel else ("none" if anchor is not None else "global")
    is_diff = spec.kind == "groundiff"

    prior_full = None
    if is_diff and prior == "global":
        # keep the aspect ratio: the scene's long side becomes one tile
        f = t / max(H, W)
        sh, sw = max(1, round(H * f)), max(1, round(W * f))
        small = {n: _window(_resize(arrs[n], sh, sw, nearest=n in NEAREST_CHANNELS), 0, 0, t)
                 for n in spec.needed_channels}
        lo, sc = tile_norm(small, spec)
        g0, _ = sample(net, prepare(small, spec, lo, sc)[None], spec, init="dsm_noise", rng=rng,
                       add_noise=add_noise)
        coarse = (g0[0, 0, :sh, :sw].astype(np.float64) + 1) * 0.5 * sc + lo
        prior_full = _resize(coarse, H, W)
    elif prior == "channel" or not is_diff:
        if not spec.prior_channel:
            raise ValueError("prior='channel' needs a prior channel in the spec")
        prior_full = arrs[spec.prior_channel].astype(np.float64)   # filled per tile below, as in training
    if init is None:
        init = "prior" if (is_diff and prior_full is not None) else "dsm_noise"
    if is_diff and init not in INITS:
        raise ValueError(f"init must be one of {INITS}, got {init!r}")

    # Reference DTM for models that do not take lasground_new as an input (DSM -> DTM):
    # when the tiles carry lasground_new classes, their ground TIN (dtm_before) is
    # compared with every sample to give dz_before and a sampled edit probability.
    edit_gated = bool(spec.prior_channel) and spec.gate_channel == spec.prior_channel
    ref = None
    if is_diff and not edit_gated and "dtm_before" in arrs:
        ref = arrs["dtm_before"].astype(np.float64)
    pe_acc = np.zeros((H, W)) if ref is not None else None

    ar, ac = anchor if anchor is not None else (None, None)
    r0g, c0g, hg, wg = region if region is not None else (0, 0, H, W)
    rows = _tile_starts(H, t, stride, ar, r0g, r0g + hg)
    cols = _tile_starts(W, t, stride, ac, c0g, c0g + wg)
    acc = np.full((H, W), np.inf) if blend == "min" else np.zeros((H, W))
    wsum = np.zeros((H, W))
    pg_acc, sd_acc, cnt = np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
    wt = _ramp_weights(t, stride) if blend == "linear" else np.ones((t, t))
    views = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]

    window = lambda a, r, c: _window(a, r, c, t)

    jobs = [(r, c) for r in rows for c in cols]
    # progress after every network call (a batch is views x samples x steps calls: minutes on a CPU)
    steps = (spec.T if t_start is None else int(t_start)) if is_diff else 1
    per_batch = len(views) * (n_samples if is_diff else 1) * steps
    n_batches = max(1, -(-len(jobs) // batch_size))
    calls = [0]
    net_main = net

    def net(*a, _f=net_main):
        out = _f(*a)
        calls[0] += 1
        if progress:
            progress(min(calls[0] / (n_batches * per_batch), 1.0))
        return out

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
                pw = window(prior_full, r, c)
                if spec.fill_empty == "nearest" and prior == "channel":
                    pw = fill_nearest(pw)                  # per tile, as dataset._finish does
                pn = channel_transform("dtm_before", pw, lo, sc)
                priors.append(np.where(np.isfinite(pn), pn, 0.0).astype(np.float32)[None])
        cond = np.stack(tiles)
        pri = np.stack(priors) if priors else None
        if anchor is not None:      # noise depends only on the tile's global position
            tile_rngs = [np.random.default_rng([seed, r + ar + 2 ** 30, c + ac + 2 ** 30]) for r, c in chunk]
        else:
            tile_rngs = rng
        preds, probs = [], []
        for k, f in views:
            c_v = np.ascontiguousarray(_d4(cond, k, f))
            p_v = np.ascontiguousarray(_d4(pri, k, f)) if pri is not None else None
            for _ in range(n_samples if is_diff else 1):
                if is_diff:
                    g0, logit = sample(net, c_v, spec, init=init, prior=p_v, t_start=t_start, rng=tile_rngs,
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
            if pe_acc is not None:
                ra_, ca_, rb_, cb_ = max(r, 0), max(c, 0), min(r + t, H), min(c + t, W)
                if rb_ > ra_ and cb_ > ca_:
                    tl_ = (slice(ra_ - r, rb_ - r), slice(ca_ - c, cb_ - c))
                    bw = ref[ra_:rb_, ca_:cb_]
                    ms = (P[:, i, 0][(slice(None),) + tl_].astype(np.float64) + 1) * 0.5 * scs[i] + los[i]
                    pe_acc[ra_:rb_, ca_:cb_] += (np.abs(ms - bw) > spec.alpha).mean(0)
            # part of the tile inside the array
            ra, ca, rb, cb = max(r, 0), max(c, 0), min(r + t, H), min(c + t, W)
            if rb <= ra or cb <= ca:
                continue
            tl = (slice(ra - r, rb - r), slice(ca - c, cb - c))
            m = ((mean[i, 0].astype(np.float64) + 1) * 0.5 * scs[i] + los[i])[tl]
            sd = (std[i, 0].astype(np.float64) * 0.5 * scs[i])[tl]
            sl = (slice(ra, rb), slice(ca, cb))
            if blend == "min":
                acc[sl] = np.minimum(acc[sl], m)
            else:
                acc[sl] += m * wt[tl]
                wsum[sl] += wt[tl]
            sd_acc[sl] += sd
            cnt[sl] += 1
            if pg is not None:
                pg_acc[sl] += pg[i, 0][tl]
        if progress:
            progress(min(j0 // batch_size + 1, n_batches) / n_batches)

    dtm = acc if blend == "min" else acc / np.maximum(wsum, 1e-12)
    has_data = survey_mask(arrs, spec, gsd) & (cnt > 0)
    out = {"dtm": np.where(has_data, dtm, np.nan).astype(np.float32),
           "coverage": cnt.astype(np.float32)}
    if is_diff:
        out["p_ground"] = np.where(has_data, pg_acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
        if spec.prior_channel and spec.gate_channel == spec.prior_channel:
            # gate on the lasground_new DTM: sigmoid(l) = "keep lasground_new here"
            out["p_edit"] = (1.0 - out["p_ground"]).astype(np.float32)
    if n_samples > 1 or tta:
        out["std"] = np.where(has_data, sd_acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
    if pe_acc is not None:
        # share of samples (x TTA views) whose DTM differs from lasground_new's by more than alpha
        ok = has_data & np.isfinite(ref)
        out["p_edit"] = np.where(ok, pe_acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
        out["dz_before"] = np.where(ok, out["dtm"] - ref, np.nan).astype(np.float32)
    if spec.prior_channel and spec.prior_channel in arrs:
        # only where lasground_new has a DTM of its own (not the filled prior)
        out["dz_before"] = (out["dtm"] - arrs[spec.prior_channel]).astype(np.float32)
        if "p_edit" in out:
            out["p_edit"] = np.where(np.isfinite(arrs[spec.prior_channel]), out["p_edit"], np.nan
                                     ).astype(np.float32)
    if prior_full is not None and prior == "global":
        out["prior"] = prior_full.astype(np.float32)
    return out
