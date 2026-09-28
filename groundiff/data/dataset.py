"""Tile datasets over preprocessed scenes (see preprocess.py).

Each item is a dict of float32 tensors:
  cond    [C, T, T]  conditioning channels in `cond_channels` order (normalised)
  target  [1, T, T]  ground-truth DTM (normalised)
  m_alpha [1, T, T]  1 where |s - g| < alpha (GrounDiff Eq. 14 target), s = gate channel
  valid   [1, T, T]  loss mask
  prior   [1, T, T]  normalised lasground_new DTM (zeros if absent)
  lo, scale          per-tile normalisation (metres) to map back
Invalid pixels are 0 in every normalised channel (GrounDiff supplement §7.2).

Training augmentation follows GrounDiff supplement §7.1: each step with
probability 0.5 - k*90° rotation, ±5° jitter, multi-scale resize to
{256, 512, 1024} followed by a random 256 crop, horizontal flip, vertical
flip. Rotation, zoom and crop are done in one resampling (bilinear for
continuous rasters with no-data-aware weighting, nearest for masks and
labels); when there is no jitter and no zoom the tile is cut exactly.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from collections import OrderedDict

from ..normalise import (FILL_CHANNELS, NEAREST_CHANNELS, VIRTUAL_CHANNELS, base_channels, channel_transform,
                         context_base, context_pool, context_window, fill_nearest, is_context, tile_range,
                         virtual_channel)

# Open memmaps are cached per process with a cap: every open .npy holds a file
# descriptor, and macOS allows only 256 per process by default.
MAX_OPEN_ARRAYS = 96
_OPEN: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _open_array(path: Path) -> np.ndarray:
    key = str(path)
    a = _OPEN.get(key)
    if a is not None:
        _OPEN.move_to_end(key)
        return a
    a = np.load(path, mmap_mode="r")
    _OPEN[key] = a
    while len(_OPEN) > MAX_OPEN_ARRAYS:
        _OPEN.popitem(last=False)          # dropping the last reference closes the file
    return a



@dataclass
class DataConfig:
    root: str = "data/scenes"
    split_file: str | None = None
    cond_channels: list = field(default_factory=lambda: ["dsm_max", "dsm_min"])
    gate_channel: str = "dsm_max"
    norm_channels: list = field(default_factory=lambda: ["dsm_max", "dsm_min"])
    prior_channel: str | None = None        # e.g. "dtm_before"
    tile: int = 256
    alpha: float = 0.2                       # metres, M_alpha threshold (not given in the paper)
    min_range: float = 2.0                   # metres, see normalise.py
    norm_mode: str = "minmax"                # "minmax" (GrounDiff §7.2) or "mean_std" (ResDepth)
    norm_std: float | None = None            # metres, mean_std mode (estimated from training tiles if None)
    norm_quantile: float = 0.0               # >0: robust range from [q, 1-q] quantiles (stray noise returns)
    coverage_close_m: float = 30.0           # voids narrower than 2x this (and enclosed voids) count as covered
    loss_mask: str = "gt_and_dsm"            # or "gt": also learn to fill no-return cells
    m_alpha_mode: str = "residual"           # "residual": |s - g| < alpha (paper Eq. 14);
                                             # "top_class": highest return is ground (DSM gate only)
    fill_empty: str = "zero"                 # "zero" (paper §7.2) or "nearest": fill empty cells of
                                             # height inputs with the nearest value (has_return/density tell the net)
    augment: bool = True
    p_rot90: float = 0.5
    p_jitter: float = 0.5
    jitter_deg: float = 5.0
    p_multiscale: float = 0.5
    multiscale_sizes: tuple = (256, 512, 1024)
    p_hflip: float = 0.5
    p_vflip: float = 0.5
    samples_per_epoch: int = 20000
    min_valid_frac: float = 0.05
    val_stride: int | None = None
    include_suspect: bool = False            # also use scenes the preprocess quality gate flagged
    bridge_oversample: float = 0.0           # share of training tiles centred on a bridge (preprocess --osm)
    context_factor: int = 4                  # ctx_* channels: window this many times wider than the tile
    # SAR2SAR (Dalsasso et al. 2021, Noise2Noise across acquisition dates): every training tile takes
    # its input from one survey year and its target from another year b of the same place, compensated
    # for change with pre-estimates x_hat of each year's DTM (data.multiyear compensate):
    # label_b - x_hat_b + x_hat_a. compensation: the name of those pre-estimates (xhat_a, xhat_b).
    cross_year: bool = False
    compensation: str = "xhat_a"
    eval_target: str = "own"                 # "consensus": validate against the multi-year median DTM where it exists


def check_context_order(cond_channels) -> None:
    """Context channels feed the network's context branch and must come last."""
    names = list(cond_channels)
    first = next((i for i, n in enumerate(names) if is_context(n)), len(names))
    if any(not is_context(n) for n in names[first:]):
        raise ValueError(f"ctx_* channels must come last in cond_channels, got {names}")


def _window(a: np.ndarray, r0: int, c0: int, h: int, w: int, fill=np.nan) -> np.ndarray:
    out = np.full((h, w), fill, np.float32)
    H, W = a.shape
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + h, H), min(c0 + w, W)
    if re > rs and ce > cs:
        out[rs - r0:re - r0, cs - c0:ce - c0] = a[rs:re, cs:ce]
    return out


class Scene:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        g = self.meta["grid"]
        self.height, self.width, self.gsd = g["height"], g["width"], g["gsd"]

    def array(self, name: str) -> np.ndarray:
        f = self.path / f"{name}.npy"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing (re-run preprocess with --before-dir?)")
        return _open_array(f)

    def window(self, name: str, r0: int, c0: int, h: int, w: int) -> np.ndarray:
        """Read [r0:r0+h, c0:c0+w], NaN outside the scene."""
        return _window(self.array(name), r0, c0, h, w)


class PairScene:
    """SAR2SAR pair (Dalsasso et al. 2021, eq. in Sec. IV-B): inputs from scene a, target from scene
    b (another survey year of the same place) compensated for change: y_b - x_hat_b + x_hat_a, where
    x_hat are pre-estimates of each year's DTM (the `compensation` rasters). Valid where b's label and
    both pre-estimates are."""

    TARGETS = ("gt_dtm", "gt_valid")

    def __init__(self, a: Scene, b: Scene, compensation: str):
        self.a, self.b, self.comp = a, b, compensation
        self.path, self.meta = a.path, a.meta
        self.height, self.width, self.gsd = a.height, a.width, a.gsd

    def array(self, name: str) -> np.ndarray:
        if name in self.TARGETS:
            raise KeyError(f"{name}: read windows of a PairScene")
        return self.a.array(name)

    def _target(self, r0, c0, h, w):
        gb = self.b.window("gt_dtm", r0, c0, h, w)
        xb = self.b.window(self.comp, r0, c0, h, w)
        xa = self.a.window(self.comp, r0, c0, h, w)
        ok = (np.nan_to_num(self.b.window("gt_valid", r0, c0, h, w)) > 0.5) & np.isfinite(gb) \
            & np.isfinite(xb) & np.isfinite(xa)
        return np.where(ok, gb - xb + xa, np.nan), ok

    def window(self, name: str, r0: int, c0: int, h: int, w: int) -> np.ndarray:
        if name == "gt_dtm":
            return self._target(r0, c0, h, w)[0]
        if name == "gt_valid":
            return self._target(r0, c0, h, w)[1].astype(np.float32)
        return self.a.window(name, r0, c0, h, w)


class ConsensusScene(Scene):
    """A scene whose target is the multi-year median DTM (data.multiyear.pairs), for validation."""

    def __init__(self, sc: Scene):
        self.__dict__.update(sc.__dict__)
        self._cons = None

    def array(self, name: str) -> np.ndarray:
        if name == "gt_dtm":
            return super().array("gt_consensus")
        if name == "gt_valid":
            if self._cons is None:
                self._cons = np.isfinite(np.asarray(super().array("gt_consensus"))).astype(np.float32)
            return self._cons
        return super().array(name)


def load_scenes(cfg: DataConfig, split: str | None) -> list[Scene]:
    root = Path(cfg.root)
    names = sorted(p.parent.name for p in root.glob("*/meta.json"))
    if split is not None:
        if not cfg.split_file:
            raise ValueError("split_file is required to select a split")
        names_split = set(json.loads(Path(cfg.split_file).read_text())[split])
        names = [n for n in names if n in names_split]
    scenes = [Scene(root / n) for n in names]
    if not cfg.include_suspect:
        bad = [s.path.name for s in scenes if s.meta.get("quality", {}).get("suspect")]
        if bad:
            print(f"[info] skipping {len(bad)} scenes flagged suspect by preprocess: {bad[:5]}"
                  + ("..." if len(bad) > 5 else ""))
            scenes = [s for s in scenes if s.path.name not in set(bad)]
    if not scenes:
        hint = (" (all were flagged suspect by preprocess: see its summary; to keep them set "
                "data.include_suspect=true, or rerun preprocess with a looser gate)") if names else ""
        raise FileNotFoundError(f"no scenes for split {split!r} under {root}{hint}")
    return scenes


def _starts(n: int, t: int, stride: int) -> list:
    if n <= t:
        return [0]
    s = list(range(0, n - t + 1, stride))
    if s[-1] != n - t:
        s.append(n - t)
    return s


def _own(starts: list, i: int, t: int, stride: int) -> tuple:
    """Part of tile i (in tile coordinates) not already covered by tile i-1,
    so pooled metrics count every pixel once."""
    lo = 0 if i == 0 else max(0, starts[i - 1] + t - starts[i])
    return lo, t


class TileDataset(Dataset):
    """mode="train": random augmented tiles, `samples_per_epoch` per epoch.
    mode="eval": every tile of a regular grid (stride `val_stride`), no augmentation."""

    def __init__(self, cfg: DataConfig, split: str | None, mode: str = "train",
                 scenes: list[Scene] | None = None, max_tiles: int | None = None):
        if cfg.m_alpha_mode == "top_class" and cfg.gate_channel not in ("dsm_max", "dsm_min", "dsm_last"):
            raise ValueError("m_alpha_mode='top_class' only makes sense with a DSM gate channel; "
                             f"got gate_channel={cfg.gate_channel!r}")
        self.cfg = cfg
        self.mode = mode
        self.epoch = 0
        self.scenes = scenes if scenes is not None else load_scenes(cfg, split)
        self.needed = base_channels(set(cfg.cond_channels) | set(cfg.norm_channels)
                                    | {cfg.gate_channel, "gt_dtm", "gt_valid", "dsm_max"}
                                    | ({cfg.prior_channel} if cfg.prior_channel else set())
                                    | ({"top_ground"} if cfg.m_alpha_mode == "top_class" else set()))
        self.context = [n for n in cfg.cond_channels if is_context(n)]
        check_context_order(cfg.cond_channels)
        if mode == "eval" and cfg.eval_target == "consensus":
            self.scenes = [ConsensusScene(sc) if (sc.path / "gt_consensus.npy").exists() else sc for sc in self.scenes]
        if mode == "eval":
            t, stride = cfg.tile, cfg.val_stride or cfg.tile
            self.index = []
            for si, sc in enumerate(self.scenes):
                rows = _starts(sc.height, t, stride)
                cols = _starts(sc.width, t, stride)
                gv = sc.array("gt_valid")
                for i, r in enumerate(rows):
                    for j, c in enumerate(cols):
                        if not np.any(gv[r:r + t, c:c + t] > 0.5):
                            continue                          # nothing to evaluate
                        own = (_own(rows, i, t, stride), _own(cols, j, t, stride))
                        self.index.append((si, r, c, own))
            if max_tiles and len(self.index) > max_tiles:
                # spread the subset over all scenes rather than taking the first ones
                pick = np.unique(np.linspace(0, len(self.index) - 1, max_tiles).round().astype(int))
                self.index = [self.index[k] for k in pick]
        else:
            self.partners = self._partners() if cfg.cross_year else {}
            if cfg.cross_year:
                # SAR2SAR trains on pairs of dates only: places surveyed once take no part
                self.scenes = [sc for sc in self.scenes if sc.path.name in self.partners]
                if not self.scenes:
                    raise ValueError(f"cross_year: no scene has another year with '{cfg.compensation}' "
                                     "pre-estimates (run data.multiyear compensate first)")
            areas = np.array([s.height * s.width for s in self.scenes], np.float64)
            self.scene_p = areas / areas.sum()
            self.bridges = self._bridge_cells() if cfg.bridge_oversample > 0 else []

    def _partners(self) -> dict:
        """{scene: [scenes of the other years]} among the loaded scenes (same split) that have the
        compensation pre-estimates."""
        comp = self.cfg.compensation
        by = {sc.path.name: sc for sc in self.scenes if (sc.path / f"{comp}.npy").exists()}
        out = {}
        for name, sc in by.items():
            lst = [by[p["scene"]] for y, p in sorted((sc.meta.get("pairs") or {}).items()) if p["scene"] in by]
            if lst:
                out[name] = lst
        n = sum(len(v) for v in out.values())
        print(f"[info] SAR2SAR pairs ({comp}): {len(out)} scenes with {n} other-year targets")
        return out

    def _bridge_cells(self, per_scene: int = 4000) -> list:
        """[(scene index, cell rows, cell cols)] for scenes with bridge cells (a subsample of each)."""
        out, rng = [], np.random.default_rng(0)
        for si, sc in enumerate(self.scenes):
            if not sc.meta.get("bridge_cells") or not (sc.path / "bridge.npy").exists():
                continue
            r, c = np.nonzero(np.asarray(sc.array("bridge")) > 0.5)
            if r.size > per_scene:
                k = rng.choice(r.size, per_scene, replace=False)
                r, c = r[k], c[k]
            out.append((si, r.astype(np.int32), c.astype(np.int32)))
        if out:
            print(f"[info] {len(out)} training scenes with bridges; {self.cfg.bridge_oversample:.0%} of tiles "
                  "are centred on one")
        return out

    def set_epoch(self, epoch: int):
        """Call before building each DataLoader iterator so random tiles
        differ between epochs (also with num_workers=0)."""
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.index) if self.mode == "eval" else self.cfg.samples_per_epoch

    # ------------------------------------------------------------------ sampling

    def _exact(self, sc: Scene, r0: int, c0: int, k: int) -> dict:
        t = self.cfg.tile
        out = {n: np.rot90(sc.window(n, r0, c0, t, t), k).copy() for n in self.needed}
        for n in self.context:                  # the same function the runtime uses
            out[n] = np.rot90(context_window(sc.array(context_base(n)), r0, c0, t, self.cfg.context_factor,
                                             context_pool(n)), k).copy()
        return out

    def _resampled(self, sc: Scene, rng, theta_deg: float, win: float, centre=None) -> dict:
        from scipy.ndimage import map_coordinates

        t = self.cfg.tile
        R = int(math.ceil(win * math.sqrt(2) / 2)) + 2
        if centre is not None:
            cy, cx = centre
        else:
            cy = int(rng.integers(min(R, sc.height // 2), max(sc.height - R, sc.height // 2) + 1))
            cx = int(rng.integers(min(R, sc.width // 2), max(sc.width - R, sc.width // 2) + 1))
        th = math.radians(theta_deg)
        u = (np.arange(t) + 0.5) / t - 0.5                  # output offsets in [-0.5, 0.5)
        V, U = np.meshgrid(u * win, u * win, indexing="ij")
        src_r = R + (math.cos(th) * V - math.sin(th) * U) - 0.5
        src_c = R + (math.sin(th) * V + math.cos(th) * U) - 0.5
        coords = np.stack([src_r, src_c])
        out = {}
        for n in self.needed:
            reg = sc.window(n, cy - R, cx - R, 2 * R, 2 * R)
            if n in NEAREST_CHANNELS:
                out[n] = map_coordinates(np.nan_to_num(reg, nan=0.0), coords, order=0, mode="constant", cval=0.0)
            else:
                ok = np.isfinite(reg)
                v = map_coordinates(np.where(ok, reg, 0.0), coords, order=1, mode="constant", cval=0.0)
                w = map_coordinates(ok.astype(np.float64), coords, order=1, mode="constant", cval=0.0)
                val = np.full(v.shape, np.nan, np.float32)
                good = w > 0.5
                val[good] = (v[good] / w[good]).astype(np.float32)
                out[n] = val
        if self.context:
            out.update(self._resampled_context(sc, cy, cx, th, win, V, U))
        return out

    def _resampled_context(self, sc: Scene, cy: int, cx: int, th: float, win: float, V, U) -> dict:
        """Context for a rotated/rescaled window: the same centre and rotation over a
        context_factor times wider area, pooled over the matching footprint, then resampled."""
        from scipy.ndimage import map_coordinates, maximum_filter, minimum_filter, uniform_filter

        F, t = self.cfg.context_factor, self.cfg.tile
        Rc = int(math.ceil(F * win * math.sqrt(2) / 2)) + 2
        src_r = Rc + F * (math.cos(th) * V - math.sin(th) * U) - 0.5
        src_c = Rc + F * (math.sin(th) * V + math.cos(th) * U) - 0.5
        coords = np.stack([src_r, src_c])
        size = max(1, int(round(F * win / t)))            # source pixels per output pixel
        out = {}
        for n in self.context:
            reg = sc.window(context_base(n), cy - Rc, cx - Rc, 2 * Rc, 2 * Rc)
            ok = np.isfinite(reg)
            pool = context_pool(n)
            if pool == "min":
                f = minimum_filter(np.where(ok, reg, np.inf), size=size)
            elif pool == "max":
                f = maximum_filter(np.where(ok, reg, -np.inf), size=size)
            else:
                cnt = uniform_filter(ok.astype(np.float64), size=size)
                f = np.where(cnt > 0, uniform_filter(np.where(ok, reg, 0.0), size=size) / np.maximum(cnt, 1e-12),
                             np.nan)
            good = np.isfinite(f)
            v = map_coordinates(np.where(good, f, 0.0), coords, order=1, mode="constant", cval=0.0)
            w = map_coordinates(good.astype(np.float64), coords, order=1, mode="constant", cval=0.0)
            val = np.full(v.shape, np.nan, np.float32)
            m = w > 0.5
            val[m] = (v[m] / w[m]).astype(np.float32)
            out[n] = val
        return out

    def _sample_train(self, rng) -> dict:
        cfg, t = self.cfg, self.cfg.tile
        aug = cfg.augment
        centre = None
        if self.bridges and rng.random() < cfg.bridge_oversample:
            si, br, bc = self.bridges[int(rng.integers(len(self.bridges)))]
            sc = self.scenes[si]
            j = int(rng.integers(br.size))
            off = rng.integers(-(t // 4), t // 4 + 1, 2)       # the bridge anywhere in the middle half
            centre = (int(br[j] + off[0]), int(bc[j] + off[1]))
        else:
            sc = self.scenes[int(rng.choice(len(self.scenes), p=self.scene_p))]
        other = self.partners.get(sc.path.name)
        if other:                                        # SAR2SAR: the target is another date
            sc = PairScene(sc, other[int(rng.integers(len(other)))], cfg.compensation)
        k = int(rng.integers(0, 4)) if aug and rng.random() < cfg.p_rot90 else 0
        jitter = float(rng.uniform(-cfg.jitter_deg, cfg.jitter_deg)) if aug and rng.random() < cfg.p_jitter else 0.0
        size = int(rng.choice(cfg.multiscale_sizes)) if aug and rng.random() < cfg.p_multiscale else t
        win = t * t / size                                  # source window width in pixels
        if jitter == 0.0 and size == t:
            if centre is not None:
                r0 = min(max(centre[0] - t // 2, 0), max(sc.height - t, 0))
                c0 = min(max(centre[1] - t // 2, 0), max(sc.width - t, 0))
            else:
                r0 = int(rng.integers(0, max(sc.height - t, 0) + 1))
                c0 = int(rng.integers(0, max(sc.width - t, 0) + 1))
            arrs = self._exact(sc, r0, c0, k)
        else:
            arrs = self._resampled(sc, rng, 90.0 * k + jitter, win, centre)
        if aug and rng.random() < cfg.p_hflip:
            arrs = {n: a[:, ::-1] for n, a in arrs.items()}
        if aug and rng.random() < cfg.p_vflip:
            arrs = {n: a[::-1, :] for n, a in arrs.items()}
        return arrs

    def __getitem__(self, i):
        if self.mode == "eval":
            si, r0, c0, ((or0, _), (oc0, _)) = self.index[i]
            arrs = self._exact(self.scenes[si], r0, c0, 0)
            item = self._finish(arrs)
            own = torch.zeros_like(item["valid"])
            own[:, or0:, oc0:] = 1.0
            item["own"] = own
            item["scene_index"] = torch.tensor(si)
            item["origin"] = torch.tensor([r0, c0])
            return item
        rng = np.random.default_rng([torch.initial_seed() % (2 ** 32), self.epoch, i])
        for _ in range(10):
            arrs = self._sample_train(rng)
            item = self._finish(arrs)
            if item["valid"].mean() >= self.cfg.min_valid_frac:
                return item
        return item

    def _finish(self, arrs: dict) -> dict:
        cfg = self.cfg
        gate = arrs[cfg.gate_channel]
        gt = arrs["gt_dtm"]
        gt_ok = np.isfinite(gt) & (np.nan_to_num(arrs["gt_valid"]) > 0.5)
        valid = gt_ok & np.isfinite(gate) if cfg.loss_mask == "gt_and_dsm" else gt_ok
        if cfg.m_alpha_mode == "top_class":
            m_alpha = gt_ok & (np.nan_to_num(arrs["top_ground"]) > 0.5)
        else:
            m_alpha = gt_ok & np.isfinite(gate) & (np.abs(np.nan_to_num(gate) - np.nan_to_num(gt)) < cfg.alpha)
        if cfg.norm_mode == "mean_std":
            # ResDepth: centre on the tile's mean height of the initial raster,
            # divide by a global std: x_n = (x - mean) / std
            base = arrs[cfg.norm_channels[0]]
            ok = np.isfinite(base)
            mean = float(base[ok].mean()) if ok.any() else 0.0
            std = float(cfg.norm_std or 1.0)
            lo, scale = mean - std, 2.0 * std
        else:
            ref = np.stack([arrs[n] for n in cfg.norm_channels])
            lo, scale = tile_range(ref, None, cfg.min_range, cfg.norm_quantile)

        def prep(name, fill=False):
            if name in VIRTUAL_CHANNELS:
                return virtual_channel(name, gate.shape, lo, scale)
            a = arrs[name].astype(np.float64)
            if fill and name in FILL_CHANNELS:
                a = fill_nearest(a)
            x = channel_transform(name, a, lo, scale)
            return np.where(np.isfinite(x), x, 0.0).astype(np.float32)

        fill = cfg.fill_empty == "nearest"
        cond = np.stack([prep(n, fill) for n in cfg.cond_channels])
        target = np.where(gt_ok, prep("gt_dtm"), 0.0).astype(np.float32)
        prior = prep(cfg.prior_channel, fill) if cfg.prior_channel else np.zeros_like(target)
        prior_valid = (np.isfinite(arrs[cfg.prior_channel]) if cfg.prior_channel
                       else np.zeros(target.shape, bool))
        return {
            "cond": torch.from_numpy(cond),
            "target": torch.from_numpy(target[None]),
            "m_alpha": torch.from_numpy(m_alpha[None].astype(np.float32)),
            "valid": torch.from_numpy(valid[None].astype(np.float32)),
            "prior": torch.from_numpy(prior[None]),
            "prior_valid": torch.from_numpy(prior_valid[None].astype(np.float32)),
            # highest-return DSM in network units, NaN where empty (metrics only)
            "surface": torch.from_numpy(channel_transform("dsm_max", arrs["dsm_max"].astype(np.float64), lo, scale
                                                          ).astype(np.float32)[None]),
            "lo": torch.tensor(lo, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
        }
