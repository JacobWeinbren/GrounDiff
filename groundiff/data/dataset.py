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

from ..normalise import HEIGHT_CHANNELS, NEAREST_CHANNELS, channel_transform, tile_range



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
    loss_mask: str = "gt_and_dsm"            # or "gt": also learn to fill no-return cells
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


class Scene:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        g = self.meta["grid"]
        self.height, self.width, self.gsd = g["height"], g["width"], g["gsd"]
        self._arrays: dict[str, np.ndarray] = {}

    def __getstate__(self):
        # memmaps would be pickled as full copies when DataLoader workers are
        # spawned (the default on macOS); reopen them lazily in each worker
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state

    def array(self, name: str) -> np.ndarray:
        if name not in self._arrays:
            f = self.path / f"{name}.npy"
            if not f.exists():
                raise FileNotFoundError(f"{f} missing (re-run preprocess with --before-dir?)")
            self._arrays[name] = np.load(f, mmap_mode="r")
        return self._arrays[name]

    def window(self, name: str, r0: int, c0: int, h: int, w: int) -> np.ndarray:
        """Read [r0:r0+h, c0:c0+w], NaN outside the scene."""
        out = np.full((h, w), np.nan, np.float32)
        a = self.array(name)
        rs, cs = max(r0, 0), max(c0, 0)
        re, ce = min(r0 + h, self.height), min(c0 + w, self.width)
        if re > rs and ce > cs:
            out[rs - r0:re - r0, cs - c0:ce - c0] = a[rs:re, cs:ce]
        return out


def load_scenes(cfg: DataConfig, split: str | None) -> list[Scene]:
    root = Path(cfg.root)
    names = sorted(p.parent.name for p in root.glob("*/meta.json"))
    if split is not None:
        if not cfg.split_file:
            raise ValueError("split_file is required to select a split")
        names_split = set(json.loads(Path(cfg.split_file).read_text())[split])
        names = [n for n in names if n in names_split]
    if not names:
        raise FileNotFoundError(f"no scenes for split {split!r} under {root}")
    return [Scene(root / n) for n in names]


class TileDataset(Dataset):
    """mode="train": random augmented tiles, `samples_per_epoch` per epoch.
    mode="eval": every tile of a regular grid (stride `val_stride`), no augmentation."""

    def __init__(self, cfg: DataConfig, split: str | None, mode: str = "train",
                 scenes: list[Scene] | None = None):
        self.cfg = cfg
        self.mode = mode
        self.epoch = 0
        self.scenes = scenes if scenes is not None else load_scenes(cfg, split)
        self.needed = sorted(set(cfg.cond_channels) | set(cfg.norm_channels) | {cfg.gate_channel, "gt_dtm", "gt_valid"}
                             | ({cfg.prior_channel} if cfg.prior_channel else set()))
        if mode == "eval":
            t, stride = cfg.tile, cfg.val_stride or cfg.tile
            self.index = []
            for si, sc in enumerate(self.scenes):
                rows = range(0, max(sc.height - t, 0) + 1, stride) if sc.height > t else [0]
                cols = range(0, max(sc.width - t, 0) + 1, stride) if sc.width > t else [0]
                rows = sorted(set(rows) | {max(sc.height - t, 0)})
                cols = sorted(set(cols) | {max(sc.width - t, 0)})
                self.index += [(si, r, c) for r in rows for c in cols]
        else:
            areas = np.array([s.height * s.width for s in self.scenes], np.float64)
            self.scene_p = areas / areas.sum()

    def set_epoch(self, epoch: int):
        """Call before building each DataLoader iterator so random tiles
        differ between epochs (also with num_workers=0)."""
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.index) if self.mode == "eval" else self.cfg.samples_per_epoch

    # ------------------------------------------------------------------ sampling

    def _exact(self, sc: Scene, r0: int, c0: int, k: int) -> dict:
        t = self.cfg.tile
        return {n: np.rot90(sc.window(n, r0, c0, t, t), k).copy() for n in self.needed}

    def _resampled(self, sc: Scene, rng, theta_deg: float, win: float) -> dict:
        from scipy.ndimage import map_coordinates

        t = self.cfg.tile
        R = int(math.ceil(win * math.sqrt(2) / 2)) + 2
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
        return out

    def _sample_train(self, rng) -> dict:
        cfg, t = self.cfg, self.cfg.tile
        aug = cfg.augment
        sc = self.scenes[int(rng.choice(len(self.scenes), p=self.scene_p))]
        k = int(rng.integers(0, 4)) if aug and rng.random() < cfg.p_rot90 else 0
        jitter = float(rng.uniform(-cfg.jitter_deg, cfg.jitter_deg)) if aug and rng.random() < cfg.p_jitter else 0.0
        size = int(rng.choice(cfg.multiscale_sizes)) if aug and rng.random() < cfg.p_multiscale else t
        win = t * t / size                                  # source window width in pixels
        if jitter == 0.0 and size == t:
            r0 = int(rng.integers(0, max(sc.height - t, 0) + 1))
            c0 = int(rng.integers(0, max(sc.width - t, 0) + 1))
            arrs = self._exact(sc, r0, c0, k)
        else:
            arrs = self._resampled(sc, rng, 90.0 * k + jitter, win)
        if aug and rng.random() < cfg.p_hflip:
            arrs = {n: a[:, ::-1] for n, a in arrs.items()}
        if aug and rng.random() < cfg.p_vflip:
            arrs = {n: a[::-1, :] for n, a in arrs.items()}
        return arrs

    def __getitem__(self, i):
        if self.mode == "eval":
            si, r0, c0 = self.index[i]
            arrs = self._exact(self.scenes[si], r0, c0, 0)
            item = self._finish(arrs)
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
            lo, scale = tile_range(ref, np.isfinite(ref).all(0), cfg.min_range)

        def prep(name):
            x = channel_transform(name, arrs[name].astype(np.float64), lo, scale)
            return np.where(np.isfinite(x), x, 0.0).astype(np.float32)

        cond = np.stack([prep(n) for n in cfg.cond_channels])
        target = np.where(gt_ok, prep("gt_dtm"), 0.0).astype(np.float32)
        prior = prep(cfg.prior_channel) if cfg.prior_channel else np.zeros_like(target)
        return {
            "cond": torch.from_numpy(cond),
            "target": torch.from_numpy(target[None]),
            "m_alpha": torch.from_numpy(m_alpha[None].astype(np.float32)),
            "valid": torch.from_numpy(valid[None].astype(np.float32)),
            "prior": torch.from_numpy(prior[None]),
            "lo": torch.tensor(lo, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
        }
