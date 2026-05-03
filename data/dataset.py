"""DSM→DTM tile dataset (paper §7.1, §7.2).

Each tile file is a compressed npz produced by scripts/preprocess.py:

    dsm_norm   : float16 [H, W]   DSM normalised to [-1, 1]
    dtm_norm   : float16 [H, W]   DTM normalised to [-1, 1]
    valid_mask : float32 [H, W]   1 where both DSM and DTM are valid
    stats      : dict             {'vmin': float, 'vmax': float, 'has_data': bool}

For training we yield:
    cond_dsm: [1, crop, crop] float32 in [-1, 1]   — UNet condition
    target_dtm: [1, crop, crop] float32 in [-1, 1] — y_0 in DDPM terms
    valid:    [crop, crop]   float32 in {0, 1}     — for masked loss

Sampling weights (paper-deviation, optional).
DEFRA UK is mostly rural — ~95% of pixels per tile are ground. The
training sampler can be biased to oversample tiles with more non-
ground content using `compute_sampling_weights(mode=...)`. This
deviation is sampling-only; the loss stays paper-faithful.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .normalize import joint_minmax_normalise
from .augment import groundiff_augment


def _load_tile(path: Path):
    """Load a tile npz, returning a dict. Returns None if the file is
    missing required keys (corrupt / partially-written / stale schema).
    The caller must handle None."""
    try:
        with np.load(str(path), allow_pickle=True) as f:
            d = {k: f[k] for k in f.files}
    except Exception:
        return None
    if 'stats' in d and isinstance(d['stats'], np.ndarray) \
            and d['stats'].dtype == object:
        d['stats'] = d['stats'].item()
    # Required keys schema. dsm_min_norm is optional (set when use_min_dsm).
    required = ('dsm_norm', 'dtm_norm', 'valid_mask', 'stats')
    if not all(k in d for k in required):
        return None
    return d


class DSMDTMTileDataset(Dataset):
    """Tile dataset that returns (cond_dsm, target_dtm, valid).

    Args:
        tile_dir:   directory containing per-scene subdirs of `*.npz` tiles
        split:      'train' or 'test' subdir under tile_dir; falls back to
                    tile_dir itself if the subdir doesn't exist
        crop_px:    output crop size (paper: 256)
        augment:    apply the §7.1 pipeline (turn off for val/test)
        seed:       optional RNG seed for reproducibility
        min_valid_frac: tiles with `valid_mask.mean() < min_valid_frac` are
                    skipped at construction. Default 0.0 = paper-faithful
                    (no filter). DEFRA UK uses 0.10 by config to drop
                    truly-empty corner tiles that contribute mostly noise
                    to gradients. Tiles with 10-50% valid coverage (real
                    partial-coverage cases) are kept.
    """

    def __init__(self, tile_dir, split: str = 'train',
                 crop_px: int = 256, augment: bool = True,
                 seed: int | None = None,
                 min_valid_frac: float = 0.0,
                 alpha_metres: float = 0.20,
                 use_min_dsm: bool = False):
        self.tile_dir = Path(tile_dir)
        self.split = split
        self.crop_px = int(crop_px)
        self.augment = bool(augment)
        self._rng_seed = seed
        self.min_valid_frac = float(min_valid_frac)
        self.alpha_metres = float(alpha_metres)
        self.use_min_dsm = bool(use_min_dsm)

        sd = self.tile_dir / split
        if not sd.exists():
            sd = self.tile_dir
        self.paths = sorted(p for p in sd.rglob('*.npz')
                            if not p.name.startswith('_scene_'))
        self.names = [f"{p.parent.name}/{p.stem}" for p in self.paths]
        scenes = sorted({n.split('/')[0] for n in self.names})
        n_before = len(self.paths)
        print(f"DSMDTMTileDataset({split}): "
              f"{len(self.paths)} tiles across {len(scenes)} scenes "
              f"(augment={augment}, crop={crop_px}, "
              f"min_valid_frac={self.min_valid_frac})")

        if self.min_valid_frac > 0.0 and self.paths:
            self._filter_by_valid_frac()
            scenes_after = sorted({n.split('/')[0] for n in self.names})
            print(f"  → after valid_frac filter: "
                  f"{len(self.paths)} tiles across {len(scenes_after)} scenes "
                  f"(removed {n_before - len(self.paths)} low-coverage tiles)")

    def _filter_by_valid_frac(self) -> None:
        """Drop tiles with valid coverage below threshold. Cached."""
        cache_path = self.tile_dir / '.valid_fracs.npz'
        fracs = self._load_or_compute_valid_fracs(cache_path)
        keep = fracs >= self.min_valid_frac
        self.paths = [p for p, k in zip(self.paths, keep) if k]
        self.names = [n for n, k in zip(self.names, keep) if k]

    def _load_or_compute_valid_fracs(self, cache_path: Path) -> np.ndarray:
        path_hash = self._paths_hash()
        if cache_path.exists():
            try:
                d = np.load(str(cache_path), allow_pickle=True)
                if str(d['paths_hash']) == path_hash and \
                        len(d['fracs']) == len(self.paths):
                    print(f"  loaded valid_fracs from {cache_path}")
                    return d['fracs']
            except Exception as e:
                print(f"  cache read failed ({e}); recomputing")

        print(f"  computing valid_frac for {len(self.paths)} tiles...")
        fracs = np.zeros(len(self.paths), dtype=np.float32)
        for i, p in enumerate(self.paths):
            try:
                with np.load(str(p), allow_pickle=True) as f:
                    fracs[i] = float(f['valid_mask'].astype(bool).mean())
            except Exception:
                fracs[i] = 0.0
            if (i + 1) % 10000 == 0:
                print(f"    {i+1}/{len(self.paths)}  "
                      f"mean valid_frac so far = {fracs[:i+1].mean():.3f}")
        print(f"  done. valid_frac stats: "
              f"mean={fracs.mean():.3f}  median={float(np.median(fracs)):.3f}  "
              f"p10={float(np.quantile(fracs, 0.1)):.3f}  "
              f"min={fracs.min():.3f}")
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(cache_path), fracs=fracs, paths_hash=path_hash)
            print(f"  cached to {cache_path}")
        except Exception as e:
            print(f"  cache write failed: {e}")
        return fracs

    def __len__(self) -> int:
        return len(self.paths)

    def _make_rng(self, idx: int) -> np.random.Generator:
        if self._rng_seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self._rng_seed + idx)

    def __getitem__(self, idx: int):
        # Robust against rare corrupt / partial-write tiles. We try the
        # requested tile first; if it can't be loaded, we walk forward
        # to the next loadable tile. This avoids killing a DataLoader
        # worker for one bad file. We also cache the bad-tile set so
        # we skip them on subsequent epochs.
        if not hasattr(self, '_bad_tiles'):
            self._bad_tiles = set()
        n = len(self.paths)
        tries = 0
        while True:
            cur = (idx + tries) % n
            if cur in self._bad_tiles:
                tries += 1
                if tries >= n:
                    raise RuntimeError(
                        "all tiles failed to load — dataset is corrupt")
                continue
            d = _load_tile(self.paths[cur])
            if d is not None:
                idx = cur
                break
            self._bad_tiles.add(cur)
            tries += 1
            if tries >= n:
                raise RuntimeError(
                    "all tiles failed to load — dataset is corrupt")
        # fp16 → fp32 promotion; downstream torch ops are fp32 by default
        dsm = torch.from_numpy(d['dsm_norm'].astype(np.float32))[None]
        dtm = torch.from_numpy(d['dtm_norm'].astype(np.float32))[None]
        valid = torch.from_numpy(d['valid_mask'].astype(np.float32))

        # Optional dsm_min channel (preprocessing must have written it)
        dsm_min = None
        if self.use_min_dsm:
            if 'dsm_min_norm' in d:
                dsm_min = torch.from_numpy(
                    d['dsm_min_norm'].astype(np.float32))[None]
            else:
                # Fall back to dsm if min channel wasn't preprocessed.
                # This means use_min_dsm has zero effect for tiles missing
                # the channel — equivalent to running paper-baseline.
                dsm_min = dsm.clone()

        if self.augment:
            rng = self._make_rng(idx)
            if dsm_min is None:
                dsm, dtm, valid = groundiff_augment(
                    dsm, dtm, valid, crop=self.crop_px, rng=rng,
                )
            else:
                dsm, dtm, valid, dsm_min = groundiff_augment(
                    dsm, dtm, valid, crop=self.crop_px, rng=rng,
                    extra=dsm_min,
                )
        else:
            # Centre crop / pad to (crop_px, crop_px) deterministically
            H, W = dsm.shape[-2:]
            crop = self.crop_px
            pad_h = max(0, crop - H)
            pad_w = max(0, crop - W)
            if pad_h or pad_w:
                import torch.nn.functional as _F
                dsm = _F.pad(dsm, (0, pad_w, 0, pad_h), mode='replicate')
                dtm = _F.pad(dtm, (0, pad_w, 0, pad_h), mode='replicate')
                if dsm_min is not None:
                    dsm_min = _F.pad(dsm_min, (0, pad_w, 0, pad_h),
                                      mode='replicate')
                valid = _F.pad(valid[None, None],
                               (0, pad_w, 0, pad_h),
                               mode='constant', value=0)[0, 0]
                H, W = dsm.shape[-2:]
            i = max(0, (H - crop) // 2)
            j = max(0, (W - crop) // 2)
            dsm = dsm[..., i:i + crop, j:j + crop]
            dtm = dtm[..., i:i + crop, j:j + crop]
            valid = valid[..., i:i + crop, j:j + crop]
            if dsm_min is not None:
                dsm_min = dsm_min[..., i:i + crop, j:j + crop]

        # ---- Per-tile re-normalisation by DSM only --------------------
        # Paper Eq.15 uses min(s_m, g_m) and max(s_m, g_m) — both rasters.
        # That works at training time but BREAKS AT INFERENCE: PrioStitch
        # never sees the GT DTM. To keep train/inference distributions
        # identical, we normalise by DSM-only valid-pixel min/max here AND
        # in PrioStitch. DTM physically satisfies DTM ≤ DSM so this only
        # very rarely clips DTM (water/voids) and is consistent everywhere.
        #
        # M_α is computed BEFORE re-normalisation in metres so the
        # threshold has a fixed physical meaning across tiles
        # (Sithole-Vosselman 2003 §4.2.1: 0.20m).
        scene_stats = d['stats']
        if isinstance(scene_stats, np.ndarray) and scene_stats.dtype == object:
            scene_stats = scene_stats.item()
        m_alpha_t = torch.zeros_like(dsm)  # [1, H, W]
        if scene_stats.get('has_data', True):
            v_b = (valid > 0.5)
            if v_b.any():
                s_vmin = float(scene_stats['vmin'])
                s_vmax = float(scene_stats['vmax'])
                s_span = max(s_vmax - s_vmin, 1e-6)
                # to metres
                dsm_m = (dsm + 1.0) * 0.5 * s_span + s_vmin
                dtm_m = (dtm + 1.0) * 0.5 * s_span + s_vmin
                if dsm_min is not None:
                    dsm_min_m = (dsm_min + 1.0) * 0.5 * s_span + s_vmin

                # Compute M_α in metres — paper-faithful and tile-agnostic.
                # Stored as [1, H, W] with same shape as dsm.
                v3 = v_b.unsqueeze(0) if v_b.dim() == 2 else v_b
                m_alpha_t = ((dsm_m - dtm_m).abs() < self.alpha_metres
                              ).to(dsm.dtype) * v3.to(dsm.dtype)

                # Per-tile DSM-only stats — identical to inference path
                t_vmin = float(dsm_m[v3].min().item())
                t_vmax = float(dsm_m[v3].max().item())
                t_span = max(t_vmax - t_vmin, 1e-6)
                dsm = torch.where(v3,
                                   2.0 * (dsm_m - t_vmin) / t_span - 1.0,
                                   torch.zeros_like(dsm_m))
                dtm = torch.where(v3,
                                   2.0 * (dtm_m - t_vmin) / t_span - 1.0,
                                   torch.zeros_like(dtm_m))
                if dsm_min is not None:
                    dsm_min = torch.where(
                        v3,
                        2.0 * (dsm_min_m - t_vmin) / t_span - 1.0,
                        torch.zeros_like(dsm_min_m))
                    # dsm_min ≤ dsm so it can fall below -1 if there's a
                    # ground-pierce return well below the DSM-min reference.
                    # That's OK and informative — clip to [-1, 1] to keep
                    # the network input well-conditioned.
                    dsm_min = dsm_min.clamp(-1.0, 1.0)
                # DTM may slightly exceed [-1, 1] if DTM elevations fall
                # below DSM min (rare; e.g. underground voids in TIN). Clip.
                dtm = dtm.clamp(-1.0, 1.0)
                tile_stats = dict(vmin=t_vmin, vmax=t_vmax, has_data=True)
            else:
                tile_stats = dict(scene_stats)
        else:
            tile_stats = dict(scene_stats)

        out = dict(
            cond_dsm=dsm,
            target_dtm=dtm,
            valid=valid,
            m_alpha=m_alpha_t,
            stats=tile_stats,
            name=self.names[idx],
        )
        if dsm_min is not None:
            out['cond_dsm_min'] = dsm_min
        return out

    # ---- Sampling weights for class imbalance --------------------------

    def compute_sampling_weights(self, *, mode: str = 'mild',
                                 alpha_norm: float = 0.05,
                                 cache_path: Path | None = None,
                                 verbose: bool = True) -> np.ndarray:
        """Compute per-tile sampling weights based on non-ground content.

        For each tile we compute `nonground_frac` = fraction of valid
        pixels with `|s_n − g_n| > alpha_norm` (in normalised units —
        matches training-time M_α target). Then convert to a sampling
        weight via one of the modes below. On a typical DEFRA tile set
        with ~5% urban content (frac≥0.05), urban tile sampling
        probability comes out roughly:

          'uniform':    5% (no skew, paper-faithful baseline)
          'mild':       8% (1 + 5·frac)
          'moderate':  12% (0.1 + frac)
          'strong':    30% (frac + 0.02) — between moderate and aggressive
          'aggressive': 50% (two-bucket equalise)

        The smooth modes ('mild'/'moderate'/'strong') preserve gradation:
        a tile with 30% non-ground gets more weight than one with 5%.
        'aggressive' treats all urban tiles the same.

        Args:
            mode: weighting policy (see above)
            alpha_norm: threshold on normalised |s_n − g_n| (default 0.05
                        matches loss α and metric Method A)
            cache_path: optional path to cache computed nonground_fracs.
                        Re-uses if present and valid.
            verbose: print diagnostics

        Returns:
            np.float64 array of length len(self), sampler-compatible.
        """
        # Load or compute nonground_frac per tile
        fracs = self._load_or_compute_fracs(alpha_norm, cache_path, verbose)
        return self._fracs_to_weights(fracs, mode, verbose)

    def _load_or_compute_fracs(self, alpha_norm, cache_path, verbose):
        if cache_path is not None and Path(cache_path).exists():
            try:
                d = np.load(str(cache_path), allow_pickle=True)
                if (str(d['paths_hash']) == self._paths_hash()
                        and float(d['alpha_norm']) == float(alpha_norm)):
                    if verbose:
                        print(f"  loaded sampling fracs from {cache_path}")
                    return d['fracs']
            except Exception as e:
                if verbose:
                    print(f"  cache read failed ({e}); recomputing")

        if verbose:
            print(f"  computing per-tile non-ground fraction "
                  f"(α_norm={alpha_norm}) for {len(self.paths)} tiles...")
        fracs = np.zeros(len(self.paths), dtype=np.float32)
        for i, p in enumerate(self.paths):
            try:
                with np.load(str(p), allow_pickle=True) as f:
                    s = f['dsm_norm'].astype(np.float32)
                    g = f['dtm_norm'].astype(np.float32)
                    v = f['valid_mask'].astype(bool)
                if v.any():
                    r = np.abs(s - g)[v]
                    fracs[i] = float((r > alpha_norm).mean())
                else:
                    fracs[i] = 0.0
            except Exception:
                fracs[i] = 0.0
            if verbose and (i + 1) % 5000 == 0:
                print(f"    {i+1}/{len(self.paths)}  "
                      f"mean_frac so far = {fracs[:i+1].mean():.4f}")

        if verbose:
            print(f"  done. nonground_frac stats: "
                  f"mean={fracs.mean():.4f}  median={np.median(fracs):.4f}  "
                  f"p90={np.quantile(fracs, 0.9):.4f}  "
                  f"p99={np.quantile(fracs, 0.99):.4f}  "
                  f"max={fracs.max():.4f}")

        if cache_path is not None:
            try:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez(str(cache_path), fracs=fracs,
                          paths_hash=self._paths_hash(),
                          alpha_norm=float(alpha_norm))
                if verbose:
                    print(f"  cached to {cache_path}")
            except Exception as e:
                if verbose:
                    print(f"  cache write failed: {e}")
        return fracs

    def _fracs_to_weights(self, fracs, mode, verbose):
        fracs = fracs.astype(np.float64)
        if mode == 'uniform':
            w = np.ones_like(fracs)
        elif mode == 'mild':
            # weight = 1 + 5·frac → ~3× for tiles with 40% non-ground
            w = 1.0 + 5.0 * fracs
        elif mode == 'moderate':
            # weight = 0.1 + frac → tiles with 30% non-ground get 4×,
            # mostly-ground tiles still sampled at 0.1×
            w = 0.1 + fracs
        elif mode == 'strong':
            # weight = frac + 0.02 — between moderate and aggressive.
            # Targets ~30% urban (frac≥0.05) sampling probability on
            # DEFRA. Smooth, respects gradation: a tile with 40% non-
            # ground is sampled 20× more than one with 0% non-ground.
            w = fracs + 0.02
        elif mode == 'aggressive':
            # Two-bucket equalise: urban tiles ≈ 50% of any batch.
            urban = (fracs >= 0.05).astype(np.float64)
            n_urban = max(int(urban.sum()), 1)
            n_rural = max(int((1 - urban).sum()), 1)
            w = np.where(urban, n_rural / n_urban, 1.0)
        else:
            raise ValueError(f"unknown sampling mode {mode!r}")
        # Normalise so weights sum to len (mean 1.0) — easier to interpret
        w = w * (len(w) / max(w.sum(), 1e-12))
        if verbose:
            print(f"  sampling weights ({mode}): "
                  f"min={w.min():.3f}  median={float(np.median(w)):.3f}  "
                  f"max={w.max():.3f}  effective N = {len(w):.0f}")
            urban_mask = fracs >= 0.05
            if urban_mask.any():
                p_urban = w[urban_mask].sum() / w.sum()
                p_urban_baseline = float(urban_mask.mean())
                print(f"  urban tile (frac≥0.05) sampling probability: "
                      f"{p_urban:.1%} (uniform baseline: {p_urban_baseline:.1%})")
        return w

    def _paths_hash(self) -> str:
        # Cheap hash for cache invalidation
        import hashlib
        h = hashlib.sha1()
        for p in self.paths:
            h.update(p.name.encode())
        return h.hexdigest()[:16]
