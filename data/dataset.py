"""Tile-based dataset over preprocessed raster scenes.

Iterates over all scenes in a preprocess root, samples tiles at training
time. Per-tile normalisation to [-1, +1] is computed from the tile's own
`dsm_max` so the three DSM channels (max/min/mean) share a frame.

Hard mining
-----------
Per-tile importance sampling weighted by recent loss. The training
script (scripts/train.py) calls `update_tile_loss(scene, ix, iy, loss)`
after each step; this updates a rolling EMA per tile-key. Tiles with
higher recent loss are sampled more often. When a tile has never been
seen, it starts with weight 1.0 (uniform).

We track tiles by (scene, ix, iy) where (ix, iy) is the top-left pixel
of the tile in the scene's raster.

For deterministic eval, we provide `RasterTileDataset.eval_tiles()`
which yields all non-overlapping tiles tiled across each scene.

Schema expected from preprocess.py:
    dsm_max, dsm_min, dsm_mean, dsm_mask, gt_dtm, valid, m_alpha,
    bbox, gsd, alpha_metres
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


# Sentinel used at empty cells in dsm_max (preprocess.py uses
# `dsm_max[mask].min() - 1.0`). Per-tile normalisation should not use
# this — only valid (masked) cells.
def _fill_nodata_nearest(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace no-data cells (valid==0) with their nearest valid
    neighbour's value. Used in two places, and it MUST be the same in
    both so the network sees identical no-data handling at train and
    inference time:
      1. before augmentation resampling (stops the sentinel/0 fill
         bleeding into valid cells through bilinear interpolation), and
      2. inside _tile_normalise (so val/inference, which are not
         augmented, also never feed the raw sentinel to the network).
    The filled cells are masked out of the loss; the fill only protects
    their valid neighbours and keeps the input distribution consistent.
    """
    vb = valid.astype(bool)
    if vb.all() or not vb.any():
        return arr
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(~vb, return_distances=False,
                                  return_indices=True)
    return arr[tuple(idx)]


def _tile_normalise(
    dsm_max: np.ndarray,
    dsm_min: np.ndarray,
    dsm_mean: np.ndarray,
    gt_dtm: np.ndarray,
    dsm_mask: np.ndarray,
    valid_gt: np.ndarray,
    *,
    min_span_metres: float = 1.0,
):
    """Compute per-tile elevation range from the max-z DSM (`dsm_max`)
    only — NOT from GT (inference has no GT) and NOT from dsm_min (a deep
    pierce return there would stretch the span and compress the terrain).
    Normalise all channels to [-1, +1] in that frame. This matches the
    reference GrounDiff dataset's per-tile DSM frame exactly.

    Returns (dsm_max_n, dsm_min_n, dsm_mean_n, gt_dtm_n, z_lo, z_hi,
              gt_in_range_frac).

    `gt_in_range_frac` is the fraction of valid GT cells whose normalised
    value falls inside [-1.5, +1.5]. The training dataset uses this to
    reject pathological tiles (e.g. a 12.8m tile landing entirely on a
    building, where dsm_min is the roof and GT_DTM is far below the
    normalised range).

    `min_span_metres` enforces a minimum normalisation span so that flat
    tiles (where dsm_min ≈ dsm_max) don't get a degenerate range.
    """
    real = dsm_mask.astype(bool) & valid_gt.astype(bool)
    if not real.any():
        z_lo, z_hi = 0.0, 1.0
    else:
        # Reference-parity normalisation frame: BOTH bounds come from the
        # max-z DSM (`dsm_max`), matching the official GrounDiff dataset
        # (t_vmin = dsm_m[valid].min(), t_vmax = dsm_m[valid].max()).
        #
        # Earlier we anchored z_lo on dsm_min[real].min(), but dsm_min is
        # the LOWEST return per cell (often a ground-pierce point well
        # below the surface). A single deep return there widened the span
        # and compressed the terrain into a sub-band of [-1, 1], costing
        # ~25% of the normalised resolution on the terrain itself and
        # de-calibrating the gating residual r = s - g. Anchoring on the
        # max-z DSM keeps the residual well-scaled and matches the
        # distribution the architecture/loss were tuned for.
        z_lo = float(dsm_max[real].min())
        z_hi = float(dsm_max[real].max())
        # Enforce minimum span so very flat tiles get a sane range.
        if (z_hi - z_lo) < min_span_metres:
            mid = 0.5 * (z_hi + z_lo)
            z_lo = mid - 0.5 * min_span_metres
            z_hi = mid + 0.5 * min_span_metres
    span = max(z_hi - z_lo, 1e-3)
    dmask = dsm_mask.astype(bool)
    vmask = valid_gt.astype(bool)

    def _n(arr):
        return ((arr - z_lo) / span * 2.0 - 1.0).astype(np.float32)

    # Paper §7.2 / Eq. 15: normalise valid pixels to [-1, 1], then SET
    # INVALID REGIONS TO ZERO (the normalised mid-range). This is applied
    # identically here (train + val) and in the PrioStitch inference
    # normaliser, so the network sees the same no-data encoding in every
    # path. The preprocessor's sentinel never reaches the network — it is
    # overwritten by the 0 fill. (The augmentation path additionally
    # nearest-fills BEFORE its bilinear resample so the sentinel cannot
    # bleed into valid neighbours during interpolation — that protects
    # the *valid* cells on sparse-coverage tiles, a case the paper's dense
    # data never hit; the invalid cells themselves still end up at 0.)
    dsm_max_n = np.clip(_n(dsm_max), -1.0, 1.0).astype(np.float32)
    dsm_min_n = np.clip(_n(dsm_min), -1.0, 1.0).astype(np.float32)
    dsm_mean_n = np.clip(_n(dsm_mean), -1.0, 1.0).astype(np.float32)
    gt_dtm_n = _n(gt_dtm)
    dsm_max_n[~dmask] = 0.0
    dsm_min_n[~dmask] = 0.0
    dsm_mean_n[~dmask] = 0.0
    gt_dtm_n[~vmask] = 0.0

    # GT-in-range check for the loss-valid region.
    if real.any():
        gt_n_valid = gt_dtm_n[real]
        gt_in_range_frac = float(((gt_n_valid >= -1.5) & (gt_n_valid <= 1.5)).mean())
    else:
        gt_in_range_frac = 0.0

    return (dsm_max_n, dsm_min_n, dsm_mean_n, gt_dtm_n,
             z_lo, z_hi, gt_in_range_frac)


class HardMiner:
    """Per-tile EMA loss tracker for importance sampling.

    Keys are `(scene_name, ix, iy)`. Update after each step; query before
    sampling. Initial weight for unseen tiles is `init_weight` so they're
    sampled at uniform probability.
    """

    def __init__(self, ema_decay: float = 0.95, init_weight: float = 1.0):
        self.ema_decay = float(ema_decay)
        self.init_weight = float(init_weight)
        # tile_key -> EMA loss
        self.losses: dict[tuple, float] = {}

    def update(self, scene: str, ix: int, iy: int, loss: float) -> None:
        key = (scene, int(ix), int(iy))
        loss = float(loss)
        prev = self.losses.get(key, loss)
        self.losses[key] = self.ema_decay * prev + (1 - self.ema_decay) * loss

    def weight(self, scene: str, ix: int, iy: int) -> float:
        key = (scene, int(ix), int(iy))
        return self.losses.get(key, self.init_weight)

    def state_dict(self) -> dict:
        # JSON-safe: dict[str, float]
        return {f"{s}|{ix}|{iy}": v for (s, ix, iy), v in self.losses.items()}

    def load_state_dict(self, d: dict) -> None:
        self.losses = {}
        for k, v in d.items():
            s, ix, iy = k.split('|')
            self.losses[(s, int(ix), int(iy))] = float(v)


class RasterTileDataset(IterableDataset):
    """Iterable tile sampler with optional hard mining.

    Each iteration yields a dict with:
        dsm_max, dsm_min, dsm_mean: float32 [1, tile, tile]   [-1, +1]
        dsm_mask: float32 [1, tile, tile]                     {0, 1}
        gt_dtm:   float32 [1, tile, tile]                     [-1, +1]
        valid:    float32 [1, tile, tile]   = valid_gt & dsm_mask  ("both data", paper §7.2)
        m_alpha:  float32 [1, tile, tile]                     {0, 1}
        meta:     dict {scene, ix, iy, z_lo, z_hi, gsd}

    Tiles are sampled uniformly from scenes weighted by area (large
    scenes get more tiles), then random (ix, iy) is drawn. If a hard
    miner is attached, the position is rejected with probability
    proportional to inverse weight (Metropolis-style; keeps sampling
    cheap O(1) per attempt).
    """

    def __init__(
        self,
        root: str | Path,
        tile_size: int = 512,
        *,
        scene_filter: Optional[list[str]] = None,
        min_valid_frac: float = 0.50,
        hard_miner: Optional[HardMiner] = None,
        hard_mining_strength: float = 0.5,
        max_reject: int = 8,
        seed: int = 0,
        max_cached_scenes: int = 1,
        tiles_per_scene_burst: int = 64,
        augment: bool = True,
        multiscale_sizes: tuple = (256, 512, 1024),
        multiscale_prob: float = 0.5,
        rot90_prob: float = 0.5,
        rot_jitter_deg: float = 5.0,
        rot_jitter_prob: float = 0.5,
        flip_prob: float = 0.5,
        alpha_metres: float = 0.20,
    ):
        super().__init__()
        self.root = Path(root)
        self.tile = int(tile_size)
        self.alpha_metres = float(alpha_metres)
        self.min_valid_frac = float(min_valid_frac)
        self.hard_miner = hard_miner
        self.hard_mining_strength = float(hard_mining_strength)
        self.max_reject = int(max_reject)
        self.seed = int(seed)
        # Paper §7.1 augmentation pipeline. Sampling stays at the native
        # tile_size (256x256); multi-scale is delivered via resize-and-
        # crop INSIDE _build_tile (paper's "zoom-in" via interpolation).
        # `multiscale_sizes`: the candidate target sizes — paper uses
        # {256, 512, 1024}. With multiscale_prob, one is chosen and the
        # 256x256 source is resized to it before a random 256-crop. The
        # rot_jitter_deg ∈ (-5°, +5°) is added on top of k×90° rotation.
        self.augment = bool(augment)
        self.multiscale_sizes = tuple(int(s) for s in multiscale_sizes)
        self.multiscale_prob = float(multiscale_prob)
        self.rot90_prob = float(rot90_prob)
        self.rot_jitter_deg = float(rot_jitter_deg)
        self.rot_jitter_prob = float(rot_jitter_prob)
        self.flip_prob = float(flip_prob)
        # `tiles_per_scene_burst`: after a worker picks a scene, it pulls
        # this many tiles from that scene before moving to a new scene.
        # Critical for throughput: at ~1-8 s per scene decompression (on
        # a typical DEFRA-sized .npz), bursting amortises the cost across
        # the burst. 64 tiles means decompression cost / 64 ≈ near-zero
        # per tile. Set to 1 to recover the old behaviour (one scene
        # transition per tile).
        self.tiles_per_scene_burst = max(1, int(tiles_per_scene_burst))

        # Enumerate scenes (each is a subdir with raster.npz + .done).
        # Peeking just the shape is critical: with 1900+ scenes, opening
        # each npz with np.load() and touching z['dsm_max'] decompresses
        # the array (np.savez_compressed -> deflate), which on a
        # 20000x20000 fp32 tile reads ~400 MB per scene. Multiplied
        # across 1900 scenes that's hundreds of GB of disk I/O *just*
        # to learn the shape. Instead we read the array header bytes
        # directly from the zip, which is O(1) per scene.
        import zipfile, struct
        def _peek_shape(npz_path: Path) -> tuple[int, int]:
            """Read just the (H, W) from the npz's dsm_max header
            without decompressing any data. ~1 ms per scene."""
            with zipfile.ZipFile(str(npz_path), 'r') as zf:
                with zf.open('dsm_max.npy', 'r') as f:
                    magic = f.read(6)
                    if magic[:6] != b'\x93NUMPY':
                        raise ValueError(f"bad magic in {npz_path}: {magic!r}")
                    major, minor = f.read(2)
                    if major == 1:
                        hlen = struct.unpack('<H', f.read(2))[0]
                    else:
                        hlen = struct.unpack('<I', f.read(4))[0]
                    header = f.read(hlen).decode('latin-1')
            # Header is a Python-literal dict string. Use ast.literal_eval
            # rather than eval for safety.
            import ast
            d = ast.literal_eval(header.strip())
            shape = d['shape']
            return int(shape[0]), int(shape[1])

        self.scenes: list[dict] = []
        scene_dirs = [d for d in sorted(self.root.iterdir())
                      if d.is_dir()]
        # Wrap with tqdm so on a large corpus (1900+ scenes) the user
        # sees the metadata scan progressing rather than wondering if
        # the process has hung. Falls back to a silent no-op if tqdm
        # isn't installed.
        try:
            from tqdm.auto import tqdm as _tqdm
            scene_iter = _tqdm(scene_dirs, desc=f"scan {self.root.name}",
                                unit="scene", leave=False, dynamic_ncols=True)
        except ImportError:
            scene_iter = scene_dirs
        for scene_dir in scene_iter:
            npz = scene_dir / 'raster.npz'
            done = scene_dir / '.done'
            if not (npz.exists() and done.exists()):
                continue
            name = scene_dir.name
            if scene_filter is not None and name not in scene_filter:
                continue
            try:
                H, W = _peek_shape(npz)
                with np.load(str(npz)) as z:
                    gsd = float(z['gsd'])
                    alpha = float(z['alpha_metres'])
            except Exception as e:
                print(f"[dataset] skipping {name}: {e}")
                continue
            if H < self.tile or W < self.tile:
                continue
            self.scenes.append(dict(name=name, path=npz, H=H, W=W,
                                     gsd=gsd, alpha=alpha))

        if not self.scenes:
            raise RuntimeError(f"No usable scenes under {self.root}. "
                                "Did you run preprocess.py?")

        # Per-scene area for weighted sampling.
        areas = np.array([s['H'] * s['W'] for s in self.scenes], dtype=np.float64)
        self.scene_p = areas / areas.sum()
        # Cache raster arrays in memory the first time we read them.
        # Each scene is small enough (5km × 5km × 0.1m = 2500×2500 px ≈
        # 200 MB for all 7 channels at fp32+uint8 mixed). For training
        # Per-worker scene-array cache. Decompressed npz arrays are
        # large (a 20000² scene at fp32 is ~1.6 GB per channel × 7 ≈
        # 11 GB), and with N workers all caching everything they touch
        # we'd OOM the host. Cap with an LRU policy: only keep the
        # `max_cached_scenes` most recently used.
        #
        # Worth tuning if you have lots of host RAM: each cached scene
        # uses ~(H*W*4*7) bytes. On a 20000² corpus that's ~11 GB/scene.
        # `max_cached_scenes=2` is the safe default (~22 GB per worker,
        # × 8 workers = ~176 GB peak — fits a 256 GB box with room).
        # Set higher on bigger boxes to avoid repeated decompression.
        from collections import OrderedDict
        self.max_cached_scenes = int(max_cached_scenes)
        self._cache: OrderedDict[str, dict] = OrderedDict()

    def __len__(self) -> int:
        # Reported length is "approximately one epoch's worth of tiles"
        # — we don't actually iterate exactly this many; the iter is
        # infinite. Used by schedulers for nominal step counts.
        total_area = sum(s['H'] * s['W'] for s in self.scenes)
        return total_area // (self.tile * self.tile)

    def _get_scene_arrays(self, scene_idx: int) -> dict:
        name = self.scenes[scene_idx]['name']
        if name in self._cache:
            # Touch -> move to end of LRU.
            self._cache.move_to_end(name)
            return self._cache[name]
        with np.load(str(self.scenes[scene_idx]['path'])) as z:
            entry = {
                'dsm_max': z['dsm_max'].copy(),
                'dsm_min': z['dsm_min'].copy(),
                'dsm_mean': z['dsm_mean'].copy(),
                'dsm_mask': z['dsm_mask'].copy(),
                'gt_dtm': z['gt_dtm'].copy(),
                'valid': z['valid'].copy(),
                'm_alpha': z['m_alpha'].copy(),
            }
        self._cache[name] = entry
        # Evict oldest entries beyond the cap. Free `entry` references
        # explicitly to let GC reclaim the ~11 GB promptly.
        while len(self._cache) > self.max_cached_scenes:
            _, old = self._cache.popitem(last=False)
            del old
        return entry

    def _pick_scene_only(self, rng: random.Random,
                          forced_si: Optional[int] = None):
        if forced_si is not None:
            return forced_si
        return rng.choices(
            range(len(self.scenes)), weights=self.scene_p, k=1)[0]

    def _sample_tile(self, rng: random.Random,
                      worker_id: int,
                      *,
                      forced_si: Optional[int] = None) -> dict:
        """One tile. Paper §7.1 augmentation lives in _build_tile —
        sampling is always at the native `tile_size` (256x256). The
        paper's multi-scale step resizes the 256x256 tile up to 512 or
        1024 then crops 256 — a zoom-in via interpolation, not a
        coarser-GSD crop of a larger area. This sidesteps the
        scene-too-small fallback we needed before.

        `forced_si`: if not None, sample only from that scene index
        (used by the burst path to amortise decompression).
        """
        check_validity = self.min_valid_frac > 0

        for _attempt in range(self.max_reject + 1):
            si = self._pick_scene_only(rng, forced_si)
            scene = self.scenes[si]
            H, W = scene['H'], scene['W']
            ix = rng.randint(0, W - self.tile)
            iy = rng.randint(0, H - self.tile)
            if check_validity:
                arr = self._get_scene_arrays(si)
                valid_gt = arr['valid'][iy:iy + self.tile, ix:ix + self.tile]
                dsm_mask = arr['dsm_mask'][iy:iy + self.tile, ix:ix + self.tile]
                # Sample-acceptance gate (disabled by default; config
                # sets min_valid_frac=0.0). Uses same definition as
                # loss mask: valid = (DSM exists) AND (DTM exists).
                lv = valid_gt.astype(bool) & dsm_mask.astype(bool)
                if float(lv.mean()) < self.min_valid_frac:
                    continue

            # Hard mining: probabilistic acceptance scaled by weight.
            if self.hard_miner is not None and self.hard_mining_strength > 0:
                w = self.hard_miner.weight(scene['name'], ix, iy)
                p_accept = float(np.clip(w, 0.05, 5.0)
                                  / 1.0) ** self.hard_mining_strength
                p_accept = float(np.clip(p_accept, 0.05, 1.0))
                if rng.random() > p_accept:
                    continue

            return self._build_tile(si, ix, iy, rng=rng)

        # Fallback: take any tile from the (forced or random) scene.
        si = self._pick_scene_only(rng, forced_si)
        scene = self.scenes[si]
        ix = rng.randint(0, scene['W'] - self.tile)
        iy = rng.randint(0, scene['H'] - self.tile)
        return self._build_tile(si, ix, iy, rng=rng)

    # Channel order used by augmentation helpers. Tuple of
    # (name, is_binary) so the helpers know which interpolation order
    # to use (binary masks → nearest; continuous height channels →
    # bilinear).
    _CHANNELS = (
        ('dsm_max',  False),
        ('dsm_min',  False),
        ('dsm_mean', False),
        ('dsm_mask', True),
        ('gt_dtm',   False),
        ('valid_gt', True),
        ('m_alpha',  True),
    )

    @staticmethod
    def _fill_nodata_nearest(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Augmentation-time alias for the module-level
        :func:`_fill_nodata_nearest`. Kept as a static method so the
        augmentation helpers can call it as
        ``RasterTileDataset._fill_nodata_nearest`` while sharing one
        implementation with the normaliser (so they can never drift).
        """
        return _fill_nodata_nearest(arr, valid)

    @staticmethod
    def _resize_zoom_in(arrs, target_size, rng, tile_size):
        """GrounDiff §7.1: "Multi-scale resizing to {256, 512, 1024}"
        followed by "Random cropping of a 256x256 tile". Resizes the
        256x256 source up to target_size (with bilinear for floats,
        nearest for binary masks), then random-crops a tile_size window.
        Trivially a no-op when target_size == tile_size.

        No-data cells in the float channels are filled with their nearest
        valid value BEFORE the bilinear zoom (dsm_* keyed on dsm_mask,
        gt_dtm keyed on valid_gt) so the sentinel/zero fill cannot bleed
        into valid cells. M_α is rebuilt downstream from the resampled
        dsm_max/gt_dtm, so it stays exactly consistent.
        """
        if target_size == tile_size:
            return arrs
        from scipy.ndimage import zoom as nd_zoom
        scale = float(target_size) / float(tile_size)
        # Masks needed for the no-data fill (pre-resample geometry).
        dsm_mask = arrs[3]
        valid_gt = arrs[5]
        fill_masks = {0: dsm_mask, 1: dsm_mask, 2: dsm_mask, 4: valid_gt}
        resized = []
        for ci, (arr, (_name, is_binary)) in enumerate(
                zip(arrs, RasterTileDataset._CHANNELS)):
            if not is_binary and ci in fill_masks:
                arr = RasterTileDataset._fill_nodata_nearest(
                    arr, fill_masks[ci])
            order = 0 if is_binary else 1
            r = nd_zoom(arr, scale, order=order)
            r = r[:target_size, :target_size]
            resized.append(r)
        if target_size > tile_size:
            oy = rng.randint(0, target_size - tile_size)
            ox = rng.randint(0, target_size - tile_size)
            return [np.ascontiguousarray(r[oy:oy + tile_size, ox:ox + tile_size])
                     for r in resized]
        return [np.ascontiguousarray(r) for r in resized]

    @staticmethod
    def _rotate_with_jitter(arrs, delta_deg):
        """GrounDiff §7.1: "additional jittering within the (-5, 5)
        range". Applies sub-90° rotation via scipy.ndimage.rotate.
        Tracks a rotation-valid mask of which pixels came from the
        source (vs the out-of-frame corners) and AND-s that into
        dsm_mask, valid_gt and m_alpha so the loss masks out the corner
        triangles where the rotated tile didn't reach.

        Float channels are (a) nearest-filled over their no-data cells
        and (b) rotated with mode='nearest' (edge extension) rather than
        a constant 0 fill, so neither internal no-data nor the rotated
        corners create an out-of-range cliff that bilinear interpolation
        could bleed into valid cells. The corner triangles are still
        excluded from the loss via the rotation-valid mask below.
        """
        if abs(delta_deg) < 0.1:
            return arrs
        from scipy.ndimage import rotate as nd_rotate
        H, W = arrs[0].shape
        rot_valid = nd_rotate(
            np.ones((H, W), dtype=np.float32),
            float(delta_deg), reshape=False, order=0,
            mode='constant', cval=0.0)
        dsm_mask = arrs[3]
        valid_gt = arrs[5]
        fill_masks = {0: dsm_mask, 1: dsm_mask, 2: dsm_mask, 4: valid_gt}
        new_arrs = []
        for ci, (arr, (_name, is_binary)) in enumerate(
                zip(arrs, RasterTileDataset._CHANNELS)):
            if is_binary:
                # Binary masks: rotate with constant-0 so out-of-frame
                # corners become invalid, then re-threshold.
                r = nd_rotate(arr, float(delta_deg), reshape=False,
                               order=0, mode='constant', cval=0.0)
                r = (r > 0.5).astype(arr.dtype)
            else:
                if ci in fill_masks:
                    arr = RasterTileDataset._fill_nodata_nearest(
                        arr, fill_masks[ci])
                # Edge-extension fill avoids a 0-metre cliff at the
                # rotated corners; those corners are masked out anyway.
                r = nd_rotate(arr, float(delta_deg), reshape=False,
                               order=1, mode='nearest')
            new_arrs.append(np.ascontiguousarray(r))
        rv_f = (rot_valid > 0.5).astype(np.float32)
        new_arrs[3] = (new_arrs[3] * rv_f).astype(new_arrs[3].dtype)
        new_arrs[5] = (new_arrs[5] * rv_f).astype(new_arrs[5].dtype)
        new_arrs[6] = (new_arrs[6] * rv_f).astype(new_arrs[6].dtype)
        return new_arrs

    def _apply_augmentation(self, arrs, rng):
        """Paper §7.1 pipeline. Each step independent at 0.5 prob:
          1. k×90° rotation
          2. sub-90° jitter ∈ (-5°, +5°)
          3. multi-scale resize-and-crop (zoom in to 2x or 4x)
          4. horizontal flip
          5. vertical flip
        Operates on raw arrays (pre-normalisation) so masks and channels
        stay consistent.
        """
        if rng.random() < self.rot90_prob:
            k = rng.randint(1, 3)
            arrs = [np.ascontiguousarray(np.rot90(a, k=k)) for a in arrs]
        if rng.random() < self.rot_jitter_prob and self.rot_jitter_deg > 0:
            delta = rng.uniform(-self.rot_jitter_deg, self.rot_jitter_deg)
            arrs = self._rotate_with_jitter(arrs, delta)
        if rng.random() < self.multiscale_prob:
            target = rng.choice(list(self.multiscale_sizes))
            arrs = self._resize_zoom_in(arrs, target, rng, self.tile)
        if rng.random() < self.flip_prob:
            arrs = [np.ascontiguousarray(a[:, ::-1]) for a in arrs]
        if rng.random() < self.flip_prob:
            arrs = [np.ascontiguousarray(a[::-1, :]) for a in arrs]
        return arrs

    def _build_tile(self, scene_idx, ix, iy, *, rng=None):
        scene = self.scenes[scene_idx]
        arr = self._get_scene_arrays(scene_idx)
        t = self.tile
        dsm_max = arr['dsm_max'][iy:iy + t, ix:ix + t].astype(np.float32, copy=False)
        dsm_min = arr['dsm_min'][iy:iy + t, ix:ix + t].astype(np.float32, copy=False)
        dsm_mean = arr['dsm_mean'][iy:iy + t, ix:ix + t].astype(np.float32, copy=False)
        dsm_mask = arr['dsm_mask'][iy:iy + t, ix:ix + t].astype(np.float32, copy=False)
        gt_dtm = arr['gt_dtm'][iy:iy + t, ix:ix + t].astype(np.float32, copy=False)
        valid_gt = arr['valid'][iy:iy + t, ix:ix + t].astype(np.float32, copy=False)
        m_alpha = arr['m_alpha'][iy:iy + t, ix:ix + t].astype(np.float32, copy=False)

        if self.augment and rng is not None:
            arrs = [dsm_max, dsm_min, dsm_mean, dsm_mask, gt_dtm, valid_gt, m_alpha]
            arrs = self._apply_augmentation(arrs, rng)
            dsm_max, dsm_min, dsm_mean, dsm_mask, gt_dtm, valid_gt, m_alpha = arrs

        # RECOMPUTE M_α from the (augmented) DSM and GT in METRES, rather
        # than carrying the pre-baked mask through resampling. This
        # matches the reference dataset, which computes M_α fresh in
        # __getitem__. Critical because: augmentation (rotation jitter,
        # multi-scale resize) resamples dsm_max and gt_dtm continuously,
        # so a baked binary M_α no longer equals |dsm_max_aug − gt_aug|<α
        # — the confidence head would chase a label inconsistent with its
        # own input, pinning the BCE loss high. Recomputing here keeps
        # M_α exactly consistent with the augmented inputs the network
        # sees. (dsm_max, gt_dtm are still in metres at this point.)
        m_alpha = ((np.abs(dsm_max - gt_dtm) < self.alpha_metres)
                    & valid_gt.astype(bool) & dsm_mask.astype(bool)
                    ).astype(np.float32)

        # Per-tile normalise heights to [-1, +1] using DSM-only frame.
        # This deviates from paper Eq. 15 (which also includes GT DTM
        # in the min/max calculation) — but is required for inference
        # where GT is unavailable. Train-inference consistency wins.
        (dsm_max_n, dsm_min_n, dsm_mean_n, gt_dtm_n,
          z_lo, z_hi, gt_in_range) = _tile_normalise(
            dsm_max, dsm_min, dsm_mean, gt_dtm, dsm_mask, valid_gt)

        gt_dtm_n = np.clip(gt_dtm_n, -1.0, 1.0)
        # `valid` = (DSM exists) AND (DTM exists). Paper-faithful per
        # §7.2 Eq.15: a single mask `m` defines valid pixels in BOTH s
        # and g simultaneously, which only makes sense as the
        # intersection. Excludes:
        #   - pixels with DSM-only (no GT target → can't compute loss)
        #   - pixels with DTM-only (DSM input is sentinel → gating
        #     σ(ℓ)·s would poison the prediction)
        #   - pixels with neither
        # Paper's dense training tiles (DALES/NB) have intersection = 1
        # everywhere, so the loss reduces to no masking on their setup.
        loss_valid = (valid_gt.astype(bool) & dsm_mask.astype(bool)).astype(np.float32)

        out = dict(
            dsm_max=torch.from_numpy(dsm_max_n[None]).contiguous(),
            dsm_min=torch.from_numpy(dsm_min_n[None]).contiguous(),
            dsm_mean=torch.from_numpy(dsm_mean_n[None]).contiguous(),
            dsm_mask=torch.from_numpy(dsm_mask.astype(np.float32)[None]).contiguous(),
            gt_dtm=torch.from_numpy(gt_dtm_n[None]).contiguous(),
            valid=torch.from_numpy(loss_valid[None]).contiguous(),
            m_alpha=torch.from_numpy(m_alpha.astype(np.float32)[None]).contiguous(),
            meta=dict(scene=scene['name'], ix=int(ix), iy=int(iy),
                       z_lo=z_lo, z_hi=z_hi, gsd=scene['gsd'],
                       gt_in_range=gt_in_range),
        )
        return out

    def __iter__(self) -> Iterator[dict]:
        # Per-worker RNG seeding.
        info = torch.utils.data.get_worker_info()
        if info is None:
            worker_id = 0
            seed = self.seed
        else:
            worker_id = info.id
            seed = self.seed + 1000 * info.id
        rng = random.Random(seed)
        while True:
            # Scene-locality burst: pick ONE scene, decompress it once
            # (via _get_scene_arrays in the inner sample), then yield
            # `tiles_per_scene_burst` tiles from it. With burst=64 and a
            # ~8s decompression per scene, the per-tile decompression
            # cost is ~0.13s instead of ~8s.
            si = rng.choices(
                range(len(self.scenes)), weights=self.scene_p, k=1)[0]
            for _ in range(self.tiles_per_scene_burst):
                yield self._sample_tile(rng, worker_id, forced_si=si)


# --------------------------------------------------------------------- #
#  Pre-tiled dataset: reads per-tile .npz files written by scripts/pretile.py
# --------------------------------------------------------------------- #

class RasterPretiledDataset(IterableDataset):
    """Same training-time interface as `RasterTileDataset` but reads
    individual tile .npz files written by `scripts/pretile.py`.

    Each __iter__ yield is one tile loaded from disk + augmentation
    pipeline (same as RasterTileDataset). Eliminates the scene-load
    stalls visible at finer GSDs where each scene .npz is multi-GB.

    Layout expected (matches `scripts/pretile.py` output):
        {root}/{scene_name}/tile_{iy:05d}_{ix:05d}.npz
        {root}/{scene_name}/.done   <- presence indicates scene complete

    Worker sharding: each worker iterates an independent random stream
    over ALL tiles. Slight overlap between workers is harmless since
    each tile yields a different sample under augmentation.

    Hard mining: same `(scene_name, ix, iy)` key as before. Tile
    positions are now from the pre-tile grid rather than continuous,
    so the miner state space is naturally smaller.
    """

    # Channel order is shared with RasterTileDataset (same on-disk schema).
    _CHANNELS = RasterTileDataset._CHANNELS

    def __init__(
        self,
        root: str | Path,
        tile_size: int = 256,
        *,
        scene_filter: Optional[list[str]] = None,
        hard_miner: Optional[HardMiner] = None,
        hard_mining_strength: float = 0.5,
        max_reject: int = 8,
        seed: int = 0,
        augment: bool = True,
        multiscale_sizes: tuple = (256, 512, 1024),
        multiscale_prob: float = 0.5,
        rot90_prob: float = 0.5,
        rot_jitter_deg: float = 5.0,
        rot_jitter_prob: float = 0.5,
        flip_prob: float = 0.5,
        alpha_metres: float = 0.20,
    ):
        super().__init__()
        self.root = Path(root)
        self.tile = int(tile_size)
        self.alpha_metres = float(alpha_metres)
        self.hard_miner = hard_miner
        self.hard_mining_strength = float(hard_mining_strength)
        self.max_reject = int(max_reject)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.multiscale_sizes = tuple(int(s) for s in multiscale_sizes)
        self.multiscale_prob = float(multiscale_prob)
        self.rot90_prob = float(rot90_prob)
        self.rot_jitter_deg = float(rot_jitter_deg)
        self.rot_jitter_prob = float(rot_jitter_prob)
        self.flip_prob = float(flip_prob)

        if not self.root.exists():
            raise RuntimeError(
                f"Pretiled root does not exist: {self.root}. Run "
                f"`python -m stage2_raster.scripts.pretile` first.")

        # Enumerate tile files. Index format: list of (scene_name, path, ix, iy).
        # `self.scenes` mirrors the dict-shape contract from
        # `RasterTileDataset` so train.py can iterate either:
        #     {s['name'] for s in train_ds.scenes}
        self.tiles: list[tuple[str, Path, int, int]] = []
        scene_names_seen: list[str] = []
        scene_names_set: set[str] = set()

        scene_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        try:
            from tqdm.auto import tqdm as _tqdm
            scene_iter = _tqdm(scene_dirs, desc=f"scan {self.root.name}",
                                unit="scene", leave=False, dynamic_ncols=True)
        except ImportError:
            scene_iter = scene_dirs

        for sd in scene_iter:
            if not (sd / '.done').exists():
                continue
            name = sd.name
            if scene_filter is not None and name not in scene_filter:
                continue
            scene_tile_count = 0
            for tf in sorted(sd.glob('tile_*.npz')):
                parts = tf.stem.split('_')
                if len(parts) >= 3:
                    try:
                        iy = int(parts[1])
                        ix = int(parts[2])
                    except ValueError:
                        continue
                    self.tiles.append((name, tf, ix, iy))
                    scene_tile_count += 1
            if scene_tile_count > 0 and name not in scene_names_set:
                scene_names_set.add(name)
                scene_names_seen.append(name)

        if not self.tiles:
            raise RuntimeError(
                f"No tile files under {self.root}. Did pretile.py finish?")

        # Same shape as RasterTileDataset.scenes: list of dicts with at
        # least a 'name' key. n_tiles is also stored for diagnostics.
        self.scenes = [
            {'name': n,
              'n_tiles': sum(1 for t in self.tiles if t[0] == n)}
            for n in scene_names_seen
        ]

    def __len__(self):
        # Approximate. IterableDataset has no true length; this is used
        # by the trainer for progress display only.
        return len(self.tiles)

    def _apply_augmentation(self, arrs, rng: random.Random):
        # Identical pipeline to RasterTileDataset._apply_augmentation.
        if rng.random() < self.rot90_prob:
            k = rng.randint(1, 3)
            arrs = [np.ascontiguousarray(np.rot90(a, k=k)) for a in arrs]
        if rng.random() < self.rot_jitter_prob and self.rot_jitter_deg > 0:
            delta = rng.uniform(-self.rot_jitter_deg, self.rot_jitter_deg)
            arrs = RasterTileDataset._rotate_with_jitter(arrs, delta)
        if rng.random() < self.multiscale_prob:
            target = rng.choice(list(self.multiscale_sizes))
            arrs = RasterTileDataset._resize_zoom_in(
                arrs, target, rng, self.tile)
        if rng.random() < self.flip_prob:
            arrs = [np.ascontiguousarray(a[:, ::-1]) for a in arrs]
        if rng.random() < self.flip_prob:
            arrs = [np.ascontiguousarray(a[::-1, :]) for a in arrs]
        return arrs

    def _build_tile(self, scene_name: str, tile_path: Path,
                     ix: int, iy: int, rng: Optional[random.Random]) -> dict:
        with np.load(str(tile_path)) as z:
            dsm_max = z['dsm_max'].astype(np.float32, copy=False)
            dsm_min = z['dsm_min'].astype(np.float32, copy=False)
            dsm_mean = z['dsm_mean'].astype(np.float32, copy=False)
            dsm_mask = z['dsm_mask'].astype(np.float32, copy=False)
            gt_dtm = z['gt_dtm'].astype(np.float32, copy=False)
            valid_gt = z['valid'].astype(np.float32, copy=False)
            m_alpha = z['m_alpha'].astype(np.float32, copy=False)
            gsd = float(z['gsd'])

        if self.augment and rng is not None:
            arrs = [dsm_max, dsm_min, dsm_mean, dsm_mask,
                     gt_dtm, valid_gt, m_alpha]
            arrs = self._apply_augmentation(arrs, rng)
            dsm_max, dsm_min, dsm_mean, dsm_mask, gt_dtm, valid_gt, m_alpha = arrs

        # Recompute M_α from the augmented DSM/GT in metres — matches the
        # reference and keeps the confidence target consistent with the
        # augmented inputs (see RasterTileDataset._build_tile for the full
        # rationale). dsm_max/gt_dtm are still in metres here.
        m_alpha = ((np.abs(dsm_max - gt_dtm) < self.alpha_metres)
                    & valid_gt.astype(bool) & dsm_mask.astype(bool)
                    ).astype(np.float32)

        (dsm_max_n, dsm_min_n, dsm_mean_n, gt_dtm_n,
          z_lo, z_hi, gt_in_range) = _tile_normalise(
            dsm_max, dsm_min, dsm_mean, gt_dtm, dsm_mask, valid_gt)

        gt_dtm_n = np.clip(gt_dtm_n, -1.0, 1.0)
        # `valid` = (DSM exists) AND (DTM exists). Paper §7.2 Eq.15
        # intersection. See RasterPretiledDataset for full rationale.
        loss_valid = (valid_gt.astype(bool) & dsm_mask.astype(bool)).astype(np.float32)

        return dict(
            dsm_max=torch.from_numpy(dsm_max_n[None]).contiguous(),
            dsm_min=torch.from_numpy(dsm_min_n[None]).contiguous(),
            dsm_mean=torch.from_numpy(dsm_mean_n[None]).contiguous(),
            dsm_mask=torch.from_numpy(dsm_mask.astype(np.float32)[None]).contiguous(),
            gt_dtm=torch.from_numpy(gt_dtm_n[None]).contiguous(),
            valid=torch.from_numpy(loss_valid[None]).contiguous(),
            m_alpha=torch.from_numpy(m_alpha.astype(np.float32)[None]).contiguous(),
            meta=dict(scene=scene_name, ix=int(ix), iy=int(iy),
                       z_lo=z_lo, z_hi=z_hi, gsd=gsd,
                       gt_in_range=gt_in_range),
        )

    def __iter__(self) -> Iterator[dict]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            worker_id = 0
            seed = self.seed
        else:
            worker_id = info.id
            seed = self.seed + 1000 * info.id
        rng = random.Random(seed)
        n = len(self.tiles)
        bad_tiles: set[str] = set()  # per-worker memo of paths that failed
        while True:
            # Hard-mining acceptance test, otherwise uniform.
            if self.hard_miner is not None and self.hard_mining_strength > 0:
                for _ in range(self.max_reject + 1):
                    idx = rng.randrange(n)
                    name, path, ix, iy = self.tiles[idx]
                    w = self.hard_miner.weight(name, ix, iy)
                    p_accept = float(np.clip(w, 0.05, 5.0)) ** self.hard_mining_strength
                    p_accept = float(np.clip(p_accept, 0.05, 1.0))
                    if rng.random() <= p_accept:
                        break
            else:
                idx = rng.randrange(n)
                name, path, ix, iy = self.tiles[idx]
            # Bad-tile resilience: a single corrupt .npz out of 3.5M shouldn't
            # take down a worker. Catch Python-level exceptions on tile load
            # and pick another tile. (Note: this can't catch C-level
            # segfaults from malformed numpy data — those still kill the
            # worker. PyTorch will raise RuntimeError in the main process;
            # the trainer should catch that and recreate the DataLoader.)
            path_str = str(path)
            if path_str in bad_tiles:
                continue
            try:
                yield self._build_tile(name, path, ix, iy, rng)
            except Exception as e:
                bad_tiles.add(path_str)
                print(f"[worker {worker_id}] skipping bad tile "
                       f"{path_str}: {type(e).__name__}: {e}", flush=True)
                continue


# --------------------------------------------------------------------- #
#  Deterministic eval-time tile iterator (no random sampling)
# --------------------------------------------------------------------- #

class RasterEvalTiles(Dataset):
    """Non-overlapping tile enumeration over all scenes, for validation.

    Yields the same dict shape as `RasterTileDataset` but in a fixed
    order and with no hard mining. Tiles whose `loss_valid` fraction is
    below `min_valid_frac` are dropped.
    """

    def __init__(
        self,
        root: str | Path,
        tile_size: int = 512,
        *,
        scene_filter: Optional[list[str]] = None,
        min_valid_frac: float = 0.50,
        max_cached_scenes: int = 1,
    ):
        super().__init__()
        self.root = Path(root)
        self.tile = int(tile_size)
        self.min_valid_frac = float(min_valid_frac)

        scenes = []
        self._tiles: list[tuple[int, int, int]] = []  # (scene_idx, ix, iy)
        si = 0
        # Same fast shape-peek trick as RasterTileDataset — avoids
        # decompressing the full dsm_max array per scene.
        import zipfile, struct, ast
        from collections import OrderedDict
        def _peek_shape(npz_path: Path) -> tuple[int, int]:
            with zipfile.ZipFile(str(npz_path), 'r') as zf:
                with zf.open('dsm_max.npy', 'r') as f:
                    f.read(6)
                    major, _ = f.read(2)
                    hlen = struct.unpack('<H' if major == 1 else '<I',
                                          f.read(2 if major == 1 else 4))[0]
                    header = f.read(hlen).decode('latin-1')
            return tuple(int(s) for s in ast.literal_eval(header.strip())['shape'])

        scene_dirs = [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        try:
            from tqdm.auto import tqdm as _tqdm
            scene_iter = _tqdm(scene_dirs, desc=f"eval-scan {self.root.name}",
                                 unit="scene", leave=False, dynamic_ncols=True)
        except ImportError:
            scene_iter = scene_dirs
        for scene_dir in scene_iter:
            npz = scene_dir / 'raster.npz'
            if not npz.exists():
                continue
            name = scene_dir.name
            if scene_filter is not None and name not in scene_filter:
                continue
            try:
                H, W = _peek_shape(npz)
                if H < self.tile or W < self.tile:
                    continue
                with np.load(str(npz)) as z:
                    gsd = float(z['gsd'])
                scenes.append(dict(name=name, path=npz, H=H, W=W, gsd=gsd))

                if self.min_valid_frac <= 0:
                    # No validity check — every grid tile is enumerated.
                    # The loss itself is masked by (valid & dsm_mask) so
                    # all-empty tiles contribute zero to the average.
                    for iy in range(0, H - self.tile + 1, self.tile):
                        for ix in range(0, W - self.tile + 1, self.tile):
                            self._tiles.append((si, ix, iy))
                else:
                    # Subsampled validity estimate. Full read of all val
                    # scenes' valid+dsm_mask would be ~160 GB; subsampling
                    # by 8 is plenty for an estimate of the mean.
                    ssf = 8
                    with np.load(str(npz)) as z:
                        valid = z['valid'][::ssf, ::ssf]
                        dsm_mask = z['dsm_mask'][::ssf, ::ssf]
                    tile_ss = self.tile // ssf
                    for iy in range(0, H - self.tile + 1, self.tile):
                        iy_s = iy // ssf
                        for ix in range(0, W - self.tile + 1, self.tile):
                            ix_s = ix // ssf
                            vt = valid[iy_s:iy_s + tile_ss,
                                        ix_s:ix_s + tile_ss]
                            mt = dsm_mask[iy_s:iy_s + tile_ss,
                                           ix_s:ix_s + tile_ss]
                            if vt.size == 0:
                                continue
                            lv = vt.astype(bool) & mt.astype(bool)
                            if lv.mean() >= self.min_valid_frac:
                                self._tiles.append((si, ix, iy))
                    del valid, dsm_mask
                si += 1
            except Exception as e:
                print(f"[eval-dataset] skipping {name}: {e}")
                continue
        self.scenes = scenes
        # Same LRU-bounded cache shape as the train dataset to keep
        # worker memory predictable.
        self.max_cached_scenes = int(max_cached_scenes)
        self._cache: OrderedDict[str, dict] = OrderedDict()

        if not self._tiles:
            raise RuntimeError(
                f"No eval tiles found under {self.root} (tile={self.tile}, "
                f"min_valid={self.min_valid_frac})")

    def __len__(self) -> int:
        return len(self._tiles)

    def _get_scene_arrays(self, si: int) -> dict:
        name = self.scenes[si]['name']
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        with np.load(str(self.scenes[si]['path'])) as z:
            entry = {k: z[k].copy() for k in
                      ['dsm_max', 'dsm_min', 'dsm_mean',
                       'dsm_mask', 'gt_dtm', 'valid', 'm_alpha']}
        self._cache[name] = entry
        while len(self._cache) > self.max_cached_scenes:
            _, old = self._cache.popitem(last=False)
            del old
        return entry

    def __getitem__(self, idx: int) -> dict:
        si, ix, iy = self._tiles[idx]
        arr = self._get_scene_arrays(si)
        t = self.tile
        dsm_max = arr['dsm_max'][iy:iy + t, ix:ix + t]
        dsm_min = arr['dsm_min'][iy:iy + t, ix:ix + t]
        dsm_mean = arr['dsm_mean'][iy:iy + t, ix:ix + t]
        dsm_mask = arr['dsm_mask'][iy:iy + t, ix:ix + t]
        gt_dtm = arr['gt_dtm'][iy:iy + t, ix:ix + t]
        valid_gt = arr['valid'][iy:iy + t, ix:ix + t]
        m_alpha = arr['m_alpha'][iy:iy + t, ix:ix + t]

        (dsm_max_n, dsm_min_n, dsm_mean_n, gt_dtm_n,
          z_lo, z_hi, gt_in_range) = _tile_normalise(
            dsm_max, dsm_min, dsm_mean, gt_dtm, dsm_mask, valid_gt)
        gt_dtm_n = np.clip(gt_dtm_n, -1.0, 1.0)
        # `valid` = (DSM exists) AND (DTM exists). Paper §7.2 Eq.15
        # intersection. Validation metrics share the same definition
        # so train and eval measure the same set of pixels.
        loss_valid = (valid_gt.astype(bool) & dsm_mask.astype(bool)).astype(np.float32)
        return dict(
            dsm_max=torch.from_numpy(dsm_max_n[None]).contiguous(),
            dsm_min=torch.from_numpy(dsm_min_n[None]).contiguous(),
            dsm_mean=torch.from_numpy(dsm_mean_n[None]).contiguous(),
            dsm_mask=torch.from_numpy(dsm_mask.astype(np.float32)[None]).contiguous(),
            gt_dtm=torch.from_numpy(gt_dtm_n[None]).contiguous(),
            valid=torch.from_numpy(loss_valid[None]).contiguous(),
            m_alpha=torch.from_numpy(m_alpha.astype(np.float32)[None]).contiguous(),
            meta=dict(scene=self.scenes[si]['name'], ix=int(ix), iy=int(iy),
                       z_lo=float(z_lo), z_hi=float(z_hi),
                       gsd=self.scenes[si]['gsd'],
                       gt_in_range=float(gt_in_range)),
        )
