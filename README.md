# GrounDiff for EA LiDAR DTM production

Learns what editors change after `lasground_new`, and turns that into a
per-pixel **edit probability**, the **predicted change** to the
`lasground_new` DTM, a ranked list of blocks to edit, and ready-to-view
overlays for LP360 and QGIS.

* **GrounDiff** — Dhaouadi, Meier, Kaiser & Cremers, *GrounDiff: Diffusion-Based
  Ground Surface Generation from Digital Surface Models*, WACV 2026
  ([arXiv 2511.10391](https://arxiv.org/abs/2511.10391)). No official code exists;
  the first author advised building on
  [Palette](https://github.com/janspiry/palette-image-to-image-diffusion-models).
  The denoiser here is Palette's U-Net, ported and checked output-for-output
  against Palette (62.64M parameters = the 62.6M in GrounDiff §8.1).
* **ResDepth** — Stucker & Schindler, ISPRS J. 2022
  ([code](https://github.com/prs-eth/ResDepth), MIT): deterministic
  before → after baseline, vendored unchanged.
* **ALS2DTM / DeepTerRa** — Lê et al., JSTARS 2022: source of the extra input
  rasters (lowest return, density / height-spread / echo rasters, and the
  2-channel ground/non-ground raster, here built from `lasground_new` classes).

Training runs on Apple Silicon (MPS), CUDA or CPU; inference also runs without
PyTorch through ONNX Runtime (QGIS plugin, Windows GPU via CUDA or DirectML).

![edit-probability overlay on a light-green DTM](docs/overlay_example.png)

*Overlay colours on a real EA tile (TL4378nw) shaded light green. The values
here are a stand-in (canopy height), only to show the colours: transparent
below 0.2, then magenta → deep purple, darker and more opaque as the value
rises.*

---

## Two models: pick by whether you have a LAStools licence

| | `configs/no_lastools.json` (**default**) | `configs/before_after.json` |
|---|---|---|
| Needs LAStools to train | no | yes, licensed `lasground_new` (in Docker on the Mac) |
| Learns | the EA DTM from the point clouds (GrounDiff DSM → DTM + ALS2DTM rasters) | what editors change in the `lasground_new` result |
| In QGIS, on your `lasground_new` tiles | predicted DTM; `dz_before` = predicted − `lasground_new` DTM; `p_edit` = share of diffusion samples that differ from `lasground_new` by > 0.2 m | the same, with `p_edit` from the trained confidence head |

Unlicensed `lasground_new` may be used free for non-profit personal and
educational purposes, but its output is distorted above a point limit (a few
million points), and full EA 500 m tiles have a median of ~6M points, so it
cannot make usable training data. Without a licence, use `no_lastools`.

## What is learned from what

**Production chain.** Tiles are classified by `lasground_new` with default
settings (every point becomes 1 = non-ground or 2 = ground), then edited by
hand in LP360 using only unclassified, ground, bridge and low noise. The EA
DTM is built from the edited ground class.

**Inputs, before → after model** (per 1 m cell), all from the `lasground_new`
output, so training and production see the same thing: highest / lowest / lowest-last return,
point density, height spread, echoes, the `lasground_new` DTM (TIN of its
ground points) and its ground / non-ground raster. No point is dropped by
class. The `no_lastools` model uses only the point rasters (no classes at all),
so it trains on the published tiles as downloaded.

**Target: the EA's published DTM raster.** The classes in the EA's published
LAZ/COPC files come from a different, automated process (they contain 1–7
including vegetation height bands and buildings) and do not match the DTM
rasters, so they are never used. `groundiff.data.ea_dtm` downloads the DTM
for each training tile; alternatively, your own LP360-edited tiles can be the
target (`--after-dir`: TIN of class 2).

**No-LAStools model.** GrounDiff as published (gate on the highest-return
DSM) with the ALS2DTM point rasters, trained on the EA DTM. At inference on
`lasground_new` tiles, each of N diffusion samples is compared with their
ground: `p_edit` is the share of samples whose DTM is more than α = 0.2 m from
`lasground_new`'s, `dz_before` the difference of the mean prediction. Use
several samples (QGIS default 4).

**Before → after model.** GrounDiff's gate (Eq. 5) is anchored on the `lasground_new` DTM:

    DTM = σ(ℓ) · DTM_lasground + (1 − σ(ℓ)) · (DTM_lasground − r̂)

so the model keeps the automated surface where it is right and corrects it
where editors would. Its confidence head (Eq. 14, M_α = |DTM_lasground − DTM_EA| < α)
therefore learns "no edit needed": **p_edit = 1 − σ(ℓ)** is the per-pixel
probability that an editor changes the ground there, trained with the paper's
own loss. `dz_before = DTM_pred − DTM_lasground` is the size and sign of the
predicted correction (negative: `lasground_new` kept something as ground that
editors remove; positive: it cut off real ground, e.g. an embankment crest).

**Quality gate.** A tile whose DTM raster disagrees with its points (a
different survey, misregistration) is flagged *suspect* by `preprocess` and
left out of training and evaluation. It is measured where `lasground_new`
says ground, or, without `lasground_new`, on open ground (single returns,
< 5 cm height spread), where the lowest return should lie on the DTM.

---

## Mac quickstart (MacBook Pro M3 Max, 36 GB)

Everything up to the trained model happens on the Mac; the Windows PC only
runs the model in QGIS. No LAStools needed (the `no_lastools` model; for the
before → after model with a licence see *The "before" classification* below).

```bash
# 0. one-off setup
git clone -b claude/rewrite-before-after https://github.com/JacobWeinbren/GrounDiff.git && cd GrounDiff
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install torch && pip install -e ".[onnx,dev]"
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
pytest -q                                   # ~2 min, all should pass

# 1. EA 2022 point clouds (open data; --dry-run shows count and size first)
python -m groundiff.data.download --out data/laz/ea --target 300 --min-per-grid 8 --dry-run
python -m groundiff.data.download --out data/laz/ea --target 300 --min-per-grid 8

# 2. the EA DTM rasters for those tiles (the target; ~50-150 MB per 5 km tile)
python -m groundiff.data.ea_dtm --tiles data/laz/ea --out data/ea_dtm --dry-run
python -m groundiff.data.ea_dtm --tiles data/laz/ea --out data/ea_dtm

# 3. rasterise at 1 m (the EA DTM grid) straight from the downloaded tiles, and split by 10 km blocks
python -m groundiff.data.preprocess --points-dir data/laz/ea --dtm-dir data/ea_dtm \
    --out data/scenes_1m --gsd 1.0 --workers 4
python -m groundiff.data.split --root data/scenes_1m --out data/split.json

# 4. five-minute smoke test
python -m groundiff.train configs/no_lastools.json --set train.out_dir=runs/smoke \
    optim.total_steps=50 train.val_every=50 train.val_max_tiles=16

# 5. full run in the background; caffeinate keeps the Mac awake (keep it on power)
mkdir -p runs
caffeinate -dimsu nohup python -m groundiff.train configs/no_lastools.json \
    --set train.num_workers=6 > runs/no_lastools.out 2>&1 &

# 6. watch it
python -m groundiff.monitor runs/no_lastools --follow    # step, time left, loss curve, memory, validation
tail -f runs/no_lastools.out                             # raw log
# stop: pkill -f groundiff.train      resume: run step 5 again (continues from last.pt)

# 7. evaluate on the held-out tiles, export for the PC / QGIS
python -m groundiff.infer --checkpoint runs/no_lastools/best.pt --scenes data/scenes_1m \
    --split-file data/split.json --split test --out results/test --samples 4
python -m groundiff.export runs/no_lastools/best.pt --out models/no_lastools      # .onnx + .json
python tools/build_qgis_plugin.py                                                # dist/groundiff_qgis.zip
```

`environment.data.gov.uk` (step 2) must be reachable; it is from a normal
connection. Step 3 needs roughly 1-3 GB RAM per worker for 500 m tiles and
more for the 2 km tiles the archive also contains; lower `--workers` if the
Mac starts swapping. Tiles flagged *suspect* are listed at the end of step 3.

Configs default to batch 4 × 4 accumulation (≈ batch 16, as in the paper).
If MPS runs out of memory: `--set train.batch_size=2 train.grad_accum=8`, or
add `model.use_checkpoint=true` (≈ 25 % slower, far less activation memory).

---

## Workflow details

### Point clouds

```bash
python -m groundiff.data.download --out data/laz/ea --target 1000 --min-per-grid 20
python -m groundiff.data.download --out data/laz/ea --squares TL4378 --blocks 3      # an area + neighbours
```

The DEFRA `LIDAR_2022` folder of the public `open-lidar-data` bucket holds
500 m quadrants (`TL4378nw_P_12534_...`) and 2 km tiles (`NX9410_P_12706_...`)
from ~86 surveys; it appears to be the EA's time-stamped survey archive rather
than the National LIDAR Programme. The files carry **no CRS** (they are BNG,
EPSG:27700, heights ODN); outputs default to EPSG:27700. `--blocks N` adds the
N × N neighbours of each sampled tile (every survey at each position).

### Target DTMs

```bash
python -m groundiff.data.ea_dtm --tiles data/laz/ea --out data/ea_dtm
```

For each 5 km tile the point clouds touch, and the survey year in their file
names, this asks the Defra survey service what exists and downloads, by
default, `lidar_tiles_dtm` (the time-stamped DTM of the same surveys) at the
finest resolution up to 1 m, else the National LIDAR Programme DTM of that
year (`--product` forces one). It resumes where it stopped. Zips hold float32
GeoTIFFs (nodata −3.4e38) on whole-metre cell edges, plus a GeoPackage
recording which survey fed each area. The service's public key (`dspui`, from
its web page) is undocumented and may change.

`preprocess --dtm-dir` accepts any folder of GeoTIFF / ASCII grid / VRT DTMs
in any tiling, so rasters downloaded by hand from
<https://environment.data.gov.uk/survey> work too.

### The "before" classification (before → after model; needs a LAStools licence)

Production runs `lasground_new` with **default settings**, which reclassifies
every point to 1 or 2 regardless of the classes it had. With a licence, and
Docker Desktop on the Mac (keep "Use Rosetta for x86_64/amd64 emulation" on;
Resources → Memory 16 GB+):

```bash
python -m groundiff.data.lasground docker --in data/laz/ea --out data/laz/before --license lastoolslicense.txt
```

builds (once) a Docker image with the LAStools Linux release and the
libraries its README lists, then runs
`lasground_new64 -lof <tiles> -odir <out> -olaz -cores N -v` with your
licence mounted read-only (`LAStoolsLicenseFile`). Outputs keep the input
names, so tiles pair up; tiles already done are skipped; the `-v` log goes to
`<out>/lasground_new.log` (the `lasground_new` README gives two default steps,
25 m in the text and 5.0 in the argument list; the log shows which the binary
used). It then runs `check`: equal point counts, **unchanged coordinates**
(LAStools without a valid licence perturbs files above its free point limit,
~1.5–5M points), classes 1/2 only and a sane ground share. On Apple Silicon
the x86-64 build runs under Rosetta emulation, slower than native; `--cores`
tiles are processed in parallel. `--lastools-tar` uses a downloaded `LAStools.tar.gz` instead
of fetching it; `--rebuild` picks up a new release. With LAStools on Windows
instead: `python -m groundiff.data.lasground script --windows ...` writes a
`.bat`. Then preprocess with `--before-dir data/laz/before` and train
`configs/before_after.json`.

### Files from different producers

```bash
python -m groundiff.data.lasinspect ea_tile.laz --compare lp360_tile.las
```

Reports version, point format, WKT bit, CRS storage (VLRs and EVLRs; parsed
without pyproj too), extra bytes, flags (overlap / synthetic / withheld /
key-point), classes and return statistics, with warnings for things that make
PDAL / QGIS reject files. The reader tolerates all of them; `--drop-overlap` /
`--drop-synthetic` exist everywhere.

### Rasterise and split

```bash
python -m groundiff.data.preprocess --before-dir data/laz/before --dtm-dir data/ea_dtm \
    --out data/scenes_1m --gsd 1.0 --workers 4
# or, with your own hand-edited tiles as the target (same file names, ground = class 2):
python -m groundiff.data.preprocess --before-dir data/laz/before --after-dir data/laz/edited \
    --out data/scenes_1m --gsd 1.0
python -m groundiff.data.split --root data/scenes_1m --out data/split.json --block-km 10
```

`preprocess` refuses "before" files with classes other than 1/2 (plus 7/18) —
usually a sign that the published EA file was given instead of the
`lasground_new` output. The TIN of `lasground_new` ground and the target are
cut to the LiDAR coverage (returns plus voids narrower than 60 m, e.g. rivers
and small ponds; larger water bodies are left out, consistently in training
and in batch). Re-running only redoes tiles whose inputs or settings
changed. `--geotiff` also writes every channel as GeoTIFF (for the raster tool
in QGIS). Splits assign whole 10 km blocks, balancing the three sets by size.

### Train

| Config | What |
|---|---|
| `configs/no_lastools.json` | GrounDiff DSM → DTM + ALS2DTM rasters, no LAStools (default) |
| `configs/before_after.json` | GrounDiff, before → after (needs licensed `lasground_new`) |
| `configs/paper_dsm2dtm.json` | GrounDiff as published (DSM → DTM) |
| `configs/resdepth_before_after.json` | ResDepth baseline, as published (fp32) |

```bash
# M3 Max, 36 GB (fp32; batch 4 x 4 is the default)
python -m groundiff.train configs/no_lastools.json --set train.num_workers=6
# 16 GB CUDA GPU (bf16 automatic on RTX 30xx and newer)
python -m groundiff.train configs/no_lastools.json --set train.num_workers=8
#   batch 8 needs model.use_checkpoint=true on 16 GB
# fine-tune from another checkpoint (input channels are matched by name; new ones start at zero)
python -m groundiff.train configs/before_after.json --init-from runs/paper_dsm2dtm/best.pt
```

Runs resume from `last.pt` (weights, EMA, optimiser, schedule, grad scaler,
random state, data position). Validation uses tiles spread over all validation
scenes, counts each pixel once, and reports for the model and for
`lasground_new` against the EA DTM: RMSE, MAE, bias, MedAE, NMAD, ground /
non-ground RMSE, Type I/II/total (against the highest-return DSM), and edit
detection precision / recall / F1 (|DTM − DTM_lasground| > α). The monitor's
time estimate excludes validation.

Cost (estimates): ≈ 1 TFLOP per 256² tile per training step, so the paper's
10–20k iterations at batch 16 is roughly half a day to a day on an M3 Max and a
few hours on a recent 16 GB NVIDIA card. Memory at 256² tiles: batch 4 fp32
≈ 10–13 GB; batch 8 bf16 ≈ 15.5–17 GB (too much for 16 GB without
checkpointing); batch 16 fp32 ≈ 35–46 GB. Accumulation 4 × 4 is close to, not
exactly, batch 16 (loss means are per micro-batch). Inference needs ≈ 2–3 GB of GPU memory at the default 8 network tiles per batch
(lower "Network tiles per batch" on small GPUs).

### Evaluate on held-out scenes

```bash
python -m groundiff.infer --checkpoint runs/no_lastools/best.pt --scenes data/scenes_1m \
    --split-file data/split.json --split test --out results/test --samples 4
```

Per scene: GeoTIFFs, overlays, `metrics.json` (as above, plus roughness) and
`priority.csv` (100 m blocks with coordinates, ranked by predicted edit
volume). With a reference, `capture_topK` reports how much of the true edit
volume the top 5/10/20 % of blocks contain; this ranking measure is ours,
neither paper defines one.

### Run on new tiles (command line, no QGIS)

```bash
python -m groundiff.export runs/no_lastools/best.pt --out models/no_lastools     # checked vs PyTorch
python -m groundiff.batch --onnx models/no_lastools.onnx --tiles "lasground/*.laz" --out results/area1 --samples 4
```

Inputs are tiles as `lasground_new` wrote them (quote wildcards; folders work
too, and Windows is handled). Each tile is read with a buffer of neighbour
points (one network tile + 32 m), network tiles lie on one lattice anchored to
the National Grid, and each tile's sampling noise is seeded by its position,
so neighbouring tiles agree exactly where they meet (and re-running part of
an area reproduces the same values, except within one buffer of the new
selection's edge, where neighbouring points are missing). A tile is prepared in the background while the network runs
(each job needs roughly 550 bytes per point read, ~6-7 GB for a 500 m tile at
EA density; the log prints an estimate; raise `--workers` only with RAM to
spare); a bad file is reported in `batch_summary.json` and the rest
carry on. Header extents are checked against point counts and tile names and
replaced by the points' own extent when they cannot be right (stale LP360
headers). Cell size and read options default to what the model was trained
with.

Outputs: `p_edit.tif`, `dz_before.tif`, `dtm.tif` (`std.tif` with
`--samples > 1` or `--tta`), their LP360 overlays and QGIS styles, overviews,
`.tfw`/`.prj`, and `priority.csv` / `priority.geojson` (+ `priority.shp` where
GDAL's Python bindings exist, e.g. inside QGIS): blocks ranked by predicted
edit volume, with coordinates, edit area and mean p_edit.

## Using the outputs in LP360

For each map there are two pre-coloured GeoTIFFs:

* `*_overlay.tif` — RGBA with an alpha channel: transparent where nothing
  needs attention. Add it as a raster layer above the DTM; try this first.
* `*_overlay_rgb.tif` — RGB with no-data = 0, for viewers that ignore alpha;
  transparent pixels are no-data, set the layer's transparency in the viewer.

Both have `.tfw` and `.prj` side-cars and internal overviews. The colour ramp
(one hue, the complement of the light-green DTM shading) was checked
numerically after blending over pale, mid and hill-shade greens, white and
grey: lightness decreases monotonically, steps stay distinguishable, and even
the palest visible step has ≥ 2.3 : 1 contrast with the background.
`priority.shp` / `.geojson` give the same information as a work list of
squares.

Two colours show the two edit tools:

* **purple**: `lasground_new` is too **high** there (points kept as ground
  that should become unclassified: vegetation, buildings, noise);
* **orange**: `lasground_new` is too **low** (ground it missed, e.g.
  embankment crests, banks): points to classify as ground.

| Overlay | Shows | Transparent below |
|---|---|---|
| `p_edit` | probability that editors change the ground, coloured by direction | 0.2 |
| `dz_before` | size of the predicted correction, purple = lower, orange = raise | 0.15 m |
| `std` | spread across samples / flips (purple only) | 0.1 m |

The `dz_before` layer in QGIS gets the same two-colour style.

## QGIS plugin (QGIS 3.22 – 4.x, Windows and Mac)

```bash
python tools/build_qgis_plugin.py        # -> dist/groundiff_qgis.zip
```

1. *Plugins → Manage and Install Plugins → Install from ZIP* → `groundiff_qgis.zip`.
2. The first time, QGIS shows "GrounDiff needs a few components" → **Install now**
   (or *Plugins → GrounDiff → Install / check GrounDiff components*). It installs
   ONNX Runtime (DirectML on Windows, which uses any GPU including NVIDIA
   without CUDA; CPU + CoreML on Mac) and laspy into your QGIS profile with
   QGIS's own Python, about 160 MB, once. Nothing to type, and QGIS's own
   packages are left alone.
3. Click the **GrounDiff** toolbar button (*Find edits in point clouds*):
   * pick your `lasground_new` tiles, either point-cloud layers already in the
     project or LAS/LAZ files (… → Add File(s) / Add Directory);
   * pick the model `.onnx` the first time (it is remembered; the model's
     settings are stored inside the `.onnx`, so that one file is all you copy);
   * Run. The edit probability, predicted edit (styled) and ranked priority
     blocks are added to the map. Results go to a temporary folder unless you
     choose an output folder (do so when you want the files for LP360:
     `*_overlay.tif`, `priority.shp`).

Everything else (device, samples, cell size, buffer, workers, …) is under
*Advanced* and can normally be left alone. The log says which device ONNX
Runtime used; if a GPU cannot take the model it falls back to the CPU and
says so. Also in the toolbox: **Predict DTM from rasters** and **Inspect
LAS/LAZ file**. The plugin reads point clouds itself (laspy), so files that
QGIS's own point-cloud layers (PDAL) refuse can still be given as files.

## Faithfulness to the papers

| Item | Here | Paper |
|---|---|---|
| Denoiser | Palette U-Net, 62.64M | U-Net "inspired by DDPM", 62.6M |
| Gating, loss (Eq. 5, 11–14), λ | as published | |
| T, schedule | T = 10, Palette cosine; noise levels sampled as Palette does | "cosine … from 0.0001 to 0.02" (ambiguous; `cosine_range` implements the other reading) |
| Sampler init | N(s, I) in DSM → DTM mode; before → after starts from the `lasground_new` DTM (Table 6 "DSM" init) | §3.2, Table 6 |
| PrioStitch | global coarse prior (aspect kept), 50 % overlap, linear blending by default; min / mean available | §3.3, Table 7 (min = best RMSE, linear = best balance) |
| Augmentation | rotations + ±5° jitter, zoom {256, 512, 1024} + crop, flips, p = 0.5 each | §7.1 |
| Normalisation | per-tile min–max to [−1, 1] from inputs only, 2 m minimum range (before → after: 0.1 / 99.9 % quantiles) | min–max over DSM **and GT** (GT unknown at inference) |
| Optimiser | AdamW 1e-4, wd 0.01, 500 warm-up, cosine, batch 16 | §7.3 |
| α for M_α | 0.2 m (option: class of highest return, DSM → DTM only) | not given |
| EMA | 0.999 with warm-up | not stated (Palette uses one) |
| Before → after | gate on `lasground_new` DTM, extra inputs from ALS2DTM | extension |

Known gap: with Palette's cosine schedule the last step is almost pure noise
(√ᾱ_T = 0.005), so a prior injected at t = T barely reaches the network. The
paper reports a large PrioStitch gain (Table 7), which suggests their setup
kept more of it. Worth ablating: `diffusion.schedule=cosine_range` (then also
try `--init dsm_q`, which matches the forward process at t = T), or
`--t-start 5` at inference. In before → after mode the `lasground_new` DTM is
an input channel, so the model sees it at every step regardless.

## Suggested ablations

* schedule `cosine` vs `cosine_range` (with `--init dsm_q`); `--t-start`
* `loss.units=metres` (stops flat tiles dominating)
* gate on `dsm_min` in DSM → DTM mode (on a real EA tile, 94 % of cells have
  the lowest return within 0.2 m of the ground surface vs 81 % for the highest)
* ResDepth vs GrounDiff on the same split

## Not done yet

* Breaklines: no breakline output yet. Raster breakline targets need a sample
  of how they are stored in production.
* Point reclassification from the predicted DTM (a height-above-DTM rule, or
  a dense classification head as the GrounDiff author suggested).
* Checkpoints from the earlier version of this repository are not compatible.

## Tests

`pytest` covers: the U-Net matching Palette, checkpointed dropout gradients,
schedules and posterior maths, every sampler init, loss and metric
definitions, rasterisation, TIN, DTM-raster targets (exact copy and
resampling), the quality gate, rejection of published-class files, dataset and
augmentation consistency, spatial splits, training / resume / fine-tune,
numpy vs PyTorch sampler equivalence, ONNX export, scene inference, seamless
batch mosaics (also with a noise-dependent network and different job splits),
stale headers, cancelling, overlays, LAS variants (LAS 1.2/1.4, missing WKT
bit, extra bytes, broken return numbers, flags, CRS without pyproj), the DTM
downloader (offline), and the QGIS algorithms run against a stand-in
`qgis.core` in QGIS 3 and 4 styles.
