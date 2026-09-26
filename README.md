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

## What is learned from what

**Production chain.** Tiles are classified by `lasground_new` with default
settings (every point becomes 1 = non-ground or 2 = ground), then edited by
hand in LP360 using only unclassified, ground, bridge and low noise. The EA
DTM is built from the edited ground class.

**Inputs** (per 1 m cell), all from the `lasground_new` output, so training
and production see the same thing: highest / lowest / lowest-last return,
point density, height spread, echoes, the `lasground_new` DTM (TIN of its
ground points) and its ground / non-ground raster. No point is dropped by
class.

**Target: the EA's published DTM raster.** The classes in the EA's published
LAZ/COPC files come from a different, automated process (they contain 1–7
including vegetation height bands and buildings) and do not match the DTM
rasters, so they are never used. `groundiff.data.ea_dtm` downloads the DTM
for each training tile; alternatively, your own LP360-edited tiles can be the
target (`--after-dir`: TIN of class 2).

**Model.** GrounDiff's gate (Eq. 5) is anchored on the `lasground_new` DTM:

    DTM = σ(ℓ) · DTM_lasground + (1 − σ(ℓ)) · (DTM_lasground − r̂)

so the model keeps the automated surface where it is right and corrects it
where editors would. Its confidence head (Eq. 14, M_α = |DTM_lasground − DTM_EA| < α)
therefore learns "no edit needed": **p_edit = 1 − σ(ℓ)** is the per-pixel
probability that an editor changes the ground there, trained with the paper's
own loss. `dz_before = DTM_pred − DTM_lasground` is the size and sign of the
predicted correction (negative: `lasground_new` kept something as ground that
editors remove; positive: it cut off real ground, e.g. an embankment crest).

**Quality gate.** A tile whose DTM raster disagrees with `lasground_new` on
the cells `lasground_new` calls ground (a different survey, misregistration)
is flagged *suspect* by `preprocess` and left out of training and evaluation.

---

## Mac quickstart (MacBook Pro M3 Max, 36 GB)

Everything up to the trained model happens on the Mac; the Windows PC only
runs the model in QGIS. One-off installs: Python 3.10–3.12 and
[Docker Desktop](https://www.docker.com/products/docker-desktop/) (for
LAStools' Linux build; in its settings keep "Use Rosetta for x86_64/amd64
emulation" on and set Resources → Memory to 16 GB or more). Have your
`lastoolslicense.txt` to hand.

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

# 3. "before" tiles: lasground_new, default settings, in Docker on the Mac (builds the image
#    the first time; re-run to continue after an interruption; checks the results at the end)
python -m groundiff.data.lasground docker --in data/laz/ea --out data/laz/before \
    --license ~/lastools/lastoolslicense.txt --cores 6

# 4. rasterise at 1 m (the EA DTM grid) and split by 10 km blocks
python -m groundiff.data.preprocess --before-dir data/laz/before --dtm-dir data/ea_dtm \
    --out data/scenes_1m --gsd 1.0 --workers 4
python -m groundiff.data.split --root data/scenes_1m --out data/split.json

# 5. five-minute smoke test
python -m groundiff.train configs/before_after.json --set train.out_dir=runs/smoke \
    optim.total_steps=50 train.val_every=50 train.val_max_tiles=16

# 6. full run in the background; caffeinate keeps the Mac awake (keep it on power)
mkdir -p runs
caffeinate -dimsu nohup python -m groundiff.train configs/before_after.json \
    --set train.num_workers=6 > runs/before_after.out 2>&1 &

# 7. watch it
python -m groundiff.monitor runs/before_after --follow   # step, time left, loss curve, memory, validation
tail -f runs/before_after.out                            # raw log
# stop: pkill -f groundiff.train      resume: run step 6 again (continues from last.pt)

# 8. evaluate on the held-out tiles, export for the PC / QGIS
python -m groundiff.infer --checkpoint runs/before_after/best.pt --scenes data/scenes_1m \
    --split-file data/split.json --split test --out results/test
python -m groundiff.export runs/before_after/best.pt --out models/before_after   # .onnx + .json
python tools/build_qgis_plugin.py                                                # dist/groundiff_qgis.zip
```

`environment.data.gov.uk` (step 2) must be reachable; it is from a normal
connection. Step 4 needs roughly 1-3 GB RAM per worker for 500 m tiles and
more for the 2 km tiles the archive also contains; lower `--workers` if the
Mac starts swapping. Tiles flagged *suspect* are listed at the end of step 4.

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

### The "before" classification

Production runs `lasground_new` with **default settings**, which reclassifies
every point to 1 or 2 regardless of the classes it had.

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
`.bat`.

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
cut to the LiDAR coverage (returns plus voids narrower than 60 m and enclosed
voids such as lakes). Re-running only redoes tiles whose inputs or settings
changed. `--geotiff` also writes every channel as GeoTIFF (for the raster tool
in QGIS). Splits assign whole 10 km blocks, balancing the three sets by size.

### Train

| Config | What |
|---|---|
| `configs/before_after.json` | GrounDiff, before → after (recommended) |
| `configs/paper_dsm2dtm.json` | GrounDiff as published (DSM → DTM) |
| `configs/resdepth_before_after.json` | ResDepth baseline, as published (fp32) |

```bash
# M3 Max, 36 GB (fp32; batch 4 x 4 is the default)
python -m groundiff.train configs/before_after.json --set train.num_workers=6
# 16 GB CUDA GPU (bf16 automatic on RTX 30xx and newer)
python -m groundiff.train configs/before_after.json --set train.num_workers=8
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
exactly, batch 16 (loss means are per micro-batch). Inference needs ≈ 0.5 GB.

### Evaluate on held-out scenes

```bash
python -m groundiff.infer --checkpoint runs/before_after/best.pt --scenes data/scenes_1m \
    --split-file data/split.json --split test --out results/test --samples 4
```

Per scene: GeoTIFFs, overlays, `metrics.json` (as above, plus roughness) and
`priority.csv` (100 m blocks with coordinates, ranked by predicted edit
volume). With a reference, `capture_topK` reports how much of the true edit
volume the top 5/10/20 % of blocks contain; this ranking measure is ours,
neither paper defines one.

### Run on new tiles (command line, no QGIS)

```bash
python -m groundiff.export runs/before_after/best.pt --out models/before_after   # checked vs PyTorch
python -m groundiff.batch --onnx models/before_after.onnx --tiles "lasground/*.laz" --out results/area1 --workers 3
```

Inputs are tiles as `lasground_new` wrote them (quote wildcards; folders work
too, and Windows is handled). Each tile is read with a buffer of neighbour
points (one network tile + 32 m), network tiles lie on one lattice anchored to
the National Grid, and each tile's sampling noise is seeded by its position,
so neighbouring tiles agree exactly where they meet and re-running part of an
area reproduces the same values. Tiles are prepared in parallel while the
network runs; a bad file is reported in `batch_summary.json` and the rest
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

| Overlay | Shows | Transparent below |
|---|---|---|
| `p_edit` | probability that editors change the ground | 0.2 |
| `dz_before` | size of the predicted correction (either sign) | 0.15 m |
| `std` | spread across samples / flips | 0.1 m |

## QGIS plugin (QGIS 3.22 – 4.x)

```bash
python tools/build_qgis_plugin.py        # -> dist/groundiff_qgis.zip
```

Install with *Plugins → Manage and Install Plugins → Install from ZIP*, then
add the runtime to QGIS's Python:

* Windows (OSGeo4W Shell): `python -m pip install numpy scipy "laspy[lazrs]" onnxruntime-directml`
  (any GPU), or replace `onnxruntime-directml` with `"onnxruntime-gpu[cuda,cudnn]"`
  (NVIDIA, CUDA libraries included) or `onnxruntime` (CPU). Install only one
  onnxruntime package.
* macOS: `/Applications/QGIS.app/Contents/MacOS/bin/python3 -m pip install scipy "laspy[lazrs]" onnxruntime`
* pyproj is optional.

Copy `models/before_after.onnx` and `models/before_after.json` together.
The plugin's input is the tiles as they come out of `lasground_new` in your
normal production chain (no extra processing on the PC).
Processing Toolbox → GrounDiff:

* **Predict DTM and edit priorities from point-cloud tiles**: select any
  number of `lasground_new` tiles (… → Add File(s) / Add Directory), an output
  folder and how many tiles to prepare in parallel. Loads the edit probability
  and predicted edit (styled), the DTM and the priority blocks when done. The
  log says which device ONNX Runtime used and warns if a requested GPU was not
  available.
* **Predict DTM from rasters**: channel rasters already on one grid (e.g. from
  `preprocess --geotiff`); outputs the model cannot produce are skipped.
* **Inspect LAS/LAZ file**: the inspector above, optionally comparing two files.

The plugin reads point clouds itself (laspy), so files that QGIS's own
point-cloud layers (PDAL) refuse can still be processed.

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
