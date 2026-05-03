#!/usr/bin/env python3
"""Export training metrics to Datawrapper-ready CSVs.

Reads a TensorBoard events file from a training run and writes one CSV
per chart you might want to make on Datawrapper. Each CSV is in the
exact format Datawrapper expects — one header row, comma-separated, no
ragged rows.

Usage:
    python scripts/datawrapper.py <run_dir> [--out_dir charts/]

Examples:
    # Auto-find the latest run, write CSVs to <run_dir>/datawrapper/
    python scripts/datawrapper.py $(ls -td experiments/groundiff_defra_v10_*/ | head -1)

    # Custom output directory
    python scripts/datawrapper.py experiments/groundiff_defra_v10_260501_235609 \\
        --out_dir charts_v10/

CSV outputs (1 file per chart):
    01_training_loss.csv          Per-iter total loss (line chart)
    02_loss_components.csv         Per-iter L1/L2/grad/BCE (multi-line, log scale)
    03_val_rmse_mae.csv            Per-epoch RMSE+MAE for val and PrioStitch
    04_classification_errors.csv   Per-epoch E_T1/E_T2/E_tot for val + PrioStitch
    05_mcc_miou.csv                Per-epoch MCC and mIoU (val + PrioStitch)
    06_learning_rate.csv           LR schedule
    07_final_epoch_summary.csv     Bar chart: final ep val vs PrioStitch metrics
    08_per_epoch_all_metrics.csv   Wide-format master sheet — every per-epoch metric
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# TB parser (same as chart_training.py — kept self-contained here)
# ============================================================================

def _parse_tb(events_path: Path) -> dict:
    """Parse a TF events file → {tag: (steps[], values[])}.

    Falls back to log parsing if tensorboard isn't installed.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator)
        ea = EventAccumulator(str(events_path),
                              size_guidance={'scalars': 0})
        ea.Reload()
        out = {}
        for tag in ea.Tags().get('scalars', []):
            evs = ea.Scalars(tag)
            steps = np.array([e.step for e in evs], dtype=np.int64)
            vals = np.array([e.value for e in evs], dtype=np.float64)
            wall = np.array([e.wall_time for e in evs], dtype=np.float64)
            out[tag] = (steps, vals, wall)
        return out
    except ImportError:
        return {}


# Log fallback parser (subset — only what's strictly needed for charts)
import re
_RE_TRAIN = re.compile(
    r'ep\s+(\d+)\s+it\s+(\d+)\s+'
    r'loss=([\d.eE+-]+)\s+l1=([\d.eE+-]+)\s+l2=([\d.eE+-]+)\s+'
    r'\S+=([\d.eE+-]+)\s+conf=([\d.eE+-]+)\s+lr=([\d.eE+-]+)'
)
_RE_VAL = re.compile(
    r'val \(n=[\d,]+\):\s*RMSE=([\d.]+)m\s+MAE=([\d.]+)m\s+'
    r'E_T1=\s*([\d.]+)%\s+E_T2=\s*([\d.]+)%\s+E_tot=\s*([\d.]+)%\s+'
    r'MCC=([\d.]+)\s+bal_acc=([\d.]+)\s+mIoU=([\d.]+)')
_RE_PSTITCH = re.compile(
    r'priostitch \(n=[\d,]+\):\s*RMSE=([\d.]+)m\s+MAE=([\d.]+)m\s+'
    r'E_T1=\s*([\d.]+)%\s+E_T2=\s*([\d.]+)%\s+E_tot=\s*([\d.]+)%\s+'
    r'MCC=([\d.]+)\s+bal_acc=([\d.]+)\s+mIoU=([\d.]+)')


def _parse_log(log_path: Path) -> dict:
    """Parse train.log → same {tag: (steps[], values[])} format as _parse_tb,
    minus wall times."""
    from collections import defaultdict
    out = defaultdict(lambda: ([], []))
    last_iter = 0
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = _RE_TRAIN.search(line)
            if m:
                _, it, loss, l1, l2, grad, conf, lr = m.groups()
                it = int(it); last_iter = it
                for tag, val in [
                    ('train/loss', loss), ('train/l1', l1),
                    ('train/l2', l2),     ('train/grad', grad),
                    ('train/conf', conf), ('train/lr', lr),
                ]:
                    out[tag][0].append(it)
                    out[tag][1].append(float(val))
                continue
            m = _RE_VAL.search(line)
            if m:
                rmse, mae, et1, et2, etot, mcc, ba, miou = m.groups()
                # Stored as fractions (0-1) in TB but percentages here
                # — normalised later
                for tag, val in [
                    ('val/rmse', rmse),  ('val/mae', mae),
                    ('val/E_T1', et1),   ('val/E_T2', et2),
                    ('val/E_tot', etot), ('val/MCC', mcc),
                    ('val/bal_acc', ba), ('val/mIoU', miou),
                ]:
                    out[tag][0].append(last_iter)
                    out[tag][1].append(float(val))
                continue
            m = _RE_PSTITCH.search(line)
            if m:
                rmse, mae, et1, et2, etot, mcc, ba, miou = m.groups()
                for tag, val in [
                    ('priostitch/rmse', rmse),  ('priostitch/mae', mae),
                    ('priostitch/E_T1', et1),   ('priostitch/E_T2', et2),
                    ('priostitch/E_tot', etot), ('priostitch/MCC', mcc),
                    ('priostitch/bal_acc', ba), ('priostitch/mIoU', miou),
                ]:
                    out[tag][0].append(last_iter)
                    out[tag][1].append(float(val))
    # Convert: log values for E_T* are in % (0-100); normalise to fraction
    # (0-1) so they're consistent with TB scale
    final = {}
    for k, (s, v) in out.items():
        s = np.asarray(s); v = np.asarray(v)
        if any(x in k for x in ('E_T1', 'E_T2', 'E_tot')) and len(v) > 0:
            if v.max() > 1.5:
                v = v / 100.0
        final[k] = (s, v, np.zeros_like(v))   # no wall times from log
    return final


def _normalise_classification(metrics: dict) -> dict:
    """Ensure E_T1/E_T2/E_tot are stored as fractions 0-1 (TB stores
    them as fractions usually; log parser stores as percentages and
    converts above)."""
    for tag, (s, v, w) in list(metrics.items()):
        if any(x in tag for x in ('E_T1', 'E_T2', 'E_tot')) and len(v) > 0:
            if v.max() > 1.5:
                metrics[tag] = (s, v / 100.0, w)
    return metrics


# ============================================================================
# Helpers — interpolation / lookup / formatting
# ============================================================================

def _ema(x, alpha=0.05):
    """EMA-smooth a 1D array (used for loss curves)."""
    if len(x) == 0:
        return x
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _join_at_iters(metrics: dict, tag: str, target_iters):
    """Return values from `metrics[tag]` aligned to `target_iters`.
    Uses exact match if available, else nearest neighbour. Missing
    → empty string."""
    if tag not in metrics:
        return [''] * len(target_iters)
    src_steps, src_vals, _ = metrics[tag]
    if len(src_steps) == 0:
        return [''] * len(target_iters)
    out = []
    for it in target_iters:
        idx = np.searchsorted(src_steps, it)
        if idx < len(src_steps) and src_steps[idx] == it:
            out.append(src_vals[idx])
        elif idx > 0 and (idx == len(src_steps)
                           or abs(src_steps[idx - 1] - it)
                              < abs(src_steps[idx] - it)):
            out.append(src_vals[idx - 1])
        else:
            out.append(src_vals[idx] if idx < len(src_steps) else '')
    return out


def _epoch_iters(metrics: dict):
    """Return the set of iterations where val metrics were logged
    (these are the per-epoch checkpoints), and a parallel epoch number
    array. Returns (iters, epochs) arrays."""
    if 'val/rmse' not in metrics:
        return np.array([]), np.array([])
    iters, _, _ = metrics['val/rmse']
    epochs = np.arange(1, len(iters) + 1)
    return iters, epochs


def _write_csv(path: Path, fieldnames, rows):
    """Write rows (list of dicts) to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fieldnames})


def _round(v, n=4):
    """Round numeric, pass-through empty/string."""
    if v == '' or v is None:
        return ''
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return v


# ============================================================================
# CSV builders — one per chart
# ============================================================================

def _build_training_loss(metrics, out_dir):
    """01_training_loss.csv — total loss per iter, with EMA-smoothed.

    Datawrapper line chart: iter on X, total + smoothed on Y.
    """
    if 'train/loss' not in metrics:
        return None
    its, vals, _ = metrics['train/loss']
    smoothed = _ema(vals, alpha=0.02)
    rows = [
        {'iteration': int(it),
         'loss_raw':      _round(v, 5),
         'loss_smoothed': _round(s, 5)}
        for it, v, s in zip(its, vals, smoothed)
    ]
    _write_csv(out_dir / '01_training_loss.csv',
                ['iteration', 'loss_raw', 'loss_smoothed'], rows)
    return len(rows)


def _build_loss_components(metrics, out_dir):
    """02_loss_components.csv — L1, L2, gradient, BCE per iter.

    Datawrapper multi-line chart on log Y-axis. EMA-smoothed only
    (raw is too noisy for a multi-series chart).
    """
    components = ['l1', 'l2', 'grad', 'conf']
    available = [c for c in components if f'train/{c}' in metrics]
    if not available:
        return None

    # Use train/loss iterations as the master list (they all share the
    # same iter index since they're logged together)
    its = metrics[f'train/{available[0]}'][0]
    rows = []
    smoothed = {}
    for c in available:
        _, vals, _ = metrics[f'train/{c}']
        smoothed[c] = _ema(vals, alpha=0.02)

    label_map = {'l1': 'L1', 'l2': 'L2', 'grad': 'Gradient', 'conf': 'BCE'}
    fieldnames = ['iteration'] + [label_map[c] for c in available]
    for i, it in enumerate(its):
        row = {'iteration': int(it)}
        for c in available:
            row[label_map[c]] = _round(smoothed[c][i], 5)
        rows.append(row)
    _write_csv(out_dir / '02_loss_components.csv', fieldnames, rows)
    return len(rows)


def _build_val_rmse_mae(metrics, out_dir):
    """03_val_rmse_mae.csv — per-epoch RMSE and MAE for val + PrioStitch.

    Datawrapper multi-line chart: epoch on X, four series on Y
    (val RMSE, val MAE, priostitch RMSE, priostitch MAE).
    """
    iters, epochs = _epoch_iters(metrics)
    if len(iters) == 0:
        return None

    val_rmse  = _join_at_iters(metrics, 'val/rmse', iters)
    val_mae   = _join_at_iters(metrics, 'val/mae', iters)
    pstitch_rmse = _join_at_iters(metrics, 'priostitch/rmse', iters)
    pstitch_mae  = _join_at_iters(metrics, 'priostitch/mae', iters)

    rows = []
    for ep, it, vr, vm, pr, pm in zip(
            epochs, iters, val_rmse, val_mae, pstitch_rmse, pstitch_mae):
        rows.append({
            'epoch': int(ep),
            'iteration': int(it),
            'val RMSE (m)':         _round(vr, 4),
            'val MAE (m)':          _round(vm, 4),
            'PrioStitch RMSE (m)':  _round(pr, 4),
            'PrioStitch MAE (m)':   _round(pm, 4),
        })
    _write_csv(out_dir / '03_val_rmse_mae.csv',
                ['epoch', 'iteration',
                 'val RMSE (m)', 'val MAE (m)',
                 'PrioStitch RMSE (m)', 'PrioStitch MAE (m)'],
                rows)
    return len(rows)


def _build_classification_errors(metrics, out_dir):
    """04_classification_errors.csv — Sithole–Vosselman E_T1/E_T2/E_tot
    over training, for val and PrioStitch.

    All values converted to PERCENT (0-100) for readability.
    """
    iters, epochs = _epoch_iters(metrics)
    if len(iters) == 0:
        return None

    series = [
        ('val/E_T1',          'val E_T1 (%)'),
        ('val/E_T2',          'val E_T2 (%)'),
        ('val/E_tot',         'val E_tot (%)'),
        ('priostitch/E_T1',   'PrioStitch E_T1 (%)'),
        ('priostitch/E_T2',   'PrioStitch E_T2 (%)'),
        ('priostitch/E_tot',  'PrioStitch E_tot (%)'),
    ]

    rows = []
    for i, (ep, it) in enumerate(zip(epochs, iters)):
        row = {'epoch': int(ep), 'iteration': int(it)}
        for tag, label in series:
            v = _join_at_iters(metrics, tag, [it])[0]
            if v != '' and v is not None:
                # All E_* are stored 0-1, convert to %
                try:
                    row[label] = round(float(v) * 100.0, 3)
                except (TypeError, ValueError):
                    row[label] = ''
            else:
                row[label] = ''
        rows.append(row)

    fieldnames = ['epoch', 'iteration'] + [lbl for _, lbl in series]
    _write_csv(out_dir / '04_classification_errors.csv',
                fieldnames, rows)
    return len(rows)


def _build_mcc_miou(metrics, out_dir):
    """05_mcc_miou.csv — MCC and mIoU over time, val + PrioStitch."""
    iters, epochs = _epoch_iters(metrics)
    if len(iters) == 0:
        return None
    series = [
        ('val/MCC',         'val MCC'),
        ('priostitch/MCC',  'PrioStitch MCC'),
        ('val/mIoU',        'val mIoU'),
        ('priostitch/mIoU', 'PrioStitch mIoU'),
    ]
    rows = []
    for ep, it in zip(epochs, iters):
        row = {'epoch': int(ep), 'iteration': int(it)}
        for tag, label in series:
            v = _join_at_iters(metrics, tag, [it])[0]
            row[label] = _round(v, 4)
        rows.append(row)
    fieldnames = ['epoch', 'iteration'] + [lbl for _, lbl in series]
    _write_csv(out_dir / '05_mcc_miou.csv', fieldnames, rows)
    return len(rows)


def _build_lr_schedule(metrics, out_dir):
    """06_learning_rate.csv — LR schedule (cosine decay typically)."""
    if 'train/lr' not in metrics:
        return None
    its, vals, _ = metrics['train/lr']
    # Subsample if very dense (every 100th iter is plenty for a chart)
    if len(its) > 1000:
        idx = np.linspace(0, len(its) - 1, 1000, dtype=int)
        its, vals = its[idx], vals[idx]
    rows = [
        {'iteration': int(it),
         'learning_rate': float(f"{v:.4e}")}
        for it, v in zip(its, vals)
    ]
    _write_csv(out_dir / '06_learning_rate.csv',
                ['iteration', 'learning_rate'], rows)
    return len(rows)


def _build_final_summary(metrics, out_dir):
    """07_final_epoch_summary.csv — bar chart comparing val vs PrioStitch
    on the final-epoch checkpoint.

    Datawrapper grouped bar chart: rows = metric, columns = val / pstitch.
    """
    iters, epochs = _epoch_iters(metrics)
    if len(iters) == 0:
        return None
    last_it = iters[-1]
    last_ep = int(epochs[-1])

    spec = [
        # (metric label, val_tag, pstitch_tag, units, scale-to-pct)
        ('RMSE',  'val/rmse',  'priostitch/rmse',  'm',  False),
        ('MAE',   'val/mae',   'priostitch/mae',   'm',  False),
        ('E_T1',  'val/E_T1',  'priostitch/E_T1',  '%',  True),
        ('E_T2',  'val/E_T2',  'priostitch/E_T2',  '%',  True),
        ('E_tot', 'val/E_tot', 'priostitch/E_tot', '%',  True),
        ('MCC',   'val/MCC',   'priostitch/MCC',   '',   False),
        ('mIoU',  'val/mIoU',  'priostitch/mIoU',  '',   False),
    ]

    rows = []
    for label, vt, pt, unit, to_pct in spec:
        v = _join_at_iters(metrics, vt, [last_it])[0]
        p = _join_at_iters(metrics, pt, [last_it])[0]
        if to_pct:
            v = float(v) * 100.0 if v != '' else ''
            p = float(p) * 100.0 if p != '' else ''
        label_full = f"{label} ({unit})" if unit else label
        rows.append({
            'metric': label_full,
            'Per-tile val':   _round(v, 3),
            'PrioStitch':     _round(p, 3),
        })
    _write_csv(out_dir / '07_final_epoch_summary.csv',
                ['metric', 'Per-tile val', 'PrioStitch'], rows)
    return len(rows), last_ep


def _build_master_sheet(metrics, out_dir):
    """08_per_epoch_all_metrics.csv — wide-format master sheet with EVERY
    per-epoch metric in one place. Useful for ad-hoc filtering / further
    analysis in pandas / Excel without re-running the parser."""
    iters, epochs = _epoch_iters(metrics)
    if len(iters) == 0:
        return None

    spec = [
        ('val/rmse',         'val RMSE (m)',          False),
        ('val/mae',          'val MAE (m)',           False),
        ('val/E_T1',         'val E_T1 (%)',          True),
        ('val/E_T2',         'val E_T2 (%)',          True),
        ('val/E_tot',        'val E_tot (%)',         True),
        ('val/MCC',          'val MCC',               False),
        ('val/bal_acc',      'val balanced acc',      False),
        ('val/mIoU',         'val mIoU',              False),
        ('priostitch/rmse',  'PrioStitch RMSE (m)',   False),
        ('priostitch/mae',   'PrioStitch MAE (m)',    False),
        ('priostitch/E_T1',  'PrioStitch E_T1 (%)',   True),
        ('priostitch/E_T2',  'PrioStitch E_T2 (%)',   True),
        ('priostitch/E_tot', 'PrioStitch E_tot (%)',  True),
        ('priostitch/MCC',   'PrioStitch MCC',        False),
        ('priostitch/bal_acc', 'PrioStitch balanced acc', False),
        ('priostitch/mIoU',  'PrioStitch mIoU',       False),
    ]

    rows = []
    for ep, it in zip(epochs, iters):
        row = {'epoch': int(ep), 'iteration': int(it)}
        for tag, label, to_pct in spec:
            v = _join_at_iters(metrics, tag, [it])[0]
            if v == '' or v is None:
                row[label] = ''
            else:
                try:
                    fv = float(v)
                    if to_pct:
                        fv *= 100.0
                    row[label] = round(fv, 4)
                except (TypeError, ValueError):
                    row[label] = ''
        rows.append(row)
    fieldnames = ['epoch', 'iteration'] + [lbl for _, lbl, _ in spec]
    _write_csv(out_dir / '08_per_epoch_all_metrics.csv',
                fieldnames, rows)
    return len(rows)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir', help="Experiment directory.")
    p.add_argument('--out_dir', default=None,
                   help="Output directory for CSVs. Defaults to "
                        "<run_dir>/datawrapper/.")
    p.add_argument('--source', choices=['auto', 'tb', 'log'],
                   default='auto')
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"error: {run_dir} not found", file=sys.stderr)
        return 1
    out_dir = (Path(args.out_dir) if args.out_dir
                else run_dir / 'datawrapper')

    metrics = {}
    if args.source in ('auto', 'tb'):
        tbs = sorted(run_dir.glob('tb/events.out.tfevents.*'))
        if tbs:
            print(f"reading TB events: {tbs[-1]}")
            metrics = _parse_tb(tbs[-1])
            if not metrics:
                print("  (TB parse empty — install tensorboard for "
                      "denser per-iter data)")
        else:
            print(f"no TB events under {run_dir}/tb/")

    has_train = any(k.startswith('train/') for k in metrics)
    if args.source == 'log' or (args.source == 'auto' and not has_train):
        log = run_dir / 'train.log'
        if log.exists():
            print(f"reading log: {log}")
            log_metrics = _parse_log(log)
            for k, v in log_metrics.items():
                metrics.setdefault(k, v)
        elif not metrics:
            print(f"error: no train.log at {log}", file=sys.stderr)
            return 1

    if not metrics:
        print("error: no metrics parsed from any source", file=sys.stderr)
        return 1

    metrics = _normalise_classification(metrics)
    print(f"parsed {len(metrics)} series\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing CSVs to {out_dir}/\n")

    results = []
    n = _build_training_loss(metrics, out_dir)
    if n:
        results.append(('01_training_loss.csv', n,
                        'Line chart — iter vs total loss'))
    n = _build_loss_components(metrics, out_dir)
    if n:
        results.append(('02_loss_components.csv', n,
                        'Multi-line log chart — L1, L2, grad, BCE'))
    n = _build_val_rmse_mae(metrics, out_dir)
    if n:
        results.append(('03_val_rmse_mae.csv', n,
                        'Multi-line chart — val + PrioStitch RMSE/MAE'))
    n = _build_classification_errors(metrics, out_dir)
    if n:
        results.append(('04_classification_errors.csv', n,
                        'Multi-line chart — Sithole-Vosselman errors (%)'))
    n = _build_mcc_miou(metrics, out_dir)
    if n:
        results.append(('05_mcc_miou.csv', n,
                        'Multi-line chart — MCC and mIoU'))
    n = _build_lr_schedule(metrics, out_dir)
    if n:
        results.append(('06_learning_rate.csv', n,
                        'Line chart — learning rate schedule'))
    final = _build_final_summary(metrics, out_dir)
    if final:
        n_rows, ep = final
        results.append(('07_final_epoch_summary.csv', n_rows,
                        f'Grouped bar chart — final epoch ({ep}) '
                        f'val vs PrioStitch'))
    n = _build_master_sheet(metrics, out_dir)
    if n:
        results.append(('08_per_epoch_all_metrics.csv', n,
                        'Master sheet — every per-epoch metric'))

    print("Wrote:")
    for path, nrows, desc in results:
        print(f"  {path:<40s}  {nrows:>5d} rows   {desc}")
    print(f"\nUpload any of these CSVs directly to https://datawrapper.de/.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
