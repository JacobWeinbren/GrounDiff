#!/usr/bin/env python3
"""Chart all major training metrics over time.

Reads either a TensorBoard events file (preferred — has the per-iter
loss curves) or a train.log (parses the human-readable INFO lines)
and produces a single multi-panel PNG with the full trajectory.

Usage:
  python scripts/chart_training.py <run_dir> [--out chart.png]

  <run_dir> is the experiment directory, e.g.
    experiments/groundiff_defra_v10_260501_235609/

Examples:
  # Auto-discover and chart latest run
  python scripts/chart_training.py $(ls -td experiments/groundiff_defra_v10_*/ | head -1)

  # Specific run, custom output
  python scripts/chart_training.py experiments/groundiff_defra_v10_260501_235609 \\
      --out runs/chart.png

  # Force log-parse path (skip TB)
  python scripts/chart_training.py <run_dir> --source log
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


# ---- TensorBoard parser ---------------------------------------------------

def _parse_tb(events_path: Path) -> dict:
    """Parse a TF events file into {tag: (steps[], values[])}.

    Tries tensorboard's EventAccumulator if installed (richer); falls
    back to a minimal protobuf reader if not.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator)
        ea = EventAccumulator(str(events_path),
                              size_guidance={'scalars': 0})  # 0 = all
        ea.Reload()
        out = {}
        for tag in ea.Tags().get('scalars', []):
            evs = ea.Scalars(tag)
            steps = np.array([e.step for e in evs], dtype=np.int64)
            vals = np.array([e.value for e in evs], dtype=np.float64)
            out[tag] = (steps, vals)
        return out
    except ImportError:
        pass
    # Fallback: use tensorflow-less event reading via crc32c-skipped
    # protobuf scanning. tf is heavy; we'd rather pip install tensorboard.
    print("tensorboard not installed — install with "
          "'pip install tensorboard' for TF events parsing.",
          file=sys.stderr)
    return {}


# ---- train.log parser ----------------------------------------------------

# "ep   20  it   100800  loss=0.0346  l1=0.0214  l2=0.0021  ∇=0.0299  conf=0.0812  lr=9.66e-05"
# (∇ may also appear escaped; regex permits Unicode)
_RE_TRAIN = re.compile(
    r'ep\s+(\d+)\s+it\s+(\d+)\s+'
    r'loss=([\d.eE+-]+)\s+'
    r'l1=([\d.eE+-]+)\s+'
    r'l2=([\d.eE+-]+)\s+'
    r'\S+=([\d.eE+-]+)\s+'                 # ∇=...
    r'conf=([\d.eE+-]+)\s+'
    r'lr=([\d.eE+-]+)'
)
# "val (n=8,610,496): RMSE=0.410m  MAE=0.151m  E_T1= 6.74%  E_T2= 1.64%  E_tot= 3.24%  MCC=0.924  bal_acc=0.958  mIoU=0.927 ..."
_RE_VAL = re.compile(
    r'val \(n=([\d,]+)\):\s*'
    r'RMSE=([\d.]+)m\s+'
    r'MAE=([\d.]+)m\s+'
    r'E_T1=\s*([\d.]+)%\s+'
    r'E_T2=\s*([\d.]+)%\s+'
    r'E_tot=\s*([\d.]+)%\s+'
    r'MCC=([\d.]+)\s+'
    r'bal_acc=([\d.]+)\s+'
    r'mIoU=([\d.]+)'
)
_RE_PSTITCH = re.compile(
    r'priostitch \(n=([\d,]+)\):\s*'
    r'RMSE=([\d.]+)m\s+'
    r'MAE=([\d.]+)m\s+'
    r'E_T1=\s*([\d.]+)%\s+'
    r'E_T2=\s*([\d.]+)%\s+'
    r'E_tot=\s*([\d.]+)%\s+'
    r'MCC=([\d.]+)\s+'
    r'bal_acc=([\d.]+)\s+'
    r'mIoU=([\d.]+)'
)
_RE_EPOCH_DONE = re.compile(r'epoch\s+(\d+)\s+done\s+in\s+(\d+)s')


def _parse_log(log_path: Path) -> dict:
    """Parse train.log into the same dict format as _parse_tb.

    Returned tags (selection):
      train/loss, train/l1, train/l2, train/grad, train/conf, train/lr   (per-iter)
      val/rmse, val/mae, val/E_T1, val/E_T2, val/MCC, val/mIoU            (per-epoch)
      priostitch/rmse, ... (per-epoch)
    """
    out = defaultdict(lambda: ([], []))
    cur_epoch = 1
    last_iter = 0
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = _RE_TRAIN.search(line)
            if m:
                ep, it, loss, l1, l2, grad, conf, lr = m.groups()
                ep, it = int(ep), int(it)
                cur_epoch = ep
                last_iter = it
                out['train/loss'][0].append(it); out['train/loss'][1].append(float(loss))
                out['train/l1'][0].append(it);   out['train/l1'][1].append(float(l1))
                out['train/l2'][0].append(it);   out['train/l2'][1].append(float(l2))
                out['train/grad'][0].append(it); out['train/grad'][1].append(float(grad))
                out['train/conf'][0].append(it); out['train/conf'][1].append(float(conf))
                out['train/lr'][0].append(it);   out['train/lr'][1].append(float(lr))
                continue
            m = _RE_VAL.search(line)
            if m:
                _, rmse, mae, et1, et2, etot, mcc, ba, miou = m.groups()
                # Use last_iter as x; epoch boundary is recorded as the
                # iteration we'd just finished.
                x = last_iter
                out['val/rmse'][0].append(x);  out['val/rmse'][1].append(float(rmse))
                out['val/mae'][0].append(x);   out['val/mae'][1].append(float(mae))
                out['val/E_T1'][0].append(x);  out['val/E_T1'][1].append(float(et1))
                out['val/E_T2'][0].append(x);  out['val/E_T2'][1].append(float(et2))
                out['val/E_tot'][0].append(x); out['val/E_tot'][1].append(float(etot))
                out['val/MCC'][0].append(x);   out['val/MCC'][1].append(float(mcc))
                out['val/mIoU'][0].append(x);  out['val/mIoU'][1].append(float(miou))
                continue
            m = _RE_PSTITCH.search(line)
            if m:
                _, rmse, mae, et1, et2, etot, mcc, ba, miou = m.groups()
                x = last_iter
                out['priostitch/rmse'][0].append(x);  out['priostitch/rmse'][1].append(float(rmse))
                out['priostitch/mae'][0].append(x);   out['priostitch/mae'][1].append(float(mae))
                out['priostitch/E_T1'][0].append(x);  out['priostitch/E_T1'][1].append(float(et1))
                out['priostitch/E_T2'][0].append(x);  out['priostitch/E_T2'][1].append(float(et2))
                out['priostitch/E_tot'][0].append(x); out['priostitch/E_tot'][1].append(float(etot))
                out['priostitch/MCC'][0].append(x);   out['priostitch/MCC'][1].append(float(mcc))
                out['priostitch/mIoU'][0].append(x);  out['priostitch/mIoU'][1].append(float(miou))
    # Convert to arrays
    return {k: (np.asarray(s), np.asarray(v)) for k, (s, v) in out.items()}


# ---- Plotting ------------------------------------------------------------

def _ema(x, alpha=0.05):
    """Exponential moving average for noisy per-iter curves."""
    if len(x) == 0:
        return x
    out = np.zeros_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _plot(metrics: dict, out_path: Path, title: str = ''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    preferred = ['Inter', 'Helvetica Neue', 'Helvetica', 'Arial',
                 'DejaVu Sans']
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in preferred if f in available), 'DejaVu Sans')
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   [chosen],
        'figure.facecolor':  'white',
        'axes.facecolor':    'white',
        'axes.edgecolor':    '#333333',
        'axes.linewidth':    0.6,
        'axes.titleweight':  'semibold',
        'axes.titlesize':    12,
        'axes.labelsize':    10,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'xtick.color':       '#444444',
        'ytick.color':       '#444444',
        'legend.frameon':    False,
        'legend.fontsize':   9,
        'grid.color':        '#eeeeee',
        'grid.linewidth':    0.6,
    })

    # 4×2 grid of panels
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.05,
                         hspace=0.35, wspace=0.20)
    if title:
        fig.suptitle(title, fontsize=15, weight='semibold',
                      y=0.985, color='#111111')

    def panel(ax, title_, ylabel, items, ylog=False, ypct=False):
        ax.set_title(title_)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.6)
        for label, key, color, kind in items:
            if key not in metrics:
                continue
            x, y = metrics[key]
            if len(x) == 0:
                continue
            if kind == 'iter':
                ax.plot(x, y, color=color, alpha=0.18, linewidth=0.6,
                         zorder=1)
                ax.plot(x, _ema(y, 0.02), color=color, linewidth=1.6,
                         label=label, zorder=2)
            else:
                ax.plot(x, y, color=color, marker='o', markersize=4,
                         linewidth=1.4, label=label, zorder=3)
        if ylog:
            ax.set_yscale('log')
        if ypct:
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f'{v:g}%'))
        ax.legend(loc='best')

    # 1. Total training loss
    panel(axes[0, 0], 'Training loss', 'loss',
           [('total loss', 'train/loss', '#1f6feb', 'iter')])

    # 2. Component losses
    panel(axes[0, 1], 'Loss components', 'magnitude',
           [('L1',     'train/l1',   '#0969da', 'iter'),
            ('L2',     'train/l2',   '#bf8700', 'iter'),
            ('grad',   'train/grad', '#1a7f37', 'iter'),
            ('BCE',    'train/conf', '#cf222e', 'iter')],
           ylog=True)

    # 3. Validation RMSE / MAE (per-tile + priostitch)
    panel(axes[1, 0], 'RMSE & MAE', 'metres',
           [('val RMSE',        'val/rmse',          '#1f6feb', 'epoch'),
            ('val MAE',         'val/mae',           '#0969da', 'epoch'),
            ('priostitch RMSE', 'priostitch/rmse',   '#cf222e', 'epoch'),
            ('priostitch MAE',  'priostitch/mae',    '#a40e26', 'epoch')])

    # 4. Sithole-Vosselman E_T1 / E_T2 / E_tot
    panel(axes[1, 1], 'Sithole–Vosselman classification errors',
           'percent',
           [('val E_T1',         'val/E_T1',          '#1f6feb', 'epoch'),
            ('val E_T2',         'val/E_T2',          '#0969da', 'epoch'),
            ('val E_tot',        'val/E_tot',         '#0a3069', 'epoch'),
            ('pstitch E_T1',     'priostitch/E_T1',   '#cf222e', 'epoch'),
            ('pstitch E_T2',     'priostitch/E_T2',   '#a40e26', 'epoch'),
            ('pstitch E_tot',    'priostitch/E_tot',  '#660018', 'epoch')],
           ypct=True)

    # 5. MCC
    panel(axes[2, 0], 'MCC (Matthews correlation)', 'MCC',
           [('val MCC',        'val/MCC',         '#1f6feb', 'epoch'),
            ('priostitch MCC', 'priostitch/MCC',  '#cf222e', 'epoch')])

    # 6. mIoU
    panel(axes[2, 1], 'mIoU (mean intersection-over-union)', 'mIoU',
           [('val mIoU',        'val/mIoU',         '#1f6feb', 'epoch'),
            ('priostitch mIoU', 'priostitch/mIoU',  '#cf222e', 'epoch')])

    # 7. Learning rate
    panel(axes[3, 0], 'Learning rate', 'lr',
           [('lr', 'train/lr', '#6e7781', 'iter')])

    # 8. Conf BCE alone (zoomed view of cls training)
    panel(axes[3, 1], 'Classification BCE loss (zoomed)', 'BCE',
           [('train conf', 'train/conf', '#cf222e', 'iter')])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ---- Main ----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir', help="Experiment directory, e.g. "
                                    "experiments/groundiff_defra_v10_*/")
    p.add_argument('--out', default=None,
                   help="Output PNG path. Defaults to "
                        "<run_dir>/training_chart.png")
    p.add_argument('--source', choices=['auto', 'tb', 'log'],
                   default='auto',
                   help="Where to read metrics from. 'auto' tries TB, "
                        "falls back to log.")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"error: run_dir not found: {run_dir}", file=sys.stderr)
        return 1
    out_path = Path(args.out) if args.out else run_dir / 'training_chart.png'

    metrics = {}
    if args.source in ('auto', 'tb'):
        # Find the TB events file
        tb_files = sorted(run_dir.glob('tb/events.out.tfevents.*'))
        if tb_files:
            print(f"reading TB events: {tb_files[-1]}")
            metrics = _parse_tb(tb_files[-1])
            if not metrics:
                print("  TB parse returned no scalars")
        else:
            print(f"no TB events file under {run_dir}/tb/")

    # If TB had nothing useful (e.g. tensorboard not installed), fall
    # back to log parsing — it's slightly less detailed but always works.
    has_train_curves = any(k.startswith('train/') for k in metrics)
    if args.source == 'log' or (args.source == 'auto' and not has_train_curves):
        log = run_dir / 'train.log'
        if not log.exists():
            print(f"error: no train.log at {log}", file=sys.stderr)
            return 1
        print(f"reading log: {log}")
        log_metrics = _parse_log(log)
        # Merge: TB keys win where present, log fills gaps
        for k, v in log_metrics.items():
            metrics.setdefault(k, v)

    if not metrics:
        print("error: no metrics parsed", file=sys.stderr)
        return 1

    # Normalise classification-error scales. Log-parsed values are
    # percentages (0-100); TB-stored values are usually fractions (0-1).
    # Convert any fractional-scale series to percent for consistent
    # axis labelling.
    for tag in list(metrics.keys()):
        if any(s in tag for s in ('E_T1', 'E_T2', 'E_tot',
                                    'err_gt', 'errgt')):
            steps, vals = metrics[tag]
            if len(vals) > 0 and vals.max() <= 1.5:  # fractional scale
                metrics[tag] = (steps, vals * 100.0)

    print(f"parsed {len(metrics)} series:")
    for k, (s, v) in sorted(metrics.items()):
        print(f"  {k:24s}  n={len(s):>6d}  range={v.min():.4g} → {v.max():.4g}")

    title = run_dir.name
    _plot(metrics, out_path, title=title)
    print(f"\nwrote: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
