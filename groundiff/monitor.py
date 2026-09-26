"""Watch a training run.

    python -m groundiff.monitor runs/before_after            # one snapshot
    python -m groundiff.monitor runs/before_after --follow   # refresh every 30 s (Ctrl-C to stop)
    python -m groundiff.monitor runs/before_after --csv curves.csv

Reads <run>/log.jsonl written by groundiff.train: progress, time left, recent
losses, memory, and the latest validation of the model vs lasground_new.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

BARS = "▁▂▃▄▅▆▇█"


def read_log(run: Path) -> list[dict]:
    path = run / "log.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue                      # a line being written right now
    return events


def sparkline(values: list[float], width: int = 40) -> str:
    vals = [v for v in values if v == v][-width:]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1.0
    return "".join(BARS[min(len(BARS) - 1, int((v - lo) / span * (len(BARS) - 1)))] for v in vals)


def _fmt_time(sec: float) -> str:
    if sec != sec or sec < 0:
        return "?"
    h, m = divmod(int(sec) // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def summary(run: Path) -> str:
    events = read_log(run)
    if not events:
        return f"{run}: no log.jsonl yet"
    cfg = {}
    if (run / "config.json").exists():
        cfg = json.loads((run / "config.json").read_text())
    total = cfg.get("optim", {}).get("total_steps")
    start = next((e for e in events if e.get("event") == "start"), {})
    train = [e for e in events if e.get("event") == "train"]
    vals = [e for e in events if e.get("event") == "val"]
    lines = [f"run: {run}   device: {start.get('device', '?')}   params: {start.get('params_M', 0):.1f}M"]
    if train:
        last = train[-1]
        step = last["step"]
        recent = [t.get("sec_per_step", float("nan")) for t in train[-5:]]
        sps = sum(recent) / len(recent)
        left = (total - step) * sps if total else float("nan")
        pct = f"{100 * step / total:.1f}%" if total else "?"
        lines.append(f"step {step}/{total or '?'} ({pct})   {sps:.2f} s/step   time left ≈ {_fmt_time(left)}"
                     f"   lr {last.get('lr', 0):.2e}")
        mem = {k: v for k, v in last.items() if k.endswith("_gb")}
        if mem:
            lines.append("memory: " + ", ".join(f"{k} {v:.1f} GB" for k, v in mem.items()))
        losses = [t.get("loss", float("nan")) for t in train]
        lines.append(f"loss {last.get('loss', float('nan')):.4f}  {sparkline(losses)}")
        parts = [f"{k} {last[k]:.4f}" for k in ("l1", "l2", "grad", "conf") if k in last]
        if parts:
            lines.append("   " + "  ".join(parts))
        age = time.time() - (run / "log.jsonl").stat().st_mtime
        if age > max(600, 20 * sps * cfg.get("train", {}).get("log_every", 50)):
            lines.append(f"WARNING: log not updated for {_fmt_time(age)} (stopped? Mac asleep? use caffeinate)")
    if vals:
        v = vals[-1]
        lines.append(f"validation @ step {v['step']}:")
        for name in ("raw", "ema"):
            if name in v:
                m, b = v[name].get("model", {}), v[name].get("lasground_new", {})
                s = f"  {name:3s} model RMSE {m.get('rmse', float('nan')):.3f} m  MAE {m.get('mae', float('nan')):.3f} m"
                if "type1_pct" in m:
                    s += f"  typeI {m['type1_pct']:.1f}%  typeII {m['type2_pct']:.1f}%"
                if b:
                    s += f"   | lasground_new RMSE {b.get('rmse', float('nan')):.3f} m"
                lines.append(s)
        rmses = [min(e[n]["model"].get("rmse", float("inf")) for n in ("raw", "ema") if n in e) for e in vals]
        lines.append(f"val RMSE history {sparkline(rmses)}  best {min(rmses):.3f} m")
    if (run / "best.pt").exists():
        lines.append(f"best.pt updated {_fmt_time(time.time() - (run / 'best.pt').stat().st_mtime)} ago")
    done = next((e for e in reversed(events) if e.get("event") == "done"), None)
    if done:
        lines.append(f"DONE at step {done['step']}: best {done.get('best')}")
    return "\n".join(lines)


def to_csv(run: Path, out: Path):
    rows = [e for e in read_log(run) if e.get("event") == "train"]
    keys = sorted({k for r in rows for k in r if not isinstance(r[k], dict)})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r.get(k) for k in keys} for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--every", type=float, default=30.0)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args(argv)
    if a.csv:
        to_csv(a.run, a.csv)
        print(f"wrote {a.csv}")
        return 0
    if not a.follow:
        print(summary(a.run))
        return 0
    try:
        while True:
            print("\033[2J\033[H" + summary(a.run), flush=True)
            time.sleep(a.every)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
