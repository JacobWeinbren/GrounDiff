"""Which compute device runs the model fastest here, with the same results?

    python -m groundiff.speedtest models/no_lastools.onnx

Runs one batch of the network on every device ONNX Runtime offers (CPU,
Apple GPU through CoreML, NVIDIA CUDA, DirectML), each in its own process so
a device that misbehaves (runs out of memory, hangs) is stopped without
harm. For each: time per 256 px tile per diffusion step (after a warm-up
that includes compiling), peak memory, and the largest difference from the
CPU's output on the same input. The fastest device whose output matches
the CPU is remembered for this model (~/.groundiff/devices.json) and used by
"Auto" in the QGIS plugin and groundiff.batch. Also in QGIS: Processing >
GrounDiff > Test compute devices.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# largest accepted difference from the CPU, in the network's output units (-1..1 over a tile's
# height range): 1e-3 is 1 cm on a tile spanning 20 m
TOLERANCE = 1e-3
TILES_5KM = 1600            # network tiles for a 5 km tile at 50 % overlap


def _state_file() -> Path:
    return Path(os.environ.get("GROUNDIFF_CACHE", Path.home() / ".groundiff")) / "devices.json"


def _model_key(model) -> str:
    p = Path(model).resolve()
    st = p.stat()
    return f"{p}|{st.st_size}|{int(st.st_mtime)}"


def best_setting(model) -> tuple[str | None, int | None]:
    """(device, batch) the last speed test chose for this model file ((None, None) if never tested
    or the file changed since)."""
    try:
        state = json.loads(_state_file().read_text()).get(_model_key(model), {})
        return state.get("best"), state.get("best_batch")
    except Exception:
        return None, None


def best_device(model) -> str | None:
    return best_setting(model)[0]


def available_devices() -> list[str]:
    import onnxruntime as ort
    from .backends import DEVICE_PROVIDERS
    avail = ort.get_available_providers()
    return [d for d in ("cpu", "coreml", "coreml_fp16acc", "coreml_ane", "cuda", "directml")
            if DEVICE_PROVIDERS[d][0] in avail]


def bench_one(model: str, device: str, batch: int, reps: int, out: str) -> dict:
    """Runs in the child process."""
    from .backends import OnnxNet
    t0 = time.time()
    net = OnnxNet(model, device, batch=batch)
    load_s = time.time() - t0
    shp = net.session.get_inputs()[0].shape
    C, T = int(shp[1]), int(shp[2])
    rng = np.random.default_rng(0)
    x = np.clip(rng.standard_normal((batch, C, T, T)), -1, 1).astype(np.float32)
    g = (0.05 + 0.9 * (np.arange(batch) % 8) / 7).astype(np.float32)     # same first tiles at any batch size
    t0 = time.time()
    y = net(x, g)                                      # warm-up (CoreML compiles here the first time)
    warm_s = time.time() - t0
    times = []
    for _ in range(reps):
        t0 = time.time()
        y = net(x, g)
        times.append(time.time() - t0)
    np.save(out, y)
    return {"device": device, "providers": net.providers, "warning": net.warning, "load_s": round(load_s, 1),
            "warmup_s": round(warm_s, 1), "s_per_tile_step": min(times) / batch}


def _rss_gb(pid: int) -> float:
    try:
        if sys.platform.startswith("win"):
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True,
                               text=True, timeout=10, creationflags=0x08000000)
            kb = r.stdout.strip().split('","')[-1].replace('"', "").replace(",", "").replace(".", "").split()[0]
            return int(kb) / 1e6
        r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip() or 0) / 1e6
    except Exception:
        return 0.0


def _child_cmd(python: str, args: list) -> tuple[list, dict]:
    # groundiff.speedtest or groundiff_qgis.core.speedtest (also when run with -m)
    mod = __spec__.name if __name__ == "__main__" and __spec__ else __name__
    root = Path(__file__).resolve().parents[len(mod.split(".")) - 1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root)] + [p for p in sys.path if p])
    env.pop("PYTHONHOME", None) if sys.platform == "darwin" else None
    return [python, "-m", mod] + args, env


def run(model: str, devices: list | None = None, batch: int | list = 8, reps: int = 3, max_gb: float = 12.0,
        timeout_s: float = 1800.0, python: str | None = None, log=print, cancelled=lambda: False) -> list[dict]:
    model = str(Path(model).resolve())
    devices = devices or available_devices()
    if "cpu" in devices:
        devices = ["cpu"] + [d for d in devices if d != "cpu"]       # the reference first
    else:
        devices = ["cpu"] + list(devices)
    python = python or sys.executable
    batches = [int(b) for b in (batch if isinstance(batch, (list, tuple)) else [batch])]
    results, ref = [], None
    tmp = Path(tempfile.mkdtemp(prefix="groundiff_speed_"))
    # the CPU once (reference, first batch size); accelerators at every batch size
    plan = [("cpu", batches[0])] + [(d, b) for d in devices if d != "cpu" for b in batches]
    for dev, bsz in plan:
        out = tmp / f"{dev}_{bsz}.npy"
        cmd, env = _child_cmd(python, ["--one", dev, "--batch", str(bsz), "--reps", str(reps),
                                       "--out", str(out), model])
        log(f"{dev}, batch {bsz}: testing (a CoreML setting compiles the model the first time, a minute or two) ...")
        flags = 0x08000000 if sys.platform.startswith("win") else 0
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                                creationflags=flags)
        peak, t0, stopped = 0.0, time.time(), None
        while proc.poll() is None:
            peak = max(peak, _rss_gb(proc.pid))
            if peak > max_gb:
                stopped = f"stopped: used more than {max_gb:g} GB"
            elif time.time() - t0 > timeout_s:
                stopped = f"stopped: no result after {timeout_s / 60:.0f} min"
            elif cancelled():
                stopped = "cancelled"
            if stopped:
                proc.kill()
                break
            time.sleep(0.5)
        so, se = proc.communicate()
        r = {"device": dev, "batch": bsz, "peak_gb": round(peak, 1)}
        if stopped or proc.returncode != 0:
            r.update(ok=False, error=stopped or (se.strip().splitlines() or ["failed"])[-1])
        else:
            r.update(json.loads(so.strip().splitlines()[-1]))
            y = np.load(out)
            if dev == "cpu":
                ref = y
            if ref is not None:
                k = min(len(y), len(ref))                   # same seeded input: the first k tiles agree
                d = float(np.abs(y[:k] - ref[:k]).max())
                r["max_diff"] = d
                r["ok"] = d <= TOLERANCE * max(1.0, float(np.abs(ref).max()))
                if not r["ok"]:
                    r["error"] = f"output differs from the CPU by {d:.2e} (limit {TOLERANCE:g})"
            else:
                r["ok"] = False
                r["error"] = "no CPU reference"
            if r.get("warning"):
                r["ok"] = False
                r["error"] = r["warning"]
        results.append(r)
        log(_line(r))
        if cancelled():
            break
    good = [r for r in results if r.get("ok")]
    bestr = min(good, key=lambda r: r["s_per_tile_step"]) if good else None
    best = bestr["device"] if bestr else None
    try:
        f = _state_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        state = json.loads(f.read_text()) if f.exists() else {}
        state[_model_key(model)] = {"best": best, "best_batch": bestr["batch"] if bestr else None,
                                    "results": results, "time": time.time()}
        f.write_text(json.dumps(state, indent=1))
    except Exception:
        pass
    if best:
        log(f"Fastest with the same results: {best}, batch {bestr['batch']} "
            f"(~{bestr['s_per_tile_step'] * TILES_5KM * 10 / 60:.0f} min per 5 km tile at 1 sample). 'Auto' now "
            "uses it for this model (with the batch size unless you set another).")
    return results


def _line(r: dict) -> str:
    name = f"{r['device']}, batch {r.get('batch')}"
    if not r.get("s_per_tile_step"):
        return f"{name}: not usable ({r.get('error')}); peak {r.get('peak_gb')} GB"
    est = r["s_per_tile_step"] * TILES_5KM * 10 / 60
    acc = f"max difference from the CPU {r['max_diff']:.1e}" if "max_diff" in r else ""
    flag = "" if r.get("ok") else f" - NOT USED: {r.get('error')}"
    return (f"{name}: {r['s_per_tile_step']:.3f} s per tile-step (~{est:.0f} min per 5 km tile at 1 sample), "
            f"peak {r['peak_gb']} GB, compile/warm-up {r.get('warmup_s')} s, {acc}{flag}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the exported .onnx")
    ap.add_argument("--devices", nargs="+", help="default: every device ONNX Runtime offers here")
    ap.add_argument("--batch", type=int, nargs="+", default=[8, 16, 32],
                    help="batch sizes to try on the accelerators (the CPU reference uses the first)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-gb", type=float, default=12.0, help="stop a device that uses more memory than this")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.one:
        print(json.dumps(bench_one(a.model, a.one, a.batch[0], a.reps, a.out)))
        return 0
    res = run(a.model, a.devices, a.batch, a.reps, a.max_gb)
    return 0 if any(r.get("ok") for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
