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
    if device.startswith("coreml") or os.environ.get("GROUNDIFF_ORT_VERBOSE"):
        import onnxruntime as ort
        ort.set_default_logger_severity(0)        # verbose: which operators CoreML takes (parsed by the parent)
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
    steps = 10                                         # network passes per tile: the diffusion steps
    try:
        spec = json.loads(net.session.get_modelmeta().custom_metadata_map.get("groundiff_spec") or "{}")
        if spec.get("kind", "groundiff") == "groundiff":
            if spec.get("one_step"):
                steps = 1
            elif spec.get("process") == "rdbm":
                steps = int(spec.get("bridge_steps", 10))
            else:
                steps = int(spec.get("T", 10))
        else:
            steps = 1
    except Exception:
        pass
    return {"device": device, "steps": steps, "providers": net.providers, "warning": net.warning,
            "load_s": round(load_s, 1),
            "warmup_s": round(warm_s, 1), "s_per_tile_step": min(times) / batch}


def _coreml_report(stderr: str) -> dict:
    """Partitions and the operator types CoreML did not take, from ONNX Runtime's verbose log."""
    import re
    out = {}
    m = re.findall(r"number of partitions supported by CoreML: (\d+).*?number of nodes in the graph: (\d+)"
                   r".*?number of nodes supported by CoreML: (\d+)", stderr)
    if m:
        p, n, k = (int(v) for v in m[-1])
        out.update(coreml_partitions=p, graph_nodes=n, coreml_nodes=k)
    bad = {}
    for op, ok in re.findall(r"Operator type: \[(\w+)\][^\n]*?supported: \[(\d)\]", stderr):
        if ok == "0":
            bad[op] = bad.get(op, 0) + 1
    if bad:
        out["not_on_coreml"] = bad
    return out


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
        timeout_s: float = 1800.0, python: str | None = None, log=print, cancelled=lambda: False,
        also: list | None = None, tolerance: float = TOLERANCE) -> list[dict]:
    """also: other model files (e.g. the float16 export) tested on the accelerators against the same
    CPU reference (this model on the CPU)."""
    model = str(Path(model).resolve())
    models = [model] + [str(Path(m).resolve()) for m in (also or [])]
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
    plan = [(model, "cpu", batches[0])] + [(m, d, b) for m in models for d in devices
                                            if not (d == "cpu" and m == model) for b in batches]
    for mdl, dev, bsz in plan:
        tag = "" if mdl == model else f" [{Path(mdl).name}]"
        out = tmp / f"{len(results)}.npy"
        cmd, env = _child_cmd(python, ["--one", dev, "--batch", str(bsz), "--reps", str(reps),
                                       "--out", str(out), mdl])
        log(f"{dev}, batch {bsz}{tag}: testing (a CoreML setting compiles the model the first time, "
            "a minute or two) ...")
        flags = 0x08000000 if sys.platform.startswith("win") else 0
        # output to files, not pipes: CoreML's verbose log is far more than a pipe holds, and a full pipe
        # would block the child until the parent read it (it only reads at the end)
        f_out, f_err = open(tmp / f"{len(results)}.out", "w+"), open(tmp / f"{len(results)}.err", "w+")
        proc = subprocess.Popen(cmd, stdout=f_out, stderr=f_err, text=True, env=env,
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
        proc.wait()
        f_out.seek(0)
        f_err.seek(0)
        so, se = f_out.read(), f_err.read()
        f_out.close()
        f_err.close()
        r = {"device": dev, "batch": bsz, "model": mdl, "tag": tag, "peak_gb": round(peak, 1)}
        if dev.startswith("coreml"):
            r.update(_coreml_report(se))
        if stopped or proc.returncode != 0:
            r.update(ok=False, error=stopped or (se.strip().splitlines() or ["failed"])[-1])
        else:
            r.update(json.loads(so.strip().splitlines()[-1]))
            y = np.load(out)
            if dev == "cpu" and mdl == model:         # the reference: this model, float32, on the CPU
                ref = y
            if ref is not None:
                k = min(len(y), len(ref))                   # same seeded input: the first k tiles agree
                d = float(np.abs(y[:k] - ref[:k]).max())
                r["max_diff"] = d
                r["mean_diff"] = float(np.abs(y[:k] - ref[:k]).mean())
                r["ok"] = d <= tolerance                  # absolute: outputs can exceed 1 (logits)
                if not r["ok"]:
                    r["error"] = f"output differs from the CPU by up to {d:.2e} (limit {tolerance:g})"
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
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    overall = None
    for mdl in models:
        good = [r for r in results if r.get("ok") and r.get("model", model) == mdl]
        bestr = min(good, key=lambda r: r["s_per_tile_step"]) if good else None
        best = bestr["device"] if bestr else None
        try:
            f = _state_file()
            f.parent.mkdir(parents=True, exist_ok=True)
            state = json.loads(f.read_text()) if f.exists() else {}
            state[_model_key(mdl)] = {"best": best, "best_batch": bestr["batch"] if bestr else None,
                                      "results": [r for r in results if r.get("model", model) == mdl],
                                      "time": time.time()}
            f.write_text(json.dumps(state, indent=1))
        except Exception:
            pass
        if bestr and (overall is None or bestr["s_per_tile_step"] < overall["s_per_tile_step"]):
            overall = bestr
        if best:
            log(f"{Path(mdl).name}: fastest with results matching the CPU: {best}, batch {bestr['batch']} "
                f"(~{_tile_min(bestr)} min per 5 km tile at 1 sample); "
                "'Auto' uses it for this file.")
    if overall and len(models) > 1:
        log(f"Overall fastest within the limit: {Path(overall['model']).name} on {overall['device']} - choose that "
            "file in QGIS.")
    return results


def _tile_min(r: dict) -> str:
    """Estimated minutes for a 5 km tile: tiles x network passes per tile (1 for a single-step model)."""
    m = r["s_per_tile_step"] * TILES_5KM * r.get("steps", 10) / 60
    return f"{m:.1f}" if m < 10 else f"{m:.0f}"


def _line(r: dict) -> str:
    name = f"{r['device']}, batch {r.get('batch')}{r.get('tag', '')}"
    if not r.get("s_per_tile_step"):
        return f"{name}: not usable ({r.get('error')}); peak {r.get('peak_gb')} GB"
    est = _tile_min(r)
    acc = (f"difference from the CPU: max {r['max_diff']:.1e}, mean {r['mean_diff']:.1e}"
           if "max_diff" in r else "")
    if "coreml_nodes" in r:
        acc += (f"; CoreML runs {r['coreml_nodes']}/{r['graph_nodes']} operators in {r['coreml_partitions']} "
                "piece(s)")
        if r.get("not_on_coreml"):
            acc += " (on the CPU: " + ", ".join(f"{k} x{v}" for k, v in sorted(r["not_on_coreml"].items())) + ")"
    flag = "" if r.get("ok") else f" - NOT USED: {r.get('error')}"
    return (f"{name}: {r['s_per_tile_step']:.3f} s per tile-step (~{est} min per 5 km tile at 1 sample), "
            f"peak {r['peak_gb']} GB, compile/warm-up {r.get('warmup_s')} s, {acc}{flag}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the exported .onnx")
    ap.add_argument("--devices", nargs="+", help="default: every device ONNX Runtime offers here")
    ap.add_argument("--batch", type=int, nargs="+", default=[8, 16, 32],
                    help="batch sizes to try on the accelerators (the CPU reference uses the first)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-gb", type=float, default=12.0, help="stop a device that uses more memory than this")
    ap.add_argument("--also", nargs="+", help="other model files to test against the same CPU reference, "
                                               "e.g. the float16 export (groundiff.export --fp16)")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE,
                    help="largest accepted difference from the CPU (1e-3 = 1 cm on a tile spanning 20 m)")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.one:
        print(json.dumps(bench_one(a.model, a.one, a.batch[0], a.reps, a.out)))
        return 0
    res = run(a.model, a.devices, a.batch, a.reps, a.max_gb, also=a.also, tolerance=a.tolerance)
    return 0 if any(r.get("ok") for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
