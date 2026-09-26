"""Recreate the automated ("before") classification with lasground_new.

Production runs lasground_new with its DEFAULT settings, i.e. exactly

    lasground_new64 -i tile.laz -o tile_before.laz

which (per the lasground_new README) re-classifies every point as ground (2)
or non-ground (1), considers only last returns, and uses the default step,
granularity, offset (0.05 m), spike (1 m) and bulge (step/10 clamped to
1-2 m). The README gives two different default steps (25 m in the text,
5.0 in the argument list): run once with --verbose and keep the log, which
records what your binary actually used.

This writes a script (Windows .bat or POSIX .sh) that runs it over a folder,
keeping file names so "before" and "after" tiles pair up by name:

    python -m groundiff.data.lasground --in EA_tiles --out before --cores 8 --verbose
    # then run the written script on the machine with licensed LAStools

Notes
  * LAStools without a licence distorts files above ~1.5M points (diagonal
    lines). Use a licensed install; `check` flags suspicious outputs.
  * Default settings do not ignore noise classes; add --ignore-noise only if
    production does (it passes -ignore_class 7 18).
  * Tiles are classified independently (no buffer), like the default command.
    Add --buffered 50 only if production uses on-the-fly buffering.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .preprocess import find_laz


def command(inputs: str, out_dir: str, *, exe: str = "lasground_new64", cores: int = 1, verbose: bool = False,
            ignore_noise: bool = False, buffered: float | None = None, extra: str = "") -> str:
    parts = [exe, "-i", f'"{inputs}"', "-odir", f'"{out_dir}"', "-olaz"]
    if cores > 1:
        parts += ["-cores", str(cores)]
    if ignore_noise:
        parts += ["-ignore_class", "7", "18"]
    if buffered:
        parts += ["-buffered", f"{buffered:g}"]
    if verbose:
        parts.append("-v")
    if extra:
        parts.append(extra)
    return " ".join(parts)


def write_script(in_dir: Path, out_dir: Path, script: Path, windows: bool, **kw) -> str:
    pattern = str(in_dir / "*.laz") if windows else str(in_dir / "*.laz")
    cmd = command(pattern.replace("/", "\\") if windows else pattern,
                  str(out_dir).replace("/", "\\") if windows else str(out_dir), **kw)
    if windows:
        text = f'@echo off\r\nif not exist "{out_dir}" mkdir "{out_dir}"\r\n{cmd} > "{out_dir}\\lasground_new.log" 2>&1\r\n'
    else:
        text = f'#!/bin/sh\nset -e\nmkdir -p "{out_dir}"\n{cmd} > "{out_dir}/lasground_new.log" 2>&1\n'
    script.write_text(text)
    return cmd


def check(before_dir: Path, after_dir: Path) -> list[str]:
    """Compare before/after pairs: same point counts and a sane ground share."""
    import numpy as np

    from .laz import read_points
    msgs = []
    after, before = find_laz(after_dir), find_laz(before_dir)
    for name, a in after.items():
        if name not in before:
            msgs.append(f"{name}: no before file")
            continue
        pa = read_points(a, drop_classes=(), drop_withheld=False)
        pb = read_points(before[name], drop_classes=(), drop_withheld=False)
        if len(pa) != len(pb):
            msgs.append(f"{name}: point counts differ ({len(pa)} after vs {len(pb)} before)")
        classes = set(np.unique(pb.cls).tolist())
        if not classes <= {1, 2, 7, 18}:
            msgs.append(f"{name}: before file has classes {sorted(classes)}; expected only 1/2 from lasground_new")
        share = float((pb.cls == 2).mean())
        if not 0.02 < share < 0.98:
            msgs.append(f"{name}: ground share {share:.1%} looks wrong (unlicensed LAStools distorts large files)")
    return msgs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    w = sub.add_parser("script", help="write the lasground_new script (default)")
    for p in (ap, w):
        p.add_argument("--in", dest="in_dir", type=Path)
        p.add_argument("--out", dest="out_dir", type=Path)
        p.add_argument("--script", type=Path)
        p.add_argument("--windows", action="store_true", help="write a .bat (default: by platform)")
        p.add_argument("--exe", default="lasground_new64")
        p.add_argument("--cores", type=int, default=1)
        p.add_argument("--verbose", action="store_true")
        p.add_argument("--ignore-noise", action="store_true")
        p.add_argument("--buffered", type=float)
    c = sub.add_parser("check", help="sanity-check before files against after files")
    c.add_argument("--before", type=Path, required=True)
    c.add_argument("--after", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        msgs = check(a.before, a.after)
        print("\n".join(msgs) if msgs else "all before/after pairs look consistent")
        return 1 if msgs else 0
    if not a.in_dir or not a.out_dir:
        ap.error("--in and --out are required")
    windows = a.windows or sys.platform.startswith("win")
    script = a.script or Path("run_lasground_new" + (".bat" if windows else ".sh"))
    cmd = write_script(a.in_dir, a.out_dir, script, windows, exe=a.exe, cores=a.cores, verbose=a.verbose,
                       ignore_noise=a.ignore_noise, buffered=a.buffered)
    print(f"wrote {script}:\n  {cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
