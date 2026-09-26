"""Recreate the automated ("before") classification with lasground_new.

Production runs lasground_new with its DEFAULT settings, i.e. exactly

    lasground_new64 -i tile.laz -o tile_before.laz

which (per the lasground_new README) re-classifies every point as ground (2)
or non-ground (1), whatever class it had (without -non_ground_unchanged or
-ignore_class nothing is kept), considers only last returns, and uses the
default step, granularity, offset (0.05 m), spike (1 m) and bulge (step/10
clamped to 1-2 m). The README gives two different default steps (25 m in the
text, 5.0 in the argument list): the -v log records what the binary used.

Run it on the EA's published tiles to recreate the "before": their own
classes (1-7 incl. vegetation/buildings) come from a different process and
are overwritten, which is what we want - the model never uses them.

On a Mac (or Linux): LAStools' Linux build in Docker, one command
------------------------------------------------------------------
    python -m groundiff.data.lasground docker --in data/laz/ea --out data/laz/before \\
        --license ~/lastools/lastoolslicense.txt --cores 6

builds a small image with LAStools for Linux (downloaded from rapidlasso, or
--lastools-tar LAStools.tar.gz) the first time, runs lasground_new on every
tile not done yet (outputs keep the input names, so tiles pair up), writes
the -v log to <out>/lasground_new.log, and runs `check`. Needs Docker Desktop;
on Apple Silicon it runs the x86-64 build under emulation (enable "Use
Rosetta for x86_64/amd64 emulation" in Docker Desktop settings, and give
Docker 16 GB+ memory for 2 km tiles).

On Windows: write a .bat for an existing LAStools install
---------------------------------------------------------
    python -m groundiff.data.lasground script --in EA_tiles --out before --cores 8 --verbose --windows

Check
-----
    python -m groundiff.data.lasground check --before data/laz/before --after data/laz/ea

compares every pair: same points, unchanged coordinates (LAStools without a
licence perturbs files above its point limit, ~1.5-5M points), classes only
1/2 (+7/18), a sane ground share.

Notes
  * Default settings do not ignore noise classes; add --ignore-noise only if
    production does (it passes -ignore_class 7 18).
  * Tiles are classified independently (no buffer), like the default command.
    Add --buffered 50 only if production uses on-the-fly buffering.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .preprocess import find_laz

IMAGE = "groundiff-lastools:latest"
LASTOOLS_URL = "https://downloads.rapidlasso.de/LAStools.tar.gz"
PLATFORM = "linux/amd64"
LIST_NAME = "_lasground_inputs.txt"

DOCKERFILE = """\
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
# dependencies listed in the LAStools README (Linux section)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget \\
        libjpeg62 libpng-dev libtiff-dev libjpeg-dev zlib1g-dev libproj-dev liblzma-dev libjbig-dev \\
        libzstd-dev libgeotiff-dev libwebp-dev libsqlite3-dev libgomp1 \\
    && rm -rf /var/lib/apt/lists/*
{fetch}
RUN mkdir -p /opt/lastools && tar xzf /tmp/LAStools.tar.gz -C /opt/lastools && rm /tmp/LAStools.tar.gz \\
    && find /opt/lastools -type f -name "*64" -exec chmod a+x {{}} + && chmod -R a+rX /opt/lastools
COPY run_lastools.sh /usr/local/bin/lastools
RUN chmod a+x /usr/local/bin/lastools
ENTRYPOINT ["/usr/local/bin/lastools"]
"""

RUN_SH = """\
#!/bin/sh
# lastools <tool> [args...]: find the tool in the unpacked release and run it
# with the release's own libraries on the library path.
set -e
tool="$1"; shift
exe=$(find /opt/lastools -type f -name "$tool" | head -n 1)
if [ -z "$exe" ]; then echo "LAStools tool '$tool' not found in the image" >&2; exit 127; fi
libs=$(find /opt/lastools -type d -name lib | tr '\\n' ':')
export LD_LIBRARY_PATH="${libs}$(dirname "$exe"):${LD_LIBRARY_PATH:-}"
cd "$(dirname "$exe")"
exec "$exe" "$@"
"""


def command(inputs: str, out_dir: str, *, exe: str = "lasground_new64", cores: int = 1, verbose: bool = False,
            ignore_noise: bool = False, buffered: float | None = None, extra: str = "") -> str:
    parts = [exe, "-i", f'"{inputs}"', "-odir", f'"{out_dir}"', "-olaz"]
    parts += _flags(cores, verbose, ignore_noise, buffered)
    if extra:
        parts.append(extra)
    return " ".join(parts)


def _flags(cores: int = 1, verbose: bool = False, ignore_noise: bool = False,
           buffered: float | None = None) -> list[str]:
    f = []
    if cores > 1:
        f += ["-cores", str(cores)]
    if ignore_noise:
        f += ["-ignore_class", "7", "18"]
    if buffered:
        f += ["-buffered", f"{buffered:g}"]
    if verbose:
        f.append("-v")
    return f


def write_script(in_dir: Path, out_dir: Path, script: Path, windows: bool, **kw) -> str:
    exts = sorted({p.suffix.lower() for p in Path(in_dir).iterdir() if p.suffix.lower() in (".laz", ".las")}) \
        if Path(in_dir).is_dir() else [".laz"]
    sep = "\\" if windows else "/"
    base = str(in_dir).replace("/", sep) if windows else str(in_dir)
    patterns = '" "'.join(f"{base}{sep}*{e}" for e in (exts or [".laz"]))
    cmd = command(patterns, str(out_dir).replace("/", sep) if windows else str(out_dir), **kw)
    if windows:
        text = f'@echo off\r\nif not exist "{out_dir}" mkdir "{out_dir}"\r\n{cmd} > "{out_dir}\\lasground_new.log" 2>&1\r\n'
    else:
        text = f'#!/bin/sh\nset -e\nmkdir -p "{out_dir}"\n{cmd} > "{out_dir}/lasground_new.log" 2>&1\n'
    script.write_text(text)
    return cmd


# ----------------------------------------------------------------------------- docker

def output_name(src: Path) -> str:
    """LAStools -odir -olaz: input name minus its last extension, plus .laz
    (TL4378nw_P_1_2_3.copc.laz -> TL4378nw_P_1_2_3.copc.laz, X.las -> X.laz)."""
    return Path(src).stem + ".laz"


def _docker(*args, check=True, capture=False, **kw):
    exe = shutil.which("docker")
    if not exe:
        raise RuntimeError("docker not found: install Docker Desktop (https://www.docker.com/products/docker-desktop/) "
                           "and start it")
    return subprocess.run([exe, *args], check=check, text=True,
                          capture_output=capture, **kw)


def docker_ready() -> None:
    r = _docker("info", "--format", "{{.MemTotal}}", check=False, capture=True)
    if r.returncode != 0:
        raise RuntimeError("Docker is installed but not running: start Docker Desktop and retry.\n" + r.stderr.strip())
    try:
        mem_gb = int(r.stdout.strip()) / 1e9
        if mem_gb < 12:
            print(f"[warn] Docker has {mem_gb:.0f} GB memory; 2 km tiles need more. Docker Desktop -> Settings -> "
                  "Resources -> Memory: 16 GB or more (or use fewer --cores).")
    except ValueError:
        pass


def image_exists(image: str = IMAGE) -> bool:
    return _docker("image", "inspect", image, check=False, capture=True).returncode == 0


def docker_build(image: str = IMAGE, lastools_tar: Path | None = None, url: str = LASTOOLS_URL,
                 rebuild: bool = False) -> None:
    if image_exists(image) and not rebuild:
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if lastools_tar:
            shutil.copy2(lastools_tar, tmp / "LAStools.tar.gz")
            fetch = "COPY LAStools.tar.gz /tmp/LAStools.tar.gz"
        else:
            fetch = f"RUN wget -q -O /tmp/LAStools.tar.gz {url}"
        (tmp / "Dockerfile").write_text(DOCKERFILE.format(fetch=fetch))
        (tmp / "run_lastools.sh").write_text(RUN_SH)
        print(f"building Docker image {image} (LAStools for Linux; first time only, a few minutes)")
        _docker("build", "--platform", PLATFORM, "-t", image, str(tmp))
    r = _docker("run", "--rm", "--platform", PLATFORM, image, "lasground_new64", "-version", check=False,
                capture=True)
    out = (r.stdout + r.stderr).strip()
    if "error while loading shared libraries" in out or r.returncode == 127:
        raise RuntimeError(f"LAStools does not run in the image:\n{out}")
    print(out.splitlines()[0] if out else "lasground_new64 runs")


def pending_inputs(in_dir: Path, out_dir: Path, overwrite: bool = False) -> list[Path]:
    files = sorted(find_laz(in_dir).values())
    return [p for p in files if overwrite or not (out_dir / output_name(p)).exists()
            or (out_dir / output_name(p)).stat().st_size == 0]


def docker_run(in_dir: Path, out_dir: Path, license_file: Path | None = None, *, cores: int = 4,
               verbose: bool = True, ignore_noise: bool = False, buffered: float | None = None,
               image: str = IMAGE, overwrite: bool = False) -> list[str]:
    """Run lasground_new (default settings) in Docker on every tile of in_dir
    that has no output yet. Returns the docker command."""
    in_dir, out_dir = Path(in_dir).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = pending_inputs(in_dir, out_dir, overwrite)
    if not todo:
        print(f"all tiles in {in_dir} already have lasground_new output in {out_dir}")
        return []
    names = [output_name(p) for p in todo]
    if len(set(names)) != len(names):
        raise ValueError("two input tiles would write the same output name (same name in different subfolders)")
    (out_dir / LIST_NAME).write_text("\n".join("/in/" + p.relative_to(in_dir).as_posix() for p in todo) + "\n")
    cmd = ["run", "--rm", "--platform", PLATFORM, "-v", f"{in_dir}:/in:ro", "-v", f"{out_dir}:/out"]
    if license_file:
        lic = Path(license_file).expanduser().resolve()
        if not lic.exists():
            raise FileNotFoundError(f"licence file {lic} not found")
        cmd += ["-v", f"{lic.parent}:/lic:ro", "-e", f"LAStoolsLicenseFile=/lic/{lic.name}"]
    if sys.platform.startswith("linux") and hasattr(os, "getuid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]          # outputs owned by you, not root
    cmd += [image, "lasground_new64", "-lof", f"/out/{LIST_NAME}", "-odir", "/out", "-olaz"]
    cmd += _flags(cores, verbose, ignore_noise, buffered)
    print(f"lasground_new on {len(todo)} tiles ({len(find_laz(in_dir)) - len(todo)} already done), "
          f"{cores} cores; log: {out_dir / 'lasground_new.log'}")
    with open(out_dir / "lasground_new.log", "a") as log:
        log.write("docker " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen([shutil.which("docker"), *cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        for line in proc.stdout:
            log.write(line)
            if not verbose or "done with" in line or "ERROR" in line or "WARNING" in line:
                print("  " + line.rstrip())
        rc = proc.wait()
    (out_dir / LIST_NAME).unlink(missing_ok=True)
    if rc != 0:
        raise RuntimeError(f"lasground_new exited with code {rc}; see {out_dir / 'lasground_new.log'}")
    return cmd


# ----------------------------------------------------------------------------- check

def check(before_dir: Path, after_dir: Path, coord_tol: float = 0.002) -> list[str]:
    """Compare before/after pairs: same points, unchanged coordinates, only
    lasground_new classes, a sane ground share."""
    import numpy as np

    from .laz import read_points
    msgs = []
    after, before = find_laz(after_dir), find_laz(before_dir)
    missing = [n for n in after if n not in before]
    if missing:
        msgs.append(f"{len(missing)} tiles have no before file, e.g. {missing[:3]}")
    for name, a in after.items():
        if name not in before:
            continue
        try:
            pa = read_points(a, drop_withheld=False)
            pb = read_points(before[name], drop_withheld=False)
        except Exception as e:
            msgs.append(f"{name}: cannot read: {e}")
            continue
        if len(pa) != len(pb):
            msgs.append(f"{name}: point counts differ ({len(pa)} after vs {len(pb)} before)")
        else:
            d = max(float(np.abs(pa.x - pb.x).max()), float(np.abs(pa.y - pb.y).max()),
                    float(np.abs(pa.z - pb.z).max()))
            if d > coord_tol:
                # order may differ: compare as sorted sets before blaming the licence
                ds = max(float(np.abs(np.sort(pa.x) - np.sort(pb.x)).max()),
                         float(np.abs(np.sort(pa.z) - np.sort(pb.z)).max()))
                if ds > coord_tol:
                    msgs.append(f"{name}: coordinates changed by up to {ds:.3f} m - LAStools without a valid "
                                "licence perturbs files above its point limit; check the licence (--license)")
        classes = set(np.unique(pb.cls).tolist())
        if not classes <= {1, 2, 7, 18}:
            msgs.append(f"{name}: before file has classes {sorted(classes)}; expected only 1/2 from lasground_new")
        share = float((pb.cls == 2).mean())
        if not 0.005 < share < 0.999:      # open farmland can legitimately exceed 98 % ground
            msgs.append(f"{name}: ground share {share:.1%} is extreme; check the log and the output visually")
    return msgs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--in", dest="in_dir", type=Path, required=True)
        p.add_argument("--out", dest="out_dir", type=Path, required=True)
        p.add_argument("--cores", type=int, default=4)
        p.add_argument("--ignore-noise", action="store_true")
        p.add_argument("--buffered", type=float)

    d = sub.add_parser("docker", help="run lasground_new in Docker (Mac/Linux)")
    common(d)
    d.add_argument("--license", type=Path, help="your lastoolslicense.txt (without it: free point limit, "
                   "larger files get perturbed)")
    d.add_argument("--lastools-tar", type=Path, help="use this LAStools.tar.gz instead of downloading it")
    d.add_argument("--rebuild", action="store_true", help="rebuild the image (e.g. for a new LAStools release)")
    d.add_argument("--overwrite", action="store_true", help="redo tiles that already have output")
    d.add_argument("--quiet", action="store_true", help="no -v log")
    w = sub.add_parser("script", help="write a .bat/.sh for an existing LAStools install")
    common(w)
    w.add_argument("--script", type=Path)
    w.add_argument("--windows", action="store_true", help="write a .bat (default: by platform)")
    w.add_argument("--exe", default="lasground_new64")
    w.add_argument("--verbose", action="store_true")
    c = sub.add_parser("check", help="sanity-check before files against after files")
    c.add_argument("--before", type=Path, required=True)
    c.add_argument("--after", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        msgs = check(a.before, a.after)
        print("\n".join(msgs) if msgs else "all before/after pairs look consistent")
        return 1 if msgs else 0
    if a.cmd == "docker":
        docker_ready()
        docker_build(lastools_tar=a.lastools_tar, rebuild=a.rebuild)
        if not a.license:
            print("[warn] no --license: LAStools runs in its free mode and perturbs files above its point limit")
        docker_run(a.in_dir, a.out_dir, a.license, cores=a.cores, verbose=not a.quiet,
                   ignore_noise=a.ignore_noise, buffered=a.buffered, overwrite=a.overwrite)
        msgs = check(a.out_dir, a.in_dir)
        print("\n".join(msgs) if msgs else "check: all before/after pairs look consistent")
        return 1 if msgs else 0
    if a.cmd == "script":
        windows = a.windows or sys.platform.startswith("win")
        script = a.script or Path("run_lasground_new" + (".bat" if windows else ".sh"))
        cmd = write_script(a.in_dir, a.out_dir, script, windows, exe=a.exe, cores=a.cores, verbose=a.verbose,
                           ignore_noise=a.ignore_noise, buffered=a.buffered)
        print(f"wrote {script}:\n  {cmd}")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
