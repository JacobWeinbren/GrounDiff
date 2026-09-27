"""Install the plugin's Python packages from inside QGIS (no terminal).

Packages go into a private folder in the QGIS profile
(<profile>/python/groundiff_deps), which is put on sys.path when the plugin
loads; QGIS's own Python is not modified. numpy (and GDAL) always come from
QGIS: pip is run with --no-deps for the packages that would otherwise pull in
their own numpy.

    onnxruntime-gpu[cuda,cudnn]  Windows / Linux with an NVIDIA GPU: CUDA and cuDNN come from pip
                           too (~2.5 GB), so only the NVIDIA driver is needed
    onnxruntime-directml   Windows without an NVIDIA GPU (any GPU)
    onnxruntime            macOS (CPU + Apple GPU through CoreML) and Linux
    laspy, lazrs           LAS/LAZ reading
    scipy                  only if QGIS does not already ship it
    triangle               fast Delaunay for the ground TIN (~10x Qhull; free for private,
                           research and institutional use)
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

# packages whose own dependencies are installed explicitly (never numpy)
ORT_SUPPORT = ["flatbuffers", "protobuf", "packaging", "coloredlogs", "humanfriendly", "sympy", "mpmath"]


def deps_dir() -> Path:
    try:
        from qgis.core import QgsApplication
        base = Path(QgsApplication.qgisSettingsDirPath()) / "python"
    except Exception:                               # outside QGIS (tests)
        base = Path.home() / ".groundiff"
    return base / "groundiff_deps"


def activate():
    """Put the private package folder on sys.path (called when the plugin loads)."""
    d = str(deps_dir())
    if d not in sys.path:
        sys.path.insert(0, d)


def has_nvidia() -> bool:
    """An NVIDIA driver is installed (nvidia-smi ships with it)."""
    import shutil
    if sys.platform == "darwin":
        return False
    if shutil.which("nvidia-smi"):
        return True
    return sys.platform.startswith("win") and Path(r"C:\Windows\System32\nvidia-smi.exe").exists()


def ort_package() -> str:
    if has_nvidia():
        return "onnxruntime-gpu[cuda,cudnn]"
    return "onnxruntime-directml" if sys.platform.startswith("win") else "onnxruntime"


def ort_flavour_ok() -> bool:
    """The installed ONNX Runtime can use this machine's GPU (CUDA with an NVIDIA driver)."""
    try:
        import onnxruntime as ort
    except Exception:
        return False
    return not has_nvidia() or "CUDAExecutionProvider" in ort.get_available_providers()


def missing() -> list[str]:
    """pip names of what is not importable yet."""
    out = []
    if not ort_flavour_ok():
        out.append(ort_package())
    for mod, pkg in (("laspy", "laspy"), ("lazrs", "lazrs"), ("scipy", "scipy"), ("triangle", "triangle")):
        try:
            importlib.import_module(mod)
        except Exception:
            out.append(pkg)
    return out


def python_exe() -> str:
    """The Python interpreter behind QGIS (sys.executable is the QGIS program
    itself on Windows and macOS)."""
    exe = Path(sys.executable)
    names = ("python.exe", "python3.exe", "python3", "python")
    cands = []
    if exe.name.lower().startswith("python"):
        cands.append(exe)
    for base in (Path(sys.exec_prefix), Path(sys.prefix), exe.parent, exe.parent / "bin",
                 Path(sys.exec_prefix) / "bin"):
        cands += [base / n for n in names]
    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cands += [Path(sys.exec_prefix) / "bin" / ver, exe.parent / "bin" / ver]
    for c in cands:
        if c.is_file():
            return str(c)
    raise RuntimeError("cannot find QGIS's Python interpreter; install the packages by hand: "
                       + " ".join(missing()))


def pip_commands(pkgs: list[str], target: Path | None = None) -> list[list[str]]:
    """pip calls that install pkgs into the private folder without touching numpy."""
    target = target or deps_dir()
    py = python_exe()
    base = [py, "-m", "pip", "install", "--upgrade", "--target", str(target), "--no-warn-script-location",
            "--disable-pip-version-check"]
    cmds = []
    # CUDA build: its extras (the NVIDIA libraries) need dependency resolution; the numpy it pulls
    # in is deleted afterwards (install)
    nodeps = [p for p in pkgs if (p.startswith("onnxruntime") and "[" not in p) or p in ("scipy", "laspy", "triangle")]
    plain = [p for p in pkgs if p not in nodeps]
    if nodeps:
        cmds.append(base + ["--no-deps"] + nodeps)
    if any(p.startswith("onnxruntime") and "[" not in p for p in pkgs):
        cmds.append(base + ["--no-deps"] + ORT_SUPPORT)
    if plain:
        cmds.append(base + plain)
    return cmds


def install(pkgs: list[str] | None = None, log=print) -> None:
    pkgs = pkgs if pkgs is not None else missing()
    if not pkgs:
        return
    target = deps_dir()
    target.mkdir(parents=True, exist_ok=True)
    flags = 0x08000000 if sys.platform.startswith("win") else 0        # CREATE_NO_WINDOW
    env = dict(os.environ)
    env.pop("PYTHONHOME", None) if sys.platform == "darwin" else None
    if any(p.startswith("onnxruntime") for p in pkgs):     # one ONNX Runtime build at a time
        import shutil
        for old in list(target.glob("onnxruntime*")):
            shutil.rmtree(old, ignore_errors=True) if old.is_dir() else old.unlink()
    for cmd in pip_commands(pkgs, target):
        log("running: " + " ".join(cmd[3:]))
        r = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"pip failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    import shutil
    for p in [target / "numpy", target / "numpy.libs", target / "numpy.dylibs"] + list(target.glob("numpy-*")):
        if p.exists():                                                   # never shadow QGIS's numpy
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
    activate()
    importlib.invalidate_caches()
