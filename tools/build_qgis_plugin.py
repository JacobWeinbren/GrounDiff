"""Build the QGIS plugin: copy the torch-free core modules into the plugin and
zip it for "Plugins > Manage and Install Plugins > Install from ZIP".

    python tools/build_qgis_plugin.py            # -> dist/groundiff_qgis.zip
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORE_FILES = ["runtime.py", "normalise.py", "io_raster.py", "backends.py", "schedule.py", "overlay.py",
              "batch.py", "data/rasterise.py", "data/laz.py", "data/lasinspect.py", "data/preprocess.py",
              "data/osgrid.py", "data/stream.py"]


def build(dest: Path) -> Path:
    """Copy the plugin sources + core into dest/groundiff_qgis; return that folder."""
    src = REPO / "qgis_plugin" / "groundiff_qgis"
    out = dest / "groundiff_qgis"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("__pycache__", "core"))
    core = out / "core"
    (core / "data").mkdir(parents=True)
    (core / "__init__.py").write_text("")
    (core / "data" / "__init__.py").write_text("")
    for f in CORE_FILES:
        shutil.copy2(REPO / "groundiff" / f, core / f)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dist", type=Path, default=REPO / "dist")
    a = ap.parse_args()
    a.dist.mkdir(parents=True, exist_ok=True)
    folder = build(a.dist)
    zpath = a.dist / "groundiff_qgis.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(a.dist))
    print(f"wrote {zpath}")


if __name__ == "__main__":
    main()
