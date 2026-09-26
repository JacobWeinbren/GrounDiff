import os
import shutil
import subprocess

import laspy
import numpy as np
import pytest

from groundiff.data import lasground as lg
from groundiff.data.lasground import check, command, output_name, pending_inputs, write_script
from tests.synthetic import write_scene


def test_default_command_has_no_tuning_flags():
    cmd = command("in/*.laz", "out")
    assert cmd == 'lasground_new64 -i "in/*.laz" -odir "out" -olaz'
    assert "-ignore_class 7 18" in command("a", "b", ignore_noise=True)


def test_script_and_check(tmp_path):
    write_scene(tmp_path, name="S0", size=40.0)
    cmd = write_script(tmp_path / "after", tmp_path / "before", tmp_path / "run.sh", windows=False, cores=4)
    assert "-cores 4" in cmd and (tmp_path / "run.sh").read_text().startswith("#!/bin/sh")
    assert '*.las' in cmd                                     # .las tiles are picked up too
    # the synthetic before files carry only classes 1/2: consistent pair
    assert check(tmp_path / "before", tmp_path / "after") == []


def test_check_catches_perturbed_coordinates(tmp_path):
    """Unlicensed LAStools perturbs coordinates of large files."""
    write_scene(tmp_path, name="S0", size=40.0)
    p = tmp_path / "before" / "S0.las"
    las = laspy.read(str(p))
    las.x = np.asarray(las.x) + np.random.default_rng(0).uniform(-0.05, 0.05, len(las.points))
    las.write(str(p))
    msgs = check(tmp_path / "before", tmp_path / "after")
    assert any("licence" in m for m in msgs)


def test_output_names_and_resume(tmp_path):
    assert output_name("TL4378nw_P_1_20220315_20220316.copc.laz") == "TL4378nw_P_1_20220315_20220316.copc.laz"
    assert output_name("X.las") == "X.laz"
    write_scene(tmp_path, name="S0", size=20.0)
    out = tmp_path / "out"
    out.mkdir()
    assert [p.name for p in pending_inputs(tmp_path / "after", out)] == ["S0.las"]
    (out / "S0.laz").write_bytes(b"x")
    assert pending_inputs(tmp_path / "after", out) == []
    assert len(pending_inputs(tmp_path / "after", out, overwrite=True)) == 1


def test_docker_run_command(tmp_path, monkeypatch):
    """The docker command line, without Docker: licence mounted read-only and
    passed by environment variable, inputs by a file list, default settings."""
    write_scene(tmp_path, name="S0", size=20.0)
    lic = tmp_path / "lastoolslicense.txt"
    lic.write_text("x")
    seen = {}

    class FakeProc:
        stdout = iter(["done with '/in/S0.las'\n"])

        def wait(self):
            return 0

    def fake_popen(args, **kw):
        seen["args"] = args
        seen["list"] = (tmp_path / "out" / lg.LIST_NAME).read_text()
        return FakeProc()

    monkeypatch.setattr(lg.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(lg.subprocess, "Popen", fake_popen)
    cmd = lg.docker_run(tmp_path / "after", tmp_path / "out", lic, cores=3)
    s = " ".join(cmd)
    assert "--platform linux/amd64" in s and ":/in:ro" in s and ":/lic:ro" in s
    assert "-e LAStoolsLicenseFile=/lic/lastoolslicense.txt" in s
    assert s.endswith("lasground_new64 -lof /out/_lasground_inputs.txt -odir /out -olaz -cores 3 -v")
    assert seen["list"].strip() == "/in/S0.las"
    assert not (tmp_path / "out" / lg.LIST_NAME).exists()


@pytest.mark.skipif(os.environ.get("GROUNDIFF_DOCKER_TESTS") != "1" or not shutil.which("docker"),
                    reason="set GROUNDIFF_DOCKER_TESTS=1 with Docker running to build the image")
def test_docker_end_to_end_with_stand_in(tmp_path):
    """Builds the real image (Ubuntu + LAStools' dependencies) around a stand-in
    lasground_new64 that copies its inputs, and runs the full flow."""
    fake = tmp_path / "fake" / "LAStools" / "bin"
    fake.mkdir(parents=True)
    (fake / "lasground_new64").write_text(
        '#!/bin/sh\nif [ "$1" = "-version" ]; then echo stand-in; exit 0; fi\n'
        'while [ $# -gt 0 ]; do case "$1" in -lof) lof="$2"; shift;; -odir) odir="$2"; shift;; esac; shift; done\n'
        'while read -r f; do [ -n "$f" ] || continue; b=$(basename "$f"); cp "$f" "$odir/${b%.*}.laz"; done < "$lof"\n')
    (fake / "lasground_new64").chmod(0o755)
    subprocess.run(["tar", "czf", str(tmp_path / "LAStools.tar.gz"), "-C", str(tmp_path / "fake"), "LAStools"],
                   check=True)
    write_scene(tmp_path, name="S0", size=20.0)
    lg.docker_ready()
    lg.docker_build("groundiff-lastools-test:latest", tmp_path / "LAStools.tar.gz", rebuild=True)
    lg.docker_run(tmp_path / "before", tmp_path / "out", None, image="groundiff-lastools-test:latest")
    assert (tmp_path / "out" / "S0.laz").exists()
    assert check(tmp_path / "out", tmp_path / "before") == []
