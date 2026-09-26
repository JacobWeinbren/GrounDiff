from groundiff.data.lasground import check, command, write_script
from tests.synthetic import write_scene


def test_default_command_has_no_tuning_flags():
    cmd = command("in/*.laz", "out")
    assert cmd == 'lasground_new64 -i "in/*.laz" -odir "out" -olaz'
    assert "-ignore_class 7 18" in command("a", "b", ignore_noise=True)


def test_script_and_check(tmp_path):
    write_scene(tmp_path, name="S0", size=40.0)
    cmd = write_script(tmp_path / "after", tmp_path / "before", tmp_path / "run.sh", windows=False, cores=4)
    assert "-cores 4" in cmd and (tmp_path / "run.sh").read_text().startswith("#!/bin/sh")
    # the synthetic before files carry only classes 1/2 (+ noise): consistent pair
    assert check(tmp_path / "before", tmp_path / "after") == []
