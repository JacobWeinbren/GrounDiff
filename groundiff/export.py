"""Export a trained checkpoint to ONNX for PyTorch-free inference (QGIS).

    python -m groundiff.export runs/before_after/best.pt --out models/before_after

writes <out>.onnx (the network only: GrounDiff denoiser or ResDepth U-Net)
and <out>.json (RuntimeSpec: channels, normalisation, noise schedule). The
diffusion loop, tiling and blending run in numpy (runtime.py).
The export is checked against PyTorch on random inputs before returning.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .backends import OnnxNet, TorchNet
from .models.build import load_model
from .runtime import RuntimeSpec


class _Denoiser(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, gamma):
        return self.unet(x, gamma)


def _inline_weights(onnx_path: Path):
    """torch.export writes weights to <name>.onnx.data; fold them into the
    .onnx so the model is one file (plus its .json) to copy to the PC."""
    data = onnx_path.with_name(onnx_path.name + ".data")
    if not data.exists():
        return
    import onnx
    m = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(m, str(onnx_path), save_as_external_data=False)
    data.unlink()


SPEC_KEY = "groundiff_spec"


def _embed_spec(onnx_path: Path, spec_json: str):
    """Store the RuntimeSpec inside the .onnx (metadata_props), so the model is
    one self-contained file; the .json next to it stays for reference."""
    try:
        import onnx
    except ImportError:
        print("[warn] onnx not installed: spec not embedded; keep the .json next to the .onnx")
        return
    m = onnx.load(str(onnx_path))
    for p in list(m.metadata_props):
        if p.key == SPEC_KEY:
            m.metadata_props.remove(p)
    m.metadata_props.add(key=SPEC_KEY, value=spec_json)
    onnx.save_model(m, str(onnx_path))


def export(ckpt: str | Path, out: str | Path, use_ema: bool = True, opset: int = 18,
           check: bool = True) -> tuple[Path, Path]:
    model, cfg, ck = load_model(ckpt, "cpu", use_ema=use_ema)
    spec = RuntimeSpec.from_config(cfg, ck.get("data_meta"))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx_path, json_path = out.with_suffix(".onnx"), out.with_suffix(".json")
    t = cfg.data.tile
    if spec.kind == "groundiff":
        net = _Denoiser(model.denoiser).eval()
        x = torch.randn(2, 1 + len(spec.cond_channels), t, t)
        args, names = (x, torch.rand(2)), ["x", "gamma"]
        dyn = {"x": {0: "batch"}, "gamma": {0: "batch"}, "out": {0: "batch"}}
    else:
        net = model.net.eval()
        x = torch.randn(2, net.n_input_channels, t, t)     # [prior, guidance...]
        args, names = (x,), ["x"]
        dyn = {"x": {0: "batch"}, "out": {0: "batch"}}
    try:
        torch.onnx.export(net, args, str(onnx_path), input_names=names, output_names=["out"],
                          dynamic_axes=dyn, opset_version=opset, dynamo=False)
    except (TypeError, RuntimeError, NotImplementedError) as e:   # legacy exporter removed
        print(f"legacy ONNX exporter unavailable ({e}); using torch.export")
        batch = torch.export.Dim("batch", min=1, max=1024)
        shapes = {"x": {0: batch}, "gamma": {0: batch}} if len(args) == 2 else {"x": {0: batch}}
        torch.onnx.export(net, args, str(onnx_path), input_names=names, output_names=["out"],
                          dynamic_shapes=shapes, opset_version=opset, dynamo=True)
        _inline_weights(onnx_path)
    spec.to_json(json_path)
    _embed_spec(onnx_path, json_path.read_text())
    if check:
        ref = TorchNet(model, "cpu")
        ox = OnnxNet(onnx_path, ["CPUExecutionProvider"])
        xs = np.random.default_rng(0).standard_normal(tuple(x.shape)).astype(np.float32)
        gs = np.array([0.9, 0.1], np.float32)
        a = ref(xs, gs) if spec.kind == "groundiff" else ref(xs)
        b = ox(xs, gs)
        err = float(np.abs(a - b).max())
        if err > 1e-3 * max(1.0, float(np.abs(a).max())):
            raise RuntimeError(f"ONNX output differs from PyTorch by {err}")
        print(f"ONNX check passed (max abs diff {err:.2e})")
    return onnx_path, json_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", action="store_true", help="export raw weights instead of EMA")
    ap.add_argument("--opset", type=int, default=18)
    a = ap.parse_args(argv)
    p, j = export(a.checkpoint, a.out, use_ema=not a.raw, opset=a.opset)
    print(f"wrote {p} and {j}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
