"""Network callables for runtime.predict_scene: PyTorch or ONNX Runtime."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class TorchNet:
    """Wraps a loaded GrounDiff (calls its denoiser) or ResDepth model."""

    def __init__(self, model, device):
        import torch
        self.torch = torch
        self.model = model.eval()
        self.device = device
        self.is_diff = hasattr(model, "denoiser")

    def __call__(self, x: np.ndarray, gamma: np.ndarray | None = None) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            xt = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
            if self.is_diff:
                out = self.model.denoiser(xt, torch.from_numpy(gamma).to(self.device))
            else:
                out = self.model.net(xt)
            return out.float().cpu().numpy()


GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")


def cpu_threads() -> int:
    """Threads for ONNX Runtime on the CPU: the performance cores on Apple
    silicon (efficiency cores would hold every step back), else all cores."""
    import os
    import subprocess
    import sys
    if sys.platform == "darwin":
        try:
            n = int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True,
                                   text=True, timeout=5).stdout.strip())
            if n > 0:
                return n
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


class OnnxNet:
    """ONNX Runtime session. providers: e.g. ["CUDAExecutionProvider",
    "CPUExecutionProvider"], ["DmlExecutionProvider", ...] on Windows
    (onnxruntime-directml), ["CoreMLExecutionProvider", ...] on macOS.
    `warning` is set when a requested GPU provider could not be used (ONNX
    Runtime then silently runs on the CPU)."""

    def __init__(self, path: str | Path, providers: list | None = None):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if providers is None:
            # not CoreML: ONNX Runtime's CoreML provider splits this network into many pieces and
            # was seen using ~40 GB on an M3 Max; the CPU provider is steady and uses all cores
            pref = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in pref if p in avail]
        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            try:        # load pip-installed CUDA/cuDNN DLLs (onnxruntime-gpu[cuda,cudnn]), e.g. inside QGIS
                ort.preload_dlls()
            except Exception:
                pass
        self.warning = None
        so = ort.SessionOptions()
        so.intra_op_num_threads = cpu_threads()
        try:
            self.session = ort.InferenceSession(str(path), so, providers=providers)
        except Exception as e:                     # e.g. CoreML/DirectML cannot take this graph: use the CPU
            if providers == ["CPUExecutionProvider"]:
                raise
            self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            self.warning = f"{providers[0]} could not load the model ({e}); running on the CPU"
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.providers = self.session.get_providers()
        wanted = [p for p in providers if p in GPU_PROVIDERS]
        if not self.warning and wanted and self.providers[0] not in GPU_PROVIDERS:
            self.warning = (f"requested {wanted[0]} but ONNX Runtime is using {self.providers[0]} "
                            f"(available: {avail}); check the GPU build of onnxruntime and its CUDA/cuDNN")

    def __call__(self, x: np.ndarray, gamma: np.ndarray | None = None) -> np.ndarray:
        feeds = {self.inputs[0]: np.ascontiguousarray(x, np.float32)}
        if len(self.inputs) > 1:
            feeds[self.inputs[1]] = np.ascontiguousarray(gamma, np.float32)
        return self.session.run(None, feeds)[0]
