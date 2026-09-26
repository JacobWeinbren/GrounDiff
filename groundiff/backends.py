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
            pref = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider",
                    "CPUExecutionProvider"]
            providers = [p for p in pref if p in avail]
        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            try:        # load pip-installed CUDA/cuDNN DLLs (onnxruntime-gpu[cuda,cudnn]), e.g. inside QGIS
                ort.preload_dlls()
            except Exception:
                pass
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.providers = self.session.get_providers()
        wanted = [p for p in providers if p in GPU_PROVIDERS]
        self.warning = None
        if wanted and self.providers[0] not in GPU_PROVIDERS:
            self.warning = (f"requested {wanted[0]} but ONNX Runtime is using {self.providers[0]} "
                            f"(available: {avail}); check the GPU build of onnxruntime and its CUDA/cuDNN")

    def __call__(self, x: np.ndarray, gamma: np.ndarray | None = None) -> np.ndarray:
        feeds = {self.inputs[0]: np.ascontiguousarray(x, np.float32)}
        if len(self.inputs) > 1:
            feeds[self.inputs[1]] = np.ascontiguousarray(gamma, np.float32)
        return self.session.run(None, feeds)[0]
