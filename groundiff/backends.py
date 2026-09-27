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


# Accelerator settings. CoreML: Apple's newer ML Program format on the GPU, with fixed input shapes
# (the default NeuralNetwork format with a free batch size splits this network into many pieces
# and used ~40 GB on an M3 Max) and a compile cache. CUDA: no TF32, so results match the CPU.
_COREML = {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU", "RequireStaticInputShapes": "1",
           "SpecializationStrategy": "FastPrediction", "AllowLowPrecisionAccumulationOnGPU": "0"}
PROVIDER_OPTIONS = {
    "CoreMLExecutionProvider": _COREML,
    "CUDAExecutionProvider": {"use_tf32": "0"},
}
DEVICE_PROVIDERS = {
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    # variants the speed test also tries (kept only if the output still matches the CPU):
    "coreml_ane": ["CoreMLExecutionProvider", "CPUExecutionProvider"],       # + Neural Engine (16-bit)
    "coreml_fp16acc": ["CoreMLExecutionProvider", "CPUExecutionProvider"],   # GPU, 16-bit accumulation
    "cpu": ["CPUExecutionProvider"],
}
DEVICE_OPTIONS = {
    "coreml_ane": {**_COREML, "MLComputeUnits": "ALL"},
    "coreml_fp16acc": {**_COREML, "AllowLowPrecisionAccumulationOnGPU": "1"},
}


def coreml_cache_dir(device: str = "coreml") -> str:
    import os
    d = Path(os.environ.get("GROUNDIFF_CACHE", Path.home() / ".groundiff")) / "coreml_cache" / device
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def auto_providers(avail: list, model_path=None) -> list:
    """CUDA, then DirectML; on a Mac the device the speed test found fastest and accurate for this
    model (python -m groundiff.speedtest, or 'Test compute devices' in QGIS), else the CPU."""
    for dev in ("cuda", "directml"):
        if DEVICE_PROVIDERS[dev][0] in avail:
            return DEVICE_PROVIDERS[dev]
    try:
        from .speedtest import best_device
        dev = best_device(model_path) if model_path else None
        if dev and DEVICE_PROVIDERS[dev][0] in avail:
            return dev
    except Exception:
        pass
    return ["CPUExecutionProvider"]


class OnnxNet:
    """ONNX Runtime session. providers: e.g. ["CUDAExecutionProvider",
    "CPUExecutionProvider"], ["DmlExecutionProvider", ...] on Windows
    (onnxruntime-directml), ["CoreMLExecutionProvider", ...] on macOS, or a
    device name from DEVICE_PROVIDERS. On an accelerator the batch size is
    fixed (batch, default 8): the graph is then fully static (ONNX Runtime
    folds away every Shape op), and smaller calls are padded, larger ones
    split. `warning` is set when a requested accelerator could not be used."""

    def __init__(self, path: str | Path, providers: list | str | None = None, batch: int = 8):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if providers is None:
            providers = auto_providers(avail, path)
        device = providers if isinstance(providers, str) else None
        self.device = device
        if device:
            providers = DEVICE_PROVIDERS[device]
        providers = [p for p in providers if p in avail or p == providers[0]]
        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            try:        # load pip-installed CUDA/cuDNN DLLs (onnxruntime-gpu[cuda,cudnn]), e.g. inside QGIS
                ort.preload_dlls()
            except Exception:
                pass
        self.warning = None
        accel = providers[0] in GPU_PROVIDERS
        self.fixed_batch = int(batch) if accel else None

        def options():
            so = ort.SessionOptions()
            so.intra_op_num_threads = cpu_threads()
            if self.fixed_batch:
                for name in ("batch", "b", "batch_size", "N"):        # the export's name for the batch axis
                    so.add_free_dimension_override_by_name(name, self.fixed_batch)
            return so

        def with_opts(ps):
            out = []
            for p in ps:
                o = dict(DEVICE_OPTIONS.get(device) or PROVIDER_OPTIONS.get(p, {})) if p == ps[0] else \
                    dict(PROVIDER_OPTIONS.get(p, {}))
                if p == "CoreMLExecutionProvider":
                    # one compiled copy per setting and batch size (the batch is baked into it)
                    o["ModelCacheDirectory"] = coreml_cache_dir(f"{device or 'coreml'}_b{self.fixed_batch}")
                out.append((p, o) if o else p)
            return out

        try:
            try:
                self.session = ort.InferenceSession(str(path), options(), providers=with_opts(providers))
            except Exception:
                if providers[0] not in PROVIDER_OPTIONS:
                    raise
                # an older ONNX Runtime that does not know some option: its defaults
                self.session = ort.InferenceSession(str(path), options(), providers=providers)
        except Exception as e:                     # e.g. CoreML/DirectML cannot take this graph: use the CPU
            if providers == ["CPUExecutionProvider"]:
                raise
            self.fixed_batch = None
            self.session = ort.InferenceSession(str(path), options(), providers=["CPUExecutionProvider"])
            self.warning = f"{providers[0]} could not load the model ({e}); running on the CPU"
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.providers = self.session.get_providers()
        wanted = [p for p in providers if p in GPU_PROVIDERS]
        if not self.warning and wanted and self.providers[0] not in GPU_PROVIDERS:
            self.warning = (f"requested {wanted[0]} but ONNX Runtime is using {self.providers[0]} "
                            f"(available: {avail}); check the GPU build of onnxruntime and its CUDA/cuDNN")
        if self.providers[0] not in GPU_PROVIDERS:
            self.fixed_batch = None

    def _run(self, x, gamma):
        feeds = {self.inputs[0]: np.ascontiguousarray(x, np.float32)}
        if len(self.inputs) > 1:
            feeds[self.inputs[1]] = np.ascontiguousarray(gamma, np.float32)
        return self.session.run(None, feeds)[0]

    def __call__(self, x: np.ndarray, gamma: np.ndarray | None = None) -> np.ndarray:
        B = self.fixed_batch
        n = x.shape[0]
        if not B or n == B:
            return self._run(x, gamma)
        outs = []
        for i in range(0, n, B):                  # fixed batch: split, and pad the last part by repetition
            xs = x[i:i + B]
            gs = gamma[i:i + B] if gamma is not None else None
            k = xs.shape[0]
            if k < B:
                xs = np.concatenate([xs, np.repeat(xs[-1:], B - k, axis=0)])
                if gs is not None:
                    gs = np.concatenate([gs, np.repeat(gs[-1:], B - k, axis=0)])
            outs.append(self._run(xs, gs)[:k])
        return np.concatenate(outs)
