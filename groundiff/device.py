"""Device, precision and memory helpers (CUDA, Apple MPS, CPU)."""
from __future__ import annotations

import contextlib

import torch


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def resolve_amp(amp: str, device: torch.device) -> torch.dtype | None:
    """auto: bf16 on CUDA GPUs that support it, full precision elsewhere
    (MPS autocast support varies across PyTorch/macOS versions; opt in with
    amp="bf16" or "fp16" after checking it on your machine)."""
    if amp == "none":
        return None
    if amp == "auto":
        if device.type == "cuda":
            # native bf16 needs Ampere or newer (is_bf16_supported() also counts
            # emulation); older GPUs train in fp32 unless fp16 is chosen explicitly
            major, _ = torch.cuda.get_device_capability(device)
            return torch.bfloat16 if major >= 8 else None
        return None
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[amp]


def autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def setup_backend(device: torch.device):
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def memory_gb(device: torch.device) -> dict:
    if device.type == "cuda":
        return {"alloc_gb": torch.cuda.memory_allocated() / 1e9,
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    if device.type == "mps":
        return {"alloc_gb": torch.mps.current_allocated_memory() / 1e9,
                "driver_gb": torch.mps.driver_allocated_memory() / 1e9}
    return {}


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
