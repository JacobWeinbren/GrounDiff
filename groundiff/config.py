"""Experiment configuration (JSON <-> dataclasses)."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from .data.dataset import DataConfig
from .diffusion import DiffusionConfig
from .losses import LossConfig


@dataclass
class ModelConfig:
    kind: str = "groundiff"          # "groundiff" or "resdepth"
    # GrounDiff denoiser = Palette U-Net (defaults give 62.64M params, as in the paper)
    inner_channel: int = 64
    channel_mults: tuple = (1, 2, 4, 8)
    res_blocks: int = 2
    attn_res: tuple = (16,)          # Palette default -> attention only in the middle block
    num_head_channels: int = 32
    dropout: float = 0.2
    use_checkpoint: bool = False     # gradient checkpointing: less memory, ~25% slower
    # ResDepth (published defaults)
    resdepth_depth: int = 5
    resdepth_start_kernel: int = 64


@dataclass
class OptimConfig:
    name: str = "adamw"              # GrounDiff: AdamW; ResDepth: adam
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    schedule: str = "warmup_cosine"  # GrounDiff: 500 warmup steps + cosine; ResDepth: "step"
    warmup_steps: int = 500
    total_steps: int = 20_000        # GrounDiff converged in 10K-20K iterations
    min_lr: float = 0.0
    step_size: int = 200_000         # for schedule="step"
    step_gamma: float = 0.1
    grad_clip: float | None = None   # not specified by the paper
    ema_decay: float | None = 0.999  # Palette keeps an EMA; the paper does not say


@dataclass
class TrainConfig:
    out_dir: str = "runs/exp"
    device: str = "auto"             # auto | cuda | mps | cpu
    amp: str = "auto"                # auto | none | bf16 | fp16
    batch_size: int = 16             # paper: 16
    grad_accum: int = 1              # effective batch = batch_size * grad_accum
    num_workers: int = 4
    seed: int = 0
    log_every: int = 50
    val_every: int = 1000
    save_every: int = 1000
    val_max_tiles: int = 256
    val_init: str = "dsm_noise"      # sampler init used for validation (see diffusion.py)
    val_t_start: int | None = None
    compile: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    notes: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _fill(dc_type, values: dict):
    names = {f.name: f for f in dataclasses.fields(dc_type)}
    unknown = set(values) - set(names) - {"_doc"}
    if unknown:
        raise KeyError(f"unknown {dc_type.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for k, v in values.items():
        if k == "_doc":
            continue
        default = getattr(dc_type(), k) if not dataclasses.is_dataclass(names[k].type) else None
        kwargs[k] = tuple(v) if isinstance(default, tuple) and isinstance(v, list) else v
    return dc_type(**kwargs)


def config_from_dict(d: dict) -> Config:
    sections = {"data": DataConfig, "model": ModelConfig, "diffusion": DiffusionConfig,
                "loss": LossConfig, "optim": OptimConfig, "train": TrainConfig}
    unknown = set(d) - set(sections) - {"notes", "_doc"}
    if unknown:
        raise KeyError(f"unknown config sections: {sorted(unknown)}")
    kw = {name: _fill(t, d.get(name, {})) for name, t in sections.items()}
    return Config(notes=d.get("notes", ""), **kw)


def _set_path(d: dict, dotted: str, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """overrides: ["train.batch_size=4", "data.root=\\"x\\""] (values parsed as JSON,
    falling back to plain strings)."""
    d = json.loads(Path(path).read_text())
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        _set_path(d, key.strip(), val)
    return config_from_dict(d)
