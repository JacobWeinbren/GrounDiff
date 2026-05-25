"""Stage-2 raster GrounDiff: pixel-space diffusion with a UNet backbone.

Public API:
    GrounDiffRaster    - top-level training/inference wrapper
    GrounDiffUNet      - Palette-derived UNet denoiser (the same lineage
                          as stage 1; Marigold-validated as still-SOTA
                          for dense pixel-aligned regression)
    GrounDiffDiffusion - noise schedule + sampling
    gating             - paper Eq. 5
    groundiff_loss     - paper §3.2 Eq. 11-14
"""
from .diffusion import GrounDiffDiffusion
from .gating import gating
from .groundiff_raster import GrounDiffRaster
from .losses import groundiff_loss
from .unet import GrounDiffUNet

__all__ = [
    "GrounDiffRaster",
    "GrounDiffUNet",
    "GrounDiffDiffusion",
    "gating",
    "groundiff_loss",
]
