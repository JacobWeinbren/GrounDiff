"""Checkpoint management.

Two persistent files:
    {dir}/best.pt          — model with lowest val RMSE seen so far
    {dir}/epoch_{N:04d}.pt — periodic snapshot

A small JSON sidecar `state.json` tracks epoch / best metric across
restarts.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

import torch


class CheckpointManager:
    def __init__(self, ckpt_dir: Path, model: torch.nn.Module,
                 optim: Optional[torch.optim.Optimizer] = None,
                 scheduler=None,
                 logger: Optional[logging.Logger] = None,
                 ema_model: Optional[torch.nn.Module] = None):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.ema_model = ema_model
        self.logger = logger or logging.getLogger('groundiff')
        self.state_path = self.dir / 'state.json'

        self.epoch = 0
        self.global_step = 0
        self.best_rmse = float('inf')

    def _state_path_for_epoch(self, epoch: int) -> Path:
        return self.dir / f"epoch_{epoch:04d}.pt"

    def save(self, *, epoch: int, global_step: int,
             rmse: Optional[float] = None) -> Path:
        path = self._state_path_for_epoch(epoch)
        payload = dict(
            unet=self.model.unet.state_dict(),
            epoch=epoch,
            global_step=global_step,
        )
        if self.ema_model is not None:
            payload['unet_ema'] = self.ema_model.unet.state_dict()
        if self.optim is not None:
            payload['optim'] = self.optim.state_dict()
        if self.scheduler is not None:
            payload['scheduler'] = self.scheduler.state_dict()
        torch.save(payload, path)
        self.epoch = epoch
        self.global_step = global_step

        if rmse is not None and rmse < self.best_rmse:
            self.best_rmse = float(rmse)
            torch.save(payload, self.dir / 'best.pt')
            self.logger.info(f"  ★ new best RMSE {rmse:.4f}m → best.pt")
        self._write_state()
        return path

    def load(self, path: Path, *, map_location='cpu',
             load_optim: bool = True) -> dict:
        d = torch.load(path, map_location=map_location, weights_only=False)
        self.model.unet.load_state_dict(d['unet'], strict=True)
        if self.ema_model is not None:
            if 'unet_ema' in d:
                self.ema_model.unet.load_state_dict(d['unet_ema'], strict=True)
                self.logger.info("  loaded EMA weights from checkpoint")
            else:
                # No EMA in checkpoint — initialise from main model
                self.ema_model.unet.load_state_dict(d['unet'], strict=True)
                self.logger.info("  no EMA in checkpoint; "
                                 "initialised EMA from main weights")
        if load_optim and 'optim' in d and self.optim is not None:
            try:
                self.optim.load_state_dict(d['optim'])
            except (ValueError, KeyError) as e:
                self.logger.warning(f"  optim state load failed: {e}; "
                                    "continuing with fresh optim state")
        if load_optim and 'scheduler' in d and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(d['scheduler'])
            except (ValueError, KeyError) as e:
                self.logger.warning(f"  scheduler state load failed: {e}")
        self.epoch = int(d.get('epoch', 0))
        self.global_step = int(d.get('global_step', 0))
        self._read_state()
        self.logger.info(f"  resumed from {path} "
                         f"(epoch {self.epoch}, step {self.global_step}, "
                         f"best RMSE {self.best_rmse:.4f}m)")
        return d

    def _read_state(self):
        if self.state_path.exists():
            try:
                d = json.loads(self.state_path.read_text())
                self.best_rmse = float(d.get('best_rmse', self.best_rmse))
            except Exception:
                pass

    def _write_state(self):
        self.state_path.write_text(json.dumps(dict(
            epoch=self.epoch,
            global_step=self.global_step,
            best_rmse=self.best_rmse,
        ), indent=2))
