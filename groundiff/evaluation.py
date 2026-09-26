"""Tile-level prediction and metric aggregation for validation during
training (held-out test metrics on whole scenes come from infer.py)."""
from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .diffusion import GrounDiff
from .metrics import dtm_metrics


@torch.no_grad()
def predict_tiles(model, batch: dict, device: torch.device, init: str = "dsm_noise",
                  t_start: int | None = None, n_samples: int = 1,
                  generator: torch.Generator | None = None) -> dict:
    """Returns normalised predictions: mean [B,1,H,W], std (over samples),
    ground probability sigmoid(logit) (GrounDiff only)."""
    cond = batch["cond"].to(device)
    prior = batch["prior"].to(device)
    if isinstance(model, GrounDiff):
        preds, probs = [], []
        for _ in range(n_samples):
            g0, logit = model.sample(cond, init=init, prior=prior, t_start=t_start, generator=generator)
            preds.append(g0.float())
            probs.append(torch.sigmoid(logit.float()))
        p = torch.stack(preds)
        return {"mean": p.mean(0), "std": p.std(0) if n_samples > 1 else torch.zeros_like(p[0]),
                "p_ground": torch.stack(probs).mean(0)}
    pred = model(prior, cond).float()
    return {"mean": pred, "std": torch.zeros_like(pred), "p_ground": None}


def to_metres(x: torch.Tensor, lo: torch.Tensor, scale: torch.Tensor) -> np.ndarray:
    lo = lo.view(-1, 1, 1, 1).to(x.device)
    scale = scale.view(-1, 1, 1, 1).to(x.device)
    return ((x + 1.0) * 0.5 * scale + lo).cpu().numpy()


@torch.no_grad()
def evaluate_loader(model, loader, cfg: Config, device: torch.device, max_tiles: int | None = None,
                    init: str = "dsm_noise", t_start: int | None = None, seed: int = 0) -> dict:
    """Pixel-pooled metrics in metres over (up to) `max_tiles` tiles, plus the
    same metrics for the lasground_new DTM when a prior channel is configured.
    Overlapping eval tiles count each pixel once (the dataset's `own` mask)."""
    model.eval()
    # CPU generator on every device: the same noise for raw and EMA weights and
    # for every validation, so their metrics are comparable
    gen = torch.Generator(device="cpu").manual_seed(seed)
    P, G, S, V, PG, B, PE = [], [], [], [], [], [], []
    seen = 0
    edit_mode = bool(cfg.data.prior_channel) and cfg.data.gate_channel == cfg.data.prior_channel
    for batch in loader:
        out = predict_tiles(model, batch, device, init=init, t_start=t_start, generator=gen)
        lo, sc = batch["lo"], batch["scale"]
        v = batch["valid"].numpy() > 0.5
        if "own" in batch:
            v &= batch["own"].numpy() > 0.5
        P.append(to_metres(out["mean"], lo, sc))
        G.append(to_metres(batch["target"], lo, sc))
        S.append(to_metres(batch["surface"], lo, sc))
        V.append(v)
        if out["p_ground"] is not None:
            pg = out["p_ground"].cpu().numpy()
            PG.append(pg > 0.5)
            if edit_mode:
                PE.append(1.0 - pg)
        if cfg.data.prior_channel:
            b = to_metres(batch["prior"], lo, sc)
            B.append(np.where(batch["prior_valid"].numpy() > 0.5, b, np.nan))
        seen += batch["cond"].shape[0]
        if max_tiles and seen >= max_tiles:
            break
    cat = lambda xs: np.concatenate([x.reshape(-1) for x in xs]) if xs else None
    p, g, s, v, b = cat(P), cat(G), cat(S), cat(V), cat(B)
    # classification by sigmoid(l) only means "ground" when the gate is the DSM
    pg = cat(PG) if cfg.data.gate_channel == "dsm_max" else None
    res = {"model": dtm_metrics(p, g, v, s, cfg.data.alpha, pred_ground=pg, before=b, p_edit=cat(PE)),
           "n_tiles": seen}
    if b is not None:
        res["lasground_new"] = dtm_metrics(b, g, v & np.isfinite(b), s, cfg.data.alpha)
    return res
