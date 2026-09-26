"""Evaluation metrics, in metres (numpy).

GrounDiff reports RMSE and MAE (m), Type I error ("retaining non-ground
points"), Type II error ("removing ground points") and total error (%), and
MAD (degrees) for surface roughness. Note that this Type I/II naming is the
reverse of Sithole & Vosselman (2004); we follow GrounDiff.

On rasters, the paper does not say how a pixel is called ground. We use the
same rule as its training target M_alpha, always against the highest-return
DSM s = dsm_max (whatever the model's gate is): a pixel is ground when
|s - DTM| < alpha. `pred_ground` may instead be passed explicitly (e.g.
sigmoid(logit) > 0.5 for a dsm_max-gated model); both are reported by infer.py.

Edit detection (before -> after mode, our addition): a pixel truly needs an
edit when |reference - lasground_new DTM| > alpha, and is predicted to when
|prediction - lasground_new DTM| > alpha (or p_edit > 0.5); precision, recall
and F1 of that.

Total error is the share of misclassified pixels (the paper's tables are
consistent with this, e.g. SU-II: 4.49 / 2.46 / 3.82, not a plain sum).

ResDepth reports MAE, RMSE, median absolute error and NMAD; these are
included too.
"""
from __future__ import annotations

import numpy as np


def _safe_pct(num: int, den: int) -> float:
    return float("nan") if den == 0 else 100.0 * num / den


def classification_errors(pred_ground: np.ndarray, gt_ground: np.ndarray, valid: np.ndarray) -> dict:
    p, g = pred_ground[valid], gt_ground[valid]
    n_ng, n_g = int((~g).sum()), int(g.sum())
    t1 = int((p & ~g).sum())          # non-ground kept as ground
    t2 = int((~p & g).sum())          # ground removed
    return {"type1_pct": _safe_pct(t1, n_ng), "type2_pct": _safe_pct(t2, n_g),
            "total_pct": _safe_pct(t1 + t2, int(valid.sum()))}


def surface_roughness_deg(z: np.ndarray, valid: np.ndarray, gsd: float) -> float:
    """Mean angle (degrees) between the normals of 4-neighbouring cells.

    Our reading of the MAD roughness measure GrounDiff takes from FlexRoad; it
    is 0 for a plane and grows with surface noise.
    """
    gy, gx = np.gradient(z.astype(np.float64), gsd)
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=0)
    n /= np.linalg.norm(n, axis=0, keepdims=True)
    angles = []
    for sl_a, sl_b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        cos = (n[(slice(None),) + sl_a] * n[(slice(None),) + sl_b]).sum(0)
        ok = valid[sl_a] & valid[sl_b] & np.isfinite(cos)      # normals next to no-data are NaN
        angles.append(np.degrees(np.arccos(np.clip(cos[ok], -1.0, 1.0))))
    a = np.concatenate(angles)
    return float(a.mean()) if a.size else float("nan")


def edit_detection(pred_edit: np.ndarray, true_edit: np.ndarray, valid: np.ndarray) -> dict:
    p, g = pred_edit[valid].astype(bool), true_edit[valid].astype(bool)
    tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else (0.0 if (fp or fn) else float("nan"))
    return {"precision": prec, "recall": rec, "f1": f1, "true_edit_pct": _safe_pct(int(g.sum()), g.size)}


def dtm_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, dsm: np.ndarray,
                alpha: float, gsd: float | None = None,
                pred_ground: np.ndarray | None = None, before: np.ndarray | None = None,
                p_edit: np.ndarray | None = None) -> dict:
    """All inputs [H, W] (or flat) in metres; `valid` is a bool mask; `dsm` is
    the highest-return DSM (dsm_max); `before` the lasground_new DTM."""
    valid = valid.astype(bool) & np.isfinite(pred) & np.isfinite(gt)
    out: dict = {"n_pixels": int(valid.sum())}
    if not valid.any():
        return out
    e = (pred - gt)[valid].astype(np.float64)
    out.update({
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "medae": float(np.median(np.abs(e))),
        "nmad": float(1.4826 * np.median(np.abs(e - np.median(e)))),
    })
    cls_valid = valid & np.isfinite(dsm)
    gt_ground = np.abs(dsm - gt) < alpha
    for name, m in (("ground", gt_ground), ("nonground", ~gt_ground)):
        sel = cls_valid & m
        out[f"rmse_{name}"] = float(np.sqrt(np.mean((pred - gt)[sel] ** 2))) if sel.any() else float("nan")
    height_ground = np.abs(dsm - pred) < alpha
    out.update(classification_errors(height_ground, gt_ground, cls_valid))
    if pred_ground is not None:
        out.update({f"{k}_logit": v for k, v in
                    classification_errors(pred_ground.astype(bool), gt_ground, cls_valid).items()})
    if before is not None:
        ev = valid & np.isfinite(before)
        if ev.any():
            true_edit = np.abs(gt - before) > alpha
            out.update({f"edit_{k}": v for k, v in
                        edit_detection(np.abs(pred - before) > alpha, true_edit, ev).items()})
            if p_edit is not None:
                pv = ev & np.isfinite(p_edit)
                out.update({f"edit_{k}_p": v for k, v in
                            edit_detection(np.nan_to_num(p_edit) > 0.5, true_edit, pv).items()})
    if gsd is not None:
        out["roughness_deg"] = surface_roughness_deg(pred, valid, gsd)
    return out
