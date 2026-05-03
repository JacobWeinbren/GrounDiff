"""Evaluation metrics — paper §4.1 + Tab.1/2/4.

The paper reports five metrics for ground generation:

  1. RMSE  ↓ — Root Mean Squared Error in metres
  2. MAE   ↓ — Mean Absolute Error in metres
  3. E_T1  ↓ — Type I error (% retaining non-ground points)
  4. E_T2  ↓ — Type II error (% removing ground points)
  5. E_tot ↓ — Total error rate

Paper §4.1: "Classification performance is assessed by Type I error
E_T1 (retaining non-ground points), Type II error E_T2 (removing
ground points), and total error E_tot (sum of both), all expressed
as percentages."

Cell-level classification.
The paper inherits Sithole-Vosselman 2003 metric definitions but does
NOT explicitly state the threshold it uses to decide ground vs. non-
ground from the residual. The S-V 2003 paper (§4.2.1) used 0.20 m for
their per-point reference comparison.

We expose two classification methods, both supported, with the
residual-based method as the default since it's self-consistent with
the training loss:

  Method A — residual-based (default)
    True ground:      |s − g_GT|   < α  (in normalised [-1,1] units)
    Predicted ground: |s − g_pred| < α  (same threshold)
    α matches the training-time confidence loss target M_α (paper
    Eq.14). Default α = 0.05 in normalised units.

  Method B — σ(ℓ)-based (paper Fig.16)
    True ground:      |s − g_GT|   < α  (same as A)
    Predicted ground: σ(ℓ) ≥ 0.5
    Uses the trained per-pixel logit ℓ. Available when the caller
    supplies `prob_ground` to update().

We also expose `err_gt_α` (fraction with absolute error > α metres)
for monitoring training-time regression progress.
"""
from __future__ import annotations
import numpy as np
import torch


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _mcc(TP: int, FP: int, FN: int, TN: int) -> float:
    """Matthews Correlation Coefficient — symmetric, robust to class
    imbalance. Returns -1 (worst) to +1 (best); 0 = random.

    Returns 0.0 (well-behaved degenerate) when either class is absent
    or only one class is predicted; this avoids -nan and matches the
    sklearn convention.
    """
    import math
    num = TP * TN - FP * FN
    den_sq = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    if den_sq <= 0:
        return 0.0
    return float(num / math.sqrt(den_sq))


def _balanced_accuracy(TP: int, FP: int, FN: int, TN: int) -> float:
    """Mean of per-class recall: 0.5 (recall_pos + recall_neg).
    Defined whenever each true class has >= 1 sample. Returns NaN if a
    class is absent."""
    n_pos = TP + FN
    n_neg = TN + FP
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    recall_pos = TP / n_pos
    recall_neg = TN / n_neg
    return 0.5 * (recall_pos + recall_neg)


def _f1(TP: int, FP: int, FN: int) -> float:
    """F1 for the positive class (non-ground in our convention).
    Returns 0.0 if no positives predicted nor present."""
    denom = 2 * TP + FP + FN
    if denom == 0:
        return 0.0
    return float(2 * TP / denom)


class MetricAggregator:
    """Streaming aggregator over many tiles or scenes.

    Tracks per-pixel sums so the final RMSE/MAE/ET* are over all valid
    pixels jointly, not the unweighted mean of per-tile means.

    Two classification methods are computed when the inputs allow:
      result()['E_T1'/'E_T2'/'E_tot']           — Method A (residual)
      result()['E_T1_logit'/'E_T2_logit'/'E_tot_logit']
                                                — Method B (σ(ℓ)) if
                                                  prob_ground supplied

    Args:
        thresholds: thresholds (metres) for `err_gt_α` reporting.
        cls_threshold_m: residual threshold in METRES for the M_α-style
                    classification. Default 0.20 m (Sithole-Vosselman
                    2003 §4.2.1). All classification (true and predicted
                    ground) is computed in metres for cross-tile
                    consistency.
        prob_threshold: σ(ℓ) cutoff for Method B (default 0.5).
        min_classify_n: minimum number of cells in a true class for
                    E_T1/E_T2 to be reported. If a tile/scene has fewer
                    than `min_classify_n` true non-ground cells, E_T1
                    is reported as NaN (and similarly for E_T2 / true
                    ground). Default 1000 — for an ocean tile with ~5
                    non-ground cells this avoids dividing by tiny
                    denominators (each individual cell is then 20% of
                    the metric, making it Bernoulli noise). MCC and
                    bal_acc are always reported regardless.
    """

    def __init__(self, thresholds=(0.5, 1.0),
                 cls_threshold_m: float = 0.20,
                 prob_threshold: float = 0.5,
                 min_classify_n: int = 1000):
        self.thresholds = tuple(float(t) for t in thresholds)
        self.cls_threshold_m = float(cls_threshold_m)
        self.prob_threshold = float(prob_threshold)
        self.min_classify_n = int(min_classify_n)
        self.sum_sq = 0.0
        self.sum_abs = 0.0
        self.n = 0
        self.over = {t: 0 for t in self.thresholds}
        # Method A: residual-based (predicted ground = |s − g_pred| < α/τ)
        self.cls_TP = 0; self.cls_FP = 0
        self.cls_FN = 0; self.cls_TN = 0
        # Method B: σ(ℓ)-based (predicted ground = σ(ℓ) ≥ 0.5)
        self.logit_TP = 0; self.logit_FP = 0
        self.logit_FN = 0; self.logit_TN = 0

    def update(self, pred_m, gt_m, valid, dsm_m=None,
               prob_ground=None, dsm_norm=None, dtm_norm=None):
        """Accumulate one tile's worth of metrics.

        Classification (both methods) needs the DSM in metres (`dsm_m`)
        to compute the residual against. Without `dsm_m`, only RMSE/MAE
        and `err_gt_α` are tracked.

        Args:
            pred_m: [H, W] predicted DTM in metres
            gt_m:   [H, W] ground-truth DTM in metres
            valid:  [H, W] bool/{0,1} valid-pixel mask
            dsm_m:  [H, W] DSM in metres. Used for both Method A
                    (residual classification) and as denominator for
                    Method B (true-class label).
            prob_ground: [H, W] σ(ℓ) ∈ [0, 1] — enables Method B.
            dsm_norm, dtm_norm: optional, kept for API compatibility but
                    not used in this simplified path.
        """
        del dsm_norm, dtm_norm  # unused in simplified metric

        pred = _to_np(pred_m)
        gt   = _to_np(gt_m)
        v    = _to_np(valid).astype(bool)
        if not v.any():
            return

        # --- Regression (always tracked) ---
        d = (pred - gt)[v]
        ad = np.abs(d)
        self.sum_sq  += float((d * d).sum())
        self.sum_abs += float(ad.sum())
        self.n       += int(v.sum())
        for t in self.thresholds:
            self.over[t] += int((ad > t).sum())

        if dsm_m is None:
            return  # can't classify without DSM

        # --- True-class M_τ = 1[|s − g_GT| > τ_metres] ---
        dsm = _to_np(dsm_m)
        tau = self.cls_threshold_m
        r_gt   = np.abs(dsm - gt)[v]
        r_pred = np.abs(dsm - pred)[v]
        is_nonground_gt = r_gt > tau

        # --- Method A: residual-based predicted class ---
        is_nonground_pred_a = r_pred > tau
        self.cls_TP += int(( is_nonground_pred_a &  is_nonground_gt).sum())
        self.cls_FP += int((~is_nonground_pred_a &  is_nonground_gt).sum())
        self.cls_FN += int(( is_nonground_pred_a & ~is_nonground_gt).sum())
        self.cls_TN += int((~is_nonground_pred_a & ~is_nonground_gt).sum())

        # --- Method B: σ(ℓ)-based predicted class (Fig.16) ---
        if prob_ground is not None:
            p = _to_np(prob_ground)[v]
            is_ground_pred_b = p >= self.prob_threshold
            is_nonground_pred_b = ~is_ground_pred_b
            self.logit_TP += int(( is_nonground_pred_b &  is_nonground_gt).sum())
            self.logit_FP += int((~is_nonground_pred_b &  is_nonground_gt).sum())
            self.logit_FN += int(( is_nonground_pred_b & ~is_nonground_gt).sum())
            self.logit_TN += int((~is_nonground_pred_b & ~is_nonground_gt).sum())

    def result(self) -> dict:
        if self.n == 0:
            r = dict(rmse=float('nan'), mae=float('nan'), n_valid=0)
            for t in self.thresholds:
                r[f"err_gt_{t}"] = float('nan')
            r.update(E_T1=float('nan'), E_T2=float('nan'),
                      E_tot=float('nan'), n_classified=0)
            return r

        out = dict(
            rmse=float(np.sqrt(self.sum_sq / self.n)),
            mae=float(self.sum_abs / self.n),
            n_valid=int(self.n),
        )
        for t in self.thresholds:
            out[f"err_gt_{t}"] = float(self.over[t] / self.n)

        # Method A — residual-based (default)
        # By our convention "positive" = non-ground:
        #   TP = pred non-ground & true non-ground (cls_TP)
        #   FN = pred ground     & true non-ground (cls_FP)
        #   FP = pred non-ground & true ground     (cls_FN)
        #   TN = pred ground     & true ground     (cls_TN)
        # E_T1 (retain non-ground) = FN / (TP+FN) = cls_FP / n_nonground
        # E_T2 (remove ground)     = FP / (TN+FP) = cls_FN / n_ground
        TP_A = self.cls_TP; FN_A = self.cls_FP
        FP_A = self.cls_FN; TN_A = self.cls_TN
        n_ng = TP_A + FN_A   # true non-ground
        n_g  = TN_A + FP_A   # true ground
        n_cls = n_ng + n_g
        if n_cls > 0:
            # E_T1/E_T2 — gated by `min_classify_n`. If n_nonground or
            # n_ground is below this threshold, the per-class error rate
            # is returned as NaN (denominator too small for stable
            # estimation; e.g. ocean tile with 5 non-ground cells).
            if n_ng >= self.min_classify_n:
                out['E_T1'] = float(FN_A / n_ng)   # retain non-ground
            else:
                out['E_T1'] = float('nan')
            if n_g >= self.min_classify_n:
                out['E_T2'] = float(FP_A / n_g)    # remove ground
            else:
                out['E_T2'] = float('nan')
            out['E_tot'] = float((FN_A + FP_A) / n_cls)
            out['n_classified'] = int(n_cls)
            out['n_ground']     = int(n_g)
            out['n_nonground']  = int(n_ng)

            # IoU per class (for non-ground = positive class):
            #   IoU_nonground = TP / (TP + FP + FN)
            #   IoU_ground    = TN / (TN + FN + FP)
            out['IoU_nonground'] = (
                float(TP_A / (TP_A + FP_A + FN_A))
                if (TP_A + FP_A + FN_A) > 0 else 0.0)
            out['IoU_ground'] = (
                float(TN_A / (TN_A + FN_A + FP_A))
                if (TN_A + FN_A + FP_A) > 0 else 0.0)
            out['mIoU'] = 0.5 * (out['IoU_ground'] + out['IoU_nonground'])

            # MCC + Balanced Acc + F1 — robust to class imbalance
            out['MCC']      = _mcc(TP_A, FP_A, FN_A, TN_A)
            out['bal_acc']  = _balanced_accuracy(TP_A, FP_A, FN_A, TN_A)
            out['F1_nonground'] = _f1(TP_A, FP_A, FN_A)
        else:
            out.update(E_T1=float('nan'), E_T2=float('nan'),
                       E_tot=float('nan'), n_classified=0,
                       IoU_ground=float('nan'), IoU_nonground=float('nan'),
                       mIoU=float('nan'),
                       MCC=float('nan'), bal_acc=float('nan'),
                       F1_nonground=float('nan'))

        # Method B — σ(ℓ)-based (paper Fig.16) when supplied
        TP_B = self.logit_TP; FN_B = self.logit_FP
        FP_B = self.logit_FN; TN_B = self.logit_TN
        n_ng_b = TP_B + FN_B
        n_g_b  = TN_B + FP_B
        n_cls_b = n_ng_b + n_g_b
        if n_cls_b > 0:
            if n_ng_b >= self.min_classify_n:
                out['E_T1_logit'] = float(FN_B / n_ng_b)
            else:
                out['E_T1_logit'] = float('nan')
            if n_g_b >= self.min_classify_n:
                out['E_T2_logit'] = float(FP_B / n_g_b)
            else:
                out['E_T2_logit'] = float('nan')
            out['E_tot_logit'] = float((FN_B + FP_B) / n_cls_b)
            out['IoU_nonground_logit'] = (
                float(TP_B / (TP_B + FP_B + FN_B))
                if (TP_B + FP_B + FN_B) > 0 else 0.0)
            out['IoU_ground_logit'] = (
                float(TN_B / (TN_B + FN_B + FP_B))
                if (TN_B + FN_B + FP_B) > 0 else 0.0)
            out['mIoU_logit'] = 0.5 * (out['IoU_ground_logit']
                                        + out['IoU_nonground_logit'])
            out['MCC_logit']      = _mcc(TP_B, FP_B, FN_B, TN_B)
            out['bal_acc_logit']  = _balanced_accuracy(TP_B, FP_B, FN_B, TN_B)
            out['F1_nonground_logit'] = _f1(TP_B, FP_B, FN_B)
        else:
            out['E_T1_logit'] = float('nan')
            out['E_T2_logit'] = float('nan')
            out['E_tot_logit'] = float('nan')
            out['IoU_ground_logit']    = float('nan')
            out['IoU_nonground_logit'] = float('nan')
            out['mIoU_logit']          = float('nan')
            out['MCC_logit']           = float('nan')
            out['bal_acc_logit']       = float('nan')
            out['F1_nonground_logit']  = float('nan')

        return out


def per_point_classification(prob_ground: np.ndarray,
                             gt_is_ground: np.ndarray) -> dict:
    """Per-point E_T1/E_T2/E_tot from gating-logit probabilities.

    Note: paper tables use cell-level classification (see
    MetricAggregator). Per-point is exposed here for completeness.
    """
    pred_g = (prob_ground >= 0.5)
    gt_g   = gt_is_ground.astype(bool)
    TP = int(( pred_g &  gt_g).sum())
    FP = int(( pred_g & ~gt_g).sum())
    FN = int((~pred_g &  gt_g).sum())
    TN = int((~pred_g & ~gt_g).sum())
    n_g  = TP + FN
    n_ng = TN + FP
    e_t1 = float(FP / max(n_ng, 1))
    e_t2 = float(FN / max(n_g, 1))
    return dict(
        E_T1=e_t1, E_T2=e_t2, E_tot=e_t1 + e_t2,
        TP=TP, FP=FP, FN=FN, TN=TN,
        n_ground=n_g, n_nonground=n_ng,
    )


def classify_points_against_dtm(x_pts: np.ndarray, y_pts: np.ndarray,
                                z_pts: np.ndarray,
                                dtm: np.ndarray, valid: np.ndarray,
                                bbox, gsd: float,
                                threshold: float = 0.20) -> np.ndarray:
    """Classify each LIDAR point as 'ground' if z is within `threshold`
    metres of the predicted DTM at its (x, y) cell — Sithole-Vosselman
    2003 §4.2.1 methodology (which used τ=0.20m).

    Args:
        x_pts, y_pts, z_pts: [N] point coordinates in scene CRS, metres
        dtm:    [H, W] predicted DTM in metres
        valid:  [H, W] valid-cell mask
        bbox:   (x0, y0, x1, y1) scene extent
        gsd:    grid spacing in metres
        threshold: max |z - DTM| to be ground (S-V 2003: 0.20m)

    Returns:
        [N] {0.0, 0.5, 1.0} — 1.0=predicted ground, 0.0=non-ground,
        0.5=invalid cell (caller should drop these before passing to
        per_point_classification).
    """
    H, W = dtm.shape
    x0, y0, _, _ = bbox
    y1 = y0 + H * gsd
    j = np.clip(((x_pts - x0) / gsd).astype(np.int64), 0, W - 1)
    i = np.clip(((y1 - y_pts) / gsd).astype(np.int64), 0, H - 1)
    in_valid = valid[i, j].astype(bool)
    dtm_z = dtm[i, j]
    return np.where(in_valid,
                    (np.abs(z_pts - dtm_z) <= threshold).astype(np.float32),
                    0.5)
