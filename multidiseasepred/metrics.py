"""Discrimination, calibration and operating points, with intervals.

Two opinions are baked in, both because leaving them to the caller produced
misleading tables in practice.

**Targets are not pooled by default.** `evaluate` takes named groups and
macro-averages within each. Predicting whether a condition is present depends on
the patient; predicting whether they get admitted also depends on whether there
is a bed. Averaging those into one number produces a figure that means nothing.

**Differences are reported through intervals, not tests.** `compare` calls a
difference a difference only when the two intervals do not overlap. That is
conservative -- non-overlapping intervals imply significance, but overlapping
ones do not imply its absence -- and being conservative in that direction is the
right way round when the alternative is claiming a 0.001 win.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class Estimate:
    point: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.point:.4f} ({self.lo:.4f}–{self.hi:.4f})"

    def overlaps(self, other: "Estimate") -> bool:
        return not (self.hi < other.lo or other.hi < self.lo)


def _safe(fn, y, s) -> float:
    if y.min() == y.max():
        return float("nan")
    return float(fn(y, s))


def bootstrap_ci(y: np.ndarray, s: np.ndarray, fn, *, n: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> Estimate:
    """Percentile bootstrap over encounters.

    Resampling rows rather than events is the right unit here: an encounter is
    what the hospital has one of, and the rare-endpoint intervals should widen
    when there are few events, which is exactly what this does.
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    point = _safe(fn, y, s)
    if np.isnan(point):
        return Estimate(point, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        v = _safe(fn, y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return Estimate(point, float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(point, float(lo), float(hi))


def evaluate(y: np.ndarray, probs: np.ndarray, *,
             target_names: Sequence[str],
             groups: Optional[Mapping[str, Sequence[int]]] = None,
             n_boot: int = 1000, seed: int = 0) -> dict:
    """AUROC and AUPRC per target, plus a macro average within each group.

    `groups` maps a label to the target indices that belong together. Pass None
    and every target is macro-averaged into a single group called "all", which
    is fine when the targets really are the same kind of thing and wrong the
    moment they are not.
    """
    y = np.asarray(y)
    probs = np.asarray(probs, dtype=float)
    names = list(target_names)
    if groups is None:
        groups = {"all": list(range(y.shape[1]))}

    out: dict = {"per_target": {}, "macro": {}}
    for t, name in enumerate(names):
        out["per_target"][name] = {
            "AUROC": bootstrap_ci(y[:, t], probs[:, t], roc_auc_score,
                                  n=n_boot, seed=seed + t),
            "AUPRC": bootstrap_ci(y[:, t], probs[:, t], average_precision_score,
                                  n=n_boot, seed=seed + 500 + t),
            "prevalence": float(y[:, t].mean()),
            "events": int(y[:, t].sum()),
        }

    for gname, idxs in groups.items():
        idxs = list(idxs)
        for metric, fn in (("AUROC", roc_auc_score),
                           ("AUPRC", average_precision_score)):
            def macro(yy, ss, _i=idxs, _f=fn):
                vals = [_safe(_f, yy[:, t], ss[:, t]) for t in _i]
                vals = [v for v in vals if not np.isnan(v)]
                return float(np.mean(vals)) if vals else float("nan")
            out["macro"].setdefault(gname, {})[metric] = bootstrap_ci(
                y, probs, macro, n=n_boot, seed=seed + 1000)
    return out


def operating_point(y: np.ndarray, probs: np.ndarray,
                    sensitivity: float = 0.90) -> dict:
    """Highest threshold that still reaches `sensitivity`.

    Sensitivity is the quantity fixed, not precision, because at the prevalences
    these models are used at the positive predictive value is bounded low by the
    base rate itself. A threshold is informative there mainly through its
    negative predictive value, and the achieved sensitivity is reported rather
    than assumed: thresholds live on a grid of observed scores and cannot always
    land exactly on the target.
    """
    y = np.asarray(y).astype(bool)
    p = np.asarray(probs, dtype=float)
    if y.sum() == 0:
        return {k: float("nan") for k in
                ("threshold", "sensitivity", "specificity", "ppv", "npv", "flagged")}
    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    need = np.searchsorted(tp, sensitivity * y.sum())
    need = min(need, len(order) - 1)
    thr = p[order][need]
    pred = p >= thr
    tp_ = int((pred & y).sum()); fp_ = int((pred & ~y).sum())
    fn_ = int((~pred & y).sum()); tn_ = int((~pred & ~y).sum())
    d = lambda a, b: float(a / b) if b else float("nan")
    return {"threshold": float(thr),
            "sensitivity": d(tp_, tp_ + fn_), "specificity": d(tn_, tn_ + fp_),
            "ppv": d(tp_, tp_ + fp_), "npv": d(tn_, tn_ + fn_),
            "flagged": float(pred.mean())}


def calibration_metrics(y: np.ndarray, probs: np.ndarray, bins: int = 10) -> dict:
    """Slope, intercept, Brier and ECE for one target.

    Slope and intercept come from a logistic fit of the outcome on the predicted
    logit: slope 1 and intercept 0 is perfect. They are reported because they say
    different things -- a slope away from 1 is a spread problem, an intercept
    away from 0 is a base-rate problem -- and a single summary hides which one
    you have.
    """
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y) ** 2))
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1])
        if m.sum():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    if y.min() == y.max():
        return {"slope": float("nan"), "intercept": float("nan"),
                "brier": brier, "ece": float(ece)}
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000).fit(z, y)
    return {"slope": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0]),
            "brier": brier, "ece": float(ece)}


def compare(a: Estimate, b: Estimate) -> str:
    """Describe two estimates without over-claiming."""
    if a.overlaps(b):
        return "indistinguishable (intervals overlap)"
    return "higher" if a.point > b.point else "lower"
