"""Per-target isotonic calibration on the logit scale.

Where the calibrator is fitted is the whole question
----------------------------------------------------
Fit it on the data you then score and you have measured nothing: the calibration
curve will look excellent because it was drawn through those same points. This
module fits on the development set and applies the frozen mapping everywhere
else, so calibration on a validation cohort is a result rather than a tautology.

`fit` therefore takes development data and nothing else, and `transform` never
looks at labels. If you want the other design -- refit locally at each new site
-- do it explicitly with a second Calibrator; the class will not do it for you by
accident.

Isotonic regression is monotone *non-decreasing*, not strictly increasing: it is
a step function, so distinct scores can be mapped onto the same value. Ranking is
therefore preserved up to ties, and AUROC is unchanged except for the half-credit
that tied pairs receive. That is usually negligible, but it is not identically
zero and it grows as a target gets rarer and the isotonic fit gets coarser: on a
low-prevalence target it can reach the third decimal. Measure it rather than
assume it -- `ranking_shift` is there for exactly that -- and if you intend to
write "recalibration does not change AUROC" in a paper, run it first and quote
the number you actually saw.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """One isotonic regression per target, fitted once, applied everywhere.

    Targets whose development split contains a single class are left
    uncalibrated and recorded in `skipped_`. Silently returning the identity
    would hide the fact; raising would make the whole model unusable because one
    rare endpoint had no events in a fold.
    """

    def __init__(self, out_of_bounds: str = "clip"):
        self.out_of_bounds = out_of_bounds
        self.models_: list[Optional[IsotonicRegression]] = []
        self.skipped_: list[int] = []

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "Calibrator":
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y)
        if probs.shape != y.shape:
            raise ValueError(f"probs {probs.shape} and y {y.shape} differ")
        self.models_, self.skipped_ = [], []
        z = _logit(probs)
        for t in range(probs.shape[1]):
            yt = y[:, t]
            if yt.min() == yt.max():
                self.models_.append(None)
                self.skipped_.append(t)
                continue
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                                    out_of_bounds=self.out_of_bounds)
            ir.fit(z[:, t], yt.astype(float))
            self.models_.append(ir)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("Calibrator is not fitted")
        probs = np.asarray(probs, dtype=float)
        z = _logit(probs)
        out = np.array(probs, copy=True)
        for t, ir in enumerate(self.models_):
            if ir is not None:
                out[:, t] = ir.predict(z[:, t])
        return np.clip(out, 0.0, 1.0)

    def fit_transform(self, probs, y):
        return self.fit(probs, y).transform(probs)

    def check_monotone(self, probs: np.ndarray, tol: float = 1e-9) -> bool:
        """Confirm calibration never reversed the order of two scores.

        This checks non-decreasing, which is what isotonic regression
        guarantees. It should never fail; it is here because an order reversal
        would mean the pipeline is not doing what this module says, and that is
        worth finding at the point of failure rather than in a referee report.
        """
        cal = self.transform(probs)
        for t in range(probs.shape[1]):
            a = np.argsort(probs[:, t], kind="stable")
            if (np.diff(cal[a, t]) < -tol).any():
                return False
        return True

    def ranking_shift(self, probs: np.ndarray, y: np.ndarray) -> list[float]:
        """AUROC after calibration minus AUROC before, per target.

        Zero or negative. Negative comes from ties: the step function collapses
        distinct scores onto one value and tied pairs score half. How negative
        depends on how coarse the fit is, so a rare target can shift by more
        than a common one. A *positive* value means something other than the
        calibrator changed the scores, and is worth chasing down.
        """
        from sklearn.metrics import roc_auc_score
        cal = self.transform(probs)
        out = []
        for t in range(probs.shape[1]):
            yt = np.asarray(y)[:, t]
            if yt.min() == yt.max():
                out.append(float("nan"))
                continue
            out.append(float(roc_auc_score(yt, cal[:, t])
                             - roc_auc_score(yt, probs[:, t])))
        return out
