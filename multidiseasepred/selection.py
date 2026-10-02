"""Greedy forward selection of a variable budget.

Why not rank by importance and take the top k
---------------------------------------------
Attribution answers "how much does the fitted model use this variable", one
variable at a time. A prefix of that ranking answers nothing in particular when
the candidates are correlated: two variables carrying the same information both
score highly, and both get taken, so the budget is spent twice on one signal.

Greedy forward selection asks the question a variable budget is actually about:
at each step, which single remaining variable most improves held-out performance
*given everything already chosen*. A redundant variable falls to the bottom on
its own, because once its partner is in, adding it changes nothing.

The cost is quadratic -- choosing 12 from 62 is 62 + 61 + ... + 51 fits -- which
is why the selector is a gradient-boosted tree rather than the final network.
A selector does not have to be the model that is finally reported; it has to
rank candidate sets in the same order the final model would.

Which targets get a vote is a real decision, not a detail. A variable set is
only ever optimal for the thing it was selected against, so `targets=` is
explicit and has no default that quietly picks for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


@dataclass
class SelectionResult:
    """The whole curve, not just the answer.

    `order` is the sequence variables were added in; `scores[i]` is the
    held-out macro score with the first i+1 of them. Keeping the curve means
    the stopping rule can be re-applied at another threshold without refitting,
    which is what `k_at` does.
    """

    order: list[int]
    names: list[str]
    scores: list[float]
    full_score: float
    k: int
    threshold: float
    targets: list[int]

    @property
    def selected(self) -> list[int]:
        return self.order[: self.k]

    @property
    def selected_names(self) -> list[str]:
        return self.names[: self.k]

    def k_at(self, threshold: float) -> Optional[int]:
        """Smallest k reaching `threshold` x the full-variable score.

        Returns None when the sweep never got there, which is a real answer and
        not an error: it means the budget you asked about is larger than the
        curve that was computed.
        """
        target = threshold * self.full_score
        for i, s in enumerate(self.scores):
            if s >= target:
                return i + 1
        return None

    def summary(self) -> str:
        lines = [
            f"greedy forward selection over {len(self.names)} steps",
            f"  selected against targets {self.targets}",
            f"  full-variable score {self.full_score:.4f}",
            f"  rule: smallest k reaching {self.threshold:.0%} "
            f"({self.threshold * self.full_score:.4f})  ->  k = {self.k}",
            "",
            f"  {'k':>3}  {'variable':<34} {'score':>8} {'gain':>8}",
        ]
        prev = None
        for i, (n, s) in enumerate(zip(self.names, self.scores), start=1):
            gain = "" if prev is None else f"{s - prev:+.4f}"
            mark = " *" if i == self.k else "  "
            lines.append(f"  {i:>3}{mark}{n:<34} {s:>8.4f} {gain:>8}")
            prev = s
        return "\n".join(lines)


def _macro_auroc(y_true: np.ndarray, scores: np.ndarray,
                 targets: Sequence[int]) -> float:
    """Mean AUROC over `targets`, skipping any target with one class present.

    A target with no positives in the validation split has no AUROC. Dropping it
    from the mean is the only defensible option; counting it as 0.5 would let the
    number of degenerate targets drive the selection.
    """
    vals = []
    for t in targets:
        yt = y_true[:, t]
        if yt.min() == yt.max():
            continue
        vals.append(roc_auc_score(yt, scores[:, t]))
    if not vals:
        raise ValueError(
            "no target had both classes present in the validation split; "
            "selection cannot be scored")
    return float(np.mean(vals))


def _default_selector(seed: int) -> Callable:
    """A fast, well-behaved stand-in for the final model.

    Histogram gradient boosting rather than the network: 825 fits at a few
    seconds each is a coffee break, at four minutes each it is a day.
    """

    def fit_score(X_tr, y_tr, X_va, cols, targets):
        out = np.zeros((len(X_va), y_tr.shape[1]))
        for t in targets:
            m = HistGradientBoostingClassifier(
                max_iter=150, max_depth=6, learning_rate=0.1,
                random_state=seed)
            m.fit(X_tr[:, cols], y_tr[:, t])
            out[:, t] = m.predict_proba(X_va[:, cols])[:, 1]
        return out

    return fit_score


def greedy_forward(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    targets: Sequence[int],
    variable_names: Sequence[str],
    max_k: int = 15,
    threshold: float = 0.98,
    sample: Optional[int] = 400_000,
    seed: int = 42,
    selector: Optional[Callable] = None,
    verbose: bool = True,
) -> SelectionResult:
    """Add variables one at a time, keeping the one that helps most.

    Parameters
    ----------
    targets
        Column indices of `y` that get a vote. This is the design decision the
        whole result hangs on, so there is no default.
    max_k
        How far to run the sweep. Run it past the k you expect to keep: the
        stopping rule is applied to the finished curve, and a curve that stops
        exactly at the answer cannot show you what the next variable would have
        bought.
    threshold
        Fraction of the full-variable score the retained set must reach. Fix
        this before you look at the curve, and say in the paper that you did.
    sample
        Rows drawn for each fit. The ordering is decided long before the last
        hundred thousand rows are read, and subsampling turns a day into an hour.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    targets = list(targets)
    names = list(variable_names)

    if X_train.shape[1] != len(names):
        raise ValueError(
            f"X has {X_train.shape[1]} columns but {len(names)} names were given")
    if not targets:
        raise ValueError("targets is empty; nothing to select against")
    max_k = min(max_k, X_train.shape[1])

    rng = np.random.default_rng(seed)
    if sample and sample < len(X_train):
        idx = rng.choice(len(X_train), sample, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    fit_score = selector or _default_selector(seed)

    full = _macro_auroc(
        y_val, fit_score(X_train, y_train, X_val,
                         list(range(X_train.shape[1])), targets), targets)
    if verbose:
        print(f"full variable set ({X_train.shape[1]}): {full:.4f}")

    chosen: list[int] = []
    curve: list[float] = []
    remaining = list(range(X_train.shape[1]))

    for step in range(max_k):
        best_col, best_score = None, -np.inf
        for c in remaining:
            s = _macro_auroc(
                y_val, fit_score(X_train, y_train, X_val, chosen + [c], targets),
                targets)
            if s > best_score:
                best_col, best_score = c, s
        chosen.append(best_col)
        remaining.remove(best_col)
        curve.append(best_score)
        if verbose:
            gain = "" if step == 0 else f"  (+{best_score - curve[-2]:.4f})"
            print(f"  {step + 1:>3}  {names[best_col]:<34} {best_score:.4f}{gain}")

    res = SelectionResult(
        order=chosen, names=[names[c] for c in chosen], scores=curve,
        full_score=full, k=len(chosen), threshold=threshold, targets=targets)
    k = res.k_at(threshold)
    if k is None and verbose:
        print(f"\nwarning: {threshold:.0%} of {full:.4f} was not reached within "
              f"{max_k} variables; keeping all {max_k}. Raise max_k or lower "
              f"the threshold.")
    res.k = k if k is not None else len(chosen)
    if verbose:
        print(f"\nrule returned k = {res.k}: {res.selected_names}")
    return res
