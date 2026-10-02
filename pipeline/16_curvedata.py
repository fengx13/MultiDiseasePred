"""
16_curvedata.py -- curve coordinates compact enough to leave through a screenshot.

    python 16_curvedata.py --dataset umn_test --arm G2_no_l1_no_drop \
        --features greedy:12 --drop triage_acuity --what roc

Nothing leaves this enclave as a file. The probability arrays are hundreds of
thousands of rows and cannot be exported; a PDF drawn here cannot be exported
either. What can leave is whatever fits legibly on a screen and gets
photographed.

So the curves are put on a FIXED x grid, agreed in advance and printed in the
header, and only the y values travel. A ROC curve becomes twenty numbers on
one line instead of forty thousand (fpr, tpr) pairs; nine endpoints become nine
lines; one screenshot carries a whole panel.

That is a real loss of resolution, and it is the right trade. A published ROC
figure is a few centimetres wide and twenty well-placed points draw it to
within the line width. The grid is denser where ROC curves bend -- below 0.1
false positive rate -- because that is the region a triage reader cares about
and the region where uniform spacing would waste most of its points.

    --what roc     TPR at each fixed FPR
    --what pr      precision at each fixed recall
    --what cal     ten equal-count bins: mean predicted, observed
    --what dca     net benefit at each fixed threshold, plus treat-all
    --what all     every block in turn

Read the numbers back with tools/fig_from_curvedata.py, which expects exactly
this layout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib                                              # noqa: E402
_t = importlib.import_module("01_train")                      # noqa: E402

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
DATA = f"{PROJ}/data"
RUNS = f"{PROJ}/runs"
RES = f"{PROJ}/results"

# Denser below 0.1: that is where a ROC curve for a rare endpoint does all of
# its bending, and where a triage reader operates.
FPR_GRID = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15,
                     0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0])
REC_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
                     0.80, 0.90, 0.95, 1.0])
THR_GRID = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075,
                     0.10, 0.15, 0.20, 0.30, 0.40, 0.50])


def roc_on_grid(y, s):
    y = np.asarray(y).astype(bool)
    if y.sum() == 0 or y.sum() == len(y):
        return np.full(len(FPR_GRID), np.nan)
    o = np.argsort(-np.asarray(s, dtype=np.float64), kind="mergesort")
    ys = y[o].astype(np.float64)
    tpr = np.cumsum(ys) / ys.sum()
    fpr = np.cumsum(1 - ys) / (len(ys) - ys.sum())
    # np.interp needs an increasing x; fpr already is, being a cumulative sum
    return np.interp(FPR_GRID, np.concatenate([[0.0], fpr]),
                     np.concatenate([[0.0], tpr]))


def pr_on_grid(y, s):
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return np.full(len(REC_GRID), np.nan)
    o = np.argsort(-np.asarray(s, dtype=np.float64), kind="mergesort")
    ys = y[o].astype(np.float64)
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tp / ys.sum()
    # recall is non-decreasing but flat in places, so keep the last point of
    # each flat run -- the highest precision achievable at that recall
    keep = np.concatenate([np.diff(rec) > 0, [True]])
    return np.interp(REC_GRID, rec[keep], prec[keep])


def cal_bins(y, p, nb=10):
    """Equal-count bins. Returns (mean predicted, observed) per bin.

    Equal-count, not equal-width: at 0.1% prevalence every prediction lands in
    the first equal-width bin and the plot becomes a description of the
    binning rather than of the model.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    o = np.argsort(p, kind="mergesort")
    ps, ys = p[o], y[o]
    e = np.linspace(0, len(ps), nb + 1).astype(int)
    return (np.array([ps[e[i]:e[i + 1]].mean() for i in range(nb)]),
            np.array([ys[e[i]:e[i + 1]].mean() for i in range(nb)]))


def net_benefit(y, p):
    """Vickers' net benefit at each threshold, and treat-all for reference.

    A decision curve without treat-all is unreadable: the whole point is
    whether the model beats flagging everybody, and at 0.1% prevalence
    treat-all is negative almost immediately.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    prev = y.mean()
    nb, all_ = [], []
    for t in THR_GRID:
        pos = np.asarray(p) >= t
        tp = float(np.sum(pos & (y == 1)))
        fp = float(np.sum(pos & (y == 0)))
        w = t / (1.0 - t)
        nb.append(tp / n - (fp / n) * w)
        all_.append(prev - (1.0 - prev) * w)
    return np.array(nb), np.array(all_)


def fmt(v, w=7, d=4):
    return "".join(("nan" if not np.isfinite(x) else f"{x:.{d}f}").rjust(w)
                   for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="umn_test",
                    choices=["umn_val", "umn_test", "bidmc", "stanford"])
    ap.add_argument("--arm", default="G2_no_l1_no_drop")
    ap.add_argument("--features", default="greedy:12")
    ap.add_argument("--drop", default="triage_acuity")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--what", default="all",
                    choices=["roc", "pr", "cal", "dca", "all"])
    a = ap.parse_args()

    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)
    sfx = _t.drop_suffix(drop)
    tag = a.features.replace(":", "")
    meta = json.load(open(f"{DATA}/preprocessing.json"))
    names = [o.replace("outcome_", "") for o in meta["outcomes"]]

    S, Y, scale = [], None, None
    for s in [int(x) for x in a.seeds.split(",")]:
        d = f"{RUNS}/{a.arm}_{tag}{sfx}_seed{s}"
        f1, f2 = f"{d}/logits_{a.dataset}.npy", f"{d}/labels_{a.dataset}.npy"
        if os.path.exists(f1) and os.path.exists(f2):
            S.append(np.load(f1))
            Y = np.load(f2)
            if scale is None and os.path.exists(f"{d}/metrics.json"):
                scale = json.load(open(f"{d}/metrics.json")).get("score_scale")
    if not S:
        raise SystemExit(f"no scores for {a.arm} at {a.features} on {a.dataset}")
    P = np.mean(S, axis=0)
    if a.arm.startswith("A5_news"):
        scale = "points"
    if scale is None:
        scale = "probability" if (P.min() >= 0 and P.max() <= 1) else "logit"
    prob = P if scale == "probability" else 1.0 / (1.0 + np.exp(-P))

    print(f"# arm      {a.arm}")
    print(f"# features {a.features}   dropped {list(drop)}")
    print(f"# dataset  {a.dataset}   n = {len(Y):,}   seeds {len(S)}")
    print(f"# scale    {scale}")
    print(f"# data_id  {meta.get('data_id')}")

    if a.what in ("roc", "all"):
        print("\n=== ROC: TPR at fixed FPR ===")
        print("FPR   " + fmt(FPR_GRID))
        for t, nm in enumerate(names):
            print(f"{nm:<18}" + fmt(roc_on_grid(Y[:, t], P[:, t])))

    if a.what in ("pr", "all"):
        print("\n=== PR: precision at fixed recall ===")
        print("REC   " + fmt(REC_GRID))
        for t, nm in enumerate(names):
            print(f"{nm:<18}" + fmt(pr_on_grid(Y[:, t], P[:, t])))

    if a.what in ("cal", "all"):
        if scale == "points":
            print("\n=== CAL: skipped, NEWS points are not probabilities ===")
        else:
            print("\n=== CAL: ten equal-count bins, predicted then observed ===")
            for t, nm in enumerate(names):
                mp, ob = cal_bins(Y[:, t], prob[:, t])
                print(f"{nm:<18}p" + fmt(mp, 8, 5))
                print(f"{'':<18}o" + fmt(ob, 8, 5))

    if a.what in ("dca", "all"):
        if scale == "points":
            print("\n=== DCA: skipped, NEWS points are not probabilities ===")
        else:
            print("\n=== DCA: net benefit at fixed threshold, then treat-all ===")
            print("THR   " + fmt(THR_GRID, 8, 4))
            for t, nm in enumerate(names):
                nb, al = net_benefit(Y[:, t], prob[:, t])
                print(f"{nm:<18}m" + fmt(nb, 8, 5))
                print(f"{'':<18}a" + fmt(al, 8, 5))


if __name__ == "__main__":
    main()
