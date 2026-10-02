"""
06_approx_r2.py -- how much of the full model does a k-variable model reproduce?

Two questions AUROC cannot answer, both computed from predictions that are
already on disk. No GPU, no retraining.

1. Approximation R^2
--------------------
The idea is Harrell's, from Regression Modeling Strategies ch. 5, "Describing,
Resampling, Validating, and Simplifying the Model". Selecting variables
against the observed outcome y is noisy: y is a coin flip given the true risk,
so the selection chases the randomness in who happened to fall ill. Regress
instead on the full model's *predictions*, which are a smoothed estimate of
that risk, and the noise drops out of the selection problem.

    Provenance, so nobody cites this loosely. The chapter title, the section
    on approximating the full model, and the use of R2 against the full
    model's fitted values (rms::fastbw) were checked against hbiostat.org.
    The conventional target of 0.95 was NOT: it is stated here from memory
    and the page fetch did not reach that part of the chapter. Check the book
    before putting a threshold in the manuscript.

    R^2_t = 1 - sum_i (L_k[i,t] - L_ref[i,t])^2 / sum_i (L_ref[i,t] - Lbar)^2

Read it as: what fraction of the variation in the full model's risk estimates
does the reduced model reproduce. Harrell's conventional target is 0.95.

Note what this is *not*. Harrell approximates the full model's linear
predictor by least squares and steps down from there, which gives the best a
subset could possibly do. We have twenty-one independently trained models, so
we measure how close the model we would actually deploy comes to the full one.
That number is lower than Harrell's, and it is the more relevant of the two
for a paper about what to put at the bedside. The Methods must say which one
was computed.

Why bother, given we already have AUROC. AUROC only sees ranking: two models
can order every patient identically and still put one at 8% risk and the
other at 20%. The bedside display shows a number, not a rank, so the absolute
agreement matters and AUROC is blind to it.

    The noise floor. all63 and rank63 are two 63-variable models differing
    only in column order and training noise. Their R^2 against each other is
    what "no difference in information" looks like on this scale. Every other
    number should be read against it, not against 1.0.

2. Decision curve analysis
--------------------------
Vickers and Elkin. Net benefit at threshold probability p:

    NB(p) = TP/N - (FP/N) * p/(1-p)

compared against treating everyone (NB = prev - (1-prev) * p/(1-p)) and
treating nobody (NB = 0). It asks whether the smaller model changes decisions
at thresholds anyone would actually use, which is the question a clinician
has, rather than whether it changes an average over every threshold at once.

Net benefit needs calibrated probabilities -- an uncalibrated model can look
wrong at a threshold purely because its scale is off -- so isotonic regression
is fitted per endpoint on the validation split and applied to the test split
before the curve is computed. Isotonic is monotone, so AUROC is untouched.

    python 06_approx_r2.py                      the parsimony grid
    python 06_approx_r2.py --drop triage_acuity the no-ESI grid
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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

ACUTE = list(range(2, 9))       # the seven acute condition endpoints
SEVERITY = [0, 1]               # hospitalisation, critical event


# --------------------------------------------------------------------------
def load_pred(tag, split):
    """Predictions for one run, on the logit scale, or None if absent.

    The file is called logits_umn_*.npy for every arm, but the tree arms write
    probabilities into it (01_train.py line ~367 says so in a comment). A
    filename is not evidence. Anything lying entirely inside [0, 1] is treated
    as probabilities and mapped back to logits, so the two can never be
    silently averaged on different scales.
    """
    p = f"{RUNS}/{tag}/logits_umn_{split}.npy"
    if not os.path.exists(p):
        return None
    a = np.load(p).astype(np.float64)
    if a.min() >= 0.0 and a.max() <= 1.0:
        a = np.clip(a, 1e-7, 1 - 1e-7)
        a = np.log(a / (1 - a))
    return a


def load_labels(tag, split):
    p = f"{RUNS}/{tag}/labels_umn_{split}.npy"
    return np.load(p).astype(np.float64) if os.path.exists(p) else None


def r2_against(Lk, Lref):
    """Per-endpoint R^2 of Lk against Lref. Not symmetric: the reference
    supplies the denominator, because the question is what fraction of *its*
    variation is recovered."""
    resid = ((Lk - Lref) ** 2).sum(axis=0)
    total = ((Lref - Lref.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - resid / np.where(total == 0, np.nan, total)


def prob_gap(Lk, Lref):
    """Mean absolute difference in predicted probability, in percentage points.
    The most directly readable number here: how far apart are the two risks
    the dashboard would show for the same patient."""
    s = lambda z: 1.0 / (1.0 + np.exp(-z))
    return np.abs(s(Lk) - s(Lref)).mean(axis=0) * 100.0


# --------------------------------------------------------------------------
def isotonic(x_fit, y_fit, x_apply):
    """Isotonic calibration on the logit scale, val -> test."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(x_fit, y_fit)
    return ir.predict(x_apply)


def net_benefit(p_hat, y, thresholds):
    """NB(p) = TP/N - (FP/N) * p/(1-p), and the treat-all comparator."""
    N = len(y)
    prev = y.mean()
    nb, nb_all = [], []
    for p in thresholds:
        flag = p_hat >= p
        tp = float((flag & (y == 1)).sum())
        fp = float((flag & (y == 0)).sum())
        w = p / (1.0 - p)
        nb.append(tp / N - (fp / N) * w)
        nb_all.append(prev - (1.0 - prev) * w)
    return np.array(nb), np.array(nb_all)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="G2_no_l1_no_drop")
    ap.add_argument("--drop", default="")
    ap.add_argument("--reference", default=None,
                    help="run tag suffix used as the full model; defaults to "
                         "rank63 (or rank62 when a variable is dropped)")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--no-dca", action="store_true")
    ap.add_argument("--dca-k", default="7,13,15",
                    help="variable counts to draw decision curves for, plus "
                         "the full model, which is always included. These "
                         "were hard-coded at 7 and 15 until the clean "
                         "parsimony curve put the elbow at 13 and the one "
                         "number the argument needed was the one not there.")
    a = ap.parse_args()

    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)
    sfx = _t.drop_suffix(drop)
    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(RES, exist_ok=True)

    meta = json.load(open(f"{DATA}/preprocessing.json"))
    names = [o.replace("outcome_", "") for o in meta["outcomes"]]

    ranking = _t.ranking_path(drop)
    if not os.path.exists(ranking):
        raise SystemExit(f"{ranking} not found")
    with open(ranking) as fh:
        n_all = len([r for r in csv.DictReader(fh) if r["feature"] not in drop])
    ref_k = a.reference or f"rank{n_all}"
    print(f"reference model: {a.arm}_{ref_k}{sfx}   ({n_all} variables)\n")

    # Match the directory name exactly rather than picking it apart with
    # string splits. The arm name itself contains "_no_", the fullrank runs
    # end in "rank<k>" too, and the dropped runs differ only by an infix --
    # three ways for a loose match to pull in the wrong series and report it
    # as this one. An anchored pattern either matches or it does not.
    pat = re.compile(rf"^{re.escape(a.arm)}_rank(\d+){re.escape(sfx)}"
                     rf"_seed{seeds[0]}$")
    ks = sorted({int(m.group(1)) for d in os.listdir(RUNS)
                 if (m := pat.match(d))})
    if not ks:
        raise SystemExit(f"no directories match {pat.pattern}")

    rows = []
    for split in ("val", "test"):
        for seed in seeds:
            ref_tag = f"{a.arm}_{ref_k}{sfx}_seed{seed}"
            Lref = load_pred(ref_tag, split)
            if Lref is None:
                print(f"  [skip] {ref_tag} has no {split} predictions")
                continue
            for k in ks:
                tag = f"{a.arm}_rank{k}{sfx}_seed{seed}"
                Lk = load_pred(tag, split)
                if Lk is None:
                    continue
                if Lk.shape != Lref.shape:
                    print(f"  [skip] {tag}: shape {Lk.shape} vs reference "
                          f"{Lref.shape}")
                    continue
                r2 = r2_against(Lk, Lref)
                gap = prob_gap(Lk, Lref)
                for t, nm in enumerate(names):
                    rows.append({"split": split, "k": k, "seed": seed,
                                 "endpoint": nm, "r2": r2[t],
                                 "prob_gap_pp": gap[t]})

    if not rows:
        raise SystemExit("nothing to compare")

    with open(f"{RES}/approx_r2{sfx}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- the noise floor -------------------------------------------------
    print("=" * 74)
    print("NOISE FLOOR   two full models, same variables, different training")
    print("=" * 74)
    floor = None
    for seed in seeds:
        A = load_pred(f"{a.arm}_all63{sfx}_seed{seed}", "test")
        B = load_pred(f"{a.arm}_{ref_k}{sfx}_seed{seed}", "test")
        if A is None or B is None or A.shape != B.shape:
            continue
        r2 = r2_against(A, B)
        floor = r2 if floor is None else np.vstack([floor, r2])
        print(f"  seed {seed}   all63 vs {ref_k}   "
              f"acute {np.nanmean(r2[ACUTE]):.4f}   "
              f"severity {np.nanmean(r2[SEVERITY]):.4f}")
    if floor is None:
        print("  not available (no all63 runs); read the table against 1.0,")
        print("  which will make every model look worse than it is")
    else:
        f = np.atleast_2d(floor)
        print(f"\n  Two models with identical information agree to "
              f"R2 = {np.nanmean(f[:, ACUTE]):.4f} (acute).")
        print("  Nothing below can beat that, so read the column against it.")

    # ---- the table -------------------------------------------------------
    for split in ("val", "test"):
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        print()
        print("=" * 74)
        print(f"APPROXIMATION R2 vs the {n_all}-variable model   [{split}]")
        print("=" * 74)
        print(f"{'k':>4} {'acute R2':>10} {'severity R2':>12} "
              f"{'all nine':>10} {'gap pp':>9}   Harrell 0.95")
        print("-" * 74)
        for k in ks:
            g = [r for r in sub if r["k"] == k]
            ac = np.nanmean([r["r2"] for r in g
                             if names.index(r["endpoint"]) in ACUTE])
            sv = np.nanmean([r["r2"] for r in g
                             if names.index(r["endpoint"]) in SEVERITY])
            al = np.nanmean([r["r2"] for r in g])
            gp = np.nanmean([r["prob_gap_pp"] for r in g])
            mark = "  reached" if ac >= 0.95 else ""
            print(f"{k:>4} {ac:>10.4f} {sv:>12.4f} {al:>10.4f} "
                  f"{gp:>9.2f}{mark}")

    # ---- per endpoint at the sizes that matter ---------------------------
    print()
    print("=" * 74)
    print("PER ENDPOINT   [test]")
    print("=" * 74)
    show = [k for k in (7, 10, 12, 13, 15, 20, 25) if k in ks]
    print(f"{'endpoint':<22}" + "".join(f"{'k=' + str(k):>9}" for k in show))
    print("-" * 74)
    for t, nm in enumerate(names):
        line = f"{nm:<22}"
        for k in show:
            v = [r["r2"] for r in rows if r["split"] == "test"
                 and r["k"] == k and r["endpoint"] == nm]
            line += f"{np.nanmean(v):>9.4f}" if v else f"{'-':>9}"
        grp = "  acute" if t in ACUTE else "  severity"
        print(line + grp)

    print(f"\nwritten to {RES}/approx_r2{sfx}.csv")

    if a.no_dca:
        return

    # ---- decision curves -------------------------------------------------
    print()
    print("=" * 74)
    print("DECISION CURVE   net benefit, isotonic calibration fitted on val")
    print("=" * 74)
    dca = []
    seed = seeds[0]
    want = [int(x) for x in a.dca_k.split(",") if x.strip()] + [n_all]
    missing = [k for k in want if k not in ks]
    if missing:
        print(f"  not in the grid, skipped: {missing}")
    for k in [k for k in dict.fromkeys(want) if k in ks]:
        tag = f"{a.arm}_rank{k}{sfx}_seed{seed}"
        Lv, Lt = load_pred(tag, "val"), load_pred(tag, "test")
        Yv, Yt = load_labels(tag, "val"), load_labels(tag, "test")
        if any(x is None for x in (Lv, Lt, Yv, Yt)):
            continue
        for t, nm in enumerate(names):
            prev = Yt[:, t].mean()
            if prev <= 0:
                continue
            p = isotonic(Lv[:, t], Yv[:, t], Lt[:, t])
            # thresholds around this endpoint's own prevalence: a grid fixed
            # in absolute terms would sit off the end for the rare endpoints
            th = np.clip(prev * np.array([0.25, 0.5, 1, 2, 4, 8]), 1e-4, 0.9)
            nb, nb_all = net_benefit(p, Yt[:, t], th)
            for j, x in enumerate(th):
                dca.append({"k": k, "seed": seed, "endpoint": nm,
                            "prevalence": prev, "threshold": x,
                            "net_benefit": nb[j], "nb_treat_all": nb_all[j]})

    if dca:
        with open(f"{RES}/decision_curve{sfx}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dca[0].keys()))
            w.writeheader(); w.writerows(dca)
        kk = sorted({d["k"] for d in dca})
        print(f"{'endpoint':<22}{'prev':>7}{'threshold':>11}"
              + "".join(f"{'k=' + str(k):>10}" for k in kk) + f"{'all':>10}")
        print("-" * 74)
        for nm in names:
            g = [d for d in dca if d["endpoint"] == nm]
            if not g:
                continue
            for x in sorted({d["threshold"] for d in g}):
                at = [d for d in g if d["threshold"] == x]
                line = (f"{nm:<22}{at[0]['prevalence']:>7.4f}{x:>11.4f}")
                for k in kk:
                    v = [d["net_benefit"] for d in at if d["k"] == k]
                    line += f"{v[0]:>10.5f}" if v else f"{'-':>10}"
                line += f"{at[0]['nb_treat_all']:>10.5f}"
                print(line)
            print()
        print(f"written to {RES}/decision_curve{sfx}.csv")
        print()
        print("  Net benefit is on the scale of true positives per patient.")
        print("  If k=15 and the full model are within a rounding error at")
        print("  the thresholds anyone would act on, that is the argument for")
        print("  the smaller model, and it is a stronger one than AUROC.")


if __name__ == "__main__":
    main()
