"""
03_ensemble.py -- average the seeds, then ask whether the difference from
gradient boosting is real.

Nothing is retrained. Every run already wrote logits_umn_test.npy and
labels_umn_test.npy, so this reads them, averages the seeds of each arm on the
probability scale, and recomputes everything. That is the whole point of
having saved the probability matrices.

Two questions:

1. How much does averaging seeds buy? A deep ensemble of three networks
   usually adds a few thousandths of AUROC for free. The table reports the
   single-seed mean with its spread and the ensemble side by side, so the gain
   is visible rather than assumed.

2. Is our best arm actually ahead of gradient boosting? A difference of 0.001
   between two numbers computed on the same 350,842 patients is not something
   to eyeball. This does a paired bootstrap: one set of resampled row indices
   per replicate, shared by both models and all nine targets, and the
   statistic is the difference. Pairing matters -- the two models are being
   scored on the same patients, so their errors are correlated, and treating
   the two confidence intervals as independent would be far too conservative.

The comparison is only fair if both sides get the same seeds, which is why
02_arch_sweep now runs the tree baselines three times as well. Gradient
boosting with the hist tree method and no subsampling is close to
deterministic, so its three seeds may be nearly identical -- if so the table
will show a spread of zero, and that is a fact about the baseline worth
reporting, not something to hide.

    python 03_ensemble.py                    everything found
    python 03_ensemble.py --boot 30          quicker, wider intervals
    python 03_ensemble.py --compare D_moe_router,A2_xgb
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
RUNS = f"{PROJ}/runs"
RES = f"{PROJ}/results"

ACUTE = list(range(2, 9))     # the seven acute conditions
DISP = [0, 1]                 # the two disposition endpoints


def load_arm(arm, features="top15"):
    """Return (S, T, E) probabilities: seeds x patients x targets, plus y."""
    dirs = sorted(glob.glob(f"{RUNS}/{arm}_{features}_seed*"))
    P, seeds = [], []
    y = None
    for d in dirs:
        f = f"{d}/logits_umn_test.npy"
        if not os.path.exists(f):
            continue
        z = np.load(f).astype(np.float64)
        if not arm.startswith("A2"):
            z = 1.0 / (1.0 + np.exp(-z))     # trees already write probabilities
        P.append(z)
        seeds.append(int(d.rsplit("seed", 1)[-1]))
        if y is None:
            y = np.load(f"{d}/labels_umn_test.npy").astype(np.int8)
    if not P:
        return None, None, []
    return np.stack(P), y, seeds


def macro(y, p, cols):
    au = [roc_auc_score(y[:, j], p[:, j]) for j in cols]
    ap = [average_precision_score(y[:, j], p[:, j]) for j in cols]
    return float(np.mean(au)), float(np.mean(ap))


def summarise(y, P):
    """Per-seed macro metrics and the ensemble's."""
    per = [dict(zip(("aa", "ap", "da", "dp"),
                    macro(y, P[s], ACUTE) + macro(y, P[s], DISP)))
           for s in range(len(P))]
    ens = dict(zip(("aa", "ap", "da", "dp"),
                   macro(y, P.mean(0), ACUTE) + macro(y, P.mean(0), DISP)))
    out = {"n_seeds": len(P)}
    for k in ("aa", "ap", "da", "dp"):
        v = [d[k] for d in per]
        out[k + "_mean"] = float(np.mean(v))
        out[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        out[k + "_ens"] = ens[k]
    return out


def paired_bootstrap(y, pa, pb, B, seed=42):
    """Difference a minus b, per outcome and averaged, percentile intervals.

    One index set per replicate, reused for both models and all targets. The
    same indices across targets are what make the macro average's interval
    correct rather than an average of nine unrelated intervals.

    Per-outcome deltas come out of the same loop the macro average is built
    from -- the macro difference is the mean of the per-outcome differences,
    so recording them costs nothing and answers the question the macro number
    hides, which is whether we are ahead everywhere or ahead on average while
    behind on some outcomes.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    rows = []
    t0 = time.time()
    T = y.shape[1]
    for b in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        rec = {}
        d_au, d_ap = {}, {}
        for j in range(T):
            if yb[:, j].sum() == 0:
                continue
            d_au[j] = (roc_auc_score(yb[:, j], pa[idx, j])
                       - roc_auc_score(yb[:, j], pb[idx, j]))
            d_ap[j] = (average_precision_score(yb[:, j], pa[idx, j])
                       - average_precision_score(yb[:, j], pb[idx, j]))
            rec["t%d_auroc" % j] = d_au[j]
            rec["t%d_auprc" % j] = d_ap[j]
        for name, cols in (("acute", ACUTE), ("disp", DISP)):
            keep = [j for j in cols if j in d_au]
            rec[name + "_auroc"] = float(np.mean([d_au[j] for j in keep]))
            rec[name + "_auprc"] = float(np.mean([d_ap[j] for j in keep]))
        rows.append(rec)
        if b == 4:
            print("      (%.1f s for 5 replicates, about %.0f s for %d)"
                  % (time.time() - t0, (time.time() - t0) / 5 * B, B),
                  flush=True)
    return rows


def ci(vals, lo=2.5, hi=97.5):
    v = np.asarray(vals)
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="top15")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--challenger", default=None,
                    help="the arm we intend to report, named in advance")
    ap.add_argument("--compare", default=None,
                    help="challenger,reference; overrides both")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    arms = sorted({os.path.basename(d).rsplit("_" + a.features, 1)[0]
                   for d in glob.glob(f"{RUNS}/*_{a.features}_seed*")})
    if not arms:
        print("no runs found")
        sys.exit(1)

    print("=" * 96)
    print("SEED ENSEMBLE   probabilities averaged over seeds, "
          "held-out 2024-2025 cohort")
    print("=" * 96)
    print(f"{'arm':<20}{'n':>3}{'acute AUROC':>26}{'acute AUPRC':>26}"
          f"{'disp AUROC':>21}")
    print(f"{'':<20}{'':>3}{'single -> ens':>26}{'single -> ens':>26}"
          f"{'single -> ens':>21}")
    print("-" * 96)

    rows, store, y = [], {}, None
    for arm in arms:
        P, yy, seeds = load_arm(arm, a.features)
        if P is None:
            continue
        y = yy
        s = summarise(yy, P)
        s["arm"] = arm
        rows.append(s)
        store[arm] = P.mean(0)
        print("%-20s%3d  %.4f+-%.4f -> %.4f  %.4f+-%.4f -> %.4f  "
              "%.4f -> %.4f"
              % (arm, s["n_seeds"], s["aa_mean"], s["aa_sd"], s["aa_ens"],
                 s["ap_mean"], s["ap_sd"], s["ap_ens"],
                 s["da_mean"], s["da_ens"]))

    with open(f"{RES}/ensemble_comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "n_seeds"] +
                           [k + suf for k in ("aa", "ap", "da", "dp")
                            for suf in ("_mean", "_sd", "_ens")])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten to {RES}/ensemble_comparison.csv")

    # ---- who fights whom ------------------------------------------------
    # Naming the challenger in advance rather than taking whichever arm came
    # out highest. D and G2 ensemble to the same acute AUROC to four decimals,
    # so letting the script pick would be deciding on a coin flip and then
    # testing the winner -- a small but real selection effect. The reference
    # is the strongest tree baseline, which makes the claim harder to support,
    # not easier.
    trees = [r for r in rows if r["arm"].startswith("A2")]
    neural = [r for r in rows if not r["arm"].startswith("A2")]
    if a.compare:
        cha, ref = [x.strip() for x in a.compare.split(",")]
    elif not neural or not trees:
        print("\nnothing to compare")
        return
    else:
        if a.challenger and a.challenger in store:
            cha = a.challenger
        else:
            if a.challenger:
                print(f"\n[warn] {a.challenger} has no runs; "
                      f"falling back to the best neural arm")
            cha = max(neural, key=lambda r: r["aa_ens"])["arm"]
        ref = max(trees, key=lambda r: r["aa_ens"])["arm"]

    print("\n" + "=" * 96)
    print(f"PAIRED BOOTSTRAP   {cha}  minus  {ref}   B = {a.boot}")
    print("=" * 96)
    print("  positive means our model is ahead; an interval containing zero")
    print("  means the data do not separate them.\n")

    reps = paired_bootstrap(y, store[cha], store[ref], a.boot)
    out = []
    for key, label in (("acute_auroc", "acute macro AUROC"),
                       ("acute_auprc", "acute macro AUPRC"),
                       ("disp_auroc", "disposition macro AUROC"),
                       ("disp_auprc", "disposition macro AUPRC")):
        v = [r[key] for r in reps]
        lo, hi = ci(v)
        star = "" if lo <= 0 <= hi else "  *"
        print("  %-26s %+.4f   95%% CI %+.4f to %+.4f%s"
              % (label, np.mean(v), lo, hi, star))
        out.append({"metric": key, "outcome": "(macro)",
                    "group": key.split("_")[0],
                    "challenger": cha, "reference": ref,
                    "delta": float(np.mean(v)), "lo": lo, "hi": hi,
                    "B": a.boot})

    # Read the outcome names rather than hardcoding them. 00_prepare_data.py
    # writes the exact column order into preprocessing.json, and that order is
    # what the label matrix uses; a hardcoded list here would silently drift
    # and put the wrong disease against the wrong number.
    try:
        import json
        NAMES = json.load(open(f"{PROJ}/data/preprocessing.json"))["outcomes"]
        NAMES = [n.replace("outcome_", "") for n in NAMES]
    except Exception:
        NAMES = ["target %d" % j for j in range(9)]
    print("\n  Per outcome, AUROC, %s minus %s:" % (cha, ref))
    for j in range(9):
        key = "t%d_auroc" % j
        if key not in reps[0]:
            continue
        v = [r[key] for r in reps]
        lo, hi = ci(v)
        mark = "  ahead" if lo > 0 else ("  behind" if hi < 0 else "")
        label = NAMES[j] if j < len(NAMES) else "target %d" % j
        group = "disposition" if j in DISP else "acute"
        print("    %-20s %-12s %+.4f   95%% CI %+.4f to %+.4f%s"
              % (label, group, np.mean(v), lo, hi, mark))
        out.append({"metric": key, "outcome": label, "group": group,
                    "challenger": cha, "reference": ref,
                    "delta": float(np.mean(v)), "lo": lo, "hi": hi,
                    "B": a.boot})

    # B sensitivity comes free: recompute the interval from prefixes
    print("\n  How many bootstrap replicates are enough (acute macro AUROC):")
    v = [r["acute_auroc"] for r in reps]
    for b in (30, 50, 100, 200, 500, 1000):
        if b > len(v):
            continue
        lo, hi = ci(v[:b])
        print("    B = %4d   %+.4f to %+.4f   width %.4f" % (b, lo, hi, hi - lo))

    with open(f"{RES}/bootstrap_delta.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwritten to {RES}/bootstrap_delta.csv")


if __name__ == "__main__":
    main()
