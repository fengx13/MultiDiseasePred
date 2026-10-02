"""
14_metrics.py -- the full results table: every dataset, every model, every
endpoint, every metric, every interval.

    python 14_metrics.py --features greedy:12 --drop triage_acuity \
        --arms G2_no_l1_no_drop,A2_xgb_sub,A4_lgbm,A2_rf,A2_lr,A2_histgb \
        --boot 200 --csv

Writes results/metrics_<featuretag>_<dropsuffix>.csv with one row per
(dataset, arm, endpoint) and columns:

    auroc auprc  sens spec ppv npv f1 accuracy alert_rate  tp fp fn tn

each with a 95% interval, plus MACRO acute and MACRO disposition rows.

Where the numbers come from
---------------------------
Nothing is refitted. Every arm has already written its scores to
logits_<dataset>.npy at training time -- for the neural arms a logit, for the
tree arms a probability, recorded in metrics.json as score_scale. Seeds are
averaged the same way 10_ci.py averages them, on whatever scale the file
holds, so this agrees with the interval tables rather than quietly disagreeing
by a fourth decimal.

Thresholds
----------
Sensitivity, specificity, PPV, NPV and F1 are not properties of a model. They
are properties of a model and a threshold, and where the threshold comes from
decides whether the numbers mean anything.

The threshold is chosen on the internal VALIDATION split, per arm and per
endpoint, and then frozen. The same number is applied to the internal test
cohort and to both external cohorts, with no re-optimisation anywhere.

Two rules are computed and both are reported:

    youden    maximises sensitivity + specificity - 1. Conventional, and
              symmetric: it prices a missed ARDS the same as a false alarm on
              a well patient. At 0.1% prevalence that gives a threshold whose
              PPV is a few per cent. That is not wrong, but it is not a
              deployable operating point either, so PPV is reported beside it
              and never omitted.

    sens90    the highest threshold still reaching 90% sensitivity on
              validation. This is how a triage instrument is actually
              specified -- you fix the miss rate you can live with and accept
              the workload that follows.

Choosing the threshold on the test set instead would bias sensitivity and
specificity upward simultaneously, and re-optimising it on MIMIC or Stanford
would silently convert zero-shot external validation into per-site tuning.
The gap between the 90% sensitivity asked for on validation and the
sensitivity actually obtained at BIDMC is not a defect of this script. It is a
measurement of how far the calibration has drifted, and it belongs in the
paper.

Intervals
---------
AUROC uses DeLong, which is exact and needs no resampling.

Everything else uses a bootstrap over encounters, with the replicate indices
SHARED across arms and endpoints within a dataset, so that differences between
arms are paired. The interval is the bootstrap standard error times 1.96
rather than the 2.5th and 97.5th percentiles: at B=200 a percentile interval
is determined by the fifth largest value and jitters visibly between runs,
while the standard error is stable to a few per cent. With B=1000 the two
agree; the normal form is used so that B can stay small.

The macro rows average across endpoints INSIDE each replicate. Averaging nine
separately computed intervals would understate the width, because the nine
endpoints are measured on the same patients and their errors are correlated.
"""

from __future__ import annotations

import argparse
import csv
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

# umn_val is loaded always -- it is where the thresholds come from -- but it is
# reported too, because a reader is entitled to see the split that the
# operating point was taken from.
DATASETS = ["umn_val", "umn_test", "bidmc", "stanford"]
LABEL = {"umn_val": "UMN validation", "umn_test": "UMN test 2024-2025",
         "bidmc": "MIMIC-IV-ED (BIDMC)", "stanford": "Stanford"}
ACUTE, DISP = list(range(2, 9)), [0, 1]
Z = 1.959963984540054


# ----------------------------------------------------------------- metrics --
def fast_auroc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - m * (m + 1) / 2.0) / (m * n))


def fast_auprc(y, s):
    """Average precision, the same estimator sklearn uses, without the import.

    Ties are handled by ranking with mergesort and walking the sorted labels,
    so a model that assigns the same score to many encounters -- which the
    NEWS arm does constantly, it being an integer between 0 and 20 -- is not
    silently given credit for an ordering it did not produce.
    """
    y = np.asarray(y).astype(np.float64)
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="mergesort")
    ys = y[order]
    ss = np.asarray(s, dtype=np.float64)[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    # collapse tied score groups to their last index
    keep = np.empty(len(ss), dtype=bool)
    keep[:-1] = ss[1:] != ss[:-1]
    keep[-1] = True
    tp, fp = tp[keep], fp[keep]
    total = y.sum()
    if total == 0:
        return float("nan")
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / total
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def counts(y, s, thr):
    p = s >= thr
    yb = y.astype(bool)
    tp = int(np.count_nonzero(p & yb))
    fp = int(np.count_nonzero(p & ~yb))
    fn = int(np.count_nonzero(~p & yb))
    tn = int(np.count_nonzero(~p & ~yb))
    return tp, fp, fn, tn


def from_counts(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    d = lambda a, b: (a / b) if b else float("nan")      # noqa: E731
    sens = d(tp, tp + fn)
    prec = d(tp, tp + fp)
    return {
        "sens": sens,
        "spec": d(tn, tn + fp),
        "ppv": prec,
        "npv": d(tn, tn + fn),
        "f1": d(2 * tp, 2 * tp + fp + fn),
        "accuracy": d(tp + tn, n),
        "alert_rate": d(tp + fp, n),
    }


THRESH_METRICS = ["sens", "spec", "ppv", "npv", "f1", "accuracy", "alert_rate"]


def pick_threshold(y, s, rule):
    """Threshold from the validation split. Returns nan if unusable."""
    y = np.asarray(y).astype(bool)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="mergesort")
    ys = y[order].astype(np.float64)
    ss = np.asarray(s, dtype=np.float64)[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    keep = np.empty(len(ss), dtype=bool)
    keep[:-1] = ss[1:] != ss[:-1]
    keep[-1] = True
    tp, fp, thr = tp[keep], fp[keep], ss[keep]
    P, N = ys.sum(), len(ys) - ys.sum()
    tpr, fpr = tp / P, fp / N
    if rule == "youden":
        return float(thr[int(np.argmax(tpr - fpr))])
    if rule.startswith("sens"):
        target = float(rule[4:]) / 100.0
        ok = np.nonzero(tpr >= target)[0]
        # thr is descending, so the first index reaching the target is the
        # highest threshold that still does. Taking the last one instead would
        # hand back a threshold near zero that flags everybody.
        return float(thr[ok[0]]) if len(ok) else float(thr[-1])
    raise SystemExit(f"unknown threshold rule {rule}")


# --------------------------------------------------------------- calibration --
def to_linear_predictor(sc, scale):
    """Scores on whatever scale they were stored, returned as a logit.

    The neural arms store the linear predictor (BCEWithLogitsLoss), the tree
    arms store predict_proba, and NEWS stores integer points. Getting this
    wrong is not hypothetical: on 2026-08-11 a logit was passed to a routine
    that clipped it to [1e-6, 1-1e-6] as though it were a probability, which
    pinned every negative value to a single constant and produced a full table
    of plausible, meaningless calibration slopes. AUROC and AUPRC did not
    notice, because they are rank statistics and the clip is monotone.

    Returns (lp, frac_at_bound). The second number matters: a random forest
    with a hundred trees returns exact zeros for rare endpoints, and those
    become a mass point at the clip. A calibration slope fitted through a mass
    point is not identified, so the fraction is reported rather than buried.
    """
    sc = np.asarray(sc, dtype=np.float64)
    if scale == "logit":
        return sc, 0.0
    if scale == "probability":
        lo, hi = 1e-12, 1.0 - 1e-12
        at = float(np.mean((sc <= lo) | (sc >= hi)))
        p = np.clip(sc, lo, hi)
        return np.log(p / (1.0 - p)), at
    return None, float("nan")          # points: no probability scale at all


def calibration(y, lp):
    """Slope and intercept with analytic standard errors from (X'WX)^-1."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, dtype=np.float64)
    nan4 = (float("nan"),) * 4
    if lp is None or y.sum() == 0 or y.sum() == len(y):
        return nan4
    lp = np.asarray(lp, dtype=np.float64)
    if not np.isfinite(lp).all() or lp.std() < 1e-9:
        return nan4
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    m.fit(lp.reshape(-1, 1), y)
    b1, b0 = float(m.coef_[0][0]), float(m.intercept_[0])
    q = 1.0 / (1.0 + np.exp(-(b0 + b1 * lp)))
    w = q * (1.0 - q)
    s00, s01, s11 = w.sum(), (w * lp).sum(), (w * lp * lp).sum()
    det = s00 * s11 - s01 * s01
    if not np.isfinite(det) or abs(det) < 1e-12:
        return b1, b0, float("nan"), float("nan")
    return b1, b0, float(np.sqrt(s00 / det)), float(np.sqrt(s11 / det))


def brier_ece(y, lp, bins=10):
    """Brier score, and expected calibration error over equal-count bins.

    Equal-count rather than equal-width. With prevalences down to 0.1% almost
    every predicted probability sits in the first equal-width bin, and an ECE
    computed that way is a description of the binning.
    """
    if lp is None:
        return float("nan"), float("nan")
    y = np.asarray(y, dtype=np.float64)
    p = 1.0 / (1.0 + np.exp(-np.asarray(lp, dtype=np.float64)))
    brier = float(np.mean((p - y) ** 2))
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    edges = np.linspace(0, len(ps), bins + 1).astype(int)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        ece += (hi - lo) * abs(ys[lo:hi].mean() - ps[lo:hi].mean())
    return brier, float(ece / len(ps))


# -------------------------------------------------------------------- DeLong --
def midrank(x):
    j = np.argsort(x, kind="mergesort")
    z = x[j]
    n = len(x)
    t = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n, dtype=np.float64)
    out[j] = t
    return out


def delong_se(scores, y):
    y = np.asarray(y).astype(bool)
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    pos, neg = np.asarray(scores)[y], np.asarray(scores)[~y]
    tx, ty = midrank(pos), midrank(neg)
    tz = midrank(np.concatenate([pos, neg]))
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    auc = tz[:m].sum() / (m * n) - (m + 1.0) / (2.0 * n)
    var = v01.var(ddof=1) / m + v10.var(ddof=1) / n
    return float(auc), float(np.sqrt(max(var, 0.0)))


# ---------------------------------------------------------------- loading --
def load_scores(arm, tag_feat, sfx, seeds, dataset):
    """Mean score over seeds, plus the scale those scores are on.

    The scale is read from metrics.json rather than guessed. Runs written
    before that field existed fall back to inference, which is safe in this
    direction only: a probability is always inside [0, 1] and a logit over
    hundreds of thousands of encounters never is. NEWS is named explicitly,
    because integer points are neither and no inference would catch it.
    """
    # NEWS does not vary with the variable set: it is the same five vitals
    # whatever k is chosen, so 12_news_baseline.py writes one directory named
    # A5_news_all63_... and never one per k. Looking for it under the current
    # tag finds nothing and drops the whole column, which is exactly what
    # happened to the first k=12 table -- silently, because a missing arm is
    # reported once at the top and then simply left out of every row.
    if arm.startswith("A5_news"):
        tag_feat = "all63"
    S, Y, scale = [], None, None
    for s in seeds:
        d = f"{RUNS}/{arm}_{tag_feat}{sfx}_seed{s}"
        fs, fy = f"{d}/logits_{dataset}.npy", f"{d}/labels_{dataset}.npy"
        if not (os.path.exists(fs) and os.path.exists(fy)):
            continue
        S.append(np.load(fs))
        yk = np.load(fy)
        # Averaging predictions across seeds is only meaningful if the seeds
        # scored the same patients. On the test and external cohorts they
        # always do. On the validation split they did not, until 2026-08-11:
        # 01_train.py drew the split with the model seed, so row i of seed 42
        # and row i of seed 43 were different people, and the mean of the
        # three was noise. It read 0.7911 for our model against 0.9002 on
        # test and went unnoticed because it looked like a plausible number.
        #
        # This is the assertion that would have caught it in one line.
        if Y is not None and not np.array_equal(Y, yk):
            raise SystemExit(
                f"{arm} at {dataset}: seed {s} has different labels from the\n"
                f"earlier seeds, so their predictions describe different\n"
                f"patients and must not be averaged.\n"
                f"\n"
                f"This is the per-seed validation split. Retrain with the\n"
                f"current 01_train.py, which fixes VAL_SPLIT_SEED, or pass\n"
                f"--seeds 42 to score a single model.")
        Y = yk
        mj = f"{d}/metrics.json"
        if scale is None and os.path.exists(mj):
            try:
                scale = json.load(open(mj)).get("score_scale")
            except Exception:
                scale = None
    if not S:
        return None, None, None
    M = np.mean(S, axis=0)
    if arm.startswith("A5_news"):
        scale = "points"
    if scale is None:
        scale = ("probability" if (M.min() >= 0.0 and M.max() <= 1.0)
                 else "logit")
    return M, Y, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="G2_no_l1_no_drop,A2_xgb_sub,A4_lgbm,"
                                      "A2_rf,A2_lr,A2_histgb")
    ap.add_argument("--features", default="greedy:12")
    ap.add_argument("--drop", default="triage_acuity")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--rules", default="youden,sens90")
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",")]
    rules = [x.strip() for x in a.rules.split(",") if x.strip()]
    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)
    sfx = _t.drop_suffix(drop)
    tag_feat = a.features.replace(":", "")

    meta = json.load(open(f"{DATA}/preprocessing.json"))
    names = [o.replace("outcome_", "") for o in meta["outcomes"]]
    T = len(names)

    print(f"variables   {a.features}")
    print(f"dropped     {list(drop)}")
    print(f"seeds       {seeds}")
    print(f"bootstrap   B = {a.boot}, paired within each dataset")
    print(f"thresholds  {rules}, fixed on umn_val and never re-optimised")
    print(f"data_id     {meta.get('data_id')}\n")

    # -- load everything first, so a missing arm is reported once and up front
    S = {}                       # (arm, dataset) -> (scores, labels)
    for arm in arms:
        for ds in DATASETS:
            sc, y, scale = load_scores(arm, tag_feat, sfx, seeds, ds)
            if sc is not None:
                S[(arm, ds)] = (sc, y, scale)
    print(f"{'arm':<22}" + "".join(f"{d:>22}" for d in DATASETS))
    for arm in arms:
        row = f"{arm:<22}"
        for ds in DATASETS:
            row += f"{('ok' if (arm, ds) in S else 'MISSING'):>22}"
        print(row)
    print()
    missing = [(arm, ds) for arm in arms for ds in DATASETS
               if (arm, ds) not in S]
    if missing:
        print("Missing cells are left out of the table rather than filled in.")
        print("An arm with no external scores was trained before 01_train.py")
        print("learned to score the external cohorts; retrain it.\n")

    # -- thresholds, from validation only
    thr = {}                     # (rule, arm, t) -> float
    for rule in rules:
        for arm in arms:
            if (arm, "umn_val") not in S:
                continue
            sv, yv, _ = S[(arm, "umn_val")]
            for t in range(T):
                thr[(rule, arm, t)] = pick_threshold(yv[:, t], sv[:, t], rule)

    rows = []
    for ds in DATASETS:
        present = [arm for arm in arms if (arm, ds) in S]
        if not present:
            continue
        n = len(S[(present[0], ds)][0])
        rng = np.random.default_rng(42)
        bidx = rng.integers(0, n, size=(a.boot, n)) if a.boot else None

        print("=" * 100)
        print(f"{LABEL[ds]}   {n:,} encounters")
        print("=" * 100)

        for rule in rules:
            for arm in present:
                sc, y, scale = S[(arm, ds)]
                # per endpoint, plus the two macro rows
                boot = {m: np.full((a.boot, T), np.nan)
                        for m in THRESH_METRICS + ["auprc", "auroc"]}
                per = []
                for t in range(T):
                    th = thr.get((rule, arm, t), float("nan"))
                    au, se = delong_se(sc[:, t], y[:, t])
                    ap_ = fast_auprc(y[:, t], sc[:, t])
                    c = counts(y[:, t], sc[:, t], th)
                    m = from_counts(*c)
                    lp, at_bound = to_linear_predictor(sc[:, t], scale)
                    b1, b0, se1, se0 = calibration(y[:, t], lp)
                    br, ece = brier_ece(y[:, t], lp)
                    m["_cal"] = (b1, b0, se1, se0, br, ece, at_bound)
                    per.append((th, au, se, ap_, c, m))
                    for b in range(a.boot):
                        ix = bidx[b]
                        yb, sb = y[ix, t], sc[ix, t]
                        if yb.sum() == 0:
                            continue
                        boot["auroc"][b, t] = fast_auroc(yb, sb)
                        boot["auprc"][b, t] = fast_auprc(yb, sb)
                        mb = from_counts(*counts(yb, sb, th))
                        for k in THRESH_METRICS:
                            boot[k][b, t] = mb[k]

                def ci(key, t):
                    v = boot[key][:, t]
                    v = v[np.isfinite(v)]
                    if len(v) < 3:
                        return float("nan"), float("nan")
                    return float(v.mean()), float(v.std(ddof=1))

                for t in range(T):
                    th, au, se, ap_, c, m = per[t]
                    _, ap_sd = ci("auprc", t)
                    r = {"dataset": ds, "rule": rule, "arm": arm,
                         "endpoint": names[t], "n": n, "events": int(y[:, t].sum()),
                         "prevalence": round(float(y[:, t].mean()), 6),
                         "threshold": round(th, 6) if np.isfinite(th) else "",
                         "auroc": round(au, 5),
                         "auroc_lo": round(au - Z * se, 5),
                         "auroc_hi": round(au + Z * se, 5),
                         "auprc": round(ap_, 5),
                         "auprc_lo": round(ap_ - Z * ap_sd, 5),
                         "auprc_hi": round(ap_ + Z * ap_sd, 5),
                         "tp": c[0], "fp": c[1], "fn": c[2], "tn": c[3],
                         "score_scale": scale}
                    b1, b0, se1, se0, br, ece, at_bound = m["_cal"]
                    r.update({
                        "cal_slope": round(b1, 5),
                        "cal_slope_lo": round(b1 - Z * se1, 5),
                        "cal_slope_hi": round(b1 + Z * se1, 5),
                        "cal_intercept": round(b0, 5),
                        "cal_intercept_lo": round(b0 - Z * se0, 5),
                        "cal_intercept_hi": round(b0 + Z * se0, 5),
                        "brier": round(br, 6), "ece": round(ece, 6),
                        "frac_at_prob_bound": round(at_bound, 5)})
                    for k in THRESH_METRICS:
                        _, sd = ci(k, t)
                        r[k] = round(m[k], 5)
                        r[k + "_lo"] = round(m[k] - Z * sd, 5)
                        r[k + "_hi"] = round(m[k] + Z * sd, 5)
                    rows.append(r)

                # macro: average across endpoints inside each replicate
                for gname, g in (("MACRO acute", ACUTE),
                                 ("MACRO disposition", DISP)):
                    r = {"dataset": ds, "rule": rule, "arm": arm,
                         "endpoint": gname, "n": n, "events": -1,
                         "prevalence": -1, "threshold": "",
                         "tp": -1, "fp": -1, "fn": -1, "tn": -1,
                         "score_scale": scale,
                         "cal_slope": -1, "cal_slope_lo": -1,
                         "cal_slope_hi": -1, "cal_intercept": -1,
                         "cal_intercept_lo": -1, "cal_intercept_hi": -1,
                         "brier": -1, "ece": -1, "frac_at_prob_bound": -1}
                    for key in ["auroc", "auprc"] + THRESH_METRICS:
                        def point(t, key=key):
                            if key == "auroc":
                                return per[t][1]
                            if key == "auprc":
                                return per[t][3]
                            return per[t][5][key]
                        pt = float(np.nanmean([point(t) for t in g]))
                        # averaged inside each replicate, not across nine
                        # separately computed intervals
                        bs = np.nanmean(boot[key][:, g], axis=1)
                        bs = bs[np.isfinite(bs)]
                        sd = float(bs.std(ddof=1)) if len(bs) > 2 else float("nan")
                        r[key] = round(pt, 5)
                        r[key + "_lo"] = round(pt - Z * sd, 5)
                        r[key + "_hi"] = round(pt + Z * sd, 5)
                    rows.append(r)

                p = [x for x in rows if x["arm"] == arm and x["dataset"] == ds
                     and x["rule"] == rule and x["endpoint"] == "MACRO acute"][0]
                print(f"  {rule:<8} {arm:<22} macro acute  "
                      f"AUROC {p['auroc']:.4f}  AUPRC {p['auprc']:.4f}  "
                      f"sens {p['sens']:.3f}  spec {p['spec']:.3f}  "
                      f"PPV {p['ppv']:.3f}  F1 {p['f1']:.3f}")
        print()

    if a.csv and rows:
        os.makedirs(RES, exist_ok=True)
        path = f"{RES}/metrics_{tag_feat}{sfx}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
