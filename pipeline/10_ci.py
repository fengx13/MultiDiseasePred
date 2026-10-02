"""
10_ci.py -- confidence intervals, and whether a difference is real.

Every number in the results so far carries a seed-to-seed standard deviation.
That is the wrong uncertainty. It measures how much the answer moves when the
weights are initialised differently, on this fixed set of patients. What a
reader needs to know is how much it would move on a different set of
patients, which is the only thing that predicts whether an external cohort
will reproduce it. The two are unrelated: three seeds can agree to 0.0005 on
an endpoint with two hundred events, where resampling the patients moves the
AUROC by 0.03.

The ARDS column is exactly that case. It has 200 events in the internal test
set, our model is 0.0134 above gradient boosting there, and the seed spread
is 0.003. Reported as "0.8808 +- 0.003" it looks like the strongest result in
the table. It is not a result at all until it has a patient-level interval.

AUROC -- DeLong
---------------
DeLong's method gives the variance of an AUROC, and the covariance between
two AUROCs computed on the same patients, in closed form. That second part is
what matters here: the models are being compared on the same test set, so
their errors are correlated, and treating the difference as if the two were
independent inflates its standard error by a large factor. Sun and Xu's
O(n log n) formulation makes it cheap at 350,000 rows.

No bootstrap is needed for AUROC, and none should be used: the closed form is
exact where a thousand resamples are still noisy.

AUPRC -- paired bootstrap
-------------------------
Average precision has no comparable closed form, so it is resampled. The same
patient indices are used for every arm in a replicate, which keeps the
comparison paired for the same reason DeLong's covariance term matters. The
interval for the *difference* is taken from the bootstrap distribution of the
difference, not from the two separate intervals -- overlapping intervals do
not imply a non-significant difference, and non-overlapping ones are not
required for a significant one.

B defaults to 1000. Fewer is tempting because it is faster, but a 95 per cent
percentile interval from 30 replicates has its endpoints at the smallest and
largest values drawn, which are the two least stable numbers in the sample.
The cost here is minutes: nothing is retrained, the saved probabilities are
read off disk.

Seeds
-----
The arms were trained with three seeds. This averages the predicted
probabilities across them and reports intervals for that ensemble, which is
what would be deployed. Note that the AUROC of an average is not the average
of the AUROCs -- the numbers here will differ slightly from the per-seed means
in 08_final_tables.py, and both are correct answers to different questions.

    python 10_ci.py --features all63 --drop triage_acuity
    python 10_ci.py --features greedy:8 --drop triage_acuity --boot 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

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

ACUTE, DISP = list(range(2, 9)), [0, 1]
Z = 1.959963984540054            # two-sided 95 per cent


# ---------------------------------------------------------------- DeLong ---
def midrank(x):
    """Ranks with ties averaged. Ties are not a corner case here: NEWS takes
    fifteen distinct values across 350,000 patients, so almost every rank is
    tied, and using ordinary ranks would silently misestimate its variance."""
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


def delong(scores, y):
    """AUROCs and their covariance matrix for several arms on one endpoint.

    `scores` is (n_arms, n_patients); y is the shared binary label. Returns
    (aucs, cov) where cov[i, j] is the covariance of AUROC_i and AUROC_j,
    following Sun and Xu (2014), IEEE Signal Processing Letters 21(11).
    """
    y = np.asarray(y).astype(bool)
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    if m == 0 or n == 0:
        k = scores.shape[0]
        return np.full(k, np.nan), np.full((k, k), np.nan)
    k = scores.shape[0]

    tz = np.empty((k, m + n))
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    for r in range(k):
        tx[r] = midrank(pos[r])
        ty[r] = midrank(neg[r])
        tz[r] = midrank(np.concatenate([pos[r], neg[r]]))

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    # structural components
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01) if k > 1 else np.array([[np.var(v01[0], ddof=1)]])
    sy = np.cov(v10) if k > 1 else np.array([[np.var(v10[0], ddof=1)]])
    cov = np.atleast_2d(sx) / m + np.atleast_2d(sy) / n
    return aucs, cov


def normal_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def fast_auroc(y, s):
    """AUROC through the rank-sum identity, vectorised.

    01_train.py's version walks the sorted scores in a Python loop to average
    ties. That is fine once per run and far too slow here: a macro interval
    needs nine endpoints times several arms times a thousand resamples, and at
    350,000 rows the loop alone would take hours. scipy's rankdata does the
    same tie-averaging in C.

        AUC = (sum of the positives' ranks - m(m+1)/2) / (m * n)
    """
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - m * (m + 1) / 2.0) / (m * n))


# -------------------------------------------------------------- loading ---
def load_arm(arm, features, drop, seeds, split, data_id):
    """Probabilities averaged over seeds, plus the labels, for one arm."""
    tag_feat = features.replace(":", "")
    mats, found = [], []
    labels = None
    for s in seeds:
        d = f"{RUNS}/{arm}_{tag_feat}{_t.drop_suffix(drop)}_seed{s}"
        p = f"{d}/metrics.json"
        if not os.path.exists(p):
            continue
        if data_id is not None:
            try:
                if json.load(open(p)).get("data_id") != data_id:
                    continue
            except Exception:
                continue
        fp = f"{d}/logits_umn_{split}.npy"
        fl = f"{d}/labels_umn_{split}.npy"
        if not (os.path.exists(fp) and os.path.exists(fl)):
            continue
        mats.append(np.load(fp).astype(np.float64))
        lab = np.load(fl)
        if labels is None:
            labels = lab
        elif labels.shape != lab.shape or not np.array_equal(labels, lab):
            # Two seeds of one arm must have scored the same patients in the
            # same order. If they did not, averaging their probabilities makes
            # a matrix that corresponds to no cohort at all.
            raise SystemExit(f"{arm} seed {s}: labels differ from the "
                             f"earlier seeds. Refusing to average.")
        found.append(s)
    if not mats:
        return None, None, []
    return np.mean(mats, axis=0), labels, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="all63")
    ap.add_argument("--drop", default="")
    ap.add_argument("--arms", default="G2_no_l1_no_drop,A2_xgb_sub,A2_rf",
                    help="comma separated; the first is ours")
    ap.add_argument("--news", default="A5_news_all63",
                    help="run-directory prefix for the NEWS arm, which does "
                         "not vary with the feature set. Empty to skip.")
    ap.add_argument("--compare-features", default="",
                    help="load the first arm a second time at this feature "
                         "set and compare the two, paired. This is how the "
                         "choice of k is settled: same model, same patients, "
                         "different variable sets, one DeLong test.")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()

    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",")]

    meta = json.load(open(f"{DATA}/preprocessing.json"))
    names = [o.replace("outcome_", "") for o in meta["outcomes"]]
    data_id = meta.get("data_id")

    print(f"variables  {a.features}"
          + (f"   excluding {', '.join(drop)}" if drop else ""))
    print(f"split      {a.split}")
    print(f"data_id    {data_id}")
    print(f"bootstrap  B = {a.boot} for AUPRC; AUROC is closed form (DeLong)")
    print()

    loaded, labels = [], None
    for arm in arms:
        P, y, found = load_arm(arm, a.features, drop, seeds, a.split, data_id)
        if P is None:
            print(f"  {arm:<22} NO RUNS FOUND -- skipped")
            continue
        print(f"  {arm:<22} seeds {found}")
        loaded.append((arm, P))
        labels = y if labels is None else labels
    # NEWS has one directory whatever the feature set, so it is fetched with
    # its own tag rather than the one being tabulated.
    if a.news:
        P, y, found = load_arm(a.news, "", drop, [42], a.split, data_id)
        if P is not None:
            if y.shape != labels.shape or not np.array_equal(y, labels):
                print(f"  {a.news:<22} labels do not match -- skipped")
            else:
                print(f"  {a.news:<22} single score, no seeds")
                loaded.append(("NEWS", P))

    # Same model at a second variable set, as another column. Everything
    # downstream is already paired, so nothing else has to know that the two
    # columns differ in their inputs rather than in their architecture.
    if a.compare_features:
        P, y, found = load_arm(arms[0], a.compare_features, drop, seeds,
                               a.split, data_id)
        label = f"{arms[0][:12]}@{a.compare_features}"
        if P is None:
            print(f"  {label:<22} NO RUNS FOUND -- skipped")
        elif labels is None or not np.array_equal(y, labels):
            print(f"  {label:<22} labels do not match -- skipped")
        else:
            print(f"  {label:<22} seeds {found}")
            loaded.append((label, P))

    if len(loaded) < 2:
        raise SystemExit("\nNeed at least two arms to compare. Give more "
                         "--arms, or --compare-features to put the same model "
                         "at two variable sets side by side.")
    ours = loaded[0][0]
    n = labels.shape[0]
    print(f"\n  {n:,} patients, {len(loaded)} arms\n")

    rng = np.random.default_rng(42)
    boot_idx = rng.integers(0, n, size=(a.boot, n)) if a.boot else None

    rows = []
    boot_auroc, boot_auprc = {}, {}
    t0 = time.time()
    for t, nm in enumerate(names):
        y = labels[:, t]
        ev = int(y.sum())
        S = np.vstack([P[:, t] for _, P in loaded])
        aucs, cov = delong(S, y)
        se = np.sqrt(np.diag(cov))

        # AUPRC, paired bootstrap on shared indices. AUROC is resampled in
        # the same loop even though DeLong already gives its interval: the
        # macro over seven endpoints is an average of correlated AUROCs, and
        # DeLong's covariance is between arms on one endpoint, not between
        # endpoints. The resampled per-endpoint values are kept so the macro
        # rows below can be given an interval rather than a bare mean.
        ap_pt = np.array([_t.auprc(y, S[i]) for i in range(len(loaded))])
        ap_bs = np.full((a.boot, len(loaded)), np.nan)
        au_bs = np.full((a.boot, len(loaded)), np.nan)
        if a.boot:
            for b in range(a.boot):
                ix = boot_idx[b]
                yb = y[ix]
                if yb.sum() == 0:
                    continue
                for i in range(len(loaded)):
                    sb = S[i][ix]
                    ap_bs[b, i] = _t.auprc(yb, sb)
                    au_bs[b, i] = fast_auroc(yb, sb)
        boot_auroc[t] = au_bs
        boot_auprc[t] = ap_bs

        print("=" * 100)
        print(f"{nm}   ({ev:,} events, {100 * ev / n:.3f}%)")
        print("=" * 100)
        print(f"{'model':<22}{'AUROC (95% CI)':>28}{'AUPRC (95% CI)':>28}"
              f"{'vs ours':>20}")
        print("-" * 100)
        for i, (arm, _) in enumerate(loaded):
            lo, hi = aucs[i] - Z * se[i], aucs[i] + Z * se[i]
            if a.boot:
                b = ap_bs[:, i]
                b = b[~np.isnan(b)]
                alo, ahi = np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan,) * 2
            else:
                alo = ahi = float("nan")
            line = (f"{arm[:20]:<22}"
                    f"{aucs[i]:.4f} ({lo:.4f}-{hi:.4f})".rjust(28)
                    + f"{ap_pt[i]:.4f} ({alo:.4f}-{ahi:.4f})".rjust(28))
            if i == 0:
                line += f"{'--':>20}"
            else:
                d = aucs[0] - aucs[i]
                sed = math.sqrt(max(cov[0, 0] + cov[i, i] - 2 * cov[0, i], 0))
                p = normal_p(d / sed) if sed > 0 else float("nan")
                line += f"{d:>+11.4f} p={p:.3f}"
                dif = ap_bs[:, 0] - ap_bs[:, i]
                dif = dif[~np.isnan(dif)]
                dlo, dhi = (np.percentile(dif, [2.5, 97.5]) if len(dif)
                            else (np.nan, np.nan))
                rows.append({
                    "endpoint": nm, "events": ev, "baseline": arm,
                    "ours_auroc": round(aucs[0], 5),
                    "ours_auroc_lo": round(aucs[0] - Z * se[0], 5),
                    "ours_auroc_hi": round(aucs[0] + Z * se[0], 5),
                    "base_auroc": round(aucs[i], 5),
                    "base_auroc_lo": round(lo, 5), "base_auroc_hi": round(hi, 5),
                    "auroc_diff": round(d, 5),
                    "auroc_diff_se": round(sed, 5),
                    "auroc_p": round(p, 6),
                    "ours_auprc": round(ap_pt[0], 5),
                    "base_auprc": round(ap_pt[i], 5),
                    "auprc_diff": round(ap_pt[0] - ap_pt[i], 5),
                    "auprc_diff_lo": round(float(dlo), 5),
                    "auprc_diff_hi": round(float(dhi), 5),
                })
            print(line)
        if len(loaded) > 1:
            print()
            for r in [x for x in rows if x["endpoint"] == nm]:
                verdict = ("higher" if r["auroc_p"] < 0.05 and r["auroc_diff"] > 0
                           else "lower" if r["auroc_p"] < 0.05
                           else "not separated")
                print(f"  vs {r['baseline'][:20]:<22} AUROC {verdict}; "
                      f"AUPRC difference {r['auprc_diff']:+.4f} "
                      f"({r['auprc_diff_lo']:+.4f} to {r['auprc_diff_hi']:+.4f})")
        print()

    # ---- macro over endpoint groups, with intervals --------------------
    if a.boot and boot_auroc:
        print("=" * 100)
        print("MACRO OVER ENDPOINT GROUPS")
        print("=" * 100)
        print("  Averaged across endpoints within each bootstrap replicate, so")
        print("  the interval carries the correlation between endpoints on the")
        print("  same patients. Averaging the nine separate intervals instead")
        print("  would understate it.")
        for gname, ix in (("acute (7 endpoints)", ACUTE),
                          ("disposition (2)", DISP)):
            print()
            print(f"{gname:<24}{'AUROC (95% CI)':>28}{'AUPRC (95% CI)':>28}"
                  f"{'vs ours':>18}")
            print("-" * 100)
            for i, (arm, _) in enumerate(loaded):
                mac_au = np.nanmean([boot_auroc[t][:, i] for t in ix], axis=0)
                mac_ap = np.nanmean([boot_auprc[t][:, i] for t in ix], axis=0)
                mu_au, mu_ap = np.nanmean(mac_au), np.nanmean(mac_ap)
                lo_au, hi_au = np.nanpercentile(mac_au, [2.5, 97.5])
                lo_ap, hi_ap = np.nanpercentile(mac_ap, [2.5, 97.5])
                line = (f"{arm[:22]:<24}"
                        + f"{mu_au:.4f} ({lo_au:.4f}-{hi_au:.4f})".rjust(28)
                        + f"{mu_ap:.4f} ({lo_ap:.4f}-{hi_ap:.4f})".rjust(28))
                if i == 0:
                    line += f"{'--':>18}"
                    base_au = mac_au
                else:
                    d = base_au - mac_au
                    dlo, dhi = np.nanpercentile(d, [2.5, 97.5])
                    star = "*" if dlo > 0 or dhi < 0 else " "
                    line += f"{np.nanmean(d):>+9.4f}{star} "
                    rows.append({
                        "endpoint": f"MACRO {gname.split()[0]}", "events": -1,
                        "baseline": arm,
                        "ours_auroc": round(float(np.nanmean(base_au)), 5),
                        "ours_auroc_lo": round(float(np.nanpercentile(base_au, 2.5)), 5),
                        "ours_auroc_hi": round(float(np.nanpercentile(base_au, 97.5)), 5),
                        "base_auroc": round(float(mu_au), 5),
                        "base_auroc_lo": round(float(lo_au), 5),
                        "base_auroc_hi": round(float(hi_au), 5),
                        "auroc_diff": round(float(np.nanmean(d)), 5),
                        "auroc_diff_se": round(float(np.nanstd(d, ddof=1)), 5),
                        "auroc_p": -1.0,
                        "ours_auprc": -1.0, "base_auprc": round(float(mu_ap), 5),
                        "auprc_diff": -1.0,
                        "auprc_diff_lo": round(float(dlo), 5),
                        "auprc_diff_hi": round(float(dhi), 5)})
                print(line)
        print()
        print("  * the 95 per cent interval of the paired difference excludes")
        print("    zero. There is no p-value here because the macro is an")
        print("    average of correlated AUROCs and DeLong does not cover it;")
        print("    the resampled difference does.")
        print()

    # ---- what the table adds up to ------------------------------------
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for base in dict.fromkeys(r["baseline"] for r in rows):
        rs = [r for r in rows if r["baseline"] == base]
        up = [r for r in rs if r["auroc_p"] < 0.05 and r["auroc_diff"] > 0]
        dn = [r for r in rs if r["auroc_p"] < 0.05 and r["auroc_diff"] < 0]
        print(f"\n  {ours} vs {base}")
        print(f"    AUROC higher on {len(up)} of {len(rs)} endpoints, "
              f"lower on {len(dn)}, not separated on "
              f"{len(rs) - len(up) - len(dn)}")
        if up:
            print(f"      higher: {', '.join(r['endpoint'] for r in up)}")
        if dn:
            print(f"      lower:  {', '.join(r['endpoint'] for r in dn)}")

    print()
    print("  A p-value here is a paired DeLong test on the same patients, not")
    print("  a comparison of two independent intervals. Two intervals can")
    print("  overlap substantially while the paired difference is firmly")
    print("  non-zero, because the two models make their mistakes on the same")
    print("  people. Read the difference column, not the overlap.")
    print()
    print("  Nine endpoints means nine tests. If a single endpoint is going to")
    print("  carry a claim in the abstract, it needs to survive a multiplicity")
    print("  adjustment -- Holm on nine tests requires p < 0.0056 for the")
    print("  smallest before it can be called significant on its own.")

    if a.csv and rows:
        sfx = _t.drop_suffix(drop)
        fn = (f"{RES}/ci_{a.features.replace(':', '')}{sfx}_{a.split}.csv")
        os.makedirs(RES, exist_ok=True)
        with open(fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  written to {fn}")
    print(f"  {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
