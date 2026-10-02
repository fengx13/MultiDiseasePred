"""
07_variable_audit.py -- is eci_PHTN's association with the endpoints credible?

Why this exists
---------------
Adding eci_PHTN as the fifteenth variable raises macro AUROC over the seven
acute endpoints by 0.021. The variable it displaced is worth 0.004, and
n_ed_30d, added just before it, is worth 0.0001. One Elixhauser flag for
pulmonary hypertension -- a chronic condition affecting a percent or two of
an emergency population -- is doing more than any other single variable in
the model except the first four.

Two explanations fit, and they have opposite consequences.

    Real. Pulmonary hypertension is an established risk factor for pulmonary
    embolism, and four of the seven acute endpoints are respiratory. The
    approximation R2 for PE does jump from -0.87 to +0.61 exactly when the
    flag enters, in both rankings, which is the pattern a genuinely specific
    marker would produce.

    Leaked. If the flag is derived from the index encounter's diagnosis codes
    rather than from prior history, it encodes the outcome. A leaked variable
    looks exactly like this: rare, uncorrelated with the legitimate
    predictors, individually decisive, and impossible to substitute.

The distinguishing evidence is the strength of the marginal association. A
chronic comorbidity should raise the odds of an acute presentation somewhat.
A code assigned during the encounter being predicted will be enriched among
cases by an order of magnitude or more.

The comparison that matters is not against zero but against the other
thirty-four comorbidity flags, all defined the same way upstream. If
eci_PHTN sits inside that distribution it is doing ordinary work. If it is a
large outlier -- especially for endpoints it has no clinical business
predicting, such as AKI or ACS -- then whatever produced it differs from
whatever produced its peers, and that difference has to be found before any
of these numbers are published.

For a binary predictor the AUROC has a closed form,

    AUROC = (sensitivity + specificity) / 2

which is worth stating because it bounds what a rare flag can achieve. At a
prevalence of 1.5% the specificity is at most 0.985, so an AUROC of 0.60
requires roughly 22% of cases to carry the flag -- a fifteen-fold enrichment.
Reading the AUROC column is therefore the same as reading the enrichment.

    python 07_variable_audit.py
    python 07_variable_audit.py --focus eci_PHTN --top 12
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
DATA = f"{PROJ}/data"
RES = f"{PROJ}/results"

ACUTE = list(range(2, 9))
SEVERITY = [0, 1]


def binary_stats(x, y):
    """Everything that follows from one binary column and one binary label.

    Returns prevalence, sensitivity (the share of cases carrying the flag),
    specificity, risk ratio, and AUROC. AUROC is computed from the closed form
    rather than by ranking; for a two-valued predictor the two agree exactly,
    and the closed form makes it obvious that AUROC and enrichment are the
    same statement.
    """
    x = x.astype(bool)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0 or x.sum() == 0 or (~x).sum() == 0:
        return dict(prev=float(x.mean()), se=np.nan, sp=np.nan,
                    rr=np.nan, auroc=np.nan, risk1=np.nan, risk0=np.nan)
    se = float((x & (y == 1)).sum() / n1)
    sp = float(((~x) & (y == 0)).sum() / n0)
    risk1 = float(y[x].mean())
    risk0 = float(y[~x].mean())
    return dict(prev=float(x.mean()), se=se, sp=sp,
                rr=(risk1 / risk0 if risk0 > 0 else np.inf),
                auroc=(se + sp) / 2.0, risk1=risk1, risk0=risk0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focus", default="eci_PHTN")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--split", default="train",
                    help="train or test; both are checked for the focus "
                         "variable regardless")
    ap.add_argument("--rr-threshold", type=float, default=20.0,
                    help="a flag-endpoint pair above this risk ratio is "
                         "reported as suspect. A chronic comorbidity raises "
                         "the odds of an acute presentation a few fold; "
                         "twenty is far outside that.")
    a = ap.parse_args()

    meta = json.load(open(f"{DATA}/preprocessing.json"))
    cols = meta["features"]
    names = [o.replace("outcome_", "") for o in meta["outcomes"]]
    flags = [c for c in cols if c.startswith(("cci_", "eci_"))]
    if a.focus not in cols:
        raise SystemExit(f"{a.focus} is not a column")

    data = {}
    for split in ("train", "test"):
        X = np.load(f"{DATA}/umn_{split}_X.npy")
        Y = np.load(f"{DATA}/umn_{split}_y.npy")
        data[split] = (X, Y)
        print(f"{split}: {X.shape[0]:,} encounters, {X.shape[1]} variables")

    X, Y = data[a.split]

    # ---- is every column actually carrying information? ------------------
    # triage_temperature came out of the ranking at position 28 of 28 with an
    # attribution of exactly 0.000, which is not what an unimportant variable
    # looks like -- an unimportant variable gets a small number, not a zero.
    # A constant column does: load_split centres by the mean and divides by
    # the scale, and where the scale is zero the code substitutes one, so the
    # column arrives at the model as all zeros and no gradient can flow
    # through it. Worth checking every column rather than just that one.
    print()
    print("=" * 78)
    print(f"COLUMN HEALTH   [{a.split}]")
    print("=" * 78)
    sub = np.random.default_rng(0).choice(len(X), min(200000, len(X)),
                                          replace=False)
    dead, stuck = [], []
    for j, c in enumerate(cols):
        v = X[sub, j]
        if v.std() == 0:
            dead.append((c, float(v[0])))
            continue
        vals, cnt = np.unique(v, return_counts=True)
        share = float(cnt.max()) / len(v)
        if share > 0.98:
            stuck.append((c, share, float(vals[int(np.argmax(cnt))]),
                          int(vals.size)))
    if dead:
        print("  CONSTANT -- these columns cannot contribute anything:")
        for c, val in dead:
            print(f"    {c:<28} every row = {val:g}")
    else:
        print("  no constant columns")
    if stuck:
        print()
        print("  nearly constant -- one value covers more than 98% of rows:")
        for c, share, val, u in stuck:
            print(f"    {c:<28} {share * 100:5.2f}% are {val:g}"
                  f"   ({u} distinct values)")
    print()
    print("  A constant column is not a weak predictor, it is an absent one.")
    print("  If a variable the model is supposed to use appears here, the")
    print("  question is upstream: was it collected, and did it survive")
    print("  extraction and imputation.")

    # The flags are supposed to be indicators. If one is not, every statistic
    # below is meaningless for it, so check rather than assume.
    bad = [c for c in flags
           if not np.isin(np.unique(X[:, cols.index(c)]), (0, 1)).all()]
    if bad:
        print(f"\nnot 0/1, excluded: {bad}")
        flags = [c for c in flags if c not in bad]
    print(f"{len(flags)} comorbidity flags, all indicators\n")

    # ---- the focus variable, in full -------------------------------------
    print("=" * 78)
    print(f"{a.focus}")
    print("=" * 78)
    for split in ("train", "test"):
        Xs, Ys = data[split]
        xs = Xs[:, cols.index(a.focus)]
        print(f"  {split:<6} prevalence {xs.mean() * 100:6.3f}%   "
              f"({int(xs.sum()):,} of {len(xs):,})")
    print()
    Xs, Ys = data[a.split]
    xs = Xs[:, cols.index(a.focus)]
    print(f"  [{a.split}]")
    print(f"  {'endpoint':<20}{'risk if 1':>11}{'risk if 0':>11}"
          f"{'risk ratio':>12}{'% of cases':>12}{'AUROC':>9}")
    print("  " + "-" * 74)
    for t, nm in enumerate(names):
        s = binary_stats(xs, Ys[:, t])
        grp = "severity" if t in SEVERITY else "acute"
        print(f"  {nm:<20}{s['risk1'] * 100:>10.2f}%{s['risk0'] * 100:>10.2f}%"
              f"{s['rr']:>12.1f}{s['se'] * 100:>11.1f}%{s['auroc']:>9.4f}"
              f"   {grp}")
    print()
    print("  'risk ratio' is how many times more likely the endpoint is when")
    print("  the flag is set. '% of cases' is how many of that endpoint's")
    print("  cases carry it. A chronic comorbidity raises risk a few fold; a")
    print("  code from the encounter being predicted raises it far more.")

    # ---- against its peers ------------------------------------------------
    rows, mat, cas = [], {}, {}
    for c in flags:
        x = Xs[:, cols.index(c)]
        st = [binary_stats(x, Ys[:, t]) for t in range(len(names))]
        mat[c] = [s["rr"] for s in st]
        cas[c] = [s["se"] for s in st]
        rows.append({
            "flag": c, "prev": st[0]["prev"],
            "acute_auroc": float(np.nanmean([st[t]["auroc"] for t in ACUTE])),
            "acute_rr": float(np.nanmean([st[t]["rr"] for t in ACUTE])),
            "sev_auroc": float(np.nanmean([st[t]["auroc"] for t in SEVERITY])),
            "max_rr": float(np.nanmax(mat[c])),
            "max_rr_at": names[int(np.nanargmax(mat[c]))],
            "max_auroc": float(np.nanmax([st[t]["auroc"] for t in range(len(names))])),
            "max_at": names[int(np.nanargmax([st[t]["auroc"] for t in range(len(names))]))],
        })
    rows.sort(key=lambda r: -r["acute_auroc"])
    rank = [r["flag"] for r in rows].index(a.focus) + 1

    print()
    print("=" * 78)
    print(f"ALL {len(flags)} COMORBIDITY FLAGS, by mean AUROC over the seven "
          f"acute endpoints")
    print("=" * 78)
    print(f"{'':>3} {'flag':<20}{'prev':>8}{'acute AUROC':>13}"
          f"{'mean RR':>10}{'severity':>10}{'best at':>18}")
    print("-" * 78)
    for i, r in enumerate(rows, 1):
        if i > a.top and r["flag"] != a.focus:
            continue
        mark = "  <==" if r["flag"] == a.focus else ""
        print(f"{i:>3} {r['flag']:<20}{r['prev'] * 100:>7.2f}%"
              f"{r['acute_auroc']:>13.4f}{r['acute_rr']:>10.1f}"
              f"{r['sev_auroc']:>10.4f}{r['max_at']:>18}{mark}")
    print()
    print(f"  {a.focus} ranks {rank} of {len(flags)} on acute AUROC.")
    v = [r["acute_auroc"] for r in rows]
    f = [r for r in rows if r["flag"] == a.focus][0]
    z = (f["acute_auroc"] - np.mean(v)) / np.std(v)
    print(f"  Its acute AUROC is {f['acute_auroc']:.4f}; the flags average "
          f"{np.mean(v):.4f} (sd {np.std(v):.4f}).")
    print(f"  That is {z:+.1f} standard deviations from its peers.")
    print()
    # The verdict must key off the maximum over endpoints, not the mean.
    # The first version of this check used the mean acute AUROC and declared
    # eci_PHTN ordinary: it scores about 0.55 on six endpoints and 0.93 on
    # pulmonary embolism, and the average of those, 0.61, looks unremarkable.
    # Averaging over nine endpoints is exactly the wrong thing to do when the
    # failure mode is a flag that is catastrophic against one of them.
    print(f"  Its worst pairing is {a.focus} -> {f['max_rr_at']}, "
          f"risk ratio {f['max_rr']:.1f}.")
    if f["max_rr"] >= a.rr_threshold:
        print()
        print(f"  That is at or above {a.rr_threshold:.0f}, which no chronic")
        print("  comorbidity produces against an acute presentation. This")
        print("  flag is reporting the diagnosis, not the history. See the")
        print("  matrix below for which other flags do the same.")
    else:
        print(f"  Every pairing is below {a.rr_threshold:.0f}, which is what a")
        print("  comorbidity window ending before the encounter looks like.")

    # ---- every flag against every endpoint --------------------------------
    # The table to rerun after the extraction is fixed. Each flag should show
    # the modest association a chronic condition has with an acute
    # presentation. A cell in the hundreds means that flag is reporting the
    # diagnosis in that column rather than the patient's history, and the
    # pairing tells you which code leaked into which endpoint.
    print()
    print("=" * 78)
    print(f"RISK RATIO, all {len(flags)} flags against all {len(names)} "
          f"endpoints   [{a.split}]")
    print("=" * 78)
    hdr = [n[:5].rstrip("_") for n in names]
    print(f"{'flag':<20}" + "".join(f"{h:>7}" for h in hdr))
    print("-" * (20 + 7 * len(names)))
    for c in sorted(flags, key=lambda f: -np.nanmax(mat[f])):
        line = f"{c:<20}"
        for v in mat[c]:
            if not np.isfinite(v):
                line += f"{'-':>7}"
            elif v >= a.rr_threshold:
                line += f"{v:>6.0f}*"
            else:
                line += f"{v:>7.1f}"
        print(line)
    print()
    print(f"  * marks a risk ratio at or above {a.rr_threshold:.0f}.")

    susp = sorted(((c, t, mat[c][t], cas[c][t])
                   for c in flags for t in range(len(names))
                   if np.isfinite(mat[c][t]) and mat[c][t] >= a.rr_threshold),
                  key=lambda r: -r[2])
    print()
    print("=" * 78)
    print(f"SUSPECT PAIRS   risk ratio >= {a.rr_threshold:.0f}")
    print("=" * 78)
    if not susp:
        print(f"  none. Every flag-endpoint pair is below {a.rr_threshold:.0f},")
        print("  which is what a comorbidity window that ends before the")
        print("  index encounter should produce.")
    else:
        print(f"  {'flag':<20}{'endpoint':<18}{'risk ratio':>12}"
              f"{'% of cases':>13}")
        print("  " + "-" * 61)
        for c, t, rr, se in susp:
            print(f"  {c:<20}{names[t]:<18}{rr:>12.1f}{se * 100:>12.1f}%")
        print()
        print("  '% of cases' is the share of that endpoint's cases carrying")
        print("  the flag. Compare it with the flag's overall prevalence: if")
        print("  nearly every case has it and few other patients do, the flag")
        print("  is the label.")

    with open(f"{RES}/risk_ratio_matrix.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["flag", "prevalence"] + names)
        for c in flags:
            w.writerow([c, float(Xs[:, cols.index(c)].mean())]
                       + [f"{v:.4f}" if np.isfinite(v) else "" for v in mat[c]])
    print(f"\nwritten to {RES}/risk_ratio_matrix.csv")

    # ---- is it orthogonal to everything else? ----------------------------
    print()
    print("=" * 78)
    print(f"WHAT ELSE MOVES WITH {a.focus}")
    print("=" * 78)
    xi = cols.index(a.focus)
    sub = np.random.default_rng(0).choice(len(Xs), min(200000, len(Xs)),
                                          replace=False)
    Xc = Xs[sub][:, :]
    base = Xc[:, xi]
    cors = []
    for j, c in enumerate(cols):
        if j == xi:
            continue
        o = Xc[:, j]
        if o.std() == 0:
            continue
        cors.append((abs(float(np.corrcoef(base, o)[0, 1])), c))
    cors.sort(reverse=True)
    for r, c in cors[:8]:
        print(f"  {r:6.3f}  {c}")
    print()
    print(f"  highest absolute correlation with any other variable: "
          f"{cors[0][0]:.3f}")
    print("  Context only. A leaked flag tends to correlate weakly with the")
    print("  legitimate predictors, because nothing measured before the visit")
    print("  knows the visit's codes -- but so does a genuinely novel marker,")
    print("  so this section cannot decide anything on its own. The risk")
    print("  ratio matrix is what decides.")

    os.makedirs(RES, exist_ok=True)
    with open(f"{RES}/variable_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwritten to {RES}/variable_audit.csv")


if __name__ == "__main__":
    main()
