"""
09_external.py -- does the model work anywhere else.

Internal validation on 2024-2025 answers whether the model survives time.
Whether it survives a different hospital is a different question, and it is
the one a reader cares about most, because a model that only works where it
was fitted is a description of that hospital rather than of triage.

Two cohorts, both already prepared by 00_prepare_data.py --external-only:

    bidmc       MIMIC-IV-ED, Beth Israel Deaconess, 445,744 encounters
    stanford    Stanford, 117,046 encounters

Zero-shot, and why that is the number that matters
--------------------------------------------------
The model is applied exactly as trained: same weights, same medians, same
scaler, no refitting of anything. That is the honest question -- if you handed
this to another emergency department tomorrow, what would it do.

The earlier work in this project reported external results from checkpoints
named best_gate_head_only_mimic_ft, which had the gate and output heads
fine-tuned on the external cohort, with calibration refitted there too. That
is transfer learning and it answers "how much local data would you need",
which is worth reporting, but it is not external validation and must not be
labelled as such.

What is expected, and what would be alarming
--------------------------------------------
Discrimination usually transports. Calibration usually does not, and here it
certainly will not: hospitalisation is 20.4% at UMN and 47.9% at BIDMC, and
critical illness is 0.93% against 6.62%. A model carrying UMN's base rates
into a sicker cohort will be systematically under-confident. So this reports
calibration slope and intercept beside AUROC, and reports them before any
recalibration, because the size of the miscalibration is a finding.

AUPRC moves with prevalence by construction, so it is reported per cohort and
never compared across them. The comparison that is valid across cohorts is
AUROC, and the comparison that is valid within one is everything.

    python 09_external.py --arm G2_no_l1_no_drop --features greedy:8
    python 09_external.py --arm G2_no_l1_no_drop --features all63 --csv
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
import models as M                                            # noqa: E402
import torch                                                  # noqa: E402

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
DATA = f"{PROJ}/data"
RUNS = f"{PROJ}/runs"
RES = f"{PROJ}/results"

COHORTS = {"bidmc": "MIMIC-IV-ED (BIDMC)", "stanford": "Stanford"}
ACUTE, DISP = list(range(2, 9)), [0, 1]
Z = 1.959963984540054            # two-sided 95 per cent


def _fast_auroc(y, s):
    """Rank-sum AUROC. Used inside the bootstrap, where the tie-averaging
    loop in 01_train.py would cost hours over the replicates."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - m * (m + 1) / 2.0) / (m * n))


def calibration(y, lp):
    """Slope and intercept of the calibration line, with standard errors.

    `lp` is the LINEAR PREDICTOR -- the model's logit -- not a probability.

    That distinction is the whole point of this signature. The first version
    took a probability, clipped it to [1e-6, 1-1e-6] and took its logit. The
    neural arms are trained with BCEWithLogitsLoss and `model(x)` returns
    logits, so what arrived was a logit being treated as a probability: every
    negative logit was clipped to 1e-6 and every logit above 1 to 1-1e-6, and
    the "linear predictor" became a two-point constant pinned at
    log(1e-6/(1-1e-6)) = -13.8155.

    A line fitted through a single point is not identified. The fit still
    converged, because any (b0, b1) with b1*(-13.8155) + b0 = logit(prevalence)
    fits equally well, and lbfgs simply stopped near its b0 = 0 start and let
    the slope absorb the offset. The output looked entirely plausible -- slopes
    of 0.19 to 0.50, intercepts near zero, a tidy story about overconfident
    predictions -- and was pure artefact. It was caught only because
    sigmoid(b1 * -13.8155 + b0) reproduced the observed prevalence to four
    decimal places on eleven of eighteen rows, which nothing but a degenerate
    design matrix does.

    AUROC and AUPRC were never affected: both are rank statistics, and the
    clip is monotone, so discrimination survived a bug that destroyed
    calibration. That is precisely why it went unnoticed.

    A perfectly calibrated model gives slope 1 and intercept 0. Slope below 1
    means the predictions are too extreme for the new cohort; an intercept
    away from 0 means the base rate moved. They are different failures with
    different fixes -- an intercept shift needs only the prevalence, a slope
    shift needs refitting -- so they are reported separately.

    The standard errors come from the fitted model's own covariance,
    (X'WX)^-1 with W = diag(q(1-q)), rather than from resampling. It is exact
    for a two-parameter logistic regression and costs one 2x2 inverse, where a
    bootstrap would need a thousand refits per endpoint per cohort and turn a
    twenty-minute job into a three-hour one.
    """
    from sklearn.linear_model import LogisticRegression
    lp = np.asarray(lp, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nan4 = (float("nan"),) * 4
    if y.sum() == 0 or y.sum() == len(y):
        return nan4
    # Guard the mistake above rather than only documenting it. Logits over a
    # third of a million encounters are never all inside [0, 1]; probabilities
    # always are. Nine endpoints times two cohorts is enough chances that this
    # should fail loudly rather than return a plausible number.
    if lp.min() >= 0.0 and lp.max() <= 1.0:
        raise SystemExit(
            "calibration() was given values inside [0, 1]. It expects the\n"
            "linear predictor (the logit), not a probability. Passing a\n"
            "probability here produced the artefactual slopes of 2026-08-11.")
    if not np.isfinite(lp).all() or lp.std() < 1e-9:
        return nan4
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    m.fit(lp.reshape(-1, 1), y)
    b1, b0 = float(m.coef_[0][0]), float(m.intercept_[0])

    q = 1.0 / (1.0 + np.exp(-(b0 + b1 * lp)))
    w = q * (1.0 - q)
    # X'WX for X = [1, lp], accumulated without forming an n x 2 matrix
    s00, s01, s11 = w.sum(), (w * lp).sum(), (w * lp * lp).sum()
    det = s00 * s11 - s01 * s01
    if not np.isfinite(det) or abs(det) < 1e-12:
        return b1, b0, float("nan"), float("nan")
    se_b0 = float(np.sqrt(s11 / det))
    se_b1 = float(np.sqrt(s00 / det))
    return b1, b0, se_b1, se_b0


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
    """AUROC and its DeLong standard error for one model on one endpoint.

    The same closed form 10_ci.py uses. External validation needs it more
    than internal validation does, because the cohorts are smaller and the
    rare endpoints have very few events -- Stanford has about 35 ARDS cases,
    where a point estimate on its own means nothing.
    """
    y = np.asarray(y).astype(bool)
    pos, neg = scores[y], scores[~y]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    tz = midrank(np.concatenate([pos, neg]))
    tx, ty = midrank(pos), midrank(neg)
    auc = tz[:m].sum() / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    var = np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n
    return float(auc), float(np.sqrt(max(var, 0.0)))


def fast_auprc(y, s):
    from sklearn.metrics import average_precision_score
    if np.asarray(y).sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def load_external(name, idx, meta):
    fx, fy = f"{DATA}/{name}_X.npy", f"{DATA}/{name}_y.npy"
    if not (os.path.exists(fx) and os.path.exists(fy)):
        return None, None
    mean = np.asarray(meta["scaler_mean"])[idx]
    scale = np.asarray(meta["scaler_scale"])[idx]
    X = np.load(fx)[:, idx]
    X = (X - mean) / np.where(scale == 0, 1, scale)
    return X.astype(np.float32), np.load(fy).astype(np.float32)


def score(arm, run, X, n_tasks):
    """Predictions from the saved model, without refitting anything.

    Neural arms saved model.pt. The tree arms did not save their estimators,
    so they cannot be scored here; they have to be refitted, which is cheap
    but belongs in a separate pass rather than hidden inside this one.
    """
    ck = f"{run}/model.pt"
    if not os.path.exists(ck):
        raise SystemExit(
            f"{ck} not found.\n"
            f"Tree arms (A2_*) do not save an estimator, so they cannot be\n"
            f"scored zero-shot from disk. Refit them against the external\n"
            f"cohorts separately if they are needed in this table.")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = M.build(arm, X.shape[1], n_tasks=n_tasks,
                    hidden=_t.HP["hidden"], n_experts=_t.HP["n_experts"])
    state = torch.load(ck, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            out.append(model(torch.from_numpy(X[i:i + 8192]).to(dev))
                       .cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="G2_no_l1_no_drop")
    ap.add_argument("--features", default="greedy:8")
    ap.add_argument("--drop", default="triage_acuity")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--boot", type=int, default=1000,
                    help="bootstrap replicates for the AUPRC and macro "
                         "intervals. AUROC uses DeLong and needs none; the "
                         "calibration standard errors are analytic.")
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()

    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)
    seeds = [int(x) for x in a.seeds.split(",")]
    meta = json.load(open(f"{DATA}/preprocessing.json"))
    cols = meta["features"]
    names = [o.replace("outcome_", "") for o in meta["outcomes"]]
    data_id = meta.get("data_id")
    idx = _t.resolve_features(a.features, cols, drop)

    print(f"arm        {a.arm}")
    print(f"variables  {a.features}  ->  {len(idx)}")
    for c in [cols[i] for i in idx]:
        print(f"   {c}")
    print(f"data_id    {data_id}")
    print("\nZero-shot: the weights, medians and scaler are exactly those")
    print("fitted at UMN. Nothing is refitted on either external cohort.\n")

    tag_feat = a.features.replace(":", "")
    runs = []
    for s in seeds:
        r = f"{RUNS}/{a.arm}_{tag_feat}{_t.drop_suffix(drop)}_seed{s}"
        if os.path.exists(f"{r}/model.pt"):
            runs.append((s, r))
    if not runs:
        raise SystemExit(f"no run directories with model.pt for {a.arm} "
                         f"at {a.features}")
    print(f"seeds found {[s for s, _ in runs]}")

    rows = []
    for name, label in COHORTS.items():
        X, y = load_external(name, idx, meta)
        if X is None:
            print(f"\n{label}: arrays not built. "
                  f"Run 00_prepare_data.py --external-only first.")
            continue
        print()
        print("=" * 94)
        print(f"{label}   {len(X):,} encounters")
        print("=" * 94)

        # Average over seeds on the logit scale. 10_ci.py averages
        # whatever logits_umn_*.npy holds, which for the neural arms is
        # also the logit, so the two agree. It is not the same as
        # averaging probabilities, and the earlier comment claiming it
        # was is wrong.
        P = np.mean([score(a.arm, r, X, y.shape[1]) for _, r in runs], axis=0)
        np.save(f"{RUNS}/{a.arm}_{tag_feat}{_t.drop_suffix(drop)}_seed42/"
                f"logits_{name}.npy", P)
        np.save(f"{RUNS}/{a.arm}_{tag_feat}{_t.drop_suffix(drop)}_seed42/"
                f"labels_{name}.npy", y)

        rng = np.random.default_rng(42)
        boot_idx = (rng.integers(0, len(X), size=(a.boot, len(X)))
                    if a.boot else None)
        au_bs = np.full((a.boot, y.shape[1]), np.nan)
        ap_bs = np.full((a.boot, y.shape[1]), np.nan)

        print(f"{'endpoint':<18}{'events':>8}{'prev':>7}"
              f"{'AUROC (95% CI)':>26}{'AUPRC (95% CI)':>26}"
              f"{'slope (95% CI)':>22}{'intercept (95% CI)':>24}")
        print("-" * 94)
        for t, nm in enumerate(names):
            ev = int(y[:, t].sum())
            au, se = delong_se(P[:, t], y[:, t])
            ap_ = fast_auprc(y[:, t], P[:, t])
            # P is the logit, not the probability -- the neural arms train
            # with BCEWithLogitsLoss and model(x) returns the linear
            # predictor. calibration() wants exactly that. AUROC and AUPRC
            # above are rank statistics and do not care either way.
            sl, ic, se_sl, se_ic = calibration(y[:, t], P[:, t])
            if a.boot:
                for b in range(a.boot):
                    ixb = boot_idx[b]
                    yb = y[ixb, t]
                    if yb.sum() == 0:
                        continue
                    ap_bs[b, t] = fast_auprc(yb, P[ixb, t])
                    au_bs[b, t] = _fast_auroc(yb, P[ixb, t])
                blo, bhi = np.nanpercentile(ap_bs[:, t], [2.5, 97.5])
            else:
                blo = bhi = float("nan")
            print(f"{nm:<18}{ev:>8,}{100 * y[:, t].mean():>6.2f}%"
                  + f"{au:.4f} ({au - Z * se:.4f}-{au + Z * se:.4f})".rjust(26)
                  + f"{ap_:.4f} ({blo:.4f}-{bhi:.4f})".rjust(26)
                  + f"{sl:.2f} ({sl - Z * se_sl:.2f}-{sl + Z * se_sl:.2f})".rjust(22)
                  + f"{ic:.2f} ({ic - Z * se_ic:.2f}-{ic + Z * se_ic:.2f})".rjust(24))
            rows.append({"cohort": name, "endpoint": nm, "n": len(X),
                         "events": ev,
                         "prevalence": round(float(y[:, t].mean()), 6),
                         "auroc": round(au, 5),
                         "auroc_lo": round(au - Z * se, 5),
                         "auroc_hi": round(au + Z * se, 5),
                         "auprc": round(ap_, 5),
                         "auprc_lo": round(float(blo), 5),
                         "auprc_hi": round(float(bhi), 5),
                         "cal_slope": round(sl, 4),
                         "cal_slope_lo": round(sl - Z * se_sl, 4),
                         "cal_slope_hi": round(sl + Z * se_sl, 4),
                         "cal_intercept": round(ic, 4),
                         "cal_intercept_lo": round(ic - Z * se_ic, 4),
                         "cal_intercept_hi": round(ic + Z * se_ic, 4)})
        print("-" * 94)
        # Macro rows are averaged inside each replicate so the interval keeps
        # the correlation between endpoints on the same patients.
        for grp, ix in (("acute macro", ACUTE), ("disposition macro", DISP)):
            if a.boot:
                mau = np.nanmean(au_bs[:, ix], axis=1)
                map_ = np.nanmean(ap_bs[:, ix], axis=1)
                a1, a2 = np.nanpercentile(mau, [2.5, 97.5])
                p1, p2 = np.nanpercentile(map_, [2.5, 97.5])
                print(f"{grp:<18}{'':>8}{'':>7}"
                      + f"{np.nanmean(mau):.4f} ({a1:.4f}-{a2:.4f})".rjust(26)
                      + f"{np.nanmean(map_):.4f} ({p1:.4f}-{p2:.4f})".rjust(26))
                rows.append({"cohort": name, "endpoint": "MACRO " + grp.split()[0],
                             "n": len(X), "events": -1, "prevalence": -1,
                             "auroc": round(float(np.nanmean(mau)), 5),
                             "auroc_lo": round(float(a1), 5),
                             "auroc_hi": round(float(a2), 5),
                             "auprc": round(float(np.nanmean(map_)), 5),
                             "auprc_lo": round(float(p1), 5),
                             "auprc_hi": round(float(p2), 5),
                             "cal_slope": -1, "cal_slope_lo": -1,
                             "cal_slope_hi": -1, "cal_intercept": -1,
                             "cal_intercept_lo": -1, "cal_intercept_hi": -1})

    if not rows:
        raise SystemExit("\nNeither external cohort was scored.")

    print()
    print("=" * 94)
    print("HOW TO READ THIS")
    print("=" * 94)
    print("  AUROC is the number that transports. Compare it with the internal")
    print("  test column; a drop of a few hundredths is normal and expected,")
    print("  and a drop of a tenth means the model learned something local.")
    print()
    print("  AUPRC moves with prevalence by construction. BIDMC has 47.9%")
    print("  hospitalisation against UMN's 20.4%, so its AUPRC will be higher")
    print("  for reasons that have nothing to do with the model. Do not")
    print("  compare AUPRC across cohorts; compare it between models within")
    print("  one.")
    print()
    print("  Slope below 1 means the predictions are too extreme for this")
    print("  cohort; intercept away from 0 means the base rate moved. Both")
    print("  are expected here and both are fixable by recalibration, which")
    print("  is deliberately not applied above -- the size of the")
    print("  miscalibration is itself a result.")

    if a.csv:
        os.makedirs(RES, exist_ok=True)
        fn = (f"{RES}/external_{a.arm}_{tag_feat}"
              f"{_t.drop_suffix(drop)}.csv")
        with open(fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  written to {fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
