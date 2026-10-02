"""
12_news_baseline.py -- the clinical score every ED already has.

NEWS is what the comparison is really against. Gradient boosting is the
machine-learning baseline; NEWS is the one a nurse is using at the bedside
right now, and a model that cannot beat it has no reason to be deployed.

What this computes, and what it leaves out
------------------------------------------
NEWS2 has seven components. Five of them are in our feature matrix:

    respiratory rate, oxygen saturation, temperature,
    systolic blood pressure, heart rate

Two are not recorded in this extract:

    supplemental oxygen (0 on air, 2 on oxygen)
    level of consciousness (0 alert, 3 for confusion/voice/pain/unresponsive)

So this is an abbreviated NEWS, and it is named that way everywhere it
appears. Both omitted components can only add points, and both add them to
sicker patients, so the abbreviated score is a conservative version of the
real one -- it will if anything understate what NEWS achieves. That is the
right direction for a baseline we intend to beat, and it has to be stated in
the methods rather than left for a reader to notice.

Why not reuse the score_NEWS column
-----------------------------------
The extract carries one, and 07_baselines.py used it. Its temperature
component is dead: the values were in Fahrenheit and fed to a Celsius
formula, which puts every single encounter in the top temperature band. A
constant adds nothing to a ranking, so that column is a four-component score
wearing a seven-component name. Our arrays are Celsius, so recomputing gives
the temperature component back.

Alignment
---------
The score is computed from umn_test_X.npy directly, not by re-reading the
source file, so every row is the row the other arms scored. The labels are
loaded alongside and the shapes are checked. Nothing here can drift out of
step with the runs it is compared against.

NEWS does not vary with the feature set: it is the same five vitals whatever
k is chosen. It therefore gets one run directory, not one per variable set,
and the same column stands beside every parsimony point.

    python 12_news_baseline.py
    python 12_news_baseline.py --drop triage_acuity
"""

from __future__ import annotations

import argparse
import json
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

ACUTE, DISP = list(range(2, 9)), [0, 1]

# NEWS2, Royal College of Physicians 2017. Each entry is a list of
# (upper_bound_inclusive, points), read in order, with the last entry
# catching everything above.
BANDS = {
    "triage_resprate": [(8, 3), (11, 1), (20, 0), (24, 2), (np.inf, 3)],
    "triage_o2sat": [(91, 3), (93, 2), (95, 1), (np.inf, 0)],
    "triage_sbp": [(90, 3), (100, 2), (110, 1), (219, 0), (np.inf, 3)],
    "triage_heartrate": [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2),
                         (np.inf, 3)],
    "triage_temperature": [(35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1),
                           (np.inf, 2)],
}


def band_score(v, bands):
    """Points for one component. np.select rather than a loop: at 350,000
    rows a Python loop over bands is still fine, but the vectorised form is
    also the one that documents the bands as data instead of control flow."""
    out = np.full(len(v), bands[-1][1], dtype=np.float32)
    # walk downwards so the tightest band wins
    for upper, pts in reversed(bands[:-1]):
        out = np.where(v <= upper, pts, out)
    # A missing vital scores zero rather than propagating NaN. The arrays are
    # already median-imputed upstream, so this should never fire; it is here
    # so that if imputation ever changes, NEWS degrades to "no points for the
    # thing we did not measure" instead of producing a column of NaN that
    # silently drops every row from the AUROC.
    return np.where(np.isnan(v), 0.0, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", default="",
                    help="only affects the run directory name; NEWS uses the "
                         "same five vitals whatever else is excluded")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)

    t0 = time.time()
    meta = json.load(open(f"{DATA}/preprocessing.json"))
    cols, data_id = meta["features"], meta.get("data_id")
    tag = a.tag or f"A5_news_all63{_t.drop_suffix(drop)}_seed42"
    out = f"{RUNS}/{tag}"
    os.makedirs(out, exist_ok=True)
    print(f"\n=== {tag} ===")
    print(f"  data_id {data_id}")

    missing = [c for c in BANDS if c not in cols]
    if missing:
        raise SystemExit(f"not in the feature matrix: {missing}")

    # Both splits, because every other arm reports both and the comparison
    # table has a column for each.
    res = {}
    for split in ("test", "train"):
        X = np.load(f"{DATA}/umn_{split}_X.npy")
        y = np.load(f"{DATA}/umn_{split}_y.npy")
        if len(X) != len(y):
            raise SystemExit(f"{split}: X has {len(X)} rows, y has {len(y)}")

        score = np.zeros(len(X), dtype=np.float32)
        print(f"\n  {split}: {len(X):,} rows")
        for c, bands in BANDS.items():
            v = X[:, cols.index(c)].astype(np.float64)
            pts = band_score(v, bands)
            score += pts
            print(f"    {c:<22} median {np.median(v):8.1f}   "
                  f"mean points {pts.mean():.3f}")
        print(f"    abbreviated NEWS  mean {score.mean():.2f}  "
              f"sd {score.std():.2f}  range {score.min():.0f}..{score.max():.0f}")
        res[split] = (score, y)

    # The validation split is a seeded slice of the training matrix, taken the
    # same way every arm takes it, so that the val column here lines up with
    # the val column everywhere else.
    Xtr_score, ytr = res["train"]
    n = len(ytr)
    n_val = int(_t.HP["val_frac"] * n)
    perm = _t.torch.randperm(
        n, generator=_t.torch.Generator().manual_seed(42)).numpy()
    va_i = perm[:n_val]
    val_score, yva = Xtr_score[va_i], ytr[va_i]

    te_score, yte = res["test"]
    T = yte.shape[1]

    # One score, nine columns: NEWS gives every endpoint the same ranking.
    # Storing it in the shape the other arms use means 08_final_tables.py and
    # 10_ci.py need no special case for it.
    np.save(f"{out}/logits_umn_test.npy",
            np.repeat(te_score.reshape(-1, 1), T, axis=1).astype(np.float32))
    np.save(f"{out}/labels_umn_test.npy", yte.astype(np.int8))
    np.save(f"{out}/logits_umn_val.npy",
            np.repeat(val_score.reshape(-1, 1), T, axis=1).astype(np.float32))
    np.save(f"{out}/labels_umn_val.npy", yva.astype(np.int8))

    # The external cohorts, on the raw feature matrix. NEWS needs no fitting,
    # so there is nothing to transfer and nothing to recalibrate -- the points
    # come from the published bands either way. That makes it the one arm
    # whose external result requires no assumption at all, and the right floor
    # to compare every transported model against.
    for name in ("bidmc", "stanford"):
        fx, fy = f"{DATA}/{name}_X.npy", f"{DATA}/{name}_y.npy"
        if not (os.path.exists(fx) and os.path.exists(fy)):
            continue
        Xe, ye = np.load(fx), np.load(fy)
        se = np.zeros(len(Xe), dtype=np.float32)
        for c, bands in BANDS.items():
            se += band_score(Xe[:, cols.index(c)].astype(np.float64), bands)
        np.save(f"{out}/logits_{name}.npy",
                np.repeat(se.reshape(-1, 1), T, axis=1).astype(np.float32))
        np.save(f"{out}/labels_{name}.npy", ye.astype(np.int8))
        print(f"  scored {name}: {len(Xe):,} rows, "
              f"mean NEWS {se.mean():.2f}")

    per = [{"target": j, "auroc": _t.auroc(yte[:, j], te_score),
            "auprc": _t.auprc(yte[:, j], te_score)} for j in range(T)]
    val_per = [{"target": j, "auroc": _t.auroc(yva[:, j], val_score),
                "auprc": _t.auprc(yva[:, j], val_score)} for j in range(T)]

    def mac(rows, ix, key):
        return float(np.nanmean([rows[i][key] for i in ix]))

    metrics = {
        "arm": "A5_news", "seed": 42, "features": "all63",
        # the NEWS points themselves, an integer score on 0-20;
        # ranks fine, is not a probability and not a logit
        "score_scale": "points",
        "dropped": list(drop), "data_id": data_id,
        "n_features": len(BANDS),
        "feature_names": sorted(BANDS),
        "n_parameters": 0,
        "note": ("abbreviated NEWS2, five of seven components; supplemental "
                 "oxygen and level of consciousness are not recorded in this "
                 "extract. Both can only add points and both add them to "
                 "sicker patients, so this understates NEWS."),
        "per_target": per, "per_target_val": val_per,
        "macro_acute_auroc": mac(per, ACUTE, "auroc"),
        "macro_acute_auprc": mac(per, ACUTE, "auprc"),
        "macro_disp_auroc": mac(per, DISP, "auroc"),
        "macro_disp_auprc": mac(per, DISP, "auprc"),
        "val_macro_acute_auroc": mac(val_per, ACUTE, "auroc"),
        "val_macro_acute_auprc": mac(val_per, ACUTE, "auprc"),
        "val_macro_disp_auroc": mac(val_per, DISP, "auroc"),
        "val_macro_disp_auprc": mac(val_per, DISP, "auprc"),
        "gate": {}, "train_log": [],
        "minutes": (time.time() - t0) / 60,
    }
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=2)

    names = [o.replace("outcome_", "") for o in meta["outcomes"]]
    print()
    print("=" * 60)
    print("ABBREVIATED NEWS, internal validation 2024-2025")
    print("=" * 60)
    print(f"{'endpoint':<22}{'AUROC':>10}{'AUPRC':>10}")
    print("-" * 60)
    for j, nm in enumerate(names):
        print(f"{nm:<22}{per[j]['auroc']:>10.4f}{per[j]['auprc']:>10.4f}")
    print("-" * 60)
    print(f"{'acute macro':<22}{metrics['macro_acute_auroc']:>10.4f}"
          f"{metrics['macro_acute_auprc']:>10.4f}")
    print(f"{'disposition macro':<22}{metrics['macro_disp_auroc']:>10.4f}"
          f"{metrics['macro_disp_auprc']:>10.4f}")
    print()
    print("  Published Table 2 reported NEWS at 0.565 hospitalisation, 0.788")
    print("  critical, 0.652 sepsis. Those came from the score_NEWS column")
    print("  with its temperature component flattened, on an earlier cohort.")
    print("  A higher number here is the temperature band coming back, not a")
    print("  different score.")
    print(f"\n  {metrics['minutes']:.2f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
