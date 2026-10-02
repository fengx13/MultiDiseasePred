"""
13_lightgbm_local.py -- the one baseline that cannot run on the cluster.

LightGBM is not in the Apptainer image and the compute nodes have no network,
so it cannot be installed there. It is installed in the Windows virtual
environment at code_ke\\MTL, which is where the published Table 2 numbers came
from, and where it was the strongest baseline on six of nine endpoints. It is
not a baseline the paper can leave out.

So this runs on the Citrix side instead. The prepared arrays are pulled down
from the cluster first, LightGBM is trained here, and only the predicted
probabilities go back -- a few tens of megabytes rather than a model.

    "Q:\\XieF-Req04048\\Workspace\\code_ke\\MTL\\python.exe" ^
        pipeline\\13_lightgbm_local.py --features greedy:8

Everything that makes the comparison fair is copied from 01_train.py rather
than reimplemented:

  the same arrays          umn_train_X.npy and umn_test_X.npy, pulled as-is
  the same scaling         (X - scaler_mean) / scaler_scale from
                           preprocessing.json
  the same validation cut  the seeded permutation, so the val column lines up
                           with every other arm's
  the same variable sets   read from feature_ranking_greedy_noesi.csv

The validation split is the one place this can silently diverge. 01_train.py
takes it with torch.randperm under a seeded generator. If torch is present
here the identical call is made; if it is not, the val split is skipped
entirely and only the test column is produced, because a validation column
computed from a *different* set of patients would look perfectly normal
sitting next to the others and would be wrong.

Output goes into local run directories with the same names the cluster uses,
so pushing them back needs no translation and 10_ci.py finds them by the
rules it already has.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ACUTE, DISP = list(range(2, 9)), [0, 1]
VAL_FRAC = 0.2          # HP["val_frac"] in 01_train.py


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def auprc(y, s):
    from sklearn.metrics import average_precision_score
    if np.asarray(y).sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def drop_suffix(drop):
    if not drop:
        return ""
    short = {"triage_acuity": "esi"}
    return "_no" + "-".join(sorted(short.get(d, d.replace("_", ""))
                                   for d in drop))


def resolve(features, cols, drop, rank_csv):
    """The same three tokens 01_train.py understands, minus the ones this
    baseline will never be asked for."""
    keep = [c for c in cols if c not in drop]
    if features == "all63":
        return [cols.index(c) for c in keep]
    if features.startswith(("greedy:", "rank:")):
        import csv as _csv
        with open(rank_csv) as fh:
            names = [r["feature"] for r in _csv.DictReader(fh)]
        names = [n for n in names if n not in drop]
        k = int(features.split(":")[1])
        if k > len(names):
            raise SystemExit(f"asked for {k}, ranking has {len(names)}")
        return [cols.index(c) for c in names[:k]]
    raise SystemExit(f"unknown --features {features}")


def val_indices(n, seed):
    """The same permutation 01_train.py uses, or nothing at all.

    Returning a numpy permutation here instead would produce a validation
    column drawn from different patients than every other arm's, which is
    exactly the kind of mistake that survives every check and ruins a table.
    """
    try:
        import torch
    except ImportError:
        return None
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).numpy()
    return perm[:int(VAL_FRAC * n)], perm[int(VAL_FRAC * n):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="lgbm_work",
                   help="folder holding the arrays pulled from the cluster")
    ap.add_argument("--features", default="greedy:8")
    ap.add_argument("--drop", default="triage_acuity")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", default="lgbm_runs")
    a = ap.parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        raise SystemExit(
            "lightgbm is not importable. Run this with the MTL interpreter:\n"
            '  "Q:\\XieF-Req04048\\Workspace\\code_ke\\MTL\\python.exe" '
            "pipeline\\13_lightgbm_local.py ...")
    print("lightgbm", lgb.__version__)

    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)
    seeds = [int(x) for x in a.seeds.split(",")]
    D = a.data
    meta = json.load(open(os.path.join(D, "preprocessing.json")))
    cols, data_id = meta["features"], meta.get("data_id")
    idx = resolve(a.features, cols, drop,
                  os.path.join(D, "feature_ranking_greedy_noesi.csv"))
    chosen = [cols[i] for i in idx]
    mean = np.asarray(meta["scaler_mean"])[idx]
    scale = np.asarray(meta["scaler_scale"])[idx]
    scale = np.where(scale == 0, 1, scale)

    print(f"data_id  {data_id}")
    print(f"features {a.features}  ->  {len(chosen)} variables")
    for c in chosen:
        print("   ", c)

    Xtr = (np.load(os.path.join(D, "umn_train_X.npy"))[:, idx] - mean) / scale
    ytr = np.load(os.path.join(D, "umn_train_y.npy"))
    Xte = (np.load(os.path.join(D, "umn_test_X.npy"))[:, idx] - mean) / scale
    yte = np.load(os.path.join(D, "umn_test_y.npy"))
    print(f"train {Xtr.shape}   test {Xte.shape}")

    # The external cohorts, if their arrays were pulled down alongside the UMN
    # ones. Without this the external columns of the results table are empty
    # for LightGBM, because -- like every other tree arm -- no estimator is
    # ever saved to disk and the model exists only inside this loop.
    ext = {}
    for nm in ("bidmc", "stanford"):
        fx = os.path.join(D, nm + "_X.npy")
        fy = os.path.join(D, nm + "_y.npy")
        if os.path.exists(fx) and os.path.exists(fy):
            ext[nm] = ((np.load(fx)[:, idx] - mean) / scale, np.load(fy))
            print(f"external {nm}: {ext[nm][0].shape}")
    if not ext:
        print("no external arrays in " + D + " -- external columns will be "
              "absent for LightGBM. Pull bidmc_*.npy and stanford_*.npy.")

    split = val_indices(len(Xtr), 42)
    if split is None:
        print("\n!! torch is not in this environment, so the validation split")
        print("   cannot be reproduced. Producing the test column only; the")
        print("   validation column for LightGBM will be absent rather than")
        print("   computed from a different set of patients.\n")
        va_i = None
        Xfit, yfit = Xtr, ytr
    else:
        va_i, tr_i = split
        Xva, yva = Xtr[va_i], ytr[va_i]
        Xfit, yfit = Xtr[tr_i], ytr[tr_i]
        print(f"fit {Xfit.shape}   val {Xva.shape}")

    T = ytr.shape[1]
    tag_feat = a.features.replace(":", "")
    for seed in seeds:
        t0 = time.time()
        tag = f"A4_lgbm_{tag_feat}{drop_suffix(drop)}_seed{seed}"
        out = os.path.join(a.out, tag)
        os.makedirs(out, exist_ok=True)
        print(f"\n=== {tag} ===", flush=True)

        params = dict(n_estimators=300, max_depth=6, num_leaves=63,
                      learning_rate=0.1, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, random_state=seed, verbose=-1)
        te = np.zeros((len(Xte), T), dtype=np.float64)
        va = (np.zeros((len(va_i), T), dtype=np.float64)
              if va_i is not None else None)
        ex = {k: np.zeros((len(v[0]), T), dtype=np.float64)
              for k, v in ext.items()}
        for t in range(T):
            # num_leaves=63, not the default 31. XGBoost grows level-wise,
            # so max_depth=6 gives it up to 2^6 = 64 leaves. LightGBM grows
            # leaf-wise and num_leaves is the real control; max_depth is only
            # a ceiling. Setting the same max_depth on both and leaving
            # num_leaves at its default gave LightGBM half the capacity, and
            # the first run showed exactly that signature -- it trailed
            # XGBoost by 0.0058 at eight variables, 0.0099 at thirteen and
            # 0.0136 at sixty-two, the gap widening with the number of
            # variables rather than staying constant. An algorithmic
            # difference would not scale that way; a capacity cap does.
            #
            # 63 = 2^6 - 1 is the tightest match to a depth-6 level-wise tree.
            m = lgb.LGBMClassifier(**params)
            m.fit(Xfit, yfit[:, t])
            te[:, t] = m.predict_proba(Xte)[:, 1]
            if va is not None:
                va[:, t] = m.predict_proba(Xva)[:, 1]
            for k, (Xe, _) in ext.items():
                ex[k][:, t] = m.predict_proba(Xe)[:, 1]
            print(f"  endpoint {t + 1}/{T} done", flush=True)

        np.save(os.path.join(out, "logits_umn_test.npy"), te.astype(np.float32))
        np.save(os.path.join(out, "labels_umn_test.npy"), yte.astype(np.int8))
        if va is not None:
            np.save(os.path.join(out, "logits_umn_val.npy"), va.astype(np.float32))
            np.save(os.path.join(out, "labels_umn_val.npy"), yva.astype(np.int8))
        for k, (_, ye) in ext.items():
            np.save(os.path.join(out, "logits_" + k + ".npy"), ex[k].astype(np.float32))
            np.save(os.path.join(out, "labels_" + k + ".npy"), ye.astype(np.int8))
            print(f"  scored {k}: {len(ye):,} encounters", flush=True)

        per = [{"target": j, "auroc": auroc(yte[:, j], te[:, j]),
                "auprc": auprc(yte[:, j], te[:, j])} for j in range(T)]
        val_per = ([{"target": j, "auroc": auroc(yva[:, j], va[:, j]),
                     "auprc": auprc(yva[:, j], va[:, j])} for j in range(T)]
                   if va is not None else [])

        def mac(rows, ix, key):
            return (float(np.nanmean([rows[i][key] for i in ix]))
                    if rows else float("nan"))

        metrics = {
            "arm": "A4_lgbm", "seed": seed, "features": a.features,
            # predict_proba, not a logit -- see 01_train.py
            "score_scale": "probability",
            "dropped": list(drop), "data_id": data_id,
            "n_features": len(chosen), "feature_names": chosen,
            "n_parameters": 0,
            "note": ("trained in the Windows venv at code_ke\\MTL because "
                     "lightgbm is not in the cluster container"),
            # Recorded so that a directory can be told apart from one produced
            # by the first version, which left num_leaves at its default of 31
            # and gave LightGBM half of XGBoost's capacity. run_all.ps1 keys
            # on this to decide whether a run needs redoing.
            "hp": {"estimator": "LGBMClassifier", "params": params,
                   "one_model_per_endpoint": True},
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
        json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"),
                  indent=2)
        print(f"  macro AUROC  acute {metrics['macro_acute_auroc']:.4f}   "
              f"disposition {metrics['macro_disp_auroc']:.4f}")
        print(f"  {metrics['minutes']:.1f} min -> {out}", flush=True)

    print("\nPush these back with the second half of lightgbm_local.ps1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
