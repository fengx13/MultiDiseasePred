"""
04_rank_features.py -- one ranking of the 63 variables, computed once.

Why this has to be redone. The published top-10 and top-15 lists are not
nested: top-10 leaves out triage_sbp and triage_heartrate while including
eci_HTN1 and gender, which sit at 11 and 13 in the top-15 list. Two lists
where the shorter is not a prefix of the longer cannot both be truncations of
one ranking, so they came from two separate attribution runs. A parsimony
curve built on that is not a curve of one ranking -- each point is a different
selection procedure, and the shape means nothing.

This produces a single ranking and writes it to results/feature_ranking.csv.
Every point of the parsimony sweep is then a prefix of that one list.

Three choices worth stating in the Methods:

Computed on training data, not test. Attribution is part of fitting the model,
so letting the 2024-2025 cohort influence which variables are kept would make
every later interval optimistic.

Averaged over three seeds. A ranking from a single network partly reflects
that network's initialisation. Averaging the attribution matrices first, then
ranking, gives a list that does not move when the seed does; the script prints
how much the three seeds disagree so that stability is a reported number
rather than an assumption.

Mean over encounters first, then over targets. That weights all nine outcomes
equally instead of letting hospitalisation, at 30% prevalence, decide the list
for outcomes at 0.4%.

    python 04_rank_features.py                 needs the all63 runs to exist
    python 04_rank_features.py --n-explain 4000 --n-samples 300
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from determinism import seed_everything, run_fingerprint  # noqa
from attribution import expected_gradients, global_ranking  # noqa
import models as M  # noqa
import importlib
_t = importlib.import_module("01_train")

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
DATA = f"{PROJ}/data"
RUNS = f"{PROJ}/runs"
RES = f"{PROJ}/results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="G2_no_l1_no_drop")
    ap.add_argument("--n-explain", type=int, default=4000,
                    help="encounters to attribute; error falls as 1/sqrt(n)")
    ap.add_argument("--n-background", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--drop", default="",
                    help="comma separated columns to exclude. Ranks what is "
                         "left, and writes to its own CSV.")
    a = ap.parse_args()
    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)
    sfx = _t.drop_suffix(drop)

    seed_everything(42)
    os.makedirs(RES, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.load(open(f"{DATA}/preprocessing.json"))
    # names must be the columns the model was actually trained on, not every
    # column in the file. With --drop the two differ, and using the full list
    # would silently pair attribution i with the wrong variable name.
    (Xtr, ytr), _, names, _ = _t.load_split("all63", drop)
    # expand_drop, not set(drop): --drop may name a group such as "comorb",
    # which is not a column and would filter nothing.
    gone = _t.expand_drop(drop, meta["features"])
    expected = [c for c in meta["features"] if c not in gone]
    assert names == expected, "column order drifted between prepare and train"

    dirs = sorted(glob.glob(f"{RUNS}/{a.arm}_all63{sfx}_seed*"))
    if not dirs:
        raise SystemExit(
            f"no {a.arm}_all63{sfx}_seed* runs found.\n"
            f"Train them first:  01_train.py --arm {a.arm} --features all63"
            + (f" --drop {a.drop}" if drop else ""))
    print(f"ranking from {len(dirs)} seeds of {a.arm} on "
          f"{len(names)} variables"
          + (f", excluding {', '.join(drop)}" if drop else "") + "\n")

    # A fixed explain set and background, shared by every seed, so that any
    # difference between seeds is the model and not the sample.
    rng = np.random.default_rng(0)
    ie = rng.choice(len(Xtr), min(a.n_explain, len(Xtr)), replace=False)
    ib = rng.choice(len(Xtr), min(a.n_background, len(Xtr)), replace=False)
    Xe = torch.from_numpy(Xtr[ie]).to(dev)
    Xb = torch.from_numpy(Xtr[ib]).to(dev)

    per_seed, A_sum = [], None
    for d in dirs:
        seed = int(d.rsplit("seed", 1)[-1])
        model = M.build(a.arm, Xtr.shape[1], n_tasks=ytr.shape[1],
                        hidden=_t.HP["hidden"], n_experts=_t.HP["n_experts"],
                        temperature=_t.HP["temperature"]).to(dev)
        # 01_train.py saves a plain state_dict, so weights_only=True is both
        # safe and the future default. Without it torch prints a six-line
        # FutureWarning per seed, which is what buried the real traceback in
        # the job 9457 log.
        model.load_state_dict(torch.load(f"{d}/model.pt", map_location=dev,
                                         weights_only=True))
        model.eval()

        t0 = time.time()
        A = expected_gradients(model, Xe, Xb, n_samples=a.n_samples,
                               seed=seed, device=dev)          # (N, D, T)
        A = np.asarray(A)
        A_sum = A if A_sum is None else A_sum + A
        order, imp = global_ranking(A)
        per_seed.append({"seed": seed, "order": [names[i] for i in order],
                         "importance": imp})
        print(f"  seed {seed}   {time.time() - t0:.0f} s   "
              f"top 5: {', '.join(names[i] for i in order[:5])}", flush=True)

    A_mean = A_sum / len(dirs)
    order, imp = global_ranking(A_mean)

    # How much do the seeds disagree? Spearman on the ranks, plus how often the
    # same variables occupy the first k positions.
    def ranks_of(lst):
        return {n: i for i, n in enumerate(lst)}
    print("\nseed agreement")
    rs = [ranks_of(p["order"]) for p in per_seed]
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            x = np.array([rs[i][n] for n in names])
            y = np.array([rs[j][n] for n in names])
            xr, yr = x.argsort().argsort(), y.argsort().argsort()
            rho = np.corrcoef(xr, yr)[0, 1]
            print(f"  seeds {per_seed[i]['seed']} vs {per_seed[j]['seed']}"
                  f"   Spearman {rho:.3f}")
    for k in (5, 10, 15, 20):
        sets = [set(p["order"][:k]) for p in per_seed]
        common = set.intersection(*sets)
        print(f"  first {k:2d}: {len(common)} of {k} variables shared by all seeds")

    ranked = [names[i] for i in order]
    with open(_t.ranking_path(drop), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "feature", "mean_abs_attribution"])
        for r, i in enumerate(order, 1):
            w.writerow([r, names[i], float(imp[i])])

    # the (D, T) matrix, for the heat map and for eTable
    per_target = np.abs(A_mean).mean(axis=0)          # (D, T)
    with open(f"{RES}/attribution_matrix{sfx}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature"] + [o.replace("outcome_", "")
                                  for o in meta["outcomes"]])
        for i in order:
            w.writerow([names[i]] + [float(v) for v in per_target[i]])

    # Stamp the ranking with the data it came from. A ranking and the models
    # trained on it have to agree about which arrays they saw; without this
    # there is no way to tell afterwards whether they did.
    json.dump({"arm": a.arm, "seeds": [p["seed"] for p in per_seed],
               "data_id": meta.get("data_id"),
               "dropped": list(drop),
               "n_explain": len(ie), "n_background": len(ib),
               "n_samples": a.n_samples, "ranking": ranked,
               "fingerprint": run_fingerprint()},
              open(f"{RES}/feature_ranking{sfx}.json", "w"), indent=2)

    print("\nranking, best first")
    for r, i in enumerate(order, 1):
        mark = "  <- published top-15" if names[i] in _t.TOP15 else ""
        print(f"  {r:2d}  {names[i]:<26} {imp[i]:.5f}{mark}")

    new15, old15 = set(ranked[:15]), set(_t.TOP15)
    print(f"\nagreement with the published fifteen: "
          f"{len(new15 & old15)} of 15 shared")
    if new15 - old15:
        print(f"  newly in:  {', '.join(sorted(new15 - old15))}")
    if old15 - new15:
        print(f"  dropped:   {', '.join(sorted(old15 - new15))}")
    print(f"\nwritten to {_t.ranking_path(drop)}")


if __name__ == "__main__":
    main()
