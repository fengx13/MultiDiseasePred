"""
11_greedy_select.py -- pick the k variables that matter, not the k that rank.

04_rank_features.py orders variables by mean absolute attribution and
05_parsimony.py takes the first k. That is the wrong operation when the
candidates are correlated. Attribution answers "how much does the model use
this variable", one variable at a time; a prefix of that ordering answers
nothing in particular, because two variables carrying the same information
both score highly and both get taken.

The current k=13 set shows it. It contains n_ed_365d, and n_ed_30d sits at
rank 15 waiting to come in -- three ED-visit counts over nested windows, each
computed from the same events. The second one buys a fraction of what its
attribution suggests.

This script asks the question that variable selection is actually about: at
each step, which single remaining variable most improves held-out
performance, given everything already chosen. That is marginal contribution,
and it drops a redundant variable to the bottom automatically because once
its partner is in, adding it changes nothing.

    variables chosen so far    S
    for each candidate c       fit on S + {c}, score on validation
    keep the best              S <- S + {argmax}

Cost, and why it is affordable
------------------------------
Greedy selection is quadratic in the number of candidates: choosing 15 from
62 is 62 + 61 + ... + 48 = 825 fits. That is only sane with a fast model,
which is why the selection runs on gradient boosting rather than on the
multi-task network -- 0.2 minutes against 4.2. Subsampling the training rows
cuts it again, and it costs almost nothing: at 1.3 million rows the ordering
is decided long before the last hundred thousand are read.

A selector does not have to be the model that is finally reported. It has to
rank candidate sets in the same order the final model would. Gradient
boosting and the network agree to within 0.002 macro AUROC at every point on
the existing parsimony curve, which is the evidence that they do.

The output is a CSV in the same shape as feature_ranking*.csv, so
05_parsimony.py --ranking greedy and 01_train.py --features greedy:N read it
without knowing anything changed.

--targets decides which endpoints get a vote, and it matters more than it
looks. Selecting on the seven acute conditions alone produced a set that beat
the attribution prefix by 0.011 macro AUROC on those seven and lost 0.055 on
the two disposition endpoints, which nobody asked for. The two orderings are
written to different files so both can be kept.

    python 11_greedy_select.py --drop triage_acuity --k 15
    python 11_greedy_select.py --drop triage_acuity --k 15 --targets all
"""

from __future__ import annotations

import argparse
import csv
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
RES = f"{PROJ}/results"

ACUTE = list(range(2, 9))


def env_report():
    """What is actually in the container. Printed rather than assumed: a run
    that quietly falls back to a different library is a run whose numbers
    cannot be compared with the ones beside it."""
    print("environment")
    for m in ("numpy", "scipy", "sklearn", "xgboost", "lightgbm", "torch"):
        try:
            mod = importlib.import_module(m)
            print(f"  {m:<10} {getattr(mod, '__version__', 'unknown')}")
        except Exception as e:
            print(f"  {m:<10} NOT AVAILABLE  ({type(e).__name__}: {e})")
    print(flush=True)


def fit_score(Xtr, ytr, Xva, yva, idx, seed, n_est, targets):
    """Macro AUROC over `targets` for one candidate set.

    One model per endpoint, as everywhere else in the pipeline. Which
    endpoints are in `targets` is the whole design decision: a variable set
    is only ever optimal for the thing it was selected against.
    """
    import xgboost as xgb
    tot = 0.0
    for t in targets:
        m = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=6, learning_rate=0.1,
            tree_method="hist", random_state=seed, verbosity=0,
            subsample=0.8, colsample_bytree=0.8,
            device="cuda" if _t.torch.cuda.is_available() else "cpu")
        m.fit(Xtr[:, idx], ytr[:, t])
        tot += _t.auroc(yva[:, t], m.predict_proba(Xva[:, idx])[:, 1])
    return tot / len(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", default="")
    ap.add_argument("--k", type=int, default=15,
                    help="how many to select; stops there")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample", type=int, default=400000,
                    help="training rows for the selection pass, 0 for all")
    ap.add_argument("--n-est", type=int, default=150,
                    help="trees per fit during selection; the final numbers "
                         "come from 05_parsimony.py, not from here")
    ap.add_argument("--targets", default="acute",
                    choices=["acute", "all", "disposition"])
    a = ap.parse_args()

    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)
    sfx = _t.drop_suffix(drop)
    # The ordering chosen against the acute endpoints and the one chosen
    # against all nine are different orderings and get different files. The
    # first greedy sweep is typically still being read by a parsimony run when
    # the second is launched, so sharing a name would overwrite the ranking
    # under a job that is halfway through using it.
    variant = {"acute": "greedy", "all": "greedyall",
               "disposition": "greedydisp"}[a.targets]
    env_report()

    (Xtr, ytr), _, cols, meta = _t.load_split("all63", drop)
    data_id = meta.get("data_id")
    # DISPOSITION is hospitalisation and critical illness -- the two endpoints
    # this file's own docstring records losing 0.055 on when the selection
    # votes with the acute conditions only. That loss was accepted at the
    # time because the paper's claim was about the acute conditions. It has
    # since become the paper's weakest number: the twelve-variable model
    # reads 0.7926 on hospitalisation against 0.8536 at 62 variables, a gap
    # of 0.061, larger than any acute endpoint's and three times the acute
    # macro's. The budget was set without this endpoint having a vote, so
    # this option gives it one and lets the two orderings be compared
    # instead of assumed.
    DISPOSITION = [0, 1]
    targets = {"acute": ACUTE, "disposition": DISPOSITION,
               "all": list(range(ytr.shape[1]))}[a.targets]

    # Same held-out fraction and same seeded permutation the other arms use,
    # so a set chosen here is chosen on the data the curve will be read on.
    n = len(Xtr)
    n_val = int(_t.HP["val_frac"] * n)
    perm = _t.torch.randperm(
        n, generator=_t.torch.Generator().manual_seed(a.seed)).numpy()
    va_i, tr_i = perm[:n_val], perm[n_val:]
    Xva, yva = Xtr[va_i], ytr[va_i]
    Xtr, ytr = Xtr[tr_i], ytr[tr_i]

    if a.sample and a.sample < len(Xtr):
        rng = np.random.default_rng(a.seed)
        keep = rng.choice(len(Xtr), a.sample, replace=False)
        Xtr, ytr = Xtr[keep], ytr[keep]

    print(f"data_id     {data_id}")
    print(f"candidates  {len(cols)}" + (f"   excluding {', '.join(drop)}"
                                        if drop else ""))
    print(f"train rows  {len(Xtr):,}   validation rows {len(Xva):,}")
    print(f"endpoints   {a.targets} ({len(targets)})")
    print(f"selecting   {a.k}, {a.n_est} trees per fit")
    n_fits = sum(len(cols) - i for i in range(a.k))
    print(f"about {n_fits:,} fits x {len(targets)} endpoints\n", flush=True)

    chosen, remaining = [], list(range(len(cols)))
    rows, t0 = [], time.time()
    prev = 0.0
    for step in range(1, a.k + 1):
        best, best_s = None, -1.0
        for c in remaining:
            s = fit_score(Xtr, ytr, Xva, yva, chosen + [c], a.seed,
                          a.n_est, targets)
            if s > best_s:
                best, best_s = c, s
        chosen.append(best)
        remaining.remove(best)
        gain = best_s - prev
        prev = best_s
        rows.append({"rank": step, "feature": cols[best],
                     "val_macro_auroc": round(best_s, 6),
                     "marginal_gain": round(gain, 6)})
        # {gain:+.4f}, not +{gain:.4f}: the best remaining candidate can still
        # make things worse once the useful variables are in, and "+-0.0037"
        # is a line no one reads correctly at four in the morning.
        print(f"[{step:>2}/{a.k}] {cols[best]:<28} "
              f"macro AUROC {best_s:.4f}   {gain:+.4f}   "
              f"{(time.time() - t0) / 60:.0f} min", flush=True)

        # Write after every step. A greedy pass is hours long and a timeout
        # halfway through should leave a usable prefix rather than nothing --
        # the first ten of a greedy ordering are the first ten regardless of
        # where it stopped, which is not true of a curve.
        out = f"{RES}/feature_ranking_{variant}{sfx}.csv"
        os.makedirs(RES, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(f"{RES}/feature_ranking_{variant}{sfx}.json", "w") as fh:
            json.dump({"data_id": data_id, "drop": list(drop),
                       "seed": a.seed, "sample": a.sample,
                       "n_estimators": a.n_est, "targets": a.targets,
                       "selected": [cols[i] for i in chosen]}, fh, indent=2)

    print(f"\nwritten to {RES}/feature_ranking_{variant}{sfx}.csv")
    print(f"total {(time.time() - t0) / 60:.0f} min\n")

    print("=" * 66)
    print("GREEDY ORDER")
    print("=" * 66)
    print(f"{'k':>3}  {'variable':<28} {'macro AUROC':>12} {'marginal':>10}")
    print("-" * 66)
    for r in rows:
        print(f"{r['rank']:>3}  {r['feature']:<28} "
              f"{r['val_macro_auroc']:>12.4f} {r['marginal_gain']:>+10.4f}")

    # Where it flattens. The comparison worth making is against the attribution
    # prefix, not against zero: both orderings reach the same place eventually,
    # and the claim being tested is that this one gets there sooner.
    att = _t.ranking_path(drop)
    if os.path.exists(att):
        with open(att) as fh:
            pre = [r["feature"] for r in csv.DictReader(fh)][:a.k]
        same = [c for c in pre if c in {r["feature"] for r in rows}]
        print(f"\n{len(same)} of the first {a.k} are in both orderings.")
        only_g = [r["feature"] for r in rows if r["feature"] not in pre]
        only_a = [c for c in pre if c not in {r["feature"] for r in rows}]
        if only_g:
            print(f"  greedy only:      {', '.join(only_g)}")
        if only_a:
            print(f"  attribution only: {', '.join(only_a)}")
        print("\n  A variable the attribution ordering takes and this one does")
        print("  not is a variable whose information was already present.")

    print(f"\nNext: 05_parsimony.py --ranking {variant} --drop ... --grid ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
