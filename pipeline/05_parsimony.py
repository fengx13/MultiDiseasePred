"""
05_parsimony.py -- how far do k variables get you, ours and the baseline.

For each k, train the multi-task model and gradient boosting on the same first
k variables of the one ranking, three seeds each, and record both. Two curves
on one axis is the figure that answers the question the paper is actually
about: how much is given up by asking for fifteen numbers at triage instead of
sixty-three, and whether the multi-task model holds its advantage as the
variable set shrinks.

Ordering. The grid runs from the informative middle outwards rather than
1, 2, 3 upward: the elbow is expected somewhere around eight, and if the job
is cut short the points that determine where the curve flattens should already
be in hand. Every point writes its own directory and is skipped if done, so a
timeout is resumable.

Choosing k. The choice is made on the validation split, which 01_train.py now
saves for every run. Reading the test curve and picking the k where it looks
best would put a selection inside the number that is supposed to be an
out-of-sample estimate. Both curves are recorded; the validation one decides.

    python 05_parsimony.py
    python 05_parsimony.py --quick        one seed, the coarse grid only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib                                              # noqa: E402
# 01_train.py owns the two naming rules -- the run-directory suffix and where
# a ranking lives -- so they are imported rather than restated. A second copy
# here would be one edit away from writing a dropped run into a full run's
# directory, which is the one mistake this whole mechanism exists to prevent.
_t = importlib.import_module("01_train")                       # noqa: E402

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
DATA = f"{PROJ}/data"
RUNS = f"{PROJ}/runs"
RES = f"{PROJ}/results"

# dense where the elbow is expected, sparse where the curve is flat
GRID = [8, 15, 5, 10, 63, 3, 6, 12, 20, 1, 2, 4, 7, 9, 11, 13, 18, 25, 30,
        40, 50]
COARSE = [1, 3, 5, 8, 10, 15, 20, 63]
ARMS = ["G2_no_l1_no_drop", "A2_xgb_sub"]
SEEDS = [42, 43, 44]


def tag(arm, k, seed, sfx="", tok="rank"):
    return f"{arm}_{tok}{k}{sfx}_seed{seed}"


def done(arm, k, seed, sfx="", data_id=None, tok="rank"):
    """Finished, and finished on the data now in place.

    The first version returned os.path.exists(metrics.json). 01_train.py knows
    to retrain when the data_id has changed, but it never got the chance: this
    function filtered the point out before the subprocess was ever launched.
    Job 9510 rebuilt the arrays, retrained its twelve part-one models, then
    skipped all 108 sweep points and printed a curve from the previous data
    under "total 0 min" and exit 0. A guard that a caller can short-circuit is
    not a guard.
    """
    p = f"{RUNS}/{tag(arm, k, seed, sfx, tok)}/metrics.json"
    if not os.path.exists(p):
        return False
    if data_id is None:
        return True
    try:
        return json.load(open(p)).get("data_id") == data_id
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--grid", default=None,
                    help="comma separated k values, overriding the built-in "
                         "grid. Used for the cheap first pass.")
    ap.add_argument("--drop", default="",
                    help="comma separated columns to exclude, passed through "
                         "to 01_train.py")
    ap.add_argument("--ranking", default="attr",
                    choices=["attr", "greedy", "greedyall"],
                    help="attr: the attribution ordering from "
                         "04_rank_features.py, first N taken as a prefix. "
                         "greedy: the marginal-gain ordering from "
                         "11_greedy_select.py.")
    a = ap.parse_args()
    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)
    sfx = _t.drop_suffix(drop)
    greedy = a.ranking.startswith("greedy")
    tok = a.ranking if greedy else "rank"
    # The curve for one ordering must not be written over the curve for the
    # other, and a run directory from one must not be read as a run from the
    # other. Both names carry the token for that reason.
    csfx = (("_" + a.ranking) if greedy else "") + sfx

    rank_csv = _t.ranking_path(drop, a.ranking if greedy else "")
    if not os.path.exists(rank_csv):
        raise SystemExit(
            f"{rank_csv} not found; run "
            + ("11_greedy_select.py" if greedy else "04_rank_features.py")
            + " first")
    with open(rank_csv) as fh:
        ranking = [r["feature"] for r in csv.DictReader(fh)]
    # expand_drop, not set(drop): a group name such as "comorb" is not a
    # column and would filter nothing, leaving the grid asking for more
    # variables than the ranking can supply.
    ranking = [r for r in ranking if r not in _t.expand_drop(drop, ranking)]
    print(f"ranking has {len(ranking)} variables, "
          f"first five: {', '.join(ranking[:5])}")
    if drop:
        print(f"excluding {', '.join(drop)}")
    print()

    if a.grid:
        grid = [int(x) for x in a.grid.split(",")]
    else:
        grid = COARSE if a.quick else GRID
    # Dropping a variable shortens the ranking, so any k past the end has to
    # go. Silently clamping it to the maximum instead would put two different
    # k values into the same row of the curve.
    too_big = [k for k in grid if k > len(ranking)]
    if too_big:
        print(f"dropping k values past the end of the ranking: {too_big}\n")
        grid = [k for k in grid if k <= len(ranking)]
        if len(ranking) not in grid:
            grid.append(len(ranking))

    seeds = [SEEDS[0]] if a.quick else SEEDS
    arms = a.arms.split(",") if a.arms else ARMS

    data_id = json.load(open(f"{DATA}/preprocessing.json")).get("data_id")
    jobs = [(arm, k, s) for k in grid for arm in arms for s in seeds]
    todo = [j for j in jobs if not done(*j, sfx=sfx, data_id=data_id, tok=tok)]
    present = [j for j in jobs if os.path.exists(
        f"{RUNS}/{tag(*j, sfx=sfx, tok=tok)}/metrics.json")]
    stale = len(present) - (len(jobs) - len(todo))
    print(f"data_id {data_id}")
    print(f"{len(jobs)} points, {len(todo)} still to do")
    if stale:
        print(f"  {stale} of them have a directory already but were built on "
              f"different data, so they are being retrained rather than "
              f"reported")
    print(f"grid: {grid}\n", flush=True)

    t0, failed = time.time(), []
    for i, (arm, k, seed) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {arm}  k={k}  seed {seed}   "
              f"elapsed {(time.time() - t0) / 60:.0f} min", flush=True)
        cmd = [sys.executable, f"{HERE}/01_train.py",
               "--arm", arm, "--seed", str(seed),
               "--features", f"{tok}:{k}"]
        if drop:
            cmd += ["--drop", ",".join(drop)]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failed.append((arm, k, seed))
            print(f"  FAILED rc={rc}, continuing", flush=True)
            # Continuing past a failure is right for one bad point in a nine
            # hour sweep. It is wrong when nothing is working: job 9497 lost
            # all 108 points to a missing ranking file in three minutes and
            # still exited 0, so the log said COMPLETED and the curve was
            # empty. Stop early when the first handful all fail, rather than
            # spending the wall clock proving it.
            if len(failed) >= 5 and len(failed) == i:
                print(f"\nThe first {i} points all failed. This is not a bad "
                      f"grid point, it is a broken configuration.")
                print("Stopping rather than running the remaining "
                      f"{len(todo) - i} to the same end.")
                break

    # ---- collect ------------------------------------------------------
    rows, wrong_data = [], 0
    for arm, k, seed in jobs:
        p = f"{RUNS}/{tag(arm, k, seed, sfx, tok)}/metrics.json"
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        # Collect only what belongs to this data build. A point that failed to
        # retrain still has its old directory sitting there, and putting it in
        # the curve would mix two datasets in one line.
        if data_id is not None and m.get("data_id") != data_id:
            wrong_data += 1
            continue
        rows.append({
            "arm": arm, "k": k, "seed": seed,
            "val_acute_auroc": m["val_macro_acute_auroc"],
            "val_acute_auprc": m["val_macro_acute_auprc"],
            "val_disp_auroc": m["val_macro_disp_auroc"],
            "test_acute_auroc": m["macro_acute_auroc"],
            "test_acute_auprc": m["macro_acute_auprc"],
            "test_disp_auroc": m["macro_disp_auroc"],
            "test_disp_auprc": m["macro_disp_auprc"],
            "minutes": m.get("minutes"),
        })
    os.makedirs(RES, exist_ok=True)
    if rows:
        with open(f"{RES}/parsimony_curve{csfx}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---- the curve, on validation -------------------------------------
    import statistics as st
    print("\n" + "=" * 78)
    print(f"PARSIMONY   validation split, mean over seeds   "
          f"[{a.ranking} ordering]")
    print("=" * 78)
    print(f"{'k':>4}", end="")
    for arm in arms:
        print(f"{arm[:16]:>22}", end="")
    print(f"{'gain vs k-1':>14}")
    print("-" * 78)

    prev = None
    for k in sorted(set(grid)):
        print(f"{k:>4}", end="")
        cur = None
        for arm in arms:
            v = [r["val_acute_auroc"] for r in rows
                 if r["arm"] == arm and r["k"] == k]
            if not v:
                print(f"{'-':>22}", end="")
                continue
            s = st.stdev(v) if len(v) > 1 else 0.0
            print(f"{st.mean(v):>15.4f}+-{s:.3f}", end="")
            if arm == arms[0]:
                cur = st.mean(v)
        if cur is not None and prev is not None:
            print(f"{cur - prev:>+14.4f}", end="")
        print()
        if cur is not None:
            prev = cur

    print("\n  The last column is what the elbow argument rests on: the gain")
    print("  from one grid point to the next. Where it stops exceeding the")
    print("  seed-to-seed spread, more variables are no longer buying")
    print("  anything the data can distinguish.")
    if rows:
        print(f"\nwritten to {RES}/parsimony_curve{csfx}.csv")
    else:
        print(f"\nNOTHING WRITTEN -- no point produced a metrics.json.")
    if failed:
        print(f"\n{len(failed)} of {len(todo)} points failed:")
        print(f"  {failed}")
    if wrong_data:
        print(f"\n{wrong_data} directories were left out of the curve because "
              f"they belong to a different data build.")
    print(f"total {(time.time() - t0) / 60:.0f} min")

    # The exit code is what a person reads first the next morning, and until
    # now it was 0 whether the sweep produced a curve or produced nothing.
    # Anything past a couple of stray points means the run should not be
    # trusted, and saying so here is cheaper than discovering it later.
    if not rows:
        print("\nExiting non-zero: the sweep produced no usable points.")
        return 1
    if failed and len(failed) > 0.5 * len(todo):
        print(f"\nExiting non-zero: {len(failed)} of {len(todo)} points "
              f"failed, so the curve is too sparse to read.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
