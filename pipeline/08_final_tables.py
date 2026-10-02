"""
08_final_tables.py -- per-endpoint results, model against baselines.

Every run already stores per-endpoint AUROC and AUPRC. 01_train.py writes
`per_target` and `per_target_val` into metrics.json for all nine endpoints and
both splits; nothing has ever read them back. This script does, and lays them
out the way a results table needs them: endpoints down the rows, models across
the columns, seeds averaged with their spread shown.

No GPU, no retraining, no network. It reads files that are already on disk.

    python 08_final_tables.py --features rank:13 --drop triage_acuity
    python 08_final_tables.py --features all63   --drop triage_acuity
    python 08_final_tables.py --features rank:13 --drop triage_acuity --csv

On reading the comparison
-------------------------
The seed spread is printed next to every mean because it is the only thing
that makes a difference interpretable. Across nine endpoints and three seeds,
a handful of differences smaller than the spread will fall either way purely
by chance. A column of nine positive differences, each smaller than its own
standard deviation, is not a model that wins everywhere -- it is a coin landing
heads nine times, and it will not do so again on an external cohort.

Differences are therefore marked against the pooled seed spread rather than
against zero:

    +   the difference exceeds one pooled standard deviation
    .   within the spread; the data does not separate the two models
    -   below by more than one pooled standard deviation

A model that is marked + on most endpoints, . on some and - on one is an
ordinary, reportable result.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib                                              # noqa: E402
_t = importlib.import_module("01_train")                      # noqa: E402

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
DATA = f"{PROJ}/data"
RUNS = f"{PROJ}/runs"
RES = f"{PROJ}/results"

ACUTE = list(range(2, 9))
SEVERITY = [0, 1]


def mean_sd(v):
    if not v:
        return float("nan"), float("nan")
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def load_runs(arm, features, drop, seeds, data_id):
    """Every seed of one arm at one variable set, as
    {split: {endpoint_index: [value per seed]}}. Runs built on other data are
    skipped and counted, never silently mixed in."""
    tag_feat = features.replace(":", "")
    out = {"val": {}, "test": {}}
    found, wrong = [], 0
    for s in seeds:
        tag = f"{arm}_{tag_feat}{_t.drop_suffix(drop)}_seed{s}"
        p = f"{RUNS}/{tag}/metrics.json"
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        if data_id is not None and m.get("data_id") != data_id:
            wrong += 1
            continue
        found.append(s)
        for split, key in (("val", "per_target_val"), ("test", "per_target")):
            for row in m.get(key, []):
                t = row["target"]
                out[split].setdefault(t, {"auroc": [], "auprc": []})
                out[split][t]["auroc"].append(row["auroc"])
                out[split][t]["auprc"].append(row["auprc"])
    return out, found, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="rank:13",
                    help="all63, top15, rank:N or fullrank:N")
    ap.add_argument("--drop", default="")
    ap.add_argument("--arms", default="G2_no_l1_no_drop,A2_xgb_sub",
                    help="comma separated; the first is treated as ours and "
                         "the rest as baselines")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--metric", default="auroc", choices=["auroc", "auprc"])
    ap.add_argument("--csv", action="store_true",
                    help="also write results/final_table_*.csv")
    a = ap.parse_args()

    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",")]

    meta = json.load(open(f"{DATA}/preprocessing.json"))
    names = [o.replace("outcome_", "") for o in meta["outcomes"]]
    data_id = meta.get("data_id")

    print(f"variables  {a.features}"
          + (f"   excluding {', '.join(drop)}" if drop else ""))
    print(f"data_id    {data_id}")
    print(f"metric     {a.metric.upper()}")

    # which variables, spelled out -- the table is unreadable without them
    try:
        cols = meta["features"]
        idx = _t.resolve_features(a.features, cols, drop)
        chosen = [cols[i] for i in idx]
        print(f"\n{len(chosen)} variables:")
        for i in range(0, len(chosen), 3):
            print("  " + "".join(f"{c:<30}" for c in chosen[i:i + 3]).rstrip())
    except SystemExit as e:
        print(f"\ncould not resolve the variable list: {e}")
        chosen = []

    data, meta_runs = {}, {}
    for arm in arms:
        d, found, wrong = load_runs(arm, a.features, drop, seeds, data_id)
        data[arm] = d
        meta_runs[arm] = (found, wrong)
        note = f"seeds {found}" if found else "NO RUNS FOUND"
        if wrong:
            note += f"   ({wrong} on other data, skipped)"
        print(f"\n  {arm:<22} {note}")

    if not data[arms[0]]["test"]:
        raise SystemExit("\nNothing to tabulate for the first arm.")

    ours = arms[0]
    for split in ("val", "test"):
        label = ("validation split, 2019-2023 held out" if split == "val"
                 else "internal validation, 2024-2025")
        print()
        print("=" * 92)
        print(f"{a.metric.upper()} BY ENDPOINT   [{label}]")
        print("=" * 92)
        head = f"{'endpoint':<20}"
        for arm in arms:
            head += f"{arm[:16]:>20}"
        head += f"{'difference':>16}"
        print(head)
        print("-" * 92)

        rows_out = []
        for t, nm in enumerate(names):
            line = f"{nm:<20}"
            vals = {}
            for arm in arms:
                v = data[arm].get(split, {}).get(t, {}).get(a.metric, [])
                m, sd = mean_sd(v)
                vals[arm] = (m, sd)
                line += (f"{m:>13.4f}+-{sd:.3f}" if v else f"{'-':>20}")
            mo, so = vals[ours]
            if len(arms) > 1:
                mb, sb = vals[arms[1]]
                if mo == mo and mb == mb:
                    diff = mo - mb
                    pooled = math.sqrt(so ** 2 + sb ** 2) or 1e-9
                    mark = "+" if diff > pooled else ("-" if diff < -pooled else ".")
                    line += f"{diff:>+14.4f} {mark}"
                    rows_out.append((nm, mo, so, mb, sb, diff, mark))
                else:
                    line += f"{'-':>16}"
            grp = "  severity" if t in SEVERITY else "  acute"
            print(line + grp)

        if rows_out:
            print("-" * 92)
            for grp, ix in (("acute macro", ACUTE), ("severity macro", SEVERITY)):
                mo = [r[1] for i, r in enumerate(rows_out) if i in ix]
                mb = [r[3] for i, r in enumerate(rows_out) if i in ix]
                if mo and mb:
                    om, bm = sum(mo) / len(mo), sum(mb) / len(mb)
                    print(f"{grp:<20}{om:>20.4f}{bm:>20.4f}{om - bm:>+14.4f}")
            n_up = sum(1 for r in rows_out if r[6] == "+")
            n_flat = sum(1 for r in rows_out if r[6] == ".")
            n_dn = sum(1 for r in rows_out if r[6] == "-")
            print()
            print(f"  {n_up} endpoints above the pooled seed spread, "
                  f"{n_flat} within it, {n_dn} below.")
            if n_flat >= 4:
                print("  Most differences are inside the spread: on this data the")
                print("  two models are not distinguishable endpoint by endpoint.")

        if a.csv and rows_out:
            sfx = _t.drop_suffix(drop)
            fn = (f"{RES}/final_table_{a.features.replace(':','')}"
                  f"{sfx}_{split}_{a.metric}.csv")
            with open(fn, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["endpoint", f"{ours}_mean", f"{ours}_sd",
                            f"{arms[1]}_mean", f"{arms[1]}_sd",
                            "difference", "mark"])
                w.writerows(rows_out)
            print(f"\n  written to {fn}")

    print()
    print("  + above one pooled seed standard deviation")
    print("  . inside it -- the data does not separate the two models")
    print("  - below by more than one pooled standard deviation")
    print()
    print("  A clean sweep of nine small positive differences, each inside its")
    print("  own spread, is not evidence of a better model. It is what chance")
    print("  produces, and an external cohort will not reproduce it.")


if __name__ == "__main__":
    main()
