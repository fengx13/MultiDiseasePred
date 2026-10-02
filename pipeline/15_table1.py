"""
15_table1.py -- Table 1, cohort characteristics, from the pre-imputation audit.

    python 00_prepare_data.py --audit          # writes results/cohort_audit.json
    python 15_table1.py --features greedy:12 --csv

Writes results/table1_<featuretag>.csv and prints the same thing formatted.
Columns are the four cohorts; rows are the reported variables, then the nine
endpoints, then the missingness block.

Why the audit is a separate step
--------------------------------
Table 1 must describe the patients, not the preprocessing. The saved arrays
have already had missing values replaced by the development-set median, so a
median computed from umn_test_X.npy returns the training median by
construction, and a missingness rate cannot be recovered from them at all.
00_prepare_data.py --audit re-reads the sources and stops one step short of
fillna. It writes no array, so data_id cannot move.

What to look at
---------------
The missingness block is not padding. A variable that is 2% missing at UMN and
40% missing at BIDMC has been imputed to the UMN median for two out of five
BIDMC patients, and any transported performance on that cohort is partly a
measurement of the imputation. That is the first thing to check when external
discrimination drops, and it is checked here rather than guessed at.

Continuous variables are reported as median [Q1, Q3] and binary ones as n (%),
with the distinction made from the observed values rather than a hand-kept
list.
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
RES = f"{PROJ}/results"

ORDER = ["umn_train", "umn_test", "bidmc", "stanford"]
LABEL = {"umn_train": "UMN development 2019-2023",
         "umn_test": "UMN test 2024-2025",
         "bidmc": "MIMIC-IV-ED (BIDMC)",
         "stanford": "Stanford"}


def fmt(e):
    if e is None:
        return ""
    if e.get("kind") == "binary":
        return f"{e['n_positive']:,} ({e['pct_positive']:.1f}%)"
    if e.get("kind") == "continuous":
        return f"{e['median']:.1f} [{e['q1']:.1f}, {e['q3']:.1f}]"
    return "not available"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="greedy:12",
                    help="which variables to report. 'all' reports every "
                         "feature in the matrix.")
    ap.add_argument("--drop", default="triage_acuity")
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()

    path = f"{RES}/cohort_audit.json"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found.\n"
                         f"Run:  python 00_prepare_data.py --audit\n"
                         f"It writes no array and cannot change data_id.")
    audit = json.load(open(path))
    meta = json.load(open(f"{DATA}/preprocessing.json"))
    cols = meta["features"]
    outcomes = meta["outcomes"]

    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)
    if a.features == "all":
        keep = [c for c in cols if c not in _t.expand_drop(drop, cols)]
    else:
        keep = [cols[i] for i in _t.resolve_features(a.features, cols, drop)]

    have = [c for c in ORDER if c in audit["cohorts"]]
    if not have:
        raise SystemExit("the audit has no cohorts in it")

    # Table 1 says "this is the population we analysed". It is worth almost
    # nothing if it describes a different population, and there is no way to
    # tell by looking: 445,744 and 422,114 are both plausible MIMIC cohorts.
    #
    # That is what happened on 2026-08-12. 00_prepare_data.py takes --patch
    # with a default path, so the chief-complaint correction was applied
    # automatically, but --patch-exclude is opt-in and submit_table1.ps1 did
    # not pass it. The arrays every result was computed from had the 23,800
    # under-ascertained-sepsis encounters removed; the audit did not. Table 1
    # reported 47.94% hospitalisation against the 45.32% the model actually
    # saw, and nothing in either file disagreed with the other.
    #
    # The arrays are the ground truth here -- they are what was scored.
    bad = []
    for name in have:
        f = f"{DATA}/{name}_y.npy" if name != "umn_train" else None
        if f is None or not os.path.exists(f):
            continue
        n_arr = len(np.load(f, mmap_mode="r"))
        n_aud = audit["cohorts"][name].get("n")
        if n_aud is not None and n_arr != n_aud:
            bad.append((name, n_aud, n_arr))
    if bad:
        print()
        print("=" * 72)
        print("THE AUDIT AND THE ARRAYS DESCRIBE DIFFERENT COHORTS")
        print("=" * 72)
        for name, n_aud, n_arr in bad:
            print(f"  {name:<10} audit {n_aud:>9,}   arrays {n_arr:>9,}"
                  f"   difference {n_aud - n_arr:>+8,}")
        print()
        print("  Every metric in the paper was computed from the arrays.")
        print("  Rerun the audit the same way the arrays were built, e.g.")
        print("    python 00_prepare_data.py --audit --patch-exclude")
        raise SystemExit(1)

    rows = []

    def add(name, getter):
        r = {"row": name}
        for c in have:
            r[c] = getter(c)
        rows.append(r)

    add("n encounters", lambda c: f"{audit['cohorts'][c]['n']:,}")

    rows.append({"row": "-- variables (median [Q1, Q3] or n (%)) --"})
    for f in keep:
        add(f, lambda c, f=f: fmt(audit["cohorts"][c]["features"].get(f)))

    rows.append({"row": "-- endpoints, events (prevalence) --"})
    for o in outcomes:
        def g(c, o=o):
            e = audit["cohorts"][c]["outcomes"].get(o)
            if e is None:
                return ""
            return f"{e['events']:,} ({e['prevalence_pct']:.2f}%)"
        add(o.replace("outcome_", ""), g)

    rows.append({"row": "-- missing before imputation (%) --"})
    for f in keep:
        def g(c, f=f):
            e = audit["cohorts"][c]["features"].get(f)
            return "" if e is None else f"{e['missing_pct']:.2f}"
        add(f, g)

    w = max(len(r["row"]) for r in rows) + 2
    print()
    print(" " * w + "".join(f"{LABEL[c]:>30}" for c in have))
    print("-" * (w + 30 * len(have)))
    for r in rows:
        print(f"{r['row']:<{w}}" + "".join(f"{r.get(c, ''):>30}" for c in have))
    print()

    # The one thing worth saying out loud rather than leaving in a table: a
    # variable imputed for a large share of an external cohort makes that
    # cohort's result partly a measurement of the imputation.
    bad = []
    for f in keep:
        for c in have:
            if c.startswith("umn"):
                continue
            e = audit["cohorts"][c]["features"].get(f)
            if e and e.get("missing_pct", 0) >= 20:
                bad.append((c, f, e["missing_pct"]))
    if bad:
        print("HEAVILY IMPUTED IN AN EXTERNAL COHORT (>= 20% missing):")
        for c, f, p in sorted(bad, key=lambda x: -x[2]):
            print(f"  {c:<10} {f:<32} {p:6.2f}% imputed to the UMN median")
        print("  Transported performance on these cohorts is partly a")
        print("  measurement of that imputation, and the paper should say so.")
        print()

    if a.csv:
        tag = a.features.replace(":", "")
        out = f"{RES}/table1_{tag}.csv"
        with open(out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["row"] + have)
            wr.writeheader()
            wr.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
