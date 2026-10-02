"""
22_cohort_extras.py -- the two cohort-level tables that were never exported.

    python 22_cohort_extras.py

Prints two blocks and nothing else. Both are reductions over the arrays
00_prepare_data.py already wrote, so this needs no GPU and no model.

Block A: endpoint co-occurrence, 9 x 9 counts, per cohort
--------------------------------------------------------
Figure 3 is drawn by code/chord_plot.R from code/<site>_chord_data.csv. Those
files predate the outcome-definition change: they give UMN sepsis as 13,446
where cohort_audit.csv gives 42,805 and UMN AKI as 16,531 against 74,041.
Those are factors of three and four, not cohort drift, so the correlation
structure in the published figure may not be the current one.

Only the joint counts are missing. The diagonal is the marginal and is
already known, which is the check: every diagonal printed here must equal the
corresponding column of cohort_audit.csv. If one does not, this script is
reading a different cohort and the off-diagonals should not be used.

Block B: prevalence of the chief complaints and comorbidities
-------------------------------------------------------------
eTable 4 still carries the pre-correction denominators -- 1,666,474 /
448,804 / 118,385 against the current 1,664,277 / 422,114 / 117,046 -- so
every percentage in it is against the wrong base. These columns are binary
and unscaled in the saved matrices, so the prevalence is the column mean.

What is NOT here
----------------
Figure 2's three "total attendances" boxes and the excluded counts. Those are
not missing: 00_prepare_data.py writes them to data/cohort_ladder.csv at
every inclusion step, starting from "as read from file". Print that file
rather than recomputing it here:

    column -s, -t < data/cohort_ladder.csv
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
DATA = f"{PROJ}/data"

def check(s):
    """The two-digit position-weighted check every emit uses.

    Defined here rather than imported from 18_shap.py. That import pulled in
    torch, which this file has no other use for and which the Citrix MTL
    environment -- built for LightGBM -- may not have. A five-line pure
    function is not worth a hard dependency on a deep learning framework.

    The copy is checked against the original whenever the original can be
    imported, so the two cannot drift apart unnoticed.
    """
    t = 0
    for i, ch in enumerate(s):
        if ch.isdigit():
            t += (i + 1) * int(ch)
    return f"{t % 97:02d}"


def _check_agrees():
    try:
        _s = importlib.import_module("18_shap")
    except Exception:
        return          # torch absent; running on the Citrix side
    probe = "umn_train  1313435  267978  12171"
    if _s.check(probe) != check(probe):
        raise SystemExit(
            "the local check() no longer agrees with 18_shap.check(). "
            "One of them was edited; the transcription checks in this "
            "file's output would not be comparable with the others.")


SITES = ["umn_train", "umn_test", "bidmc", "stanford"]
SHORT = {"hospitalization": "hosp", "critical": "crit", "sepsis": "seps",
         "pneumonia_viral": "vpne", "ards": "ards", "pe": "pemb",
         "copd_asthma": "copd", "acs_mi": "acsm", "aki": "akin"}


def load_y(data, site, n_out):
    """The label array for one site, or None if it is not in this directory.

    None rather than an exception: the arrays get to the Citrix side a few at
    a time and there is no reason a directory holding two cohorts should
    produce nothing. Whatever is present is reported and the rest is named.
    """
    for cand in (f"{data}/{site}_y.npy", f"{data}/{site}_Y.npy"):
        if os.path.exists(cand):
            y = np.load(cand)
            if y.ndim != 2 or y.shape[1] != n_out:
                raise SystemExit(
                    f"{cand} has shape {y.shape}, expected (N, {n_out})")
            return (y > 0.5).astype(np.int64)
    return None


def main() -> int:
    _check_agrees()
    # --data so this can run against a local copy of the arrays on the Citrix
    # side, the way 13_lightgbm_local.py does. Nothing here needs a GPU or a
    # model, so there is no reason to make it queue for one.
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()
    data = a.data
    if not os.path.isdir(data):
        raise SystemExit(f"--data {data} is not a directory")

    meta = json.load(open(f"{data}/preprocessing.json"))
    cols = meta["features"]
    outcomes = [o.replace("outcome_", "") for o in meta["outcomes"]]
    T = len(outcomes)
    print(f"data_id {meta.get('data_id')}   {T} outcomes   "
          f"{len(cols)} features")
    print(f"data     {data}")

    print()
    print("BLOCK A -- endpoint co-occurrence counts")
    print("diagonal is the marginal and must match cohort_audit.csv")
    seen, skipped = [], []
    for site in SITES:
        y = load_y(data, site, T)
        if y is None:
            skipped.append(site)
            continue
        seen.append(site)
        C = y.T @ y
        print()
        print(f"{site}  n={len(y)}")
        print("      " + " ".join(f"{SHORT[o]:>7s}" for o in outcomes))
        for i, o in enumerate(outcomes):
            line = f"{SHORT[o]:>4s} " + " ".join(f"{C[i, j]:7d}"
                                                 for j in range(T))
            print(f"{line}  {check(line)}")

    # Prevalence. Binary columns only: a mean is a prevalence for those and
    # is meaningless for age or a heart rate, and printing both in one table
    # is how a reader ends up quoting "mean age 0.49".
    keep = [i for i, c in enumerate(cols)
            if c.startswith(("chiefcom_", "cci_", "eci_"))]
    print()
    if skipped:
        print()
        print("no label array for " + ", ".join(skipped) + " in this "
              "directory, so those cohorts are missing from Block A")
    if not seen:
        raise SystemExit("no label array for any cohort in " + data)

    # BLOCK A needs only the label arrays, which are a tenth the size of the
    # feature matrices. If the big pull failed halfway there is no reason to
    # throw away the block that did arrive -- Figure 3 is the more urgent of
    # the two.
    absent = [s for s in SITES
              if not os.path.exists(f"{data}/{s}_X.npy")]
    if absent:
        print()
        print()
        print("BLOCK B skipped: no feature matrix for " + ", ".join(absent))
        print("Block A above is complete and is the one Figure 3 needs.")
        return 0

    print()
    print("BLOCK B -- prevalence, per cent of encounters")
    print(f"{len(keep)} binary variables; denominators are the current cohort")
    Xs, ns = {}, {}
    for site in SITES:
        X = np.load(f"{data}/{site}_X.npy")
        v = X[:, keep]
        u = np.unique(v[: min(len(v), 5000)])
        if not np.all(np.isin(u, (0.0, 1.0))):
            raise SystemExit(
                f"{site}: columns selected as binary contain {u[:6]}. "
                f"The matrix is scaled, or the prefix filter caught a "
                f"continuous variable; either way the means are not "
                f"prevalences.")
        Xs[site] = v.mean(axis=0) * 100.0
        ns[site] = len(X)
    print("n      " + " ".join(f"{ns[s]:>10d}" for s in SITES))
    print(f"{'variable':<34s}" + " ".join(f"{s:>10s}" for s in SITES))
    for k, i in enumerate(keep):
        line = f"{cols[i]:<34s}" + " ".join(f"{Xs[s][k]:10.4f}"
                                            for s in SITES)
        print(f"{line}  {check(line)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
