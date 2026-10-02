"""
23_raw_audit.py -- read the raw master_dataset CSVs on the Citrix side.

    python 23_raw_audit.py --umn      Q:\\...\\Uploads\\master_dataset.csv
                           --stanford Q:\\...\\Uploads\\stanford_master_dataset.csv
                           --mimic    Q:\\...\\MIMIC_data\\master_dataset_with_outcomes.csv

Nothing here needs the cluster, a GPU, or torch. Pandas and the CSVs.

Two questions, both raised by the co-occurrence counts of 2026-08-17.

1. Is the Stanford label structure an artefact of the extraction, or is it
   in the source?

   The prepared arrays say that at Stanford no critical-illness encounter is
   also flagged as hospitalised (0 of 2,749), and that the seven acute
   endpoints never co-occur (1 of 21 pairs, 5 encounters). At UMN and MIMIC
   critical illness is a subset of hospitalisation (99-100%) and all 21 pairs
   overlap. Every marginal matches cohort_audit.csv, so the arrays are not
   corrupt -- the joint structure is simply different.

   That pattern is what a single categorical field looks like after one-hot
   encoding. This file checks the raw CSV, which is upstream of every step
   that could have introduced it. If the raw file shows the same thing, the
   difference is in the Stanford data as delivered and the paper must say so.
   If the raw file shows overlap, something between the CSV and the .npy is
   collapsing it, and that is a bug worth finding before submission.

2. What are the pre-exclusion counts?

   Figure 2's three "total attendances" boxes are still the numbers from the
   manuscript, carried over unverified. 00_prepare_data.py writes them to
   data/cohort_ladder.csv on the cluster, but the same criteria can be
   applied here: age > 18, at least one triage vital recorded, and all nine
   outcome labels present, counted at each step.

   The check is the last row: it must equal the cohort size in
   cohort_audit.csv. If it does not, this file is reading a different
   extraction from the one the results come from and the ladder above it
   means nothing.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

OUTCOMES = [
    "outcome_hospitalization", "outcome_critical", "outcome_sepsis",
    "outcome_pneumonia_viral", "outcome_ards", "outcome_pe",
    "outcome_copd_asthma", "outcome_acs_mi", "outcome_aki",
]
RENAME = {"outcome_viral_pne": "outcome_pneumonia_viral"}
SHORT = ["hosp", "crit", "seps", "vpne", "ards", "pemb", "copd", "acsm",
         "akin"]
VITALS = ["triage_heartrate", "triage_resprate", "triage_sbp", "triage_dbp",
          "triage_o2sat", "triage_temperature"]
MIN_AGE = 18
# CONFIG["split"] in 00_prepare_data.py: temporal on anchor_year, train
# 2019-2023 and test 2024-2025. Encounters outside that window never enter
# either split, and leaving the step out is why the first UMN ladder stopped
# at 2,351,303 against the 1,664,277 in cohort_audit.csv.
YEAR_COL, YEAR_LO, YEAR_HI = "anchor_year", 2019, 2025


def check(s):
    t = 0
    for i, ch in enumerate(s):
        if ch.isdigit():
            t += (i + 1) * int(ch)
    return f"{t % 97:02d}"


def audit(name, path):
    print()
    print("=" * 70)
    print(f"{name}   {path}")
    print("=" * 70)
    if not os.path.exists(path):
        print("  not found, skipped")
        return

    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns=RENAME)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    have = [o for o in OUTCOMES if o in df.columns]
    miss = [o for o in OUTCOMES if o not in df.columns]
    if miss:
        print(f"  outcome columns absent: {', '.join(miss)}")
    if not have:
        print("  no outcome columns at all; nothing further to do here")
        print(f"  columns seen: {list(df.columns)[:40]}")
        return

    # ---- inclusion ladder -------------------------------------------------
    print()
    print("  inclusion ladder")
    rows = [("as read from file", len(df))]
    d = df
    if "age" in d.columns:
        d = d[d["age"] > MIN_AGE]
        rows.append((f"age > {MIN_AGE}", len(d)))
    v = [c for c in VITALS if c in d.columns]
    if v:
        d = d[d[v].notna().any(axis=1)]
        rows.append(("at least one triage vital recorded", len(d)))
    if len(have) == len(OUTCOMES):
        d = d[d[have].notna().all(axis=1)]
        rows.append(("all nine outcome labels present", len(d)))
    if YEAR_COL in d.columns:
        y = pd.to_numeric(d[YEAR_COL], errors="coerce")
        d = d[(y >= YEAR_LO) & (y <= YEAR_HI)]
        rows.append((f"{YEAR_COL} in {YEAR_LO}-{YEAR_HI}", len(d)))
    else:
        rows.append((f"({YEAR_COL} not in this file, step skipped)", len(d)))
    prev = None
    for lab, n in rows:
        drop = "" if prev is None else f"   -{prev - n:,}"
        line = f"    {lab:<38s} {n:>10,}{drop}"
        print(f"{line}  {check(line)}")
        prev = n

    # ---- joint structure of the labels -----------------------------------
    if len(have) != len(OUTCOMES):
        return
    Y = (d[OUTCOMES].to_numpy() > 0.5).astype(np.int64)
    C = Y.T @ Y
    print()
    print("  co-occurrence in the RAW file, after inclusion")
    print("       " + " ".join(f"{s:>7s}" for s in SHORT))
    for i, s in enumerate(SHORT):
        line = f"  {s:>4s} " + " ".join(f"{C[i, j]:7d}" for j in range(9))
        print(f"{line}  {check(line)}")

    crit, hosp = C[1][1], C[0][0]
    print()
    print(f"  critical also flagged hospitalised: {C[1][0]:,} of {crit:,}"
          f"  ({100.0 * C[1][0] / max(crit, 1):.1f}%)")
    pairs = [(i, j) for i in range(2, 9) for j in range(i + 1, 9)]
    nz = sum(1 for i, j in pairs if C[i][j] > 0)
    tot = sum(C[i][j] for i, j in pairs)
    print(f"  acute pairs with any overlap: {nz} of 21, {tot:,} encounters")
    exp = hosp * crit / max(len(Y), 1)
    print(f"  crit and hosp if independent: {exp:,.0f}; observed {C[1][0]:,}")
    print()
    print("  rows carrying more than one acute label: "
          f"{int((Y[:, 2:].sum(axis=1) > 1).sum()):,} of {len(Y):,}")


def icd_recheck(path, etable):
    """Re-derive the acute labels from the raw ICD columns.

    The outcome columns in Uploads/master_dataset.csv show almost no overlap
    among the seven acute endpoints, and the raw file it came from carries
    1.43 diagnosis codes per encounter -- enough for some overlap. Either the
    derivation dropped labels, or Stanford's code lists really are too short
    to produce any. Applying eTable 2's prefixes here settles it: if this
    re-derivation finds far more multi-label encounters than the prepared
    arrays do, the derivation is losing them.

    Prefix matching, dots stripped, exactly as eTable 2 lists them. This is
    not meant to reproduce the official labels to the encounter -- it is a
    magnitude check on how many encounters COULD carry two.
    """
    print()
    print("=" * 70)
    print(f"ICD re-derivation from the raw codes   {path}")
    print("=" * 70)
    if not os.path.exists(path):
        print("  not found, skipped")
        return
    if not os.path.exists(etable):
        print(f"  {etable} not found; cannot read the code lists")
        return

    import csv as _csv
    codes = {}
    for r in _csv.DictReader(open(etable)):
        tgt = r["Prediction target"].strip()
        pre = []
        for col in ("ICD-9-CM codes", "ICD-10-CM codes"):
            v = (r.get(col) or "").strip()
            if not v or "field" in v.lower():
                continue
            pre += [x.strip().replace(".", "").upper()
                    for x in v.split(",") if x.strip()]
        if pre:
            codes[tgt] = pre
    if not codes:
        print("  no usable code lists in the eTable")
        return
    print(f"  {len(codes)} targets with code lists: {', '.join(codes)}")

    df = pd.read_csv(path, low_memory=False)
    cols = [c for c in ("Dx_ICD9", "Dx_ICD10", "diagnosis") if c in df.columns]
    cols = [c for c in cols if c != "diagnosis"]      # text, not codes
    if not cols:
        print("  no Dx_ICD9 / Dx_ICD10 column")
        return
    joined = df[cols].fillna("").astype(str).agg(",".join, axis=1)
    joined = joined.str.upper().str.replace(".", "", regex=False)

    hits = {}
    for tgt, pre in codes.items():
        pat = "|".join(f"(?:^|[,; ]){p}" for p in pre)
        hits[tgt] = joined.str.contains(pat, regex=True, na=False)
    H = pd.DataFrame(hits)
    per = H.sum(axis=1)
    print()
    print("  encounters matching each target, by prefix")
    for tgt in codes:
        print(f"    {tgt:<28s} {int(H[tgt].sum()):>8,}")
    print()
    print(f"  encounters matching 0 targets: {int((per == 0).sum()):,}")
    print(f"  encounters matching 1 target : {int((per == 1).sum()):,}")
    print(f"  encounters matching 2 or more: {int((per >= 2).sum()):,}")
    print()
    print("  The prepared Stanford arrays carry 5 encounters with more than")
    print("  one acute label. If the number above is of that order the")
    print("  derivation is faithful and Stanford simply has short code")
    print("  lists. If it is hundreds, the derivation is dropping labels.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--umn", default="")
    ap.add_argument("--mimic", default="")
    ap.add_argument("--stanford", default="")
    ap.add_argument("--icd-recheck", default="",
                    help="raw CSV with Dx_ICD9 / Dx_ICD10 columns")
    ap.add_argument("--etable", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "transcribed", "appendix", "eTable2_outcomes_icd.csv"))
    a = ap.parse_args()
    if not any([a.umn, a.mimic, a.stanford, a.icd_recheck]):
        ap.error("give at least one of --umn, --mimic, --stanford, "
                 "--icd-recheck")
    for nm, p in (("UMN", a.umn), ("MIMIC", a.mimic),
                  ("STANFORD", a.stanford)):
        if p:
            audit(nm, p)
    if a.icd_recheck:
        icd_recheck(a.icd_recheck, a.etable)
    print()
    print("The last ladder row must equal that cohort's n in "
          "cohort_audit.csv:")
    print("  UMN 1,664,277 (train 1,313,435 + test 350,842)   "
          "MIMIC 422,114   Stanford 117,046")
    return 0


if __name__ == "__main__":
    sys.exit(main())
