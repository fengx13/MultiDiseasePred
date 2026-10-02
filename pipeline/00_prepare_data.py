"""
00_prepare_data.py -- master CSV to model-ready arrays, once.

    bash ${MDP_ROOT}/run.sh \
         python ${MDP_ROOT}/code/00_prepare_data.py

Why this exists
---------------
Parsing a 1.7 GB CSV takes 30-60 s. The rerun trains ~140 models, so parsing
once and saving arrays turns half an hour of pure waste into a second per run.

It also does the split. The old code created train.csv and test.csv by a
script that is not in the repository, which is why the temporal split could
not be verified for months. Here the split rule lives in CONFIG, is applied in
front of you, and every count is printed and written to a CSV, so the cohort
derivation ladder for Figure 2 comes out of the same run that makes the data.

What it prints
--------------
A count after every filter. That matters right now because the numbers do not
yet reconcile:

    master_dataset.csv                   3,159,376 rows
    Table 1 and masterdata_2019_2025      1,679,022
    Results text, Methods, abstract       1,666,474      difference 12,548

Whichever filter takes 3.16M to 1.68M, and whatever further step reaches
1.67M, will be visible in the ladder. Do not adjust anything to match a number
you cannot reproduce; report what the ladder says.

Outputs, all under $PROJ/data
-----------------------------
    umn_train_X.npy   (N, 63) float32, imputed, NOT scaled
    umn_train_y.npy   (N,  9) float32
    umn_test_X.npy  umn_test_y.npy
    bidmc_X.npy  bidmc_y.npy  stanford_X.npy  stanford_y.npy
    preprocessing.json    feature order, medians, winsor bounds, scaler
    cohort_ladder.csv     every count, every cohort

Arrays are stored before scaling so they stay in clinical units and can be
eyeballed. The scaler is fitted on the development set only and saved; the
training script applies it. Changing the scaler later costs nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from determinism import seed_everything, run_fingerprint  # noqa: E402

PROJ = "/scratch/ahcie-gpu2/XieF-Req04048"
OUT = os.path.join(PROJ, "p2_pipeline", "data")

# Candidate locations, tried in order. The first version of this named one
# path inside final/, which belongs to another user; on 2026-08-05 the file
# was no longer there and the rebuild stopped dead. A list costs nothing and
# the script says which one it used, so the answer to "which file produced
# these numbers" is in the log rather than in someone's memory.
#
# Q:\XieF-Req04048\Workspace\UMN_DATASET\proc_data also holds a copy, but Q:
# is the Windows share and is not visible from the cluster. It has to be
# copied to /scratch first.
CONFIG = {
    "cohorts": {
        "umn": [
            f"{PROJ}/p2_pipeline/data/master_dataset.csv",
            f"{PROJ}/p2_pipeline/data/master_dataset.parquet",
            f"{PROJ}/final/data/proc_data/master_dataset.csv",
            f"{PROJ}/final/data/proc_data/master_dataset.parquet",
        ],
        # external paths are on the Windows share as Code_MTL/...; if they are
        # not visible here the script skips them and says so.
        "bidmc": [f"{PROJ}/final/data/mimic/master_dataset_mimic.csv"],
        "stanford": [f"{PROJ}/final/data/stanford/master_dataset.csv"],
    },
    "split": {"mode": "temporal", "column": "anchor_year",
              "train": [2019, 2023], "test": [2024, 2025]},
    "min_age": 18,          # cell 51 of the manuscript: age > 18
    "seed": 42,
}

# The raw file carries outcome_viral_pne; cell 2 of the training notebook
# renames it. Doing the same here keeps one name downstream.
RENAME = {"outcome_viral_pne": "outcome_pneumonia_viral"}

OUTCOMES = [
    "outcome_hospitalization", "outcome_critical", "outcome_sepsis",
    "outcome_pneumonia_viral", "outcome_ards", "outcome_pe",
    "outcome_copd_asthma", "outcome_acs_mi", "outcome_aki",
]

# The 63 candidates, verbatim from cell 8.
FEATURES = [
    "age", "gender", "n_ed_30d", "n_ed_90d", "n_ed_365d",
    "n_hosp_30d", "n_hosp_90d", "n_hosp_365d",
    "n_icu_30d", "n_icu_90d", "n_icu_365d",
    "triage_temperature", "triage_heartrate", "triage_resprate",
    "triage_o2sat", "triage_sbp", "triage_dbp", "triage_acuity",
    "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
    "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough",
    "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope",
    "chiefcom_dizziness",
    "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia",
    "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1",
    "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2",
    "cci_Cancer2", "cci_HIV",
    "eci_Arrhythmia", "eci_Valvular", "eci_PHTN", "eci_HTN1", "eci_HTN2",
    "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy",
    "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
    "eci_Anemia", "eci_Alcohol", "eci_Drugs", "eci_Psychoses",
    "eci_Depression",
]

# Physiologic limits for winsorisation. eMethods says values were winsorised
# to "the nearest plausible physiologic limit" but never states the limits, so
# these are set here explicitly and written into preprocessing.json.
# CONFIRM WITH THE TEAM before the final run: if the original used different
# bounds, a handful of extreme encounters will differ.
WINSOR = {
    "age":                (18, 120),
    "triage_temperature": (30.0, 43.0),      # Celsius
    "triage_heartrate":   (20, 250),
    "triage_resprate":    (4, 60),
    "triage_o2sat":       (50, 100),
    "triage_sbp":         (40, 260),
    "triage_dbp":         (20, 180),
    "triage_acuity":      (1, 5),
}

# Comorbidity and chief-complaint indicators are structurally absent rather
# than missing: no matching code in the prior 365 days means 0, and eMethods
# says explicitly that this is a coding definition, not imputation.
PATCH_PATH = ""
PATCH_EXCLUDE = False

STRUCTURAL_ZERO = [c for c in FEATURES
                   if c.startswith(("cci_", "eci_", "chiefcom_", "n_ed_",
                                    "n_hosp_", "n_icu_"))]


def log(msg=""):
    print(msg, flush=True)


def load(paths, name):
    """Read only the columns we need. 142 columns times 3.2M rows is a lot of
    memory to spend on fields nobody uses.

    `paths` is a list of candidates. The one that is used is logged with its
    size and modification time, because two copies of master_dataset exist
    two months apart and which of them produced a given result is not
    something to reconstruct later from memory.
    """
    if isinstance(paths, str):
        paths = [paths]
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        log(f"  {name}: NOT FOUND. Looked in:")
        for p in paths:
            log(f"      {p}")
        return None
    st = os.stat(path)
    log(f"  {name}: using {path}")
    log(f"      {st.st_size / 1e9:.2f} GB, modified "
        f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")
    for p in paths:
        if p != path and os.path.exists(p):
            s2 = os.stat(p)
            log(f"      NOTE another copy exists and was not used: {p}")
            log(f"           {s2.st_size / 1e9:.2f} GB, modified "
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(s2.st_mtime))}")

    t0 = time.time()
    parquet = path.endswith(".parquet")
    # stay_id is not a feature and not an outcome, so it used to be dropped
    # here at read time. That is why the MIMIC chief-complaint patch of
    # 2026-08-11 silently did nothing: the corrected values are keyed by
    # stay_id, the column was present in the file on disk, and usecols threw
    # it away before anything could join on it. The job then ran to
    # completion, wrote the arrays, and reported success.
    want = set(FEATURES) | set(OUTCOMES) | {"subject_id", "stay_id"}
    want |= set(RENAME.keys())
    want.add(CONFIG["split"]["column"])
    if parquet:
        try:
            import pyarrow.parquet as pq
            have = set(pq.ParquetFile(path).schema.names)
        except ImportError:
            log("      pyarrow is absent, reading the whole parquet")
            have = None
        usecols = [c for c in have if c in want] if have else None
        df = pd.read_parquet(path, columns=usecols)
    else:
        head = pd.read_csv(path, nrows=0)
        usecols = [c for c in head.columns if c in want]
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.rename(columns=RENAME)
    log(f"  {name}: {len(df):,} rows x {len(df.columns)} cols "
        f"({time.time() - t0:.0f} s)")
    missing_f = [c for c in FEATURES if c not in df.columns]
    missing_o = [c for c in OUTCOMES if c not in df.columns]
    if missing_f:
        log(f"     features absent, will be created as NaN: {missing_f}")
    if missing_o:
        log(f"     OUTCOMES ABSENT: {missing_o}")
    return df


def cohort_ladder(df, name, rows):
    """Apply the inclusion criteria, printing a count at every step."""
    def step(label, d):
        rows.append({"cohort": name, "step": label, "n": len(d)})
        log(f"     {label:<42} {len(d):>12,}")
        return d

    log(f"  {name} derivation")
    d = step("as read from file", df)

    if "age" in d.columns:
        d = step(f"age > {CONFIG['min_age']}",
                 d[d["age"] > CONFIG["min_age"]])

    # "empty records": no arrival time and nothing beyond the identifier.
    # arrival time is not among the columns we load, so the operational test
    # is that every triage vital is missing.
    vitals = [c for c in ["triage_heartrate", "triage_resprate", "triage_sbp",
                          "triage_dbp", "triage_o2sat", "triage_temperature"]
              if c in d.columns]
    if vitals:
        d = step("not empty (>=1 triage vital recorded)",
                 d[d[vitals].notna().any(axis=1)])

    lab = [c for c in OUTCOMES if c in d.columns]
    if lab:
        d = step("all nine outcome labels present", d[d[lab].notna().all(axis=1)])

    return d


def prepare_features(d):
    """Winsorise, map sex, coerce numeric, and put the 63 columns in order.
    Imputation happens later so the medians can come from the training set."""
    for c in FEATURES:
        if c not in d.columns:
            d[c] = np.nan

    if "gender" in d.columns:
        g = d["gender"].astype(str).str.strip().str.upper()
        # the enclave files use M/F; be tolerant of MALE/FEMALE too
        d["gender"] = g.map({"M": 0, "F": 1, "MALE": 0, "FEMALE": 1})

    # Temperature arrives in Fahrenheit. The bounds below are Celsius, so
    # every value was above the upper limit and clip() flattened the whole
    # column to 43.0. A constant column survives standardisation as all
    # zeros -- load_split substitutes 1 for a zero scale -- so the model
    # simply never saw a temperature, and the attribution came out at
    # exactly 0.000, twenty-eighth of twenty-eight variables. Nothing
    # errored at any point.
    #
    # Detect rather than assume: 45 sits between any plausible Celsius
    # reading and any plausible Fahrenheit one, so the median decides.
    if "triage_temperature" in d.columns:
        t = pd.to_numeric(d["triage_temperature"], errors="coerce")
        med = float(t.median()) if t.notna().any() else float("nan")
        if med == med and med > 45:
            log(f"     triage_temperature median {med:.1f}, reading as "
                f"Fahrenheit and converting to Celsius")
            d["triage_temperature"] = (t - 32.0) * 5.0 / 9.0

    for c, (lo, hi) in WINSOR.items():
        if c in d.columns:
            v = pd.to_numeric(d[c], errors="coerce")
            n = int(v.notna().sum())
            clipped = int(((v < lo) | (v > hi)).sum())
            # Winsorisation is meant to pull in a handful of implausible
            # entries. When it moves a large share of the column, the bounds
            # are wrong for the units the data is in, and clipping silently
            # destroys the variable instead of cleaning it.
            if n and clipped / n > 0.05:
                log(f"     WARNING  {c}: winsorising to [{lo}, {hi}] moves "
                    f"{clipped / n:.1%} of {n:,} values. Check the units.")
            d[c] = v.clip(lo, hi)

    X = d[FEATURES].apply(pd.to_numeric, errors="coerce")
    return X


def apply_bidmc_patch(d, name, patch_path, do_exclude):
    """Correct the MIMIC chief-complaint columns from a pasted patch.

    The enclave holds mimic_all.csv.gz, extracted 2026-04-05. Its
    chiefcom_shortness_of_breath rule matched "SOB" and "shortness of breath"
    literally and missed "dyspnea", which appears in 4.9% of MIMIC chief
    complaints, so the flag stands at 0.29% against 7.1% at UMN and 8.7% at
    Stanford. It is the second most valuable variable in the greedy ordering.
    A cohort where it is almost never set is a cohort where the model has been
    denied its second most important input, and the four endpoints that lean on
    it -- PE, ARDS, COPD, pneumonia -- are exactly the four where MIMIC trails
    Stanford by 0.08 to 0.22 AUROC while sepsis and AKI are level.

    The corrected extraction exists but only outside the enclave and files
    cannot be transferred in, so the corrected VALUES travel as a pasted list
    of stay_ids. Values rather than a rule: several candidate regexes were
    tested against the corrected file and the closest still left 141 false
    positives and 47 false negatives, so shipping a reconstructed rule would
    replace one error with a smaller one.

    Nothing here touches UMN. data_id is a hash of the UMN matrices and this
    function is only ever called on an external cohort.
    """
    if not patch_path or not os.path.exists(patch_path):
        log(f"  no patch at {patch_path} -- {name} left as extracted")
        return d
    import base64 as _b64
    import zlib as _zlib
    P = json.load(open(patch_path))

    def ids(b64):
        raw = _zlib.decompress(_b64.b64decode(b64))
        return np.cumsum(np.frombuffer(raw, dtype=np.int32).astype(np.int64))

    key = "stay_id"
    if key not in d.columns:
        raise SystemExit(
            f"{name} has no {key} column, so the patch cannot be joined.\n"
            f"Columns present: {sorted(d.columns)[:12]} ...\n"
            f"\n"
            f"This used to be a log line and a quiet return, which is how the\n"
            f"first attempt produced a job that finished cleanly, rewrote the\n"
            f"arrays and changed nothing. A patch that cannot be applied has\n"
            f"to stop the run: unpatched output that looks patched is worse\n"
            f"than no output.")
    sid = pd.to_numeric(d[key], errors="coerce")
    log(f"  applying {patch_path}")
    for val, block in ((1, "set1"), (0, "set0")):
        for c, b64 in P.get(block, {}).items():
            if c not in d.columns:
                log(f"     {c} not in {name}, skipped")
                continue
            want = ids(b64)
            m = sid.isin(want).to_numpy()
            # Cast first. These columns arrive as bool in some extractions, and
            # assigning an int into a bool column is deprecated in pandas 2 and
            # an error later; the warning is easy to miss in a job log.
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype("int8")
            before = d[c].mean()
            d.loc[m, c] = val
            after = d[c].mean()
            log(f"     {c:<32} set {val} on {int(m.sum()):>7,} rows "
                f"({len(want):,} in patch)   {100*before:6.2f}% -> {100*after:6.2f}%")
    if do_exclude and P.get("exclude"):
        want = ids(P["exclude"])
        m = sid.isin(want).to_numpy()
        log(f"     excluding {int(m.sum()):,} encounters that the corrected "
            f"extraction dropped")
        log("       (their sepsis rate is 0.7% against 2.7% in the rest, which "
            "is\n        under-ascertainment rather than absence of sepsis)")
        d = d.loc[~m].copy()
    return d


def external_only():
    """Build the external arrays and touch nothing else.

    The medians and the scaler are properties of the development split, and
    they are already recorded in preprocessing.json. Rebuilding them means
    re-reading three million UMN rows to arrive at numbers that are on disk,
    and it means rewriting umn_train_X.npy -- which any training job is
    reading at that moment, and which defines data_id. A partial write there
    invalidates every finished run in the project.

    So this path reads the two external files, applies the transform that was
    already fitted, and writes only bidmc_* and stanford_*. It can be run
    while jobs are training, and data_id cannot change because nothing that
    feeds it is opened for writing.
    """
    meta_path = f"{OUT}/preprocessing.json"
    if not os.path.exists(meta_path):
        raise SystemExit(f"{meta_path} not found -- run the full pass first")
    meta = json.load(open(meta_path))
    med = pd.Series(meta["median"])
    log("=" * 68)
    log("EXTERNAL COHORTS ONLY")
    log("=" * 68)
    log(f"  reusing the transform from {meta_path}")
    log(f"  data_id {meta.get('data_id')}  (unchanged by definition: this")
    log("           path does not open the UMN arrays for writing)")
    if sorted(med.index) != sorted(FEATURES):
        raise SystemExit("the saved medians do not cover the current FEATURES; "
                         "the full pass has to be rerun")

    ladder, wrote = [], []
    for name in ("bidmc", "stanford"):
        log()
        d = load(CONFIG["cohorts"][name], name)
        if d is None:
            log(f"  {name} skipped; point CONFIG at the right path and rerun")
            continue
        if name == "bidmc":
            d = apply_bidmc_patch(d, name, PATCH_PATH, PATCH_EXCLUDE)
        d = cohort_ladder(d, name, ladder)
        Xe = prepare_features(d.copy()).fillna(med)
        np.save(f"{OUT}/{name}_X.npy", Xe.to_numpy(dtype=np.float32))
        np.save(f"{OUT}/{name}_y.npy", d[OUTCOMES].to_numpy(dtype=np.float32))
        wrote.append(name)
        log(f"     {name}_X.npy  {Xe.shape}")
        log(f"     prevalence")
        for j, o in enumerate(OUTCOMES):
            v = pd.to_numeric(d[o], errors="coerce").fillna(0)
            log(f"       {o:<28}{100 * v.mean():6.3f} %   "
                f"{int(v.sum()):>8,} events")

    if not wrote:
        raise SystemExit("\nNeither external cohort was found. Nothing written.")
    pd.DataFrame(ladder).to_csv(f"{OUT}/cohort_ladder_external.csv", index=False)
    log()
    log("=" * 68)
    log(f"WROTE {', '.join(wrote)}")
    log("=" * 68)
    log("  The UMN arrays and preprocessing.json were not modified, so every")
    log("  finished run stays valid and no retraining is triggered.")
    return 0


def audit():
    """Everything Table 1 needs, computed BEFORE imputation.

    The saved arrays cannot answer this. 00_prepare_data.py fills missing
    values with the development-set median and writes the result, so a median
    taken from umn_test_X.npy is a median of real values mixed with copies of
    the training median -- which is exactly the training median, by
    construction, and says nothing about the cohort. A missingness rate cannot
    be recovered from an imputed array at all.

    So this re-reads the sources, applies the same cohort ladder and the same
    feature construction, and stops one step short of fillna. It writes
    results/cohort_audit.json and touches nothing else: no array is opened for
    writing, so data_id cannot move and training jobs can run through it.

    Binary variables are detected rather than declared -- a column whose
    observed values are a subset of {0, 1} is reported as n (%), everything
    else as median [Q1, Q3]. The comorbidity flags and chief complaints fall
    out as binary without a hand-maintained list that could drift.
    """
    meta_path = f"{OUT}/preprocessing.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    sc = CONFIG["split"]["column"]
    tr_lo, tr_hi = CONFIG["split"]["train"]
    te_lo, te_hi = CONFIG["split"]["test"]

    def describe(d, name):
        X = prepare_features(d.copy())          # deliberately no fillna
        n = len(X)
        out = {"n": int(n), "features": {}, "outcomes": {}}
        for c in FEATURES:
            v = pd.to_numeric(X[c], errors="coerce")
            obs = v.dropna()
            e = {"n_observed": int(len(obs)),
                 "missing_pct": round(100.0 * (1.0 - len(obs) / n), 4)
                 if n else float("nan")}
            if len(obs) == 0:
                e["kind"] = "empty"
            elif set(np.unique(obs.to_numpy())) <= {0.0, 1.0}:
                e["kind"] = "binary"
                e["n_positive"] = int(obs.sum())
                e["pct_positive"] = round(100.0 * float(obs.mean()), 4)
            else:
                e["kind"] = "continuous"
                q1, q2, q3 = np.percentile(obs.to_numpy(), [25, 50, 75])
                e["median"] = round(float(q2), 4)
                e["q1"] = round(float(q1), 4)
                e["q3"] = round(float(q3), 4)
                e["mean"] = round(float(obs.mean()), 4)
                e["sd"] = round(float(obs.std()), 4)
            out["features"][c] = e
        for o in OUTCOMES:
            v = pd.to_numeric(d[o], errors="coerce").fillna(0)
            out["outcomes"][o] = {"events": int(v.sum()),
                                  "prevalence_pct": round(100.0 * float(v.mean()), 4)}
        log(f"  {name:<12} {n:>10,} encounters")
        return out

    log("=" * 68)
    log("COHORT AUDIT -- pre-imputation, for Table 1")
    log("=" * 68)
    res = {"data_id": meta.get("data_id"), "cohorts": {}}
    ladder = []

    umn = load(CONFIG["cohorts"]["umn"], "umn")
    if umn is not None:
        umn = cohort_ladder(umn, "umn", ladder)
        yr = pd.to_numeric(umn[sc], errors="coerce")
        res["cohorts"]["umn_train"] = describe(
            umn[(yr >= tr_lo) & (yr <= tr_hi)], "umn_train")
        res["cohorts"]["umn_test"] = describe(
            umn[(yr >= te_lo) & (yr <= te_hi)], "umn_test")
    for name in ("bidmc", "stanford"):
        d = load(CONFIG["cohorts"][name], name)
        if d is None:
            log(f"  {name} not found, skipped")
            continue
        # The patch has to be applied here as well, not only in
        # external_only(). Table 1 is what made the shortness-of-breath
        # problem visible in the first place; a Table 1 built from the
        # unpatched source while the arrays are patched would report 0.3%
        # for a cohort the model now sees at 5.6%, and the disagreement
        # would look like a new bug rather than a stale table.
        if name == "bidmc":
            d = apply_bidmc_patch(d, name, PATCH_PATH, PATCH_EXCLUDE)
        res["cohorts"][name] = describe(cohort_ladder(d, name, ladder), name)

    res_dir = os.path.join(os.path.dirname(OUT), "results")
    os.makedirs(res_dir, exist_ok=True)
    path = os.path.join(res_dir, "cohort_audit.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    log()
    log(f"wrote {path}")
    log("data_id untouched: no array was opened for writing.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external-only", action="store_true",
                    help="build bidmc_*.npy and stanford_*.npy from the "
                         "transform already in preprocessing.json, leaving "
                         "the UMN arrays and data_id untouched")
    ap.add_argument("--patch", default=f"{OUT}/bidmc_patch.json",
                    help="corrected MIMIC chief-complaint values, written by "
                         "bidmc_patch.py. Applied only to bidmc, only in "
                         "--external-only mode. Absent file means no patch.")
    ap.add_argument("--patch-exclude", action="store_true",
                    help="also drop the 23,800 encounters the corrected "
                         "extraction removed. Their sepsis labels are "
                         "under-ascertained, but this changes the reported "
                         "MIMIC cohort size, so it is opt-in")
    ap.add_argument("--audit", action="store_true",
                    help="write results/cohort_audit.json with pre-imputation "
                         "distributions and missingness for every cohort, for "
                         "Table 1. Writes no array, so data_id cannot change")
    a = ap.parse_args()

    global PATCH_PATH, PATCH_EXCLUDE
    PATCH_PATH, PATCH_EXCLUDE = a.patch, a.patch_exclude

    seed_everything(CONFIG["seed"])
    os.makedirs(OUT, exist_ok=True)
    if a.audit:
        return audit()
    if a.external_only:
        return external_only()
    ladder = []

    log("=" * 68)
    log("READING")
    log("=" * 68)
    umn = load(CONFIG["cohorts"]["umn"], "umn")
    if umn is None:
        raise SystemExit("the UMN master dataset is required")

    log()
    log("=" * 68)
    log("COHORT DERIVATION")
    log("=" * 68)
    umn = cohort_ladder(umn, "umn", ladder)

    # ---- temporal split -------------------------------------------------
    sc = CONFIG["split"]["column"]
    if sc not in umn.columns:
        raise SystemExit(f"{sc} is missing; the temporal split cannot be made")
    yr = pd.to_numeric(umn[sc], errors="coerce")
    tr_lo, tr_hi = CONFIG["split"]["train"]
    te_lo, te_hi = CONFIG["split"]["test"]
    train = umn[(yr >= tr_lo) & (yr <= tr_hi)]
    test = umn[(yr >= te_lo) & (yr <= te_hi)]

    log()
    log(f"     development  {tr_lo}-{tr_hi}                    {len(train):>12,}")
    log(f"     internal validation {te_lo}-{te_hi}             {len(test):>12,}")
    log(f"     outside both windows                       "
        f"{len(umn) - len(train) - len(test):>12,}")
    ladder += [{"cohort": "umn", "step": f"development {tr_lo}-{tr_hi}",
                "n": len(train)},
               {"cohort": "umn", "step": f"internal validation {te_lo}-{te_hi}",
                "n": len(test)}]

    log()
    log("     year distribution")
    for y, n in yr.value_counts().sort_index().items():
        log(f"       {int(y)}  {n:>12,}")

    # ---- features, medians from development only ------------------------
    log()
    log("=" * 68)
    log("PREPROCESSING")
    log("=" * 68)
    Xtr = prepare_features(train.copy())
    Xte = prepare_features(test.copy())

    med = Xtr.median()
    for c in STRUCTURAL_ZERO:
        med[c] = 0.0                       # absence of a code means absent
    if "triage_acuity" in med:
        # eMethods says mode for ESI. The old code used the median for every
        # column; both are recorded so the text can state whichever is used.
        mode_acuity = float(Xtr["triage_acuity"].mode().iloc[0])
        log(f"  triage_acuity   median {med['triage_acuity']:.1f}  "
            f"mode {mode_acuity:.1f}   (using median, as the old code did)")

    miss = Xtr.isna().mean().sort_values(ascending=False)
    log("  missingness in the development set, worst ten:")
    for c, m in miss.head(10).items():
        log(f"     {c:<28} {100 * m:6.2f} %")

    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr.to_numpy(dtype=np.float64))

    ytr = train[OUTCOMES].to_numpy(dtype=np.float32)
    yte = test[OUTCOMES].to_numpy(dtype=np.float32)

    Xtr_a = Xtr.to_numpy(dtype=np.float32)
    Xte_a = Xte.to_numpy(dtype=np.float32)
    np.save(f"{OUT}/umn_train_X.npy", Xtr_a)
    np.save(f"{OUT}/umn_train_y.npy", ytr)
    np.save(f"{OUT}/umn_test_X.npy", Xte_a)
    np.save(f"{OUT}/umn_test_y.npy", yte)

    # An identifier for this particular matrix. 01_train.py records it and
    # refuses to skip a run that was built on a different one. Without it,
    # rebuilding the data leaves three hundred run directories that look
    # finished, and the next sweep prints "[skip] already done" and reports
    # numbers from the previous version of the data. Nothing errors, and the
    # results are wrong in a way no one would catch. The temperature fix is
    # exactly such a rebuild.
    import hashlib as _h
    _d = _h.sha256()
    for _a in (Xtr_a, ytr, Xte_a, yte):
        _d.update(np.ascontiguousarray(_a).tobytes())
    DATA_ID = _d.hexdigest()[:12]
    log(f"  data_id {DATA_ID}")

    log()
    log("  prevalence (development / internal validation)")
    for j, o in enumerate(OUTCOMES):
        log(f"     {o:<28} {100 * ytr[:, j].mean():6.3f} %   "
            f"{100 * yte[:, j].mean():6.3f} %")

    # ---- external cohorts, same transform -------------------------------
    for name in ("bidmc", "stanford"):
        log()
        d = load(CONFIG["cohorts"][name], name)
        if d is None:
            log(f"  {name} skipped; point CONFIG at the right path and rerun")
            continue
        if name == "bidmc":
            d = apply_bidmc_patch(d, name, PATCH_PATH, PATCH_EXCLUDE)
        d = cohort_ladder(d, name, ladder)
        Xe = prepare_features(d.copy()).fillna(med)
        np.save(f"{OUT}/{name}_X.npy", Xe.to_numpy(dtype=np.float32))
        np.save(f"{OUT}/{name}_y.npy", d[OUTCOMES].to_numpy(dtype=np.float32))

    # ---- what we did ----------------------------------------------------
    meta = {
        "data_id": DATA_ID,
        "config": CONFIG,
        "features": FEATURES,
        "outcomes": OUTCOMES,
        "rename": RENAME,
        "winsor": WINSOR,
        "structural_zero": STRUCTURAL_ZERO,
        "median": {k: float(v) for k, v in med.items()},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "note": ("arrays are saved before scaling; apply "
                 "(X - scaler_mean) / scaler_scale at load time"),
        "fingerprint": run_fingerprint(),
    }
    with open(f"{OUT}/preprocessing.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    pd.DataFrame(ladder).to_csv(f"{OUT}/cohort_ladder.csv", index=False)

    log()
    log("=" * 68)
    log("WRITTEN")
    log("=" * 68)
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        log(f"  {os.path.getsize(p) / 1048576:8.1f} MB  {f}")

    log()
    log("Check the ladder against the manuscript before going further:")
    log("  Table 1 and masterdata_2019_2025 say 1,679,022")
    log("  Results, Methods and abstract say   1,666,474")
    log("  If neither falls out of the ladder, the discrepancy is upstream of")
    log("  this file and the manuscript number has to be traced, not matched.")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
