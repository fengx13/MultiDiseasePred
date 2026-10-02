"""20_subgroup.py -- does the model work as well for everyone?

    python 20_subgroup.py --arm G2_no_l1_no_drop --features greedy:12
    python 20_subgroup.py --datasets umn_test,bidmc,stanford

AUROC within strata of age, sex and triage acuity, for each endpoint and for
the acute macro. Reads the saved logits and the unscaled feature matrix; no
model is loaded and nothing is retrained.

Why acuity is the interesting one
---------------------------------
triage_acuity is dropped from every feature set in this paper -- the whole
point of --drop triage_acuity is that a model which reads the nurse's
acuity score is partly reading the nurse. So the model has never seen it,
which makes it a clean stratifier: the strata are defined by information the
model did not have. If discrimination collapses in ESI 4-5 it is because
those patients are hard, not because the model was told they were easy.

Age and sex are the two a reviewer will ask for by name.

What a reader should look for
-----------------------------
Not whether the AUROCs differ -- they will, because prevalence and case mix
differ and AUROC is not comparable across populations with different spectra.
What matters is whether any stratum is near 0.5, which would mean the model
is guessing for that group while reporting a good overall number, and whether
the gap between strata is larger for our arm than for the baselines, which
would mean the multi-task structure is the thing creating it.

The confidence intervals are the point. A stratum with forty events has an
interval half a unit wide, and reporting its point estimate beside a stratum
with four thousand invites a comparison the data cannot support. Intervals
are DeLong, same as 10_ci.py.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
DATA = f"{PROJ}/data"
RUNS = f"{PROJ}/runs"
RES = f"{PROJ}/results"

TASKS = ["hospitalization", "critical", "sepsis", "pneumonia_viral", "ards",
         "pe", "copd_asthma", "acs_mi", "aki"]
ACUTE = list(range(2, 9))
EP_CODE = [("hospitalization", "01"), ("critical", "02"), ("sepsis", "03"),
           ("pneumonia_viral", "04"), ("ards", "05"), ("pe", "06"),
           ("copd_asthma", "07"), ("acs_mi", "08"), ("aki", "09"),
           ("MACRO acute", "MA")]

# Two characters, and deliberately not numbers: a stratum code that looked
# like an endpoint code would be one screenshot away from a table where the
# rows mean something else.
STRATA = [
    ("AA", "age 18-39"), ("AB", "age 40-64"), ("AC", "age 65-79"),
    ("AD", "age 80+"),
    ("SM", "male"), ("SF", "female"),
    ("E1", "ESI 1-2"), ("E3", "ESI 3"), ("E4", "ESI 4-5"),
]


def check(s):
    t = 0
    for i, ch in enumerate(s):
        if ch.isdigit():
            t += (i + 1) * int(ch)
    return f"{t % 97:02d}"


def q(v):
    if v is None or (isinstance(v, float) and v != v):
        return "----"
    return f"{int(round(min(max(float(v), 0.0), 0.9999) * 10000)):04d}"


def auroc_delong(y, s):
    """AUROC with a DeLong standard error. Returns (auc, se) or (nan, nan).

    Same estimator as 10_ci.py so a subgroup number and a whole-cohort number
    can sit in the same sentence. Refuses rather than guesses when a stratum
    has no events or no non-events: an AUROC needs both.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    sr = s[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    auc = (ranks[y == 1].sum() - m * (m + 1) / 2) / (m * n)
    # structural components
    v10 = np.array([(np.sum(neg < p) + 0.5 * np.sum(neg == p)) / n
                    for p in pos])
    v01 = np.array([(np.sum(pos > q_) + 0.5 * np.sum(pos == q_)) / m
                    for q_ in neg])
    var = v10.var(ddof=1) / m + v01.var(ddof=1) / n if m > 1 and n > 1 \
        else float("nan")
    return float(auc), float(np.sqrt(var)) if var == var else float("nan")


def strata_masks(X, cols):
    """Boolean masks, or None for a stratifier this cohort does not carry."""
    out = {}
    def col(name):
        return X[:, cols.index(name)] if name in cols else None

    age = col("age")
    if age is not None:
        out["AA"] = (age >= 18) & (age < 40)
        out["AB"] = (age >= 40) & (age < 65)
        out["AC"] = (age >= 65) & (age < 80)
        out["AD"] = age >= 80
    g = col("gender")
    if g is not None:
        # 00_prepare_data.py encodes gender as 0/1 after upper-casing; which
        # way round is recorded in preprocessing.json, and guessing it would
        # silently swap the two rows.
        out["SM"] = g == 0
        out["SF"] = g == 1
    esi = col("triage_acuity")
    if esi is not None:
        out["E1"] = esi <= 2
        out["E3"] = esi == 3
        out["E4"] = esi >= 4
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="G2_no_l1_no_drop")
    ap.add_argument("--features", default="greedy:12")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--datasets", default="umn_test,bidmc,stanford")
    a = ap.parse_args()

    tag = a.features.replace(":", "")
    seeds = [s.strip() for s in a.seeds.split(",") if s.strip()]
    meta = json.load(open(f"{DATA}/preprocessing.json"))
    cols = meta["features"]
    gmap = meta.get("gender_encoding")
    print(f"arm {a.arm}   features {a.features}   seeds {','.join(seeds)}")
    if gmap:
        print(f"gender encoding from preprocessing.json: {gmap}")
    else:
        print("preprocessing.json carries no gender_encoding; the male and")
        print("female rows may be the wrong way round. Check before using.")

    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        fx = f"{DATA}/{'umn_test' if ds == 'umn_test' else ds}_X.npy"
        fy = f"{DATA}/{'umn_test' if ds == 'umn_test' else ds}_y.npy"
        if not os.path.exists(fx):
            print(f"\n{ds}: no feature matrix, skipped")
            continue
        X, Y = np.load(fx), np.load(fy)

        S = []
        for s in seeds:
            p = f"{RUNS}/{a.arm}_{tag}_noesi_seed{s}/logits_{ds}.npy"
            if os.path.exists(p):
                S.append(np.load(p))
        if not S:
            print(f"\n{ds}: no logits, skipped")
            continue
        if any(len(x) != len(X) for x in S):
            print(f"\n{ds}: {len(X)} rows in the matrix but "
                  f"{[len(x) for x in S]} in the logits -- NOT the same "
                  f"cohort, skipped")
            continue
        s_mean = np.mean(S, axis=0)

        M = strata_masks(X, cols)
        print(f"\n=== {ds}   n = {len(X):,}   {len(S)} seeds averaged ===")
        rows, body = [], []
        for code, label in STRATA:
            if code not in M:
                continue
            m = M[code]
            print(f"  {label:<12} n = {int(m.sum()):>8,}")
            for name, ec in EP_CODE:
                if name == "MACRO acute":
                    aucs = [rows[-(9 - i)][2] for i in ACUTE]
                    a_ = float(np.nanmean([x for x in aucs]))
                    se = float("nan")
                else:
                    t = TASKS.index(name)
                    a_, se = auroc_delong(Y[m, t], s_mean[m, t])
                lo = a_ - 1.959964 * se if se == se else float("nan")
                hi = a_ + 1.959964 * se if se == se else float("nan")
                ev = int(Y[m, TASKS.index(name)].sum()) if name != "MACRO acute" \
                    else int(Y[m][:, ACUTE].sum())
                rows.append((code, ec, a_, lo, hi, ev))
                line = f"{code} {ec} {q(a_)} {q(lo)} {q(hi)} {ev:>7d}"
                body.append(line + "  " + check(line))

        out = [f"subgroup {a.arm} {a.features}   dataset {ds}   n {len(X)}",
               "",
               "strata: AA 18-39  AB 40-64  AC 65-79  AD 80+   SM male  "
               "SF female",
               "        E1 ESI 1-2  E3 ESI 3  E4 ESI 4-5",
               "endpoints: 01 hospitalization 02 critical 03 sepsis "
               "04 pneumonia_viral",
               "           05 ards 06 pe 07 copd_asthma 08 acs_mi 09 aki "
               "MA macro acute",
               "",
               "AUROC as four digits, then its interval, then the event count "
               "in that stratum.",
               "MA has no interval: it is a mean of seven correlated AUROCs "
               "and DeLong does",
               "not cover that. Dashes there are honest, not missing.",
               ""]
        head = f"{'':7}{'AUROC':^6}{'LO':^6}{'HI':^6}{'EVENTS':>8}  CK"
        out += [head, "-" * len(head)] + body
        out += ["-" * len(head),
                f"LINES {len(body)}   BLOCK {check(''.join(body))}"]
        p = f"{RES}/emit_subgroup_{a.arm}_{tag}_{ds}.txt"
        with open(p, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"  wrote {p}  ({len(out)} lines)")

        thin = [(c, e, v) for c, e, v, *_ in
                [(r[0], r[1], r[5]) for r in rows] if v < 25]
        if thin:
            print(f"  {len(thin)} stratum-endpoint cells have fewer than 25 "
                  f"events; their intervals are too wide to compare.")

    print()
    print("Photograph each file whole; they are one screen each.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
