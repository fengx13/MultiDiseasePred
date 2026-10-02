"""
17_emit.py -- the metrics table in a form that survives being photographed.

    python 17_emit.py --features greedy:12 --drop triage_acuity --rule youden

Writes results/emit_<tag>_<rule>_<dataset>.txt, one file per dataset, each
about eighty lines, which is two screens.

The problem
-----------
metrics_greedy12_noesi.csv is 223 KB and there is no export route from the
enclave. The only way out is to photograph a terminal and type the numbers
back in on the other side. At 616 rows and eighteen numbers a row that is
around eleven thousand digits, and the failure mode is not that transcription
is hard -- it is that a wrong digit produces a number that looks exactly like
a right one. An AUROC of 0.8036 instead of 0.8836 is not implausible enough
for anyone to notice, and it would go into the paper.

So the format has three properties, in order of importance.

It is checkable. Every line ends in a two-digit check over its own digits,
position-weighted so that swapping two adjacent digits changes it. Each block
ends in a check over all of its lines, so a line lost entirely between the
screen and the file is caught as well -- which the per-line checks cannot do.
mod 97 rather than mod 10 because a single check digit misses one error in
ten and that is not good enough for eleven thousand digits.

It is dense. Leading "0." is dropped and everything is four digits: 0.90001
travels as 9000. A whole dataset is 83 lines, so two screenshots, and 12
variables at Youden is eight screenshots rather than the sixty it would take
to photograph the CSV.

It is aligned. Fixed width with no delimiters to be lost at a line wrap.

What is not here
----------------
Only the columns the main table needs: AUROC, AUPRC, sensitivity,
specificity, PPV, NPV, each with its interval, plus the threshold. Counts,
calibration, Brier, ECE and alert rate stay in the CSV on the cluster; they
belong in a supplement and can be emitted separately when they are needed.
There is no point carrying a number by hand until something depends on it.

Reading it back
---------------
tools/read_emit.py on the other side. Do not type these into a spreadsheet
directly -- the checks are the entire reason the format exists, and they only
run if the numbers go through the reader.
"""

from __future__ import annotations

import argparse
import csv
import os

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
RES = f"{PROJ}/results"

DATASETS = ["umn_val", "umn_test", "bidmc", "stanford"]

# Three letters each, chosen so no two share a prefix and none is a
# substring of another: a smudged character should not turn one arm into a
# different valid arm.
ARM_CODE = {
    "G2_no_l1_no_drop": "OUR",
    "A2_xgb_sub": "XGB",
    "A4_lgbm": "LGB",
    "A2_rf": "RFO",
    "A2_lr": "LRE",
    "A2_histgb": "HGB",
    "A5_news": "NEW",
}

# Two characters. Numbered rather than named because "pneumonia_viral" and
# "pe" both begin with p and a name is eleven characters that have to be read
# correctly for no benefit -- the order is fixed and printed in the header.
EP_CODE = [
    ("hospitalization", "01"), ("critical", "02"), ("sepsis", "03"),
    ("pneumonia_viral", "04"), ("ards", "05"), ("pe", "06"),
    ("copd_asthma", "07"), ("acs_mi", "08"), ("aki", "09"),
    ("MACRO acute", "MA"), ("MACRO disposition", "MD"),
]

METRICS = ["auroc", "auprc", "sens", "spec", "ppv", "npv"]

# The calibration block. Not part of METRICS because these do not live in
# [0, 1] and cannot go through q(): a calibration slope of 1.34 would clamp
# to 0.9999 and a negative intercept would lose its sign entirely, which is
# the one thing about an intercept that matters.
CAL_CI = ["cal_slope", "cal_intercept"]
CAL_FLAT = ["brier", "ece", "alert_rate", "frac_at_prob_bound"]


def check(s):
    """Two-digit position-weighted check over the digits of s.

    Weighted by position so that a transposition changes the result: an
    unweighted sum is blind to 8836 -> 8863, which is exactly the error a
    tired reader makes. mod 97 leaves a one-in-ninety-seven chance that a
    corrupted line still checks out, against one in ten for a single digit.
    """
    t = 0
    for i, ch in enumerate(s):
        if ch.isdigit():
            t += (i + 1) * int(ch)
    return f"{t % 97:02d}"


def check_signed(s):
    """check() with the signs folded in.

    check() walks characters and keeps only those where isdigit() is true, so
    '+' and '-' contribute nothing. In the main block that is a small hole:
    the only signed field is the threshold, and its sign is pinned by the arm
    -- the neural score is a logit and every tree arm is a probability, so a
    flipped sign is visible to anyone who looks at the column. It is also
    caught by the fact that a Youden threshold frozen on validation must be
    identical in all four datasets.

    Here it would be a real hole. A calibration intercept of -1.27 and one of
    +1.27 are opposite clinical claims -- over-prediction against
    under-prediction -- and both are entirely plausible numbers, so nothing
    downstream would notice. The signs are mapped onto digits before the
    walk. The mapping is one character to one character, so every position is
    unchanged and the weighting still catches transpositions.
    """
    return check(s.replace("-", "7").replace("+", "3"))


def sci(v, width, dec):
    """A signed fixed-point field, or dots if it is not there.

    14_metrics.py writes -1 for every calibration column when the fit was
    refused -- a score pinned at the probability bound gives a mass point
    that produces a plausible-looking slope from nothing. That -1 is a real
    statement and is printed as -1, not blanked: a reader who sees dots
    assumes the column was never computed, and a reader who sees -1.0000
    across all six fields can look up why.
    """
    if v is None or v == "":
        return f"{'.' * dec:>{width}}"
    try:
        f = float(v)
    except ValueError:
        return f"{'.' * dec:>{width}}"
    if f != f:                                   # NaN, as in q()
        return f"{'.' * dec:>{width}}"
    return f"{f:+{width}.{dec}f}"


def q(v):
    """A value in [0, 1] as four digits, or ---- if it is not there.

    Clamped rather than allowed to print five digits: an interval on a
    sensitivity near 1.0 can have an upper bound above 1.0, which is an
    artefact of a normal approximation and not a sensitivity that exceeds
    one. Letting it widen the field would shift every column on the line.
    """
    if v is None or v == "":
        return "----"
    try:
        f = float(v)
    except ValueError:
        return "----"
    # NaN is not caught by the except: float("nan") is a perfectly legal
    # float and only fails four lines later inside int(). It arrives here
    # for real reasons -- NEWS is an integer 0-20 score and cannot reach 90%
    # sensitivity on some endpoints, and PPV is 0/0 when nothing is flagged.
    # That is a result, not a missing value, so it prints as ---- and
    # read_emit turns it back into None rather than into a zero.
    if f != f:
        return "----"
    f = min(max(f, 0.0), 0.9999)
    return f"{int(round(f * 10000)):04d}"


def thr(v):
    """The operating threshold, in whatever units the score is in.

    Not quantised like the rest, and given six decimals rather than four.
    The scores are on three different scales -- logits around +-10 for the
    neural arm, probabilities for the tree arms, integer points for NEWS --
    and a probability threshold for ARDS sits near 0.0004, where four
    decimals would round away most of the number. This is also the one value
    a reader might actually deploy, so it is the wrong place to save eight
    characters.
    """
    if v is None or v == "":
        return f"{'.':>11}"
    f = float(v)
    # 14_metrics.py writes nan when no threshold on the validation set
    # reached the target sensitivity at all. There is no operating point to
    # report, and printing "+nan" would put three non-digits into a field the
    # check reads as digits.
    if f != f:
        return f"{'.':>11}"
    return f"{f:+11.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="greedy:12")
    ap.add_argument("--drop", default="triage_acuity")
    ap.add_argument("--rule", default="youden")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--what", default="main", choices=["main", "cal"])
    ap.add_argument("--arms", default="",
                    help="comma-separated arm codes, e.g. OUR. Empty means "
                         "all seven. The calibration of a baseline is not "
                         "something the paper claims anything about, and "
                         "every row carried by hand is a row that can be "
                         "carried wrong.")
    ap.add_argument("--endpoints", default="",
                    help="comma-separated endpoint codes, e.g. MA. Empty "
                         "means all eleven. A sweep over four values of k "
                         "needs one number per model per dataset to draw its "
                         "curve; emitting all eleven endpoints as well turns "
                         "one screenshot into eight, and the other ten "
                         "endpoints stay on the cluster until something "
                         "depends on them.")
    a = ap.parse_args()

    keep_arms = [x.strip().upper() for x in a.arms.split(",") if x.strip()]
    unknown = [x for x in keep_arms if x not in ARM_CODE.values()]
    if unknown:
        raise SystemExit(f"no such arm code: {', '.join(unknown)}  "
                         f"(have {', '.join(ARM_CODE.values())})")

    keep_eps = [x.strip().upper() for x in a.endpoints.split(",") if x.strip()]
    codes = [c for _, c in EP_CODE]
    unknown = [x for x in keep_eps if x not in codes]
    if unknown:
        raise SystemExit(f"no such endpoint code: {', '.join(unknown)}  "
                         f"(have {', '.join(codes)})")
    eps = [(n, c) for n, c in EP_CODE if not keep_eps or c in keep_eps]

    tag = a.features.replace(":", "")
    sfx = "_noesi" if a.drop else ""
    path = f"{RES}/metrics_{tag}{sfx}.csv"
    if not os.path.exists(path):
        raise SystemExit(f"no such table: {path}")

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"read {path}  ({len(rows)} rows)")

    want = [d.strip() for d in a.datasets.split(",") if d.strip()]
    ep_order = {name: code for name, code in EP_CODE}

    # Defined here rather than in the loop: a dataset with no rows at this
    # rule hits continue, and if every dataset did, the summary at the bottom
    # would fail on a name that was never bound.
    wsfx = "" if a.what == "main" else "_cal"
    asfx = "" if not keep_arms else "_" + "".join(keep_arms).lower()

    for ds in want:
        sel = [r for r in rows
               if r["dataset"] == ds and r["rule"] == a.rule]
        if not sel:
            print(f"  {ds}: nothing at rule {a.rule} -- skipped")
            continue

        out = [f"{path.split('/')[-1]}   dataset {ds}   rule {a.rule}"]
        out.append(f"n {sel[0]['n']}   scale {sel[0].get('score_scale', '?')}")
        out.append("")
        out.append("endpoint order: 01 hospitalization 02 critical "
                   "03 sepsis 04 pneumonia_viral")
        out.append("                05 ards 06 pe 07 copd_asthma "
                   "08 acs_mi 09 aki")
        out.append("                MA macro acute (03-09)  "
                   "MD macro disposition (01-02)")
        out.append("arms: OUR ours  XGB xgboost  LGB lightgbm  RFO rf  "
                   "LRE logreg  HGB histgb  NEW news")
        out.append("")
        if a.what == "main":
            out.append("every number is four digits with 0. removed: 9000 "
                       "means 0.9000")
            out.append("each line: point lo hi for each metric, then the "
                       "threshold, then a check")
        out.append("")

        # events and prevalence do not depend on the arm, so they are printed
        # once here rather than repeated on seventy-seven lines
        ev = {}
        for r in sel:
            ev.setdefault(r["endpoint"], (r["events"], r["prevalence"]))
        out.append("EVENTS   " + "  ".join(
            f"{ep_order.get(k, '??')}:{v[0]}" for k, v in ev.items()
            if ep_order.get(k, "") not in ("MA", "MD", "")))
        out.append("")

        if a.what == "main":
            head = (f"{'':7}" + "".join(f"{m.upper():^15}" for m in METRICS)
                    + f"{'THRESH':>11}  CK")
        else:
            head = (f"{'':6}" + f"{'CAL SLOPE  point / lo / hi':^27}"
                    + f"{'CAL INTERCEPT  point / lo / hi':^30}"
                    + f"{'BRIER':^11}" + f"{'ECE':^11}"
                    + f"{'ALRT':^5}" + f"{'BND':^5}" + "  CK")
            out.append("slope and intercept are signed and are NOT the "
                       "four-digit form; brier and ece")
            out.append("carry six decimals because an ARDS Brier score sits "
                       "near 0.0001. ALRT is the alert")
            out.append("rate and BND the fraction of scores at the "
                       "probability bound, both four digits.")
            out.append("-1 across the whole row means the calibration fit "
                       "was refused, which is a result.")
            out.append("")
        out.append(head)
        out.append("-" * len(head))

        body = []
        for arm, code in ARM_CODE.items():
            if keep_arms and code not in keep_arms:
                continue
            got = [r for r in sel if r["arm"] == arm]
            if not got:
                body.append(f"{code}  --  arm absent from the table")
                continue
            by_ep = {r["endpoint"]: r for r in got}
            for name, ec in EP_CODE:
                r = by_ep.get(name)
                if r is None:
                    body.append(f"{code} {ec}  row absent")
                    continue
                if a.what == "main":
                    cells = ""
                    for m in METRICS:
                        cells += (f" {q(r.get(m))} {q(r.get(m + '_lo'))} "
                                  f"{q(r.get(m + '_hi'))}")
                    line = f"{code} {ec}" + cells + thr(r.get("threshold"))
                    body.append(line + "  " + check(line))
                else:
                    cells = ""
                    for m, w in ((CAL_CI[0], 9), (CAL_CI[1], 10)):
                        cells += (sci(r.get(m), w, 4)
                                  + sci(r.get(m + "_lo"), w, 4)
                                  + sci(r.get(m + "_hi"), w, 4))
                    cells += sci(r.get("brier"), 11, 6)
                    cells += sci(r.get("ece"), 11, 6)
                    # q() clamps to [0, 0.9999], so the -1 that 14_metrics.py
                    # writes for a refused row would print as 0000 -- "no
                    # scores at the probability bound", which is the opposite
                    # of what happened. Dots instead.
                    for k in ("alert_rate", "frac_at_prob_bound"):
                        v = r.get(k)
                        neg = False
                        try:
                            neg = float(v) < 0
                        except (TypeError, ValueError):
                            pass
                        cells += " ----" if neg else f" {q(v)}"
                    line = f"{code} {ec}" + cells
                    # check_signed, not check: see the note on that function.
                    body.append(line + "  " + check_signed(line))
        out += body

        # A per-line check cannot notice a line that never arrived, and a
        # dropped line is the likeliest thing to go wrong when eighty of them
        # are being read off two screenshots.
        out.append("-" * len(head))
        blk = check(''.join(body)) if a.what == "main" \
            else check_signed(''.join(body))
        out.append(f"LINES {len(body)}   BLOCK {blk}")

        p = f"{RES}/emit_{tag}{sfx}_{a.rule}_{ds}{wsfx}{asfx}.txt"
        with open(p, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"  wrote {p}  ({len(out)} lines)")

    print()
    print("Photograph each file in two parts, top half then bottom half,")
    print("making sure the overlap includes at least one whole line:")
    for ds in want:
        print(f"  cat {RES}/emit_{tag}{sfx}_{a.rule}_{ds}{wsfx}{asfx}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
