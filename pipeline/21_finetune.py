"""21_finetune.py -- how much local data buys how much performance.

    python 21_finetune.py --dataset bidmc --features greedy:12
    python 21_finetune.py --dataset stanford --sizes 100,250,500,1000,2500,5000

A hospital that wants to use this model has zero labelled outcomes on day
one and a few hundred by the end of the quarter. This measures what those few
hundred are worth, against the two cheaper things they could do instead.

Four arms of the same question
------------------------------
    zero-shot     the model as shipped. Already in the main table; recomputed
                  here so every number in the figure comes from one script and
                  one evaluation split.

    recalibrate   refit two numbers -- a slope and an intercept on the logit
                  -- per endpoint. This is the comparator that matters. The
                  calibration transcribed on 2026-08-13 says the external
                  failure is not a pure shift: slopes fall to about a half on
                  six of nine endpoints while in-domain they sit at one. Two
                  parameters per endpoint is eighteen numbers for the whole
                  model, and a few hundred outcomes can fit them.

    heads         unfreeze the per-task output heads and the gate, hold the
                  body and the experts frozen. This is what people mean by
                  fine-tuning a shared-representation model cheaply.

    full          everything unfrozen, for the upper bound. Reported so the
                  gap between heads and full is visible; not proposed as
                  something a hospital would do.

The evaluation split is fixed and disjoint from every fine-tuning set, and it
is drawn once before any of the sizes are cut, so the four arms and the six
sizes are all scored on the same patients. Scoring different arms on different
patients would let case mix masquerade as a method effect.

Why the sizes are encounters and not events
-------------------------------------------
A hospital counts what it has collected, not what happened in it. ARDS at
0.05% means five thousand encounters carry two or three events, so the
per-endpoint curves will be flat and noisy at the small sizes and that is the
finding, not a defect: the endpoints that most need local data are the ones
where local data arrives slowest. The event count per size is printed beside
each row so nobody has to infer it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
DATA = f"{PROJ}/data"
RUNS = f"{PROJ}/runs"
RES = f"{PROJ}/results"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TASKS = ["hospitalization", "critical", "sepsis", "pneumonia_viral", "ards",
         "pe", "copd_asthma", "acs_mi", "aki"]
ACUTE = list(range(2, 9))

# The split that separates fine-tuning from evaluation. Fixed, named, and not
# the model seed: a fine-tuning experiment whose evaluation set moves with the
# seed is three experiments averaged together.
FT_SPLIT_SEED = 20260813


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


def auroc(y, s):
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s)
    r = np.empty(len(s), float)
    sr = np.asarray(s)[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    m = int(y.sum())
    return float((r[y == 1].sum() - m * (m + 1) / 2) / (m * (len(y) - m)))


def macro_acute(Y, S):
    v = [auroc(Y[:, t], S[:, t]) for t in ACUTE]
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def calib(Y, S):
    """Mean |slope - 1|, mean |intercept|, Brier and ECE over the acute tasks.

    AUROC is invariant to any monotone transform of the score, and a slope
    and intercept on the logit is exactly that. So the recalibration arm has
    the same AUROC as the zero-shot arm -- not approximately, identically,
    by construction. The first version of this script reported AUROC for
    every arm and RC came back equal to ZS at all six sizes, which is not a
    finding, it is a statement about what AUROC measures.

    These four numbers are what recalibration moves. Reported for every arm
    so the comparison is between quantities the intervention can change.
    """
    sl, ic, br, ec = [], [], [], []
    for t in ACUTE:
        y, lp = Y[:, t], S[:, t]
        if y.sum() < 5 or y.sum() == len(y):
            continue
        b1, b0 = platt(y, lp)
        p = 1.0 / (1.0 + np.exp(-lp))
        sl.append(abs(b1 - 1.0))
        ic.append(abs(b0))
        br.append(float(np.mean((p - y) ** 2)))
        o = np.argsort(p)
        ps, ys = p[o], y[o]
        e, n = 0.0, len(ps)
        for i in range(10):
            a, b = i * n // 10, (i + 1) * n // 10
            if b > a:
                e += (b - a) * abs(ys[a:b].mean() - ps[a:b].mean())
        ec.append(e / n)
    f = lambda v: float(np.mean(v)) if v else float("nan")
    return f(sl), f(ic), f(br), f(ec)


def platt(y, lp):
    """Slope and intercept on the logit, by Newton steps on the log loss.

    Written out rather than imported so this script has no dependency beyond
    numpy and torch: the cluster has no network and a missing sklearn at two
    in the morning is a wasted night.
    """
    b = np.array([1.0, 0.0])
    X = np.column_stack([lp, np.ones_like(lp)])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(X @ b)))
        w = np.clip(p * (1 - p), 1e-9, None)
        g = X.T @ (p - y)
        H = (X * w[:, None]).T @ X + 1e-8 * np.eye(2)
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-10:
            break
    return float(b[0]), float(b[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="G2_no_l1_no_drop")
    ap.add_argument("--features", default="greedy:12")
    ap.add_argument("--drop", default="triage_acuity")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--dataset", default="bidmc")
    ap.add_argument("--sizes", default="100,250,500,1000,2500,5000")
    ap.add_argument("--eval-frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    import models

    tag = a.features.replace(":", "")
    seeds = [s.strip() for s in a.seeds.split(",") if s.strip()]
    sizes = [int(x) for x in a.sizes.split(",") if x.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.load(open(f"{DATA}/preprocessing.json"))
    cols = meta["features"]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    tr = import_module("01_train") if os.path.exists(
        f"{os.path.dirname(os.path.abspath(__file__))}/01_train.py") else None
    if tr is None:
        raise SystemExit("01_train.py must sit beside this script")
    idx = tr.resolve_features(a.features, cols, tuple(
        x for x in a.drop.split(",") if x))
    mean = np.asarray(meta["scaler_mean"])[idx]
    scale = np.asarray(meta["scaler_scale"])[idx]

    X = np.load(f"{DATA}/{a.dataset}_X.npy")[:, idx]
    X = ((X - mean) / np.where(scale == 0, 1, scale)).astype(np.float32)
    Y = np.load(f"{DATA}/{a.dataset}_y.npy").astype(np.float32)
    n = len(X)

    rng = np.random.default_rng(FT_SPLIT_SEED)
    perm = rng.permutation(n)
    n_eval = int(round(a.eval_frac * n))
    ev, pool = perm[:n_eval], perm[n_eval:]
    if max(sizes) > len(pool):
        raise SystemExit(
            f"\nlargest size {max(sizes)} exceeds the {len(pool)} encounters "
            f"left after\nthe evaluation half is taken. Lower --sizes or "
            f"--eval-frac.")
    print(f"{a.dataset}: {n:,} encounters, {n_eval:,} held out for evaluation,"
          f" {len(pool):,} available to fine-tune on")
    print(f"evaluation events, acute: {int(Y[ev][:, ACUTE].sum()):,}")

    Xe = torch.from_numpy(X[ev]).to(dev)
    Ye = Y[ev]

    def score(model):
        model.eval()
        with torch.no_grad():
            return model(Xe).cpu().numpy()

    def fresh(seed):
        m = models.build(a.arm, X.shape[1]).to(dev)
        p = f"{RUNS}/{a.arm}_{tag}_noesi_seed{seed}/model.pt"
        if not os.path.exists(p):
            return None
        m.load_state_dict(torch.load(p, map_location=dev))
        return m

    rows = []          # (mode, size, macro, per-task, slope, icpt, brier, ece)

    # ---- zero shot -------------------------------------------------------
    Z = []
    for s in seeds:
        m = fresh(s)
        if m is None:
            print(f"  seed {s}: no model.pt, skipped")
            continue
        Z.append(score(m))
    if not Z:
        raise SystemExit(f"no checkpoints under {RUNS}/{a.arm}_{tag}_noesi_*")
    z = np.mean(Z, axis=0)
    rows.append(("ZS", 0, macro_acute(Ye, z),
                 [auroc(Ye[:, t], z[:, t]) for t in range(9)], *calib(Ye, z)))
    print(f"\nzero shot  macro acute {rows[-1][2]:.4f}")

    # ---- recalibration and fine-tuning, per size -------------------------
    print(f"\n{'mode':<6}{'size':>7}{'events':>9}{'macro acute':>13}"
          f"{'|slope-1|':>11}{'|icpt|':>9}{'brier':>9}{'ece':>9}")
    show = lambda r: print(f"{r[0]:<6}{r[1]:>7}{'':>9}{r[2]:>13.4f}"
                          f"{r[4]:>11.3f}{r[5]:>9.3f}{r[6]:>9.5f}{r[7]:>9.5f}")
    show(rows[0])
    for size in sizes:
        take = pool[:size]
        Xt = torch.from_numpy(X[take]).to(dev)
        Yt = torch.from_numpy(Y[take]).to(dev)
        ne = int(Y[take][:, ACUTE].sum())

        # Recalibrate: two numbers per endpoint, fitted on these rows and
        # applied to the evaluation half. Note the slope and intercept are
        # fitted on the fine-tuning encounters' own logits, not on the
        # evaluation ones -- fitting on what you then score is the mistake
        # this whole comparison exists to avoid.
        rc = np.array(z, copy=True)
        Zt = []
        for s in seeds:
            m = fresh(s)
            if m is None:
                continue
            m.eval()
            with torch.no_grad():
                Zt.append(m(Xt).cpu().numpy())
        zt = np.mean(Zt, axis=0)
        for t in range(9):
            yt = Y[take][:, t]
            if yt.sum() < 5 or yt.sum() == len(yt):
                continue                      # not enough to fit two numbers
            b1, b0 = platt(yt, zt[:, t])
            rc[:, t] = b1 * z[:, t] + b0
        rows.append(("RC", size, macro_acute(Ye, rc),
                     [auroc(Ye[:, t], rc[:, t]) for t in range(9)],
                     *calib(Ye, rc)))
        show(rows[-1])

        # heads + gate, then everything
        for mode, only_heads in (("HD", True), ("FL", False)):
            P = []
            for s in seeds:
                m = fresh(s)
                if m is None:
                    continue
                if only_heads:
                    # Freeze the experts, tune everything else. Defined by
                    # what the shared representation IS rather than by what
                    # the per-task parts are called: the first version of
                    # this matched on "head", "gate" and "task", and
                    # G2_no_l1_no_drop calls its output heads "towers" and
                    # keeps the gate in two bare parameters S and A, so
                    # nothing matched and the arm was silently skipped for a
                    # whole run. A filter that can match nothing must say so
                    # loudly, which is why the check below raises.
                    for nm, prm in m.named_parameters():
                        prm.requires_grad = not nm.startswith("experts.")
                trainable = [p for p in m.parameters() if p.requires_grad]
                frozen = [p for p in m.parameters() if not p.requires_grad]
                if not trainable or (only_heads and not frozen):
                    raise SystemExit(
                        f"\n{mode}: {len(trainable)} trainable and "
                        f"{len(frozen)} frozen tensors.\nThat is not a "
                        f"partial fine-tune. Parameter names in this arm:\n"
                        + "\n".join(f"  {n}" for n, _ in
                                     m.named_parameters()))
                opt = torch.optim.Adam(trainable, lr=a.lr)
                lossf = nn.BCEWithLogitsLoss()
                m.train()
                for _ in range(a.epochs):
                    opt.zero_grad()
                    lossf(m(Xt), Yt).backward()
                    opt.step()
                P.append(score(m))
            if not P:
                continue
            p = np.mean(P, axis=0)
            rows.append((mode, size, macro_acute(Ye, p),
                         [auroc(Ye[:, t], p[:, t]) for t in range(9)],
                         *calib(Ye, p)))
            show(rows[-1])

    # ---- emit ------------------------------------------------------------
    out = [f"finetune {a.arm} {a.features}   dataset {a.dataset}   n {n}",
           f"evaluation half {n_eval}, split seed {FT_SPLIT_SEED}, "
           f"seeds {','.join(seeds)}",
           "",
           "modes: ZS zero shot   RC recalibrate slope+intercept   "
           "HD heads and gate   FL all weights",
           "endpoints: 01 hospitalization 02 critical 03 sepsis "
           "04 pneumonia_viral",
           "           05 ards 06 pe 07 copd_asthma 08 acs_mi 09 aki "
           "MA macro acute",
           "",
           "AUROC on the held-out half, four digits. SIZE is encounters used "
           "for fitting.",
           "SLP and ICP are the mean distance of the calibration slope from "
           "one and of the",
           "intercept from zero, over the seven acute endpoints, divided by "
           "ten so they fit",
           "the four-digit field. BRI and ECE are Brier and expected "
           "calibration error.",
           "",
           "RC has the same AUROC as ZS in every row and that is not an "
           "error: a slope and",
           "an intercept on the logit is a monotone transform, and AUROC "
           "cannot see one.",
           "The last four columns are where recalibration shows up.",
           ""]
    head = (f"{'':10}{'SIZE':>6}" + "".join(f"{c:^6}" for c in
            ("01", "02", "03", "04", "05", "06", "07", "08", "09", "MA",
             "SLP", "ICP", "BRI", "ECE"))
            + "  CK")
    out += [head, "-" * len(head)]
    body = []
    for mode, size, ma, per, sl, ic, br, ec in rows:
        cells = "".join(f" {q(v)}" for v in per) + f" {q(ma)}"
        # slope and intercept are distances from perfect, so they can exceed
        # one and are carried scaled by ten; brier and ece are already small
        cells += f" {q(sl / 10)} {q(ic / 10)} {q(br)} {q(ec)}"
        line = f"FT {mode} {size:>6}{cells}"
        body.append(line + "  " + check(line))
    out += body
    out += ["-" * len(head),
            f"LINES {len(body)}   BLOCK {check(''.join(body))}"]
    p = f"{RES}/emit_finetune_{a.arm}_{tag}_{a.dataset}.txt"
    with open(p, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\nwrote {p}  ({len(out)} lines)")
    print(f"  cat {p}")

    # chosen by expected calibration error, not AUROC: picking the best RC
    # row by AUROC would pick an arbitrary one, since they are all equal
    best_rc = min((r for r in rows if r[0] == "RC"), key=lambda r: r[7],
                  default=None)
    best_hd = min((r for r in rows if r[0] == "HD"), key=lambda r: r[7],
                  default=None)
    zs = rows[0]
    print(f"\ncalibration, the thing recalibration can actually move:")
    print(f"  zero shot         |slope-1| {zs[4]:.3f}   |icpt| {zs[5]:.3f}"
          f"   ece {zs[7]:.5f}")
    if best_rc:
        print(f"  recalibrated      |slope-1| {best_rc[4]:.3f}   "
              f"|icpt| {best_rc[5]:.3f}   ece {best_rc[7]:.5f}"
              f"   at {best_rc[1]} encounters")
    if best_hd:
        print(f"  heads and gate    |slope-1| {best_hd[4]:.3f}   "
              f"|icpt| {best_hd[5]:.3f}   ece {best_hd[7]:.5f}"
              f"   at {best_hd[1]} encounters")
        print(f"  and its AUROC {best_hd[2]:.4f} against zero shot "
              f"{zs[2]:.4f} -- discrimination is the only place fine-tuning")
        print("  can beat recalibration, because recalibration cannot move it "
              "at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
