"""
18_shap.py -- expected gradients for the k=12 model, printed for screenshots.

    python 18_shap.py --features greedy:12 --drop triage_acuity

Writes results/shap_<tag>_<drop>.txt, about sixty lines, one screen.

What is computed
----------------
Expected gradients, from attribution.py:

    EG_i(x, t) = E_{b~D, a~U(0,1)} [ (x_i - b_i) * df_t/dx_i (b + a(x - b)) ]

This is not a stand-in for SHAP. DeepSHAP applies the DeepLIFT rescale rule
with a background sample, and for a network of linear layers and ReLUs -- which
is what this is -- that rule is a discrete approximation of exactly the
quantity above. The shap package cannot be installed inside the enclave (no
network, no local index), so the thing it approximates is computed directly
instead. attribution.py's self_test checks completeness, linearity against a
closed form, and device agreement.

Three seeds, and what varies between them
-----------------------------------------
Each seed is a differently initialised model that reached its own optimum, so
attributions are averaged across seeds the same way predictions are. The
spread across seeds is printed beside the mean and is the honest measure of
how firmly a variable's rank is established: a variable whose importance
ranges from third to ninth across three runs should not be described as "the
third most important variable" in a figure caption.

Signed as well as absolute
--------------------------
Mean |EG| says how much a variable moves the prediction. Mean EG says which
way. Both are printed, because a variable can matter enormously and have no
consistent direction -- age does not, oxygen saturation does -- and a figure
that shows only magnitude cannot tell those apart.

Cost
----
2,000 explained encounters against a 200-row background at 200 samples each is
about 80 million forward-backward passes per endpoint per seed, which is a few
minutes on one MIG slice. The explained rows are drawn with a fixed seed from
the internal test set, so the same rows are used for every seed and every
endpoint and the comparison between them is paired.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_t = importlib.import_module("01_train")
import models as M                                            # noqa: E402
from attribution import expected_gradients                    # noqa: E402

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
DATA = f"{PROJ}/data"
RUNS = f"{PROJ}/runs"
RES = f"{PROJ}/results"

ACUTE, DISP = list(range(2, 9)), [0, 1]
EXPLAIN_SEED = 20260812     # which test rows get explained; fixed, not the model seed


def check(s):
    """Same two-digit position-weighted check 17_emit.py prints.

    This file leaves the enclave the same way the tables do, so it needs the
    same protection. A mistyped attribution would reorder a figure.
    """
    t = 0
    for i, ch in enumerate(s):
        if ch.isdigit():
            t += (i + 1) * int(ch)
    return f"{t % 97:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="G2_no_l1_no_drop")
    ap.add_argument("--features", default="greedy:12")
    ap.add_argument("--drop", default="triage_acuity")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n-explain", type=int, default=2000)
    ap.add_argument("--n-background", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=200)
    a = ap.parse_args()

    t0 = time.time()
    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)
    seeds = [int(x) for x in a.seeds.split(",")]
    tag = a.features.replace(":", "")
    sfx = _t.drop_suffix(drop)

    meta = json.load(open(f"{DATA}/preprocessing.json"))
    cols = meta["features"]
    idx = _t.resolve_features(a.features, cols, drop)
    names = [cols[i] for i in idx]
    outcomes = [o.replace("outcome_", "") for o in meta["outcomes"]]
    mean = np.asarray(meta["scaler_mean"])[idx]
    scale = np.asarray(meta["scaler_scale"])[idx]

    X = np.load(f"{DATA}/umn_test_X.npy")[:, idx]
    X = ((X - mean) / np.where(scale == 0, 1, scale)).astype(np.float32)
    print(f"data_id {meta.get('data_id')}   test {X.shape}   "
          f"{len(names)} variables")

    rng = np.random.default_rng(EXPLAIN_SEED)
    ex_i = rng.choice(len(X), size=min(a.n_explain, len(X)), replace=False)
    bg_i = rng.choice(len(X), size=min(a.n_background, len(X)), replace=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xex = torch.from_numpy(X[ex_i]).to(dev)
    Xbg = torch.from_numpy(X[bg_i]).to(dev)
    print(f"explaining {len(ex_i):,} encounters against a "
          f"{len(bg_i)}-row background, {a.n_samples} samples, on {dev}")

    # (seed, variable, endpoint)
    absmean = np.zeros((len(seeds), len(names), len(outcomes)))
    sgnmean = np.zeros_like(absmean)

    for si, s in enumerate(seeds):
        d = f"{RUNS}/{a.arm}_{tag}{sfx}_seed{s}"
        p = f"{d}/model.pt"
        if not os.path.exists(p):
            raise SystemExit(
                f"no model at {p}.\nOnly the neural arms save model.pt; the "
                f"tree arms are scored during training and never pickled, so "
                f"this can only run on {a.arm}.")
        model = M.build(a.arm, len(names), n_tasks=len(outcomes),
                        hidden=_t.HP["hidden"], n_experts=_t.HP["n_experts"],
                        temperature=_t.HP["temperature"],
                        alpha_shared=_t.HP["alpha_shared"],
                        lambda_kl=_t.HP["lambda_kl"],
                        expert_dropout=_t.HP["expert_dropout"],
                        l1_gate=_t.HP["l1_gate"]).to(dev)
        model.load_state_dict(torch.load(p, map_location=dev))
        model.eval()

        A = expected_gradients(model, Xex, Xbg,
                               n_samples=a.n_samples, seed=s)   # (N, D, T)
        A = np.asarray(A)
        absmean[si] = np.abs(A).mean(axis=0)
        sgnmean[si] = A.mean(axis=0)
        print(f"  seed {s}: done  {time.time() - t0:.0f}s", flush=True)
        del model, A

    # Averaged over seeds; the acute macro is the mean over the seven acute
    # endpoints, matching how every other number in the paper is macro'd.
    mabs = absmean.mean(axis=0)
    msgn = sgnmean.mean(axis=0)
    acute_abs = mabs[:, ACUTE].mean(axis=1)
    acute_sgn = msgn[:, ACUTE].mean(axis=1)
    # spread across seeds of the acute macro, per variable
    per_seed_acute = absmean[:, :, ACUTE].mean(axis=2)
    spread = per_seed_acute.max(axis=0) - per_seed_acute.min(axis=0)

    order = np.argsort(-acute_abs)
    # rank each seed separately, so the printed rank range says whether the
    # ordering is a property of the model or of the initialisation
    ranks = np.argsort(np.argsort(-per_seed_acute, axis=1), axis=1) + 1

    out = [f"expected gradients   arm {a.arm}   features {a.features}   "
           f"dropped {list(drop)}",
           f"seeds {seeds}   explained {len(ex_i)}   background {len(bg_i)}   "
           f"samples {a.n_samples}",
           f"data_id {meta.get('data_id')}   explain_seed {EXPLAIN_SEED}",
           "",
           "mean |EG| and mean EG over the seven acute endpoints, x10000.",
           "SPREAD is max-min of mean|EG| across the three seeds.",
           "RANKS is each seed's own ranking of this variable.",
           ""]
    # Data lines are collected as they are made rather than recognised
    # afterwards by their shape. An earlier version picked them out with
    # "ends in two digits", which also matched the header line "... samples
    # 200" and would have folded a header into the block check -- a check
    # that disagrees for a reason having nothing to do with the data is worse
    # than no check, because it trains you to ignore it.
    body = []

    head = (f"{'#':>2} {'variable':<30}{'MEAN|EG|':>10}{'MEANEG':>9}"
            f"{'SPREAD':>8}  {'RANKS':<10} CK")
    out.append(head)
    out.append("-" * len(head))
    for r, i in enumerate(order, 1):
        rr = "/".join(str(int(x)) for x in ranks[:, i])
        line = (f"{r:>2} {names[i]:<30}"
                f"{int(round(acute_abs[i] * 10000)):>10}"
                f"{int(round(acute_sgn[i] * 10000)):>9}"
                f"{int(round(spread[i] * 10000)):>8}  {rr:<10}")
        body.append(line)
        out.append(line + " " + check(line))
    out.append("-" * len(head))

    out.append("")
    out.append("mean |EG| per endpoint, x10000, seed-averaged")
    head2 = f"{'variable':<30}" + "".join(f"{o[:7]:>8}" for o in outcomes) + " CK"
    out.append(head2)
    out.append("-" * len(head2))
    for i in order:
        line = (f"{names[i]:<30}"
                + "".join(f"{int(round(mabs[i, t] * 10000)):>8}"
                          for t in range(len(outcomes))))
        body.append(line)
        out.append(line + " " + check(line))
    out.append("-" * len(head2))
    out.append(f"LINES {len(body)}   BLOCK {check(''.join(body))}")

    os.makedirs(RES, exist_ok=True)
    p = f"{RES}/shap_{tag}{sfx}.txt"
    with open(p, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwrote {p}   {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
