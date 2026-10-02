"""
18b_shap_consistency.py -- the twelve numbers panel b of Figure 4 needs.

    python 18b_shap_consistency.py --features greedy:12 --drop triage_acuity

Prints twelve lines. Nothing else leaves the enclave.

Why this file exists
--------------------
Panel b of Figure 4 plots, for each variable:

    y   task specificity of magnitude
        max over targets of mean|EG|  /  min over targets of mean|EG|
    x   consistency of attribution across targets
        median over the 36 target pairs of corr(EG[:, d, tA], EG[:, d, tB])

The y axis is already computable outside: it is a ratio of the per-endpoint
means that results/shap_*.txt prints. The x axis is a correlation ACROSS
ENCOUNTERS, so it needs the (N, D, T) tensor. 18_shap.py builds exactly that
tensor, reduces it to means on the next line, and drops it. Twelve numbers is
all that is missing, and one screenshot carries them.

Same rows, same background, same seeds
--------------------------------------
EXPLAIN_SEED and the check digit are IMPORTED from 18_shap.py, not restated
here, so the correlations describe the same 2,000 encounters as the means
already transcribed. If the two files disagreed, the two halves of panel b
would be about different rows and nothing in the printed output would show
it. A first draft did restate EXPLAIN_SEED, and got it wrong.

The check
---------
The last block re-prints mean|EG| per endpoint for one variable. Those nine
numbers must match the corresponding row of results/shap_greedy12.txt to four
decimals. If they do not, this script is looking at a different model or a
different sample and the correlations should be discarded.

Correlation of what, exactly
----------------------------
Pearson r between two columns of the attribution tensor for the same
variable. A variable near r = 1 pushes the same encounters in the same
direction for every endpoint, which is what a shared severity axis looks
like. A variable near r = 0 is doing unrelated work on different endpoints.
The median over pairs, not the mean, because one outlying pair (usually a
rare endpoint whose attributions are noisy) otherwise drags the summary.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_t = importlib.import_module("01_train")
# Imported, not copied. EXPLAIN_SEED, the check digit and the arm default all
# have to agree with 18_shap.py or the two halves of panel b describe
# different encounters, and nothing in the printed output would reveal it.
# A first draft of this file hard-coded EXPLAIN_SEED and got it wrong
# (20260101 against the real 20260812), which would have explained a
# completely different 2,000 rows and looked entirely plausible.
_s = importlib.import_module("18_shap")
import models as M                                            # noqa: E402
from attribution import expected_gradients                    # noqa: E402

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
DATA = f"{PROJ}/data"
RUNS = f"{PROJ}/runs"
RES = f"{PROJ}/results"

EXPLAIN_SEED = _s.EXPLAIN_SEED
check = _s.check


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="G2_no_l1_no_drop")
    ap.add_argument("--features", default="greedy:12")
    ap.add_argument("--drop", default="triage_acuity")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n-explain", type=int, default=2000)
    ap.add_argument("--n-background", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--runs", default=RUNS,
                    help="a directory of seed<N>/model.pt, for running "
                         "against a local copy of the weights")
    a = ap.parse_args()

    t0 = time.time()
    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)
    seeds = [int(x) for x in a.seeds.split(",")]
    tag = a.features.replace(":", "")
    sfx = _t.drop_suffix(drop)

    meta = json.load(open(f"{a.data}/preprocessing.json"))
    cols = meta["features"]
    idx = _t.resolve_features(a.features, cols, drop)
    names = [cols[i] for i in idx]
    outcomes = [o.replace("outcome_", "") for o in meta["outcomes"]]
    mean = np.asarray(meta["scaler_mean"])[idx]
    scale = np.asarray(meta["scaler_scale"])[idx]

    X = np.load(f"{a.data}/umn_test_X.npy")[:, idx]
    X = ((X - mean) / np.where(scale == 0, 1, scale)).astype(np.float32)

    rng = np.random.default_rng(EXPLAIN_SEED)
    ex_i = rng.choice(len(X), size=min(a.n_explain, len(X)), replace=False)
    bg_i = rng.choice(len(X), size=min(a.n_background, len(X)), replace=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xex = torch.from_numpy(X[ex_i]).to(dev)
    Xbg = torch.from_numpy(X[bg_i]).to(dev)

    D, T = len(names), len(outcomes)
    pairs = list(itertools.combinations(range(T), 2))
    # (seed, variable, pair) and (seed, variable, endpoint)
    corr = np.zeros((len(seeds), D, len(pairs)))
    absmean = np.zeros((len(seeds), D, T))

    for si, s in enumerate(seeds):
        # Two layouts: the cluster's runs/<arm>_<tag><sfx>_seed<N>/model.pt
        # and the flat seed<N>/model.pt that extras_local.ps1 scps down.
        cands = [f"{a.runs}/{a.arm}_{tag}{sfx}_seed{s}/model.pt",
                 f"{a.runs}/seed{s}/model.pt"]
        p = next((c for c in cands if os.path.exists(c)), None)
        if p is None:
            raise SystemExit("no model for seed %d; looked at:\n  %s"
                             % (s, "\n  ".join(cands)))
        model = M.build(a.arm, D, n_tasks=T, hidden=_t.HP["hidden"],
                        n_experts=_t.HP["n_experts"],
                        temperature=_t.HP["temperature"],
                        alpha_shared=_t.HP["alpha_shared"],
                        lambda_kl=_t.HP["lambda_kl"],
                        expert_dropout=_t.HP["expert_dropout"],
                        l1_gate=_t.HP["l1_gate"]).to(dev)
        model.load_state_dict(torch.load(p, map_location=dev))
        model.eval()

        A = np.asarray(expected_gradients(model, Xex, Xbg,
                                          n_samples=a.n_samples, seed=s))
        absmean[si] = np.abs(A).mean(axis=0)
        for d in range(D):
            for k, (ta, tb) in enumerate(pairs):
                u, v = A[:, d, ta], A[:, d, tb]
                su, sv = u.std(), v.std()
                # A constant column has no correlation. It has never happened
                # here, but returning nan silently would poison the median.
                corr[si, d, k] = (0.0 if su == 0 or sv == 0
                                  else float(np.corrcoef(u, v)[0, 1]))
        print(f"  seed {s}: done  {time.time() - t0:.0f}s", flush=True)
        del model, A

    med = np.median(corr, axis=2).mean(axis=0)        # (D,) over seeds
    mabs = absmean.mean(axis=0)                       # (D, T)
    spec = mabs.max(axis=1) / np.maximum(mabs.min(axis=1), 1e-12)
    overall = mabs.mean(axis=1)

    print()
    print(f"consistency   arm {a.arm}   features {a.features}   "
          f"dropped {list(drop)}")
    print(f"seeds {seeds}   explained {len(ex_i)}   background {len(bg_i)}   "
          f"samples {a.n_samples}   pairs {len(pairs)}")
    print(f"data_id {meta.get('data_id')}   explain_seed {EXPLAIN_SEED}")
    print()
    print("variable                       consist   spec   meanEG  ck")
    for d in np.argsort(-overall):
        line = (f"{names[d]:30s}  {med[d]:6.3f} {spec[d]:6.2f} "
                f"{overall[d]:8.4f}")
        print(f"{line}  {check(line)}")

    # The check. These nine must match the same row of shap_greedy12.txt.
    d0 = int(np.argmax(overall))
    print()
    print(f"check: mean|EG| per endpoint for {names[d0]}")
    print("  " + "  ".join(f"{o}={mabs[d0, t]:.4f}"
                           for t, o in enumerate(outcomes)))
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
