"""
02_arch_sweep.py -- run every arm of the architecture comparison, resumably.

Submitted once with sbatch and left overnight. Each configuration writes its
own directory and its own metrics.json; a configuration that already has a
metrics.json is skipped, so if the job is killed or times out, resubmitting
picks up where it stopped rather than starting over.

Order is deliberate. The two arms that answer the central question run first:

    D_moe_router     the published model
    E_moe_norouter   the same thing with the shared router and the KL removed

If the night goes badly and only a few runs finish, those two are the ones
worth having. After them come the other neural arms, and the tree baselines
last because random forest on 1.3 million rows is the slowest thing here and
the least informative if interrupted.

    python 02_arch_sweep.py                  everything
    python 02_arch_sweep.py --quick          one seed, neural only
    python 02_arch_sweep.py --arms D,E       just these
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
RUNS = f"{PROJ}/runs"

ORDER = [
    "G1_no_l1",         # is the L1 on the gate what collapses it
    "G2_no_l1_no_drop", # or does expert dropout have to go too
    "D_moe_router",     # published architecture
    "E_moe_norouter",   # the KL test -- answered, it is not the KL
    "B_shared",         # is the mixture of experts doing anything
    "A1_independent",   # is multi-task doing anything
    "C_mmoe",           # should the gate see the input
    "F2_topk5",         # does hard sparsity help
    "F1_topk3",
    "F3_topk7",
]
TREES = ["A2_xgb_sub", "A2_xgb", "A2_histgb", "A2_rf"]
SEEDS = [42, 43, 44]


def done(arm, seed, feats="top15"):
    return os.path.exists(f"{RUNS}/{arm}_{feats}_seed{seed}/metrics.json")


def run_one(arm, seed, feats="top15"):
    cmd = [sys.executable, f"{HERE}/01_train.py",
           "--arm", arm, "--seed", str(seed), "--features", feats]
    t0 = time.time()
    r = subprocess.run(cmd)
    return r.returncode, (time.time() - t0) / 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--features", default="top15")
    a = ap.parse_args()

    seeds = [SEEDS[0]] if a.quick else SEEDS
    arms = ORDER if a.quick else ORDER + TREES
    if a.arms:
        want = {x.strip() for x in a.arms.split(",")}
        arms = [x for x in ORDER + TREES
                if x in want or x.split("_")[0] in want]

    # The tree baselines get all three seeds too. They only ran once in the
    # first sweep, which was fine for a point estimate but not for the seed
    # ensemble: averaging three of ours against one of theirs would be a
    # comparison of ensemble against single model dressed up as a comparison
    # of architectures. Gradient boosting costs eighteen seconds a fit, so
    # symmetry here is nearly free.
    jobs = [(arm, s) for arm in arms for s in seeds]
    todo = [j for j in jobs if not done(j[0], j[1], a.features)]

    print("=" * 68)
    print(f"ARCHITECTURE SWEEP   {len(jobs)} configurations, "
          f"{len(todo)} still to do")
    print("=" * 68)
    for arm, s in jobs:
        print(f"  {'done ' if done(arm, s, a.features) else 'queued'}  "
              f"{arm}  seed {s}")
    print(flush=True)

    t0, failed = time.time(), []
    for i, (arm, seed) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {arm} seed {seed}   "
              f"elapsed {(time.time() - t0) / 60:.0f} min", flush=True)
        rc, mins = run_one(arm, seed, a.features)
        if rc != 0:
            failed.append((arm, seed, rc))
            print(f"  FAILED rc={rc}, continuing", flush=True)

    # ---- collect ------------------------------------------------------
    rows = []
    for arm, seed in jobs:
        p = f"{RUNS}/{arm}_{a.features}_seed{seed}/metrics.json"
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        rows.append({
            "arm": arm, "seed": seed,
            "acute_auroc": m["macro_acute_auroc"],
            "acute_auprc": m["macro_acute_auprc"],
            "disp_auroc": m["macro_disp_auroc"],
            "disp_auprc": m["macro_disp_auprc"],
            "gate_cos": m.get("gate", {}).get("cos_mean"),
            "eff_experts": m.get("gate", {}).get("eff_experts"),
            "minutes": m.get("minutes"),
        })

    os.makedirs(f"{PROJ}/results", exist_ok=True)
    out = f"{PROJ}/results/arch_comparison.csv"
    if rows:
        import csv
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\n" + "=" * 88)
    print("SUMMARY   mean over seeds, acute = seven conditions, "
          "disp = two disposition endpoints")
    print("=" * 88)
    print(f"{'arm':<18}{'acute AUROC':>14}{'acute AUPRC':>14}"
          f"{'disp AUROC':>13}{'gate cos':>11}{'eff exp':>10}")
    print("-" * 88)
    import statistics as st
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        def mm(k):
            v = [r[k] for r in rs if r[k] is not None]
            if not v:
                return "     -    "
            if len(v) == 1:
                return f"{v[0]:.4f}    "
            return f"{st.mean(v):.4f}+-{st.stdev(v):.3f}"
        print(f"{arm:<18}{mm('acute_auroc'):>14}{mm('acute_auprc'):>14}"
              f"{mm('disp_auroc'):>13}{mm('gate_cos'):>11}"
              f"{mm('eff_experts'):>10}")

    print("\nHow to read the gate columns:")
    print("  cos is the mean pairwise cosine between the nine per-task gate")
    print("  rows. 1.000 means every target routes identically, which is what")
    print("  the published checkpoints show. Well below 1 means the targets")
    print("  really do use different experts.")
    print("  eff is the effective number of experts; 13 means uniform.")
    print(f"\nwritten to {out}")
    if failed:
        print("\nFAILED:", failed)
    print(f"total {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
