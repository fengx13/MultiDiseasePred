"""19_gate.py -- what the per-task gates actually learned.

    python 19_gate.py --arm G2_no_l1_no_drop --features greedy:12

Reads runs/<arm>_<tag>_noesi_seed<s>/gate_weights.csv, which 01_train.py has
been writing on every run since the beginning, and turns it into the handful
of numbers a figure needs. Nothing is recomputed and no model is loaded: the
matrices are already on disk.

What is being asked
-------------------
The architecture is a mixture of experts with one gate per task. The story a
reader will expect is that related endpoints -- sepsis and ARDS, say -- lean
on shared experts while unrelated ones do not, and that this is where the
multi-task advantage comes from. models.py already records the opposite in
its own docstring: the trained gates come out near-uniform, around 12.9 of
16 effective experts, and removing the KL penalty does not change it.

If that holds here then the mechanism in the paper's introduction is not the
mechanism in the trained model, and the honest version of the claim is that
the advantage comes from sharing the body under a variable budget, not from
experts specialising. That is a smaller claim and a true one. This script
exists to state it with numbers instead of adjectives.

Three quantities per arm
------------------------
    effective experts   exp of the entropy of a task's gate row, averaged
                        over tasks. Equals the number of experts when the
                        gate is uniform and 1 when a task uses one expert.

    cosine off-diagonal the mean and minimum cosine similarity between two
                        different tasks' gate rows. Near 1 means every task
                        uses the same mixture, which is a shared bottom
                        wearing a gate.

    seed spread         the same numbers across three seeds. A gate that
                        differentiates differently every seed differentiates
                        by accident.

Output is the emit format, so it travels with a check on every line.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
RUNS = f"{PROJ}/runs"
RES = f"{PROJ}/results"

TASKS = ["hospitalization", "critical", "sepsis", "pneumonia_viral", "ards",
         "pe", "copd_asthma", "acs_mi", "aki"]
CODE = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]


def check(s):
    """Same two-digit position-weighted check as 17_emit.py."""
    t = 0
    for i, ch in enumerate(s):
        if ch.isdigit():
            t += (i + 1) * int(ch)
    return f"{t % 97:02d}"


def q(v):
    """A value in [0, 1] as four digits. Used for cosines and shares."""
    if v is None or (isinstance(v, float) and v != v):
        return "----"
    return f"{int(round(min(max(float(v), 0.0), 0.9999) * 10000)):04d}"


def load(arm, tag, seed):
    p = f"{RUNS}/{arm}_{tag}_noesi_seed{seed}/gate_weights.csv"
    if not os.path.exists(p):
        return None
    g = np.loadtxt(p, delimiter=",")
    if g.ndim == 1:
        g = g[None, :]
    # 01_train.py writes the gate after a softmax, but a file that has been
    # sitting on disk for weeks is not a promise. Renormalising costs nothing
    # and a row that does not sum to one means the file is not what this
    # script thinks it is.
    s = g.sum(axis=1, keepdims=True)
    if np.abs(s - 1).max() > 1e-3:
        print(f"    {os.path.basename(os.path.dirname(p))}: rows sum to "
              f"{s.min():.4f}-{s.max():.4f}, renormalising")
        g = g / s
    return g


def summarise(g):
    nrm = g / np.linalg.norm(g, axis=1, keepdims=True)
    cos = nrm @ nrm.T
    off = cos[~np.eye(len(g), dtype=bool)]
    ent = -(g * np.log(g + 1e-12)).sum(-1)
    eff = np.exp(ent)
    return {"eff_mean": float(eff.mean()), "eff_min": float(eff.min()),
            "eff_max": float(eff.max()), "cos_mean": float(off.mean()),
            "cos_min": float(off.min()), "n_experts": int(g.shape[1]),
            "eff_per_task": eff, "top_share": g.max(axis=1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="G2_no_l1_no_drop")
    ap.add_argument("--features", default="greedy:12")
    ap.add_argument("--seeds", default="42,43,44")
    a = ap.parse_args()

    tag = a.features.replace(":", "")
    seeds = [s.strip() for s in a.seeds.split(",") if s.strip()]

    G, have = {}, []
    for s in seeds:
        g = load(a.arm, tag, s)
        if g is None:
            print(f"  seed {s}: no gate_weights.csv -- skipped")
            continue
        G[s], _ = g, have.append(s)
    if not have:
        raise SystemExit(
            f"\nno gate matrices under {RUNS}/{a.arm}_{tag}_noesi_seed*.\n"
            f"Either this arm has no gate -- the single-task and "
            f"shared-bottom arms\nreturn None from gate_matrix() -- or the "
            f"runs predate the file being written.")

    print(f"arm {a.arm}   features {a.features}   seeds {','.join(have)}")
    S = {s: summarise(G[s]) for s in have}
    E = S[have[0]]["n_experts"]
    print(f"{E} experts, {len(TASKS)} tasks\n")

    print(f"  {'seed':<6}{'eff experts':>13}{'min':>8}{'max':>8}"
          f"{'cos mean':>10}{'cos min':>9}")
    for s in have:
        d = S[s]
        print(f"  {s:<6}{d['eff_mean']:>13.2f}{d['eff_min']:>8.2f}"
              f"{d['eff_max']:>8.2f}{d['cos_mean']:>10.4f}{d['cos_min']:>9.4f}")
    em = np.mean([S[s]["eff_mean"] for s in have])
    cm = np.mean([S[s]["cos_mean"] for s in have])
    print(f"\n  averaged over seeds: {em:.2f} of {E} effective experts, "
          f"mean cross-task cosine {cm:.4f}")
    if em > 0.8 * E and cm > 0.95:
        print("  The gate is close to uniform and every task uses close to")
        print("  the same mixture. On these numbers the experts have not")
        print("  specialised, and the advantage is not coming from routing.")
    elif em < 0.5 * E:
        print("  The gate concentrates. Worth a figure of which experts go")
        print("  with which endpoint.")

    # ---- the emitted block ---------------------------------------------
    # One line per task, averaged over seeds: effective experts, the largest
    # single expert share, and the mean cosine to the other eight tasks.
    out = [f"gate {a.arm} {a.features}",
           f"experts {E}   seeds {','.join(have)}",
           "",
           "endpoint order: 01 hospitalization 02 critical 03 sepsis "
           "04 pneumonia_viral",
           "                05 ards 06 pe 07 copd_asthma 08 acs_mi 09 aki",
           "",
           "EFF is effective experts divided by the expert count, so 9999 "
           "means fully uniform.",
           "TOP is the largest single expert share. COS is the mean cosine "
           "to the other tasks.",
           "SD is the spread of EFF across seeds, on the same scale.",
           ""]
    head = f"{'':7}{'EFF':^6}{'TOP':^6}{'COS':^6}{'SD':^6}  CK"
    out += [head, "-" * len(head)]

    body = []
    for i, cd in enumerate(CODE):
        eff = np.array([S[s]["eff_per_task"][i] for s in have]) / E
        top = np.array([S[s]["top_share"][i] for s in have])
        cs = []
        for s in have:
            g = G[s]
            nrm = g / np.linalg.norm(g, axis=1, keepdims=True)
            c = nrm @ nrm[i]
            cs.append(float(np.delete(c, i).mean()))
        line = (f"GAT {cd} {q(eff.mean())} {q(top.mean())} "
                f"{q(np.mean(cs))} {q(eff.std())}")
        body.append(line + "  " + check(line))
    out += body
    out.append("-" * len(head))
    out.append(f"LINES {len(body)}   BLOCK {check(''.join(body))}")

    p = f"{RES}/emit_gate_{a.arm}_{tag}.txt"
    with open(p, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\nwrote {p}  ({len(out)} lines, one screen)")
    print(f"  cat {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
