"""
selftest_models.py -- thirty seconds of checks before committing a night to
the sweep.

The batch job runs this first and exits if anything fails, so a shape error in
one arm is discovered immediately instead of at three in the morning after the
other arms have already burned their GPU hours.

It uses random data, not the cohort. The point is that every arm builds, runs
forward and backward, produces finite gradients, and reports a gate matrix
that is a proper distribution. Six specific things:

1. every arm accepts (B, D) and returns (B, T) logits
2. loss.backward() reaches every parameter, so nothing is silently detached
3. gate rows sum to one
4. the top-k arms really do keep exactly k experts, in training and at
   evaluation, and the targets do not all pick the same k -- the published
   model never did any of this, which is why the manuscript's claim of sparse
   routing did not hold
5. A1 is genuinely independent: perturbing target 0's subnetwork must not move
   any other target's output
6. the same seed twice gives bitwise identical weights

Check 4 earned its keep on the first run: it reported thirteen active experts
for all three top-k arms. The masking was written as "keep everything greater
than or equal to the k-th largest value", and since the gate logits start at
zero every expert had weight exactly 1/13, so the comparison kept all of them.
The test now calls route(), the same method forward() calls, rather than
recomputing the masking -- a test that reimplements the thing it is testing
will agree with a wrong implementation.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from determinism import seed_everything  # noqa
import models as M  # noqa

B, D, T, E = 64, 15, 9, 13
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'pass' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def main():
    seed_everything(0, verbose=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}   torch {torch.__version__}\n")
    x = torch.randn(B, D, device=dev)
    y = (torch.rand(B, T, device=dev) < 0.1).float()
    crit = torch.nn.BCEWithLogitsLoss()

    for arm in M.ARMS:
        print(arm)
        m = M.build(arm, D, n_tasks=T, n_experts=E).to(dev)
        m.train()
        out = m(x)
        check("shape", tuple(out.shape) == (B, T), f"{tuple(out.shape)}")

        loss = crit(out, y) + m.aux_loss()
        loss.backward()
        no_grad = [n for n, p in m.named_parameters()
                   if p.requires_grad and p.grad is None]
        nan = [n for n, p in m.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        check("gradients reach every parameter", not no_grad,
              f"{len(no_grad)} missing" + (f" {no_grad[:3]}" if no_grad else ""))
        check("gradients finite", not nan, f"{len(nan)} bad")

        m.eval()
        g = m.gate_matrix()
        if g is None:
            check("no gate (expected for A1/B)", arm in ("A1_independent",
                                                         "B_shared"))
        else:
            check("gate shape", g.shape == (T, E), str(g.shape))
            check("gate rows sum to 1",
                  bool(np.allclose(g.sum(1), 1, atol=1e-5)),
                  f"max dev {abs(g.sum(1) - 1).max():.2e}")

        # -- top-k really is sparse -------------------------------------
        # Call route(), the same code path forward() uses. The first version
        # of this test recomputed the masking itself and so agreed with a
        # wrong implementation.
        if arm.startswith("F"):
            k = M.ARMS[arm]["topk"]
            with torch.no_grad():
                p_eval = m.route(x)                 # m is in eval() here
                m.train()
                p_train = m.route(x)
                m.eval()
            nz_e = (p_eval > 0).sum(-1).float()
            nz_t = (p_train > 0).sum(-1).float()
            check(f"exactly {k} experts active at eval",
                  bool((nz_e == k).all()),
                  f"min {nz_e.min():.0f} max {nz_e.max():.0f}")
            check(f"exactly {k} experts active in training too",
                  bool((nz_t == k).all()),
                  f"min {nz_t.min():.0f} max {nz_t.max():.0f}")
            check("sparse rows still sum to 1",
                  bool(torch.allclose(p_eval.sum(-1),
                                      torch.ones_like(p_eval.sum(-1)),
                                      atol=1e-5)))
            # the selection must not be identical for every target, or
            # "sparse routing" is just a smaller dense model
            sets = {tuple(sorted(torch.nonzero(p_eval[0, t]).flatten().tolist()))
                    for t in range(T)}
            check("targets do not all select the same experts",
                  len(sets) > 1, f"{len(sets)} distinct expert sets over {T} targets")
        print()

    # -- A1 independence ------------------------------------------------
    print("cross-arm checks")
    m = M.build("A1_independent", D, n_tasks=T).to(dev).eval()
    with torch.no_grad():
        base = m(x).clone()
        for p in m.nets[0].parameters():
            p.add_(0.5)
        after = m(x)
    moved = (after - base).abs().amax(0)
    check("A1: perturbing target 0 moves only target 0",
          bool(moved[0] > 1e-6 and moved[1:].max() < 1e-9),
          f"target0 {moved[0]:.3f}  others max {moved[1:].max():.2e}")

    # -- same seed, same weights ----------------------------------------
    def weights(seed):
        seed_everything(seed, verbose=False)
        mm = M.build("D_moe_router", D, n_tasks=T, n_experts=E)
        return torch.cat([p.flatten() for p in mm.parameters()])

    a, b, c = weights(7), weights(7), weights(8)
    check("same seed -> identical initialisation",
          bool(torch.equal(a, b)))
    check("different seed -> different initialisation",
          not bool(torch.equal(a, c)))

    # -- the G arms must not silently inherit the default penalties -----
    # build() takes the training script's hyperparameters as defaults and lets
    # the arm override them. With the precedence the other way round, G1 would
    # get l1_gate = 1e-4 back and be an exact duplicate of E, and the sweep
    # would answer the wrong question while looking like it worked.
    g1 = M.build("G1_no_l1", D, n_tasks=T, n_experts=E,
                 l1_gate=1e-4, expert_dropout=0.1).to(dev)
    g2 = M.build("G2_no_l1_no_drop", D, n_tasks=T, n_experts=E,
                 l1_gate=1e-4, expert_dropout=0.1).to(dev)
    check("G1 ignores the default l1_gate", g1.l1_gate == 0.0,
          f"l1_gate {g1.l1_gate}")
    check("G1 keeps expert dropout", g1.p_drop == 0.1, f"p_drop {g1.p_drop}")
    check("G2 drops both", g2.l1_gate == 0.0 and g2.p_drop == 0.0,
          f"l1_gate {g2.l1_gate}  p_drop {g2.p_drop}")
    # The gate logits start at zero, and the L1 of zero is zero, so comparing
    # aux_loss at initialisation would pass for both arms and prove nothing.
    # Give the gates something to penalise first.
    e_arm = M.build("E_moe_norouter", D, n_tasks=T, n_experts=E,
                    l1_gate=1e-4).to(dev)
    for m_ in (g1, e_arm):
        with torch.no_grad():
            m_.S.normal_(std=0.5)
            m_.A.normal_(std=0.5)
    check("G1 has no L1 term once the gate is non-zero",
          float(g1.aux_loss()) == 0.0, f"{float(g1.aux_loss()):.3e}")
    check("E does have one, so G1 is a real contrast",
          float(e_arm.aux_loss()) > 0, f"{float(e_arm.aux_loss()):.3e}")

    # -- arm D and arm E differ only by the router ----------------------
    seed_everything(1, verbose=False)
    md = M.build("D_moe_router", D, n_tasks=T, n_experts=E)
    seed_everything(1, verbose=False)
    me = M.build("E_moe_norouter", D, n_tasks=T, n_experts=E)
    only_d = set(dict(md.named_parameters())) - set(dict(me.named_parameters()))
    check("D minus E is exactly the shared router",
          only_d == {"shared_router.weight", "shared_router.bias"},
          str(sorted(only_d)))

    print()
    if FAIL:
        print(f"SELFTEST FAILED: {FAIL}")
        sys.exit(1)
    print("SELFTEST PASSED -- all arms build, train and route as intended")


if __name__ == "__main__":
    main()
