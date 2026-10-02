"""
models.py -- the nine architectures compared in stage 3.

Every arm shares the same expert body (two layers, width 256, ReLU), the same
loss, and the same optimiser. Only how information is shared across the nine
targets differs, so a difference in performance is attributable to sharing and
not to capacity or tuning.

  A1  independent   nine separate MLPs, no shared parameters at all.
                    Mathematically identical to training nine single-task
                    models: the parameters are disjoint and the loss is a sum,
                    so the gradients never mix. Doing it in one run just saves
                    bookkeeping.
  B   shared        one MLP body, nine linear heads. No mixture of experts.
  C   mmoe          E experts, per-task gate that depends on the input.
                    Ma et al., KDD 2018.
  D   moe_router    the published model: per-task gate parameters that do not
                    depend on the input, mixed 9:1 with an input-dependent
                    shared router, plus a symmetric-KL penalty pulling the two
                    together. This is what produced best_multidiseasepred.pt.
  E   moe_norouter  D with the shared router and the KL penalty removed.
  F   moe_topk_k    E plus a hard top-k over the gate.
  G   no_l1 ...     E with the L1 penalty on the gate logits removed, and a
                    second version with expert dropout removed as well.

Why G exists, and why E was not enough.

The trained checkpoints all show per-task gates that are near-uniform and
almost identical across targets: mean pairwise cosine 1.0000, effective
experts 12.98 of 13. The earlier best_dselectk checkpoints, which have no
shared router and therefore no KL term, showed cosine 0.749, so the KL penalty
looked like the cause. E was the arm that tested that, and it refuted it: over
three seeds E came back at cosine 1.0000 with 12.9998 effective experts, more
uniform than D at 12.94. The KL is not what collapses the gate.

The same sweep supplied the next suspect. C_mmoe did differentiate, at cosine
0.523 and 8.48 effective experts, and C differs from E in two ways: its gate
is a Linear layer rather than a pair of parameter matrices, and -- because
aux_loss skips the L1 term whenever input_gate is set -- it carries no L1
penalty. D and E both carry l1_gate = 1e-4 on the gate logits and both
collapse. An L1 penalty pulls every logit toward zero, and softmax of equal
logits is exactly uniform; the gradient that BCE sends back to a gate logit is
weak by comparison, especially early on when all thirteen experts still
compute much the same thing.

G1 removes the L1 and changes nothing else. G2 also removes expert dropout,
which is the other mechanism that rewards a flat gate: if you lean on one
expert and dropout removes it, you pay for it. Between them the two arms say
which mechanism is responsible, or whether it takes both.

What the answer decides. If the gate differentiates and accuracy holds, the
model really can route per target and "task-adaptive" is defensible -- with
the honest footnote that the published configuration did not do it. If it
differentiates and accuracy drops, then the shared representation is where the
performance comes from, and that is worth saying plainly instead of claiming
a routing behaviour the numbers do not support.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp_body(d_in, hidden):
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
    )


class Independent(nn.Module):
    """A1. Nine disjoint networks."""

    def __init__(self, d_in, hidden=256, n_tasks=9, **_):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Sequential(mlp_body(d_in, hidden), nn.Linear(hidden, 1))
            for _ in range(n_tasks)])

    def forward(self, x):
        return torch.cat([n(x) for n in self.nets], dim=1)

    def aux_loss(self):
        return x_zero(self)

    def gate_matrix(self):
        return None


class SharedBottom(nn.Module):
    """B. One body, nine heads."""

    def __init__(self, d_in, hidden=256, n_tasks=9, **_):
        super().__init__()
        self.body = mlp_body(d_in, hidden)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1)
                                    for _ in range(n_tasks)])

    def forward(self, x):
        h = self.body(x)
        return torch.cat([hd(h) for hd in self.heads], dim=1)

    def aux_loss(self):
        return x_zero(self)

    def gate_matrix(self):
        return None


class MoE(nn.Module):
    """C, D, E and F, selected by the flags.

    router  True  -> mix the per-task gate with an input-dependent shared
                     router and add the symmetric-KL penalty  (arm D)
    input_gate True -> the per-task gate is itself a function of the input,
                     which is what makes it MMoE                (arm C)
    topk    int   -> keep only the k largest gate weights       (arm F)
    """

    def __init__(self, d_in, hidden=256, n_experts=13, n_tasks=9,
                 temperature=0.2, router=True, input_gate=False, topk=None,
                 alpha_shared=0.1, lambda_kl=0.05, expert_dropout=0.1,
                 l1_gate=1e-4, gate_init_std=0.0, topk_noise=0.02, **_):
        super().__init__()
        self.n_experts, self.n_tasks = n_experts, n_tasks
        self.temperature, self.topk = temperature, topk
        self.router_on, self.input_gate = router, input_gate
        self.alpha, self.lambda_kl = alpha_shared, lambda_kl
        self.p_drop, self.l1_gate = expert_dropout, l1_gate
        self.topk_noise = topk_noise

        self.experts = nn.ModuleList([mlp_body(d_in, hidden)
                                      for _ in range(n_experts)])
        self.towers = nn.ModuleList([nn.Linear(hidden, 1)
                                     for _ in range(n_tasks)])

        if input_gate:
            self.gate = nn.Linear(d_in, n_tasks * n_experts)
        else:
            # the published parameterisation: two logit matrices, averaged.
            # Zero is what the original used, and for the dense arms it is
            # harmless -- the per-task gradients differ from the first step,
            # so the rows separate on their own. For the top-k arms it is not
            # harmless, which is why they pass gate_init_std: see below.
            self.S = nn.Parameter(torch.zeros(n_tasks, n_experts))
            self.A = nn.Parameter(torch.zeros(n_tasks, n_experts))
            if gate_init_std > 0:
                nn.init.normal_(self.S, std=gate_init_std)
                nn.init.normal_(self.A, std=gate_init_std)
        if router:
            self.shared_router = nn.Linear(d_in, n_experts)

        self._last_kl = None

    # -- gate ------------------------------------------------------------
    def _p_task(self, x):
        T = self.temperature
        if self.input_gate:
            g = self.gate(x).view(-1, self.n_tasks, self.n_experts) / T
            return F.softmax(g, dim=-1)                 # (B, T, E)
        p = 0.5 * F.softmax(self.S / T, -1) + 0.5 * F.softmax(self.A / T, -1)
        p = p / (p.sum(-1, keepdim=True) + 1e-8)
        return p.unsqueeze(0).expand(x.shape[0], -1, -1)

    def route(self, x):
        """The (B, T, E) mixing weights actually used. Separated out from
        forward so the self test can check the routing without having to
        reimplement it -- reimplementing it in the test is how the top-k bug
        stayed hidden the first time."""
        p = self._p_task(x)                                        # (B,T,E)

        if self.router_on:
            ps = F.softmax(self.shared_router(x) / self.temperature, -1)
            if self.training and self.p_drop > 0:
                mask = (torch.rand(self.n_experts, device=x.device)
                        > self.p_drop).float() / (1 - self.p_drop)
                ps = ps * mask
                ps = ps / (ps.sum(-1, keepdim=True) + 1e-8)
            ps_b = ps.unsqueeze(1).expand(-1, self.n_tasks, -1)
            self._last_kl = _sym_kl(p, ps_b).mean()
            p = (1 - self.alpha) * p + self.alpha * ps_b
            p = p / (p.sum(-1, keepdim=True) + 1e-8)
        else:
            self._last_kl = None
            if self.training and self.p_drop > 0:
                mask = (torch.rand(self.n_experts, device=x.device)
                        > self.p_drop).float() / (1 - self.p_drop)
                p = p * mask
                p = p / (p.sum(-1, keepdim=True) + 1e-8)

        if self.topk is not None and self.topk < self.n_experts:
            # Hard sparsity: everything outside the top k is exactly zero,
            # which is the thing the published model never actually did.
            #
            # Select by index and scatter, not by comparing against the k-th
            # value. A threshold test is wrong whenever there are ties, and
            # ties are not an edge case here: the gate logits start at zero,
            # so at initialisation all thirteen weights are exactly 1/13 and
            # "p >= kth" keeps every one of them. That is the bug the self
            # test caught -- it reported thirteen active experts for all
            # three top-k arms.
            #
            # Noise on the selection, following Shazeer 2017. Without it a
            # static gate is stuck: the k experts that happen to win at
            # initialisation are the only ones that ever receive gradient,
            # the other nine never update, and "sparse routing" degenerates
            # into a smaller dense model with an arbitrary choice of experts.
            # The noise is on selection only -- the mixing weights are the
            # clean probabilities -- and it is off at evaluation, so the
            # reported routing is the learned one.
            sel = p
            if self.training and self.topk_noise > 0:
                sel = p + torch.randn_like(p) * self.topk_noise
            # An expert that expert-dropout has already removed must not be
            # selectable, or the noise could pick one and the row would come
            # back with k-1 active experts instead of k.
            sel = torch.where(p > 0, sel, torch.full_like(sel, -1e9))
            idx = sel.topk(self.topk, dim=-1).indices
            mask = torch.zeros_like(p).scatter_(-1, idx, 1.0)
            p = p * mask
            p = p / (p.sum(-1, keepdim=True) + 1e-8)

        return p

    def forward(self, x):
        outs = torch.stack([e(x) for e in self.experts], dim=1)   # (B,E,H)
        p = self.route(x)                                         # (B,T,E)
        reps = torch.einsum("bte,beh->bth", p, outs)              # (B,T,H)
        return torch.cat([self.towers[t](reps[:, t]) for t in
                          range(self.n_tasks)], dim=1)

    # -- extras ----------------------------------------------------------
    def aux_loss(self):
        loss = x_zero(self)
        if not self.input_gate:
            loss = loss + self.l1_gate * (self.S.abs().sum() + self.A.abs().sum())
        if self.router_on and self._last_kl is not None:
            loss = loss + self.lambda_kl * self._last_kl
        return loss

    @torch.no_grad()
    def gate_matrix(self):
        """The (T, E) routing, for the differentiation summary.

        This goes through route() rather than recomputing the gate, so that
        for the top-k arms the reported matrix is the sparse one that the
        model actually uses. For arms whose gate depends on the input (C, and
        D through its router) this evaluates at the origin, which after
        standardisation is the mean patient -- indicative, not the whole
        story, but the same convention for every arm."""
        was_training = self.training
        self.eval()
        dev = next(self.parameters()).device
        d_in = self.experts[0][0].in_features
        p = self.route(torch.zeros(1, d_in, device=dev))[0]
        if was_training:
            self.train()
        return p.cpu().numpy()


def _sym_kl(p, q, eps=1e-8):
    p, q = p.clamp_min(eps), q.clamp_min(eps)
    return ((p * (p / q).log()).sum(-1) + (q * (q / p).log()).sum(-1)) * 0.5


def x_zero(m):
    return torch.zeros((), device=next(m.parameters()).device)


# ---------------------------------------------------------------------------
ARMS = {
    "A1_independent":  dict(cls="Independent"),
    "B_shared":        dict(cls="SharedBottom"),
    "C_mmoe":          dict(cls="MoE", router=False, input_gate=True),
    "D_moe_router":    dict(cls="MoE", router=True,  input_gate=False),
    "E_moe_norouter":  dict(cls="MoE", router=False, input_gate=False),
    # The top-k arms break the gate symmetry at initialisation. With zero
    # logits every task would select the same k experts by index order, which
    # is not a fair test of sparse routing.
    "F1_topk3":        dict(cls="MoE", router=False, input_gate=False, topk=3,
                            gate_init_std=0.01),
    "F2_topk5":        dict(cls="MoE", router=False, input_gate=False, topk=5,
                            gate_init_std=0.01),
    "F3_topk7":        dict(cls="MoE", router=False, input_gate=False, topk=7,
                            gate_init_std=0.01),
    # -- added after the first sweep, see the note at the top of the file --
    "G1_no_l1":        dict(cls="MoE", router=False, input_gate=False,
                            l1_gate=0.0),
    "G2_no_l1_no_drop": dict(cls="MoE", router=False, input_gate=False,
                             l1_gate=0.0, expert_dropout=0.0),
}

TREE_ARMS = ["A2_rf", "A2_histgb", "A2_xgb", "A2_xgb_sub"]


def build(arm, d_in, n_tasks=9, hidden=256, n_experts=13, **over):
    """Build one arm.

    Precedence: the caller's hyperparameters are defaults, and the arm's own
    entry in ARMS overrides them. It has to be this way round -- an entry in
    ARMS is a deliberate architectural choice that defines the arm, whereas
    the values coming from the training script are the shared settings every
    arm inherits. With the other precedence, G1_no_l1 would silently get the
    default l1_gate of 1e-4 back and would be an exact duplicate of E.
    """
    spec = dict(ARMS[arm])
    cls = spec.pop("cls")
    params = dict(over)
    params.update(spec)
    if cls == "Independent":
        return Independent(d_in, hidden=hidden, n_tasks=n_tasks)
    if cls == "SharedBottom":
        return SharedBottom(d_in, hidden=hidden, n_tasks=n_tasks)
    return MoE(d_in, hidden=hidden, n_experts=n_experts, n_tasks=n_tasks,
               **params)
