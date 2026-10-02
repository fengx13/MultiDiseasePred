"""The network: a shared pool of experts, one gate per target.

Torch is imported inside the functions that need it, so `import
multidiseasepred` works in an environment that only has scikit-learn. Variable
selection, calibration and evaluation do not need a GPU stack to be useful.

The architecture
----------------
Every target draws on the same pool of feedforward experts, but each has its own
weighting over that pool:

    p_target = 0.5 * softmax(S / T) + 0.5 * softmax(A / T)   per target

`S` and `A` are learned parameter matrices. The gate does not depend on the
input: a target applies the same weighting of experts to every encounter. The
training objective is plain binary cross-entropy summed over targets, with no
auxiliary penalty on the gate.

Three mechanisms were evaluated during development and deliberately left out:
an input-dependent shared router mixed into the per-target gates (with a
symmetric-KL term tying the two), an L1 penalty on the gate logits, and expert
dropout. Each pulls the gate toward uniform; with all three in place the nine
per-target gates collapsed onto almost the same weighting (mean pairwise cosine
1.00). Removing them left discrimination unchanged and let the gates
differentiate (cosine 0.74). The full comparison is eTable 18 of the paper.

The gate is dense. Every expert receives non-zero weight for every target, and
there is no top-k anywhere in the forward pass. What differs between targets is
the weighting, not the membership -- `gate_usage` measures how much it differs,
so the claim can be checked rather than asserted.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:                                  # pragma: no cover
        raise ImportError(
            "This step needs PyTorch. Install it with `pip install torch`, or "
            "use the parts of multidiseasepred that do not require it "
            "(selection, calibration, metrics)."
        ) from e


def build_network(n_features: int, n_targets: int, *, hidden: int = 256,
                  n_experts: int = 13, temperature: float = 0.2):
    """Construct the model. Returns a torch.nn.Module."""
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU())

        def forward(self, x):
            return self.net(x)

    class GatedMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_experts, self.n_targets = n_experts, n_targets
            self.temperature = temperature
            self.experts = nn.ModuleList([Expert() for _ in range(n_experts)])
            # Zero init is deliberate: the per-target gradients differ from the
            # first step, so the rows separate on their own without a random
            # head start that would be one more thing to seed.
            self.S = nn.Parameter(torch.zeros(n_targets, n_experts))
            self.A = nn.Parameter(torch.zeros(n_targets, n_experts))
            self.heads = nn.ModuleList(
                [nn.Linear(hidden, 1) for _ in range(n_targets)])

        def gate(self, x):
            """(B, T, E) mixing weights. Identical for every row of x: the gate
            is a property of the target, not of the encounter."""
            T = self.temperature
            p = 0.5 * F.softmax(self.S / T, -1) + 0.5 * F.softmax(self.A / T, -1)
            p = p / (p.sum(-1, keepdim=True) + 1e-8)
            return p.unsqueeze(0).expand(x.shape[0], -1, -1)

        def forward(self, x):
            outs = torch.stack([e(x) for e in self.experts], dim=1)   # B,E,H
            p = self.gate(x)                                          # B,T,E
            reps = torch.einsum("bte,beh->bth", p, outs)              # B,T,H
            return torch.cat([self.heads[t](reps[:, t])
                              for t in range(self.n_targets)], dim=1)

    return GatedMoE()


def gate_usage(model, X: np.ndarray) -> dict:
    """How differently the targets actually weight the experts.

    "Task-adaptive" is a claim about behaviour, not about architecture, so it
    should be measured. Effective experts is the exponential of the entropy of a
    target's gate row: n_experts means perfectly even, 1 means everything on one
    expert. Cosine similarity between rows says whether two targets ended up
    wanting the same mixture.
    """
    torch = _torch()
    model.eval()
    with torch.no_grad():
        p = model.gate(torch.as_tensor(np.asarray(X, dtype=np.float32)))
        p = p.mean(0).cpu().numpy()                                   # T,E
    ent = -(p * np.log(p + 1e-12)).sum(-1)
    eff = np.exp(ent)
    norm = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-12)
    cos = norm @ norm.T
    off = cos[~np.eye(len(cos), dtype=bool)]
    return {"weights": p, "effective_experts": eff,
            "largest_weight": p.max(-1),
            "mean_pairwise_cosine": float(off.mean()),
            "n_experts": p.shape[1]}


def expected_gradients(model, x: np.ndarray, background: np.ndarray, *,
                       n_samples: int = 64, seed: int = 0) -> np.ndarray:
    """Attribution by expected gradients: (n_targets, n_features) for one row.

    A sampling estimator of SHAP values. Gradients are taken on the logit rather
    than the probability, so a contribution does not shrink just because the
    predicted risk is near zero -- which is exactly the regime these models spend
    most of their time in.
    """
    torch = _torch()
    model.eval()
    x = np.asarray(x, dtype=np.float32).reshape(1, -1)
    bg = np.asarray(background, dtype=np.float32)
    rng = np.random.default_rng(seed)
    n_targets = model.n_targets
    total = np.zeros((n_targets, x.shape[1]))
    for _ in range(n_samples):
        ref = bg[rng.integers(0, len(bg))].reshape(1, -1)
        a = float(rng.random())
        pt = torch.tensor(ref + a * (x - ref), dtype=torch.float32, requires_grad=True)
        logits = model(pt)
        for t in range(n_targets):
            g, = torch.autograd.grad(logits[0, t], pt, retain_graph=True)
            total[t] += (g.detach().numpy()[0] * (x - ref)[0])
    return total / n_samples
