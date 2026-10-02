"""The user-facing model.

    from multidiseasepred import MultiDiseasePred

    model = MultiDiseasePred(variables=cols, targets=outcomes)
    model.select_variables(X, y, vote=range(7), k=12)
    model.fit(X, y)
    model.evaluate(X_test, y_test)
    model.save("mymodel")
    model.write_dashboard("myapp")

Nothing here is specific to emergency medicine or to the nine endpoints in the
paper. `targets` is whatever set of non-mutually-exclusive outcomes you have, and
the dashboard is generated from your names and your ranges.

Design choices you will notice
------------------------------
`fit` holds out a slice of the training data for the calibrator and for choosing
the checkpoint, and never touches the data you later evaluate on. It is a keyword
argument so you can point it at your own split, but it defaults to something
honest rather than to "calibrate on whatever you pass in".

Seeds are averaged, not picked. `fit` trains `n_seeds` models and averages their
logits, because a single run of a network this size moves by a few thousandths
between initialisations and you should not have to wonder whether a result is the
model or the seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .calibration import Calibrator
from .metrics import calibration_metrics, evaluate, operating_point
from .model import build_network, expected_gradients, gate_usage
from .selection import greedy_forward

__all__ = ["MultiDiseasePred"]


class MultiDiseasePred:
    """Multi-task risk model over a shared pool of experts.

    Parameters
    ----------
    variables
        Column names of X, in order.
    targets
        Column names of y, in order. These are the outcomes you predict; they
        may co-occur.
    groups
        Optional mapping from a label to target names that should be
        macro-averaged together. Outcomes that are different kinds of thing --
        "does this patient have X" versus "will this patient be admitted" --
        should not share an average, and this is where you say so.
    """

    def __init__(self, variables: Sequence[str], targets: Sequence[str], *,
                 groups: Optional[Mapping[str, Sequence[str]]] = None,
                 hidden: int = 256, n_experts: int = 13,
                 temperature: float = 0.2, seed: int = 42):
        self.variables = list(variables)
        self.targets = list(targets)
        self.groups = {k: list(v) for k, v in (groups or {}).items()}
        self.hp = dict(hidden=hidden, n_experts=n_experts,
                       temperature=temperature)
        self.seed = seed
        self.selected_: Optional[list[str]] = None
        self.selection_: Optional[object] = None
        self.mean_ = self.scale_ = None
        self.networks_: list = []
        self.calibrator_: Optional[Calibrator] = None

    # ---------------------------------------------------------------- utils
    @property
    def active_variables(self) -> list[str]:
        return self.selected_ or self.variables

    def _cols(self) -> list[int]:
        return [self.variables.index(v) for v in self.active_variables]

    def _prep(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] == len(self.active_variables):
            pass
        elif X.shape[1] == len(self.variables):
            X = X[:, self._cols()]
        else:
            raise ValueError(
                f"X has {X.shape[1]} columns; expected "
                f"{len(self.active_variables)} (the selected variables) or "
                f"{len(self.variables)} (all candidates)")
        return (X - self.mean_) / self.scale_

    def _group_idx(self) -> dict:
        if not self.groups:
            return {"all": list(range(len(self.targets)))}
        return {k: [self.targets.index(t) for t in v]
                for k, v in self.groups.items()}

    # ------------------------------------------------------------ selection
    def select_variables(self, X, y, *, vote: Sequence, k: Optional[int] = None,
                         max_k: int = 15, threshold: float = 0.98,
                         val_frac: float = 0.2, sample: Optional[int] = 400_000,
                         verbose: bool = True):
        """Choose a variable budget by greedy forward selection.

        `vote` names (or indexes) the targets the selection is scored against.
        There is no default: a variable set is only optimal for the thing it was
        chosen for, and that choice belongs in your methods section, not in a
        library's defaults.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        vote = [self.targets.index(v) if isinstance(v, str) else int(v)
                for v in vote]
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(X))
        n_val = int(round(val_frac * len(X)))
        va, tr = perm[:n_val], perm[n_val:]
        res = greedy_forward(
            X[tr], y[tr], X[va], y[va], targets=vote,
            variable_names=self.variables, max_k=max_k, threshold=threshold,
            sample=sample, seed=self.seed, verbose=verbose)
        if k is not None:
            res.k = k
        self.selection_ = res
        self.selected_ = res.selected_names
        return res

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, *, val_frac: float = 0.2, epochs: int = 10,
            batch_size: int = 256, lr: float = 1e-3, n_seeds: int = 3,
            verbose: bool = True):
        """Train, then fit the calibrator on the held-out slice.

        The held-out slice is used for two things and nothing else: choosing the
        checkpoint and fitting the calibrator. It is never scored as a result.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        cols = self._cols()
        Xs = X[:, cols] if X.shape[1] == len(self.variables) else X
        self.mean_ = Xs.mean(0)
        self.scale_ = np.where(Xs.std(0) == 0, 1.0, Xs.std(0))
        Z = (Xs - self.mean_) / self.scale_

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(Z))
        n_val = int(round(val_frac * len(Z)))
        vi, ti = perm[:n_val], perm[n_val:]

        self.networks_ = []
        for s in range(n_seeds):
            torch.manual_seed(self.seed + s)
            net = build_network(Z.shape[1], len(self.targets), **self.hp)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max", factor=0.5, patience=2)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            dl = DataLoader(
                TensorDataset(torch.as_tensor(Z[ti]), torch.as_tensor(y[ti])),
                batch_size=batch_size, shuffle=True)
            best, best_state = -np.inf, None
            for ep in range(epochs):
                net.train()
                for xb, yb in dl:
                    opt.zero_grad()
                    loss = loss_fn(net(xb), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vp = torch.sigmoid(net(torch.as_tensor(Z[vi]))).numpy()
                score = _mean_auroc(y[vi], vp)
                sched.step(score)
                if score > best:
                    best = score
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                if verbose:
                    print(f"  seed {self.seed + s}  epoch {ep + 1}/{epochs}  "
                          f"val mean AUROC {score:.4f}")
            net.load_state_dict(best_state)
            self.networks_.append(net)

        raw_val = self._raw_probs(Z[vi])
        self.calibrator_ = Calibrator().fit(raw_val, y[vi].astype(int))
        if verbose:
            shift = self.calibrator_.ranking_shift(raw_val, y[vi].astype(int))
            worst = np.nanmin(shift) if len(shift) else 0.0
            print(f"calibrated on {len(vi):,} held-out rows; largest AUROC "
                  f"change from calibration ties: {worst:+.5f}")
        return self

    def _raw_probs(self, Z: np.ndarray) -> np.ndarray:
        import torch
        outs = []
        for net in self.networks_:
            net.eval()
            with torch.no_grad():
                outs.append(net(torch.as_tensor(np.asarray(Z, np.float32))).numpy())
        return 1 / (1 + np.exp(-np.mean(outs, axis=0)))

    # ------------------------------------------------------------ inference
    def predict(self, X, *, calibrated: bool = True) -> np.ndarray:
        if not self.networks_:
            raise RuntimeError("model is not fitted")
        p = self._raw_probs(self._prep(X))
        return self.calibrator_.transform(p) if (calibrated and self.calibrator_) else p

    def predict_lift(self, X, prevalence: Optional[Sequence[float]] = None):
        """Predicted risk relative to each target's base rate, as a log ratio.

        A 2% risk for an endpoint that occurs in 0.4% of encounters is a very
        different message from a 2% risk for one that occurs in 20%, and an
        absolute probability alone does not carry that. This is a presentation
        aid; the calibrated probability remains the primary output.
        """
        p = self.predict(X)
        base = np.asarray(prevalence if prevalence is not None
                          else self.prevalence_, dtype=float)
        return np.log(np.clip(p, 1e-9, 1) / np.clip(base, 1e-9, 1))

    def explain(self, x, background, *, n_samples: int = 64) -> np.ndarray:
        """Expected-gradient attribution for one row: (n_targets, n_variables)."""
        z = self._prep(np.asarray(x).reshape(1, -1))[0]
        bg = self._prep(background)
        return np.mean([expected_gradients(net, z, bg, n_samples=n_samples,
                                           seed=self.seed + i)
                        for i, net in enumerate(self.networks_)], axis=0)

    def gate_usage(self, X):
        """Measure how differently the targets weight the experts."""
        return gate_usage(self.networks_[0], self._prep(X))

    # ----------------------------------------------------------- evaluation
    def evaluate(self, X, y, *, n_boot: int = 1000, sensitivity: float = 0.90):
        y = np.asarray(y)
        p = self.predict(X)
        res = evaluate(y, p, target_names=self.targets,
                       groups=self._group_idx(), n_boot=n_boot, seed=self.seed)
        res["calibration"] = {
            n: calibration_metrics(y[:, t], p[:, t])
            for t, n in enumerate(self.targets)}
        res["operating_point"] = {
            n: operating_point(y[:, t], p[:, t], sensitivity)
            for t, n in enumerate(self.targets)}
        return res

    # ------------------------------------------------------ persistence, IO
    def save(self, path):
        import torch
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        meta = {"variables": self.variables, "targets": self.targets,
                "groups": self.groups, "hp": self.hp, "seed": self.seed,
                "selected": self.selected_,
                "mean": np.asarray(self.mean_).tolist(),
                "scale": np.asarray(self.scale_).tolist(),
                "prevalence": list(getattr(self, "prevalence_", [])),
                "n_networks": len(self.networks_)}
        (p / "model.json").write_text(json.dumps(meta, indent=2))
        for i, net in enumerate(self.networks_):
            torch.save(net.state_dict(), p / f"network_{i}.pt")
        if self.calibrator_ is not None:
            import joblib
            joblib.dump(self.calibrator_, p / "calibrator.joblib")
        return p

    @classmethod
    def load(cls, path):
        import torch
        p = Path(path)
        meta = json.loads((p / "model.json").read_text())
        m = cls(meta["variables"], meta["targets"], groups=meta["groups"],
                seed=meta["seed"], **meta["hp"])
        m.selected_ = meta["selected"]
        m.mean_ = np.asarray(meta["mean"], dtype=np.float32)
        m.scale_ = np.asarray(meta["scale"], dtype=np.float32)
        if meta.get("prevalence"):
            m.prevalence_ = np.asarray(meta["prevalence"], dtype=float)
        for i in range(meta["n_networks"]):
            net = build_network(len(m.active_variables), len(m.targets), **m.hp)
            net.load_state_dict(torch.load(p / f"network_{i}.pt",
                                           map_location="cpu"))
            net.eval()
            m.networks_.append(net)
        cal = p / "calibrator.joblib"
        if cal.exists():
            import joblib
            m.calibrator_ = joblib.load(cal)
        return m

    def write_dashboard(self, path, **kwargs):
        """Generate a Streamlit app for *this* model's variables and targets."""
        from .dashboard import write_dashboard
        return write_dashboard(self, path, **kwargs)

    def set_prevalence(self, y):
        """Record base rates so risks can be shown as lift over them."""
        self.prevalence_ = np.asarray(y).mean(0)
        return self

    def __repr__(self):
        state = "fitted" if self.networks_ else "not fitted"
        return (f"MultiDiseasePred({len(self.active_variables)} variables, "
                f"{len(self.targets)} targets, {state})")


def _mean_auroc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    vals = []
    for t in range(y.shape[1]):
        if y[:, t].min() != y[:, t].max():
            vals.append(roc_auc_score(y[:, t], p[:, t]))
    return float(np.mean(vals)) if vals else float("nan")
