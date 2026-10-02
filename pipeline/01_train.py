"""
01_train.py -- train one configuration and save everything it produced.

    python 01_train.py --arm D_moe_router --seed 42 --features top15

Hyperparameters default to exactly what cell 14 of Dselect_k_train_test.ipynb
used, so arm D is the published model rather than an approximation of it:
13 experts, hidden 256, temperature 0.2, alpha 0.1, KL 0.05, expert dropout
0.1, Adam at 1e-3, batch 256, gradient clip 3.0, ReduceLROnPlateau on mean
validation AUROC with factor 0.5 and patience 2, ten epochs, keeping the
checkpoint with the best mean validation AUROC.

One thing is deliberately different: the seed is set, for everything. The old
code defined SEED = 42 and never called torch.manual_seed, so weight
initialisation and dropout were unseeded and no run could be repeated.

Everything lands in runs/<arm>_<features>_seed<n>/. The probability matrices
matter most: with probs and labels saved, any metric, any confidence interval
and any curve can be recomputed later without touching the cluster again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from determinism import seed_everything, run_fingerprint  # noqa
import models as M  # noqa

PROJ = os.environ.get("MDP_ROOT",
                       os.path.expanduser("~/multidiseasepred"))
DATA = f"{PROJ}/data"
RUNS = f"{PROJ}/runs"

# The fifteen retained variables, in the order the published scaler stored
# them. Stage 3 holds the feature set fixed so that only the architecture
# varies; stage 4 recomputes the ranking and sweeps k.
TOP15 = ["triage_acuity", "n_ed_365d", "age", "triage_dbp", "triage_o2sat",
         "triage_resprate", "triage_sbp", "triage_heartrate", "cci_Pulmonary",
         "eci_FluidsLytes", "eci_HTN1", "cci_Renal", "gender",
         "chiefcom_abdominal_pain", "n_ed_30d"]

HP = dict(hidden=256, n_experts=13, temperature=0.2, alpha_shared=0.1,
          lambda_kl=0.05, expert_dropout=0.1, l1_gate=1e-4,
          lr=1e-3, batch=256, epochs=10, clip=3.0, val_frac=0.2)


GROUPS = {
    # Every cci_ and eci_ column.
    #
    # An earlier version of this comment asserted that the upstream lookback
    # window includes the index encounter, so that a diagnosis made at the
    # visit being predicted would set its own comorbidity flag. That is wrong.
    # The window ends before the index encounter; the provenance was checked
    # against the extraction and confirmed on 2026-08-11. Corrected here rather
    # than deleted, because a claim of label leakage that stays in the codebase
    # unchallenged will be believed by whoever reads it next.
    #
    # What produced the suspicion is a real measurement and still stands:
    # eci_PHTN alone is worth 0.025 macro AUROC over the seven acute condition
    # endpoints and nothing at all over the two severity endpoints, and the
    # neural model and gradient boosting agree on it to three decimal places.
    #
    # The innocent reading fits that pattern exactly as well as leakage does.
    # Pulmonary hypertension is a genuine risk factor for the cardiopulmonary
    # diagnoses -- PE, ARDS, COPD -- and carries little independent information
    # about whether a patient gets admitted, which is decided by physiology and
    # bed availability. A prior condition that predicts diagnoses and not
    # dispositions is what one would expect. The asymmetry alone cannot
    # separate the two explanations; only the extraction window can, and it has
    # been checked.
    #
    # The group is kept so that "what if the comorbidity history were
    # unavailable" remains a one-flag sensitivity analysis, which is worth
    # reporting whether or not anyone suspects leakage.
    "comorb": lambda cols: [c for c in cols if c.startswith(("cci_", "eci_"))],
}


def expand_drop(drop, cols):
    """Resolve group names in --drop, leaving column names alone."""
    out = set()
    for d in (drop or ()):
        out.update(GROUPS[d](cols) if d in GROUPS else [d])
    return out


def drop_suffix(drop):
    """The tag fragment that keeps a dropped run out of an existing directory.

    This exists because of a failure mode that does not announce itself. Run
    directories are named {arm}_{features}_seed{seed}. If a run that excludes
    triage_acuity produced the same name as one that included it, 01_train.py
    would find metrics.json already there, print "[skip] already done", and
    the with-ESI numbers would be read off as the without-ESI result. Nothing
    would error. Every dropped run therefore carries the exclusion in its name.

    The short names are sorted and joined with a hyphen so that the suffix is
    both stable -- the same exclusion always yields the same directory,
    whatever order it was typed in -- and readable. Concatenating without a
    separator does neither: "_nogenderesi" could be read several ways.
    """
    if not drop:
        return ""
    short = {"triage_acuity": "esi"}
    # A group keeps its own name in the tag. Spelling out thirty-four columns
    # would produce a directory name no one could read and no one could match.
    parts = sorted(short.get(d, d.replace("_", "")) for d in drop
                   if d not in GROUPS) + sorted(d for d in drop if d in GROUPS)
    return "_no" + "-".join(parts)


def resolve_features(features, cols, drop=()):
    """Which columns this run uses.

        top15       the fifteen variables the manuscript reports
        all63       everything
        rank:N      the first N of the ranking for *this* exclusion, i.e.
                    feature_ranking.csv normally and feature_ranking_noesi.csv
                    under --drop triage_acuity
        fullrank:N  the first N of the original all-variable ranking, after
                    removing the dropped names
        greedy:N    the first N of the marginal-gain ordering from
                    11_greedy_select.py, chosen against the acute endpoints
        greedyall:N the same, chosen against all nine endpoints, so the two
                    disposition outcomes get a vote in what is selected

    rank:N takes the N variables the model leans on most, each judged alone.
    greedy:N takes the N that between them explain the most. These are
    different sets whenever two candidates carry the same information: the
    attribution ordering scores both highly and a prefix takes both, while
    greedy takes one and moves on. On this data the two disagree from the
    first position -- attribution opens with n_ed_365d, greedy opens with age
    and does not reach n_ed_365d until fourteenth.

    The ranking is recomputed by 04_rank_features.py on the training split, so
    rank:15 is not necessarily the same set as top15 -- the published top-10
    and top-15 lists are not nested, which is one of the things that made the
    parsimony figure impossible to reproduce.

    `drop` removes named columns. For both ranked forms the removal happens
    *before* the first N are taken, so dropping ESI still gives fifteen
    variables and everything below it slides up by one. That is the question
    worth asking: what does the model do when it is allowed fifteen numbers
    but not that one, rather than when it is simply given one fewer.

    rank:N and fullrank:N differ in which ranking they slide up, and the
    difference is the whole design of the ESI ablation. fullrank:N holds the
    ordering fixed at the one computed with every variable present, so the
    only thing that changed is that ESI is gone -- the cheap, tightly matched
    comparison. rank:N uses an ordering recomputed without ESI, because
    removing the strongest variable changes what the others are worth, and
    answers the different question of which fifteen you would choose if you
    never had ESI at all. Both are legitimate; conflating them is not, which
    is why they are separate tokens that land in separate directories and get
    recorded in metrics.json.

    A name that is not a column is an error rather than a no-op. The whole
    point of the run is that a variable is absent, so silently dropping
    nothing would produce a result that answers a different question while
    looking exactly like the one asked for.
    """
    # Two things are called "drop" and they are not interchangeable. `spec` is
    # what was typed, which may be a group name; `drop` is the set of columns
    # it resolves to. ranking_path needs the spec, because that is what named
    # the file when 04_rank_features.py wrote it. Passing the resolved set
    # instead asks for feature_ranking_nocciMI-nocciCHF-... which no run has
    # ever produced, and every rank:N call fails on a missing file. That is
    # what killed all 108 points of job 9497's sweep while the job still
    # reported exit 0.
    spec = tuple(drop or ())
    drop = expand_drop(spec, cols)
    unknown = drop - set(cols)
    if unknown:
        raise SystemExit(f"--drop names no such column: {sorted(unknown)}")

    if features == "top15":
        return [cols.index(c) for c in TOP15 if c not in drop]
    if features == "all63":
        return [i for i, c in enumerate(cols) if c not in drop]
    if features.startswith(("rank:", "fullrank:", "greedy:", "greedyall:")):
        kind, k = features.split(":")
        k = int(k)
        # fullrank always reads the all-variable ranking; rank reads the one
        # belonging to this exclusion; greedy and greedyall read the two
        # marginal-gain orderings 11_greedy_select.py wrote for it.
        variant = kind if kind.startswith("greedy") else ""
        path = ranking_path(() if kind == "fullrank" else spec, variant)
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found; run 04_rank_features.py first")
        import csv as _csv
        with open(path) as fh:
            names = [r["feature"] for r in _csv.DictReader(fh)]
        names = [n for n in names if n not in drop]
        if k > len(names):
            raise SystemExit(f"asked for {k} features, ranking has {len(names)}")
        return [cols.index(c) for c in names[:k]]
    raise SystemExit(f"unknown --features {features}")


def ranking_path(drop=(), variant=""):
    """Where the ranking for this exclusion lives.

    A ranking computed without ESI is a different ranking -- removing the
    strongest variable changes what the others are worth -- so it gets its own
    file rather than overwriting the one the main run depends on.

    `variant` selects which ordering. "" is the attribution ordering from
    04_rank_features.py; "greedy" and "greedyall" are the marginal-gain
    orderings from 11_greedy_select.py, chosen against the acute endpoints
    and against all nine respectively.

    Every one of these gets its own filename, and that is not tidiness. They
    answer different questions -- how much the model uses a variable alone,
    against what it adds to the set already chosen, against what it adds when
    the disposition endpoints also get a vote -- and a run that read the wrong
    file would look entirely normal and report a number for a set it never
    trained on. The greedy pass writes its ordering while an earlier sweep is
    still reading one, so a shared name is not a hypothetical collision.
    """
    v = ("_" + variant) if variant else ""
    return f"{PROJ}/results/feature_ranking{v}{drop_suffix(drop)}.csv"


def load_split(features, drop=()):
    meta = json.load(open(f"{DATA}/preprocessing.json"))
    cols = meta["features"]
    idx = resolve_features(features, cols, drop)
    mean = np.asarray(meta["scaler_mean"])[idx]
    scale = np.asarray(meta["scaler_scale"])[idx]

    def get(split):
        X = np.load(f"{DATA}/umn_{split}_X.npy")[:, idx]
        X = (X - mean) / np.where(scale == 0, 1, scale)
        y = np.load(f"{DATA}/umn_{split}_y.npy")
        return X.astype(np.float32), y.astype(np.float32)

    return get("train"), get("test"), [cols[i] for i in idx], meta


EXTERNAL = ("bidmc", "stanford")


def load_external(features, drop=()):
    """The external cohorts, scaled with UMN's own scaler and nothing else.

    Returns a dict, and returns an empty one rather than raising, because most
    runs happen with no external arrays on disk and a missing cohort must not
    stop training.

    Why this lives in the training script
    -------------------------------------
    The tree arms never saved an estimator. 09_external.py can score the
    neural arms from model.pt, but XGBoost, Random Forest, logistic regression
    and HistGradientBoosting had no way of being applied to MIMIC or Stanford
    at all, so five of the six columns of the external table did not exist.

    Pickling the estimators is the obvious fix and the wrong one: a hundred-tree
    forest over 1.3M rows is hundreds of megabytes, times nine endpoints times
    three seeds, against five gigabytes of free quota. Scoring while the model
    is still in memory costs one forward pass and about 20 MB of output.

    Nothing is refitted here. Same weights, same medians, same scaler -- the
    zero-shot question, asked at the only moment the estimator exists.
    """
    meta = json.load(open(f"{DATA}/preprocessing.json"))
    cols = meta["features"]
    idx = resolve_features(features, cols, drop)
    mean = np.asarray(meta["scaler_mean"])[idx]
    scale = np.asarray(meta["scaler_scale"])[idx]
    out = {}
    for name in EXTERNAL:
        fx, fy = f"{DATA}/{name}_X.npy", f"{DATA}/{name}_y.npy"
        if not (os.path.exists(fx) and os.path.exists(fy)):
            continue
        X = np.load(fx)[:, idx]
        X = (X - mean) / np.where(scale == 0, 1, scale)
        out[name] = (X.astype(np.float32), np.load(fy).astype(np.float32))
    return out


def auroc(y, s):
    """Rank based, so it costs one sort instead of a sweep over thresholds."""
    y = np.asarray(y)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.empty(len(s), dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, s):
    from sklearn.metrics import average_precision_score
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def train_neural(arm, seed, features, out, drop=()):
    (Xtr, ytr), (Xte, yte), cols, meta = load_split(features, drop)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # The whole training matrix is 63 MB at fifteen features and 331 MB at all
    # sixty-three, so it goes on the GPU once and stays there. A DataLoader
    # over a TensorDataset would call __getitem__ a million times per epoch in
    # Python; slicing a resident tensor turns a forty-second epoch into a
    # four-second one, and the shuffle is still seeded, so nothing about
    # reproducibility changes.
    Xg = torch.from_numpy(Xtr).to(dev)
    Yg = torch.from_numpy(ytr).to(dev)
    n = len(Xg)
    n_val = int(HP["val_frac"] * n)
    # VAL_SPLIT_SEED, not seed -- see the note beside its definition
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(VAL_SPLIT_SEED))
    val_idx = perm[:n_val].to(dev)
    tr_idx = perm[n_val:].to(dev)
    Xva, Yva = Xg[val_idx], Yg[val_idx]
    Xtr_g, Ytr_g = Xg[tr_idx], Yg[tr_idx]
    del Xg, Yg
    n_tr, bs = len(Xtr_g), HP["batch"]
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    model = M.build(arm, Xtr.shape[1], n_tasks=ytr.shape[1],
                    hidden=HP["hidden"], n_experts=HP["n_experts"],
                    temperature=HP["temperature"],
                    alpha_shared=HP["alpha_shared"],
                    lambda_kl=HP["lambda_kl"],
                    expert_dropout=HP["expert_dropout"],
                    l1_gate=HP["l1_gate"]).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  parameters {n_par:,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"])
    crit = nn.BCEWithLogitsLoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2)

    best, best_state, log = -np.inf, None, []
    for ep in range(1, HP["epochs"] + 1):
        t0 = time.time()
        model.train()
        order = torch.randperm(n_tr, generator=gen, device=dev)
        tot = 0.0
        for s in range(0, n_tr, bs):
            b = order[s:s + bs]
            xb, yb = Xtr_g[b], Ytr_g[b]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb) + model.aux_loss()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), HP["clip"])
            opt.step()
            tot += loss.item() * len(b)

        model.eval()
        P = []
        with torch.no_grad():
            for s in range(0, len(Xva), 16384):
                P.append(model(Xva[s:s + 16384]).cpu().numpy())
        P, Y = np.concatenate(P), Yva.cpu().numpy()
        aucs = [auroc(Y[:, j], P[:, j]) for j in range(Y.shape[1])]
        m = float(np.nanmean(aucs))
        sched.step(m)
        log.append({"epoch": ep, "train_loss": tot / n_tr,
                    "val_auroc": m, "sec": time.time() - t0})
        print(f"  epoch {ep:2d}  loss {tot / n_tr:.4f}  "
              f"val mAUROC {m:.4f}  {time.time() - t0:.0f}s", flush=True)
        if m > best:
            best = m
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, f"{out}/model.pt")

    # Score the validation split and keep it. Choosing anything -- the number
    # of variables, a hyperparameter, which arm to report -- from the test set
    # makes the reported intervals optimistic, because they assume the model
    # was not selected using that data. Saving validation predictions is what
    # lets those choices be made without touching 2024-2025.
    model.eval()
    with torch.no_grad():
        Pv = np.concatenate([model(Xva[s:s + 16384]).cpu().numpy()
                             for s in range(0, len(Xva), 16384)])
    save_scores(f"{out}/logits_umn_val.npy", Pv)
    save_labels(f"{out}/labels_umn_val.npy", Yva.cpu().numpy())
    val_per = [{"target": j,
                "auroc": auroc(Yva.cpu().numpy()[:, j], Pv[:, j]),
                "auprc": auprc(Yva.cpu().numpy()[:, j], Pv[:, j])}
               for j in range(Pv.shape[1])]

    # score the held-out 2024-2025 cohort
    del Xtr_g, Ytr_g
    L = []
    with torch.no_grad():
        for i in range(0, len(Xte), 8192):
            L.append(model(torch.from_numpy(Xte[i:i + 8192]).to(dev))
                     .cpu().numpy())
    logits = np.concatenate(L)
    save_scores(f"{out}/logits_umn_test.npy", logits)
    save_labels(f"{out}/labels_umn_test.npy", yte)

    # and the external cohorts, while the weights are still loaded
    for name, (Xe, ye) in load_external(features, drop).items():
        E = []
        with torch.no_grad():
            for i in range(0, len(Xe), 8192):
                E.append(model(torch.from_numpy(Xe[i:i + 8192]).to(dev))
                         .cpu().numpy())
        save_scores(f"{out}/logits_{name}.npy", np.concatenate(E))
        save_labels(f"{out}/labels_{name}.npy", ye)
        print(f"  scored {name}: {len(Xe):,} encounters", flush=True)

    g = model.gate_matrix()
    gate_summary = {}
    if g is not None:
        np.savetxt(f"{out}/gate_weights.csv", g, delimiter=",")
        nrm = g / np.linalg.norm(g, axis=1, keepdims=True)
        cos = nrm @ nrm.T
        off = cos[~np.eye(len(g), dtype=bool)]
        ent = -(g * np.log(g + 1e-12)).sum(-1)
        gate_summary = {"cos_mean": float(off.mean()),
                        "cos_min": float(off.min()),
                        "eff_experts": float(np.exp(ent).mean()),
                        "n_experts": int(g.shape[1])}
        print(f"  gate: cos {off.mean():.4f}  "
              f"eff {np.exp(ent).mean():.2f}/{g.shape[1]}", flush=True)

    return logits, yte, cols, log, gate_summary, n_par, val_per


# Filled in by train_tree so that main() records what the estimator was
# actually given rather than the neural HP dict. Until this existed, every
# tree run wrote the multi-task network's hyperparameters into its own
# metrics.json -- hidden 256, thirteen experts, temperature 0.2 -- for a model
# that has none of those things. Nothing errored and the file looked complete,
# which is the worst way for a methods section to be wrong.
# The validation split is drawn with this seed and never with the model
# seed. That distinction is not cosmetic.
#
# Until 2026-08-11 the split used manual_seed(seed), so seeds 42, 43 and 44
# each held out a different 20% of the development set. Every per-run number
# was still correct -- each model was scored on the patients it had not seen.
# What broke was the ensemble: 14_metrics.py and 10_ci.py average
# logits_umn_val.npy across seeds, and averaging row i of three arrays that
# describe three different patient i's produces nothing at all. Our model read
# 0.7911 on validation against 0.9002 on test, while LightGBM -- which had
# always fixed its split at 42 -- read a clean 0.9088.
#
# The damage went further than the validation column, because every operating
# threshold is taken from validation. Sensitivity, specificity, PPV, NPV and F1
# on the test cohort and on both external cohorts were all derived from a
# threshold chosen on scrambled data.
#
# With the seed fixed, all three models hold out the same patients, so their
# averaged prediction on that set is a genuine out-of-sample ensemble
# prediction and a threshold taken from it applies to the same quantity that
# is scored at test time.

# Labels are 0/1 and were being written as float32 -- four bytes to hold one
# bit of information, in four files per run, in twenty-one runs. That is 650 MB
# of a 3 GB working set spent on a quantity that never needed more than int8.
# Scores stay float32; predict_proba returns float64 and LightGBM was writing
# it out at full width, which doubled its share for no precision that any
# metric here can use.
#
# This is not tidiness. The metrics job died with "No space left on device" on
# 2026-08-11 and the difference between fitting and not fitting is roughly this
# saving.
def save_scores(path, a):
    np.save(path, np.asarray(a, dtype=np.float32))


def save_labels(path, a):
    np.save(path, np.asarray(a, dtype=np.int8))


VAL_SPLIT_SEED = 42

TREE_HP = {}


def tree_spec(arm, seed):
    """Every baseline's hyperparameters in one place, so the methods section
    can be read off rather than reconstructed from the code."""
    cuda = torch.cuda.is_available()
    if arm == "A2_rf":
        return "RandomForestClassifier", dict(
            n_estimators=100, min_samples_leaf=20, n_jobs=-1,
            random_state=seed)
    if arm == "A2_histgb":
        return "HistGradientBoostingClassifier", dict(
            max_iter=200, random_state=seed)
    if arm == "A2_lr":
        # Ridge-penalised, on the same standardised matrix every other arm
        # sees. lbfgs at 1000 iterations converges on this data; saga would be
        # the choice only if an L1 penalty were wanted, and it is not -- the
        # variable selection is done upstream and reported separately.
        return "LogisticRegression", dict(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
            random_state=seed)
    if arm in ("A2_xgb", "A2_xgb_sub"):
        kw = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                  tree_method="hist", random_state=seed,
                  device="cuda" if cuda else "cpu")
        if arm == "A2_xgb_sub":
            kw.update(subsample=0.8, colsample_bytree=0.8)
        return "XGBClassifier", kw
    raise ValueError(arm)


def train_tree(arm, seed, features, out, drop=()):
    from sklearn.ensemble import RandomForestClassifier, \
        HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    (Xtr, ytr), (Xte, yte), cols, _ = load_split(features, drop)

    # Hold out the same fraction, with the same seeded permutation the neural
    # arms use, so both sides pick their variable count on identical data.
    n = len(Xtr)
    n_val = int(HP["val_frac"] * n)
    # VAL_SPLIT_SEED, not seed -- see the note beside its definition
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(VAL_SPLIT_SEED)).numpy()
    va_i, tr_i = perm[:n_val], perm[n_val:]
    Xva, yva = Xtr[va_i], ytr[va_i]
    Xtr, ytr = Xtr[tr_i], ytr[tr_i]

    T = ytr.shape[1]
    scores = np.zeros_like(yte)
    val_scores = np.zeros_like(yva)
    # Scored inside the per-endpoint loop, because that is the only moment the
    # fitted estimator exists -- it is discarded on the next iteration.
    ext = load_external(features, drop)
    ext_scores = {k: np.zeros_like(v[1]) for k, v in ext.items()}

    # Subsampling for XGBoost is not a handicap and not decoration. Without
    # row or column sampling its hist method is deterministic and
    # random_state does nothing: seeds 43 and 44 came back bit-identical, so a
    # three-seed ensemble of ours was being compared against one model
    # repeated three times. Subsampling gives the baseline real seed
    # diversity, and it is standard practice that usually improves gradient
    # boosting. The point is to beat a strong baseline, not a weakened one.
    cls, kw = tree_spec(arm, seed)
    TREE_HP.clear()
    TREE_HP.update({"estimator": cls, "params": {
        k: v for k, v in kw.items()}, "one_model_per_endpoint": True})
    MAKE = {"RandomForestClassifier": RandomForestClassifier,
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
            "LogisticRegression": LogisticRegression}
    print(f"  {cls}({', '.join(f'{k}={v!r}' for k, v in kw.items())})",
          flush=True)

    for t in range(T):
        y = ytr[:, t]
        if cls == "XGBClassifier":
            import xgboost as xgb
            m = xgb.XGBClassifier(**kw)
        else:
            m = MAKE[cls](**kw)
        m.fit(Xtr, y)
        scores[:, t] = m.predict_proba(Xte)[:, 1]
        val_scores[:, t] = m.predict_proba(Xva)[:, 1]
        for name, (Xe, _) in ext.items():
            ext_scores[name][:, t] = m.predict_proba(Xe)[:, 1]
        print(f"  {arm} target {t + 1}/{T} done", flush=True)

    save_scores(f"{out}/logits_umn_test.npy", scores)   # already probabilities
    save_labels(f"{out}/labels_umn_test.npy", yte)
    save_scores(f"{out}/logits_umn_val.npy", val_scores)
    save_labels(f"{out}/labels_umn_val.npy", yva)
    for name, (Xe, ye) in ext.items():
        save_scores(f"{out}/logits_{name}.npy", ext_scores[name])
        save_labels(f"{out}/labels_{name}.npy", ye)
        print(f"  scored {name}: {len(Xe):,} encounters", flush=True)
    val_per = [{"target": j, "auroc": auroc(yva[:, j], val_scores[:, j]),
                "auprc": auprc(yva[:, j], val_scores[:, j])}
               for j in range(T)]
    return scores, yte, cols, [], {}, 0, val_per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", default="top15",
                    help="top15, all63, or rank:N")
    ap.add_argument("--drop", default="",
                    help="comma separated columns to exclude, e.g. "
                         "triage_acuity. Changes the run directory name so a "
                         "dropped run can never be mistaken for a full one.")
    a = ap.parse_args()
    drop = tuple(s for s in (x.strip() for x in a.drop.split(",")) if s)

    seed_everything(a.seed)
    tag = (f"{a.arm}_{a.features.replace(':', '')}"
           f"{drop_suffix(drop)}_seed{a.seed}")
    out = f"{RUNS}/{tag}"
    # Skipping finished runs is what makes a sixteen hour sweep resumable, but
    # "finished" has to mean finished on *this* data. 00_prepare_data.py
    # stamps every matrix with a data_id; a run built on a different one is
    # stale, not done, and saying so is the difference between rerunning it
    # and quietly reporting last week's numbers under this week's heading.
    data_id = json.load(open(f"{DATA}/preprocessing.json")).get("data_id")
    if os.path.exists(f"{out}/metrics.json"):
        prev = json.load(open(f"{out}/metrics.json")).get("data_id")
        if prev == data_id:
            print(f"[skip] {tag} already done")
            return
        print(f"[stale] {tag} was built on data {prev}, current is "
              f"{data_id} -- retraining", flush=True)
    os.makedirs(out, exist_ok=True)
    print(f"\n=== {tag} ===", flush=True)
    if drop:
        print(f"  excluding {', '.join(drop)}", flush=True)

    t0 = time.time()
    fn = train_tree if a.arm.startswith("A2") else train_neural
    scores, y, cols, log, gate, n_par, val_per = fn(a.arm, a.seed,
                                                    a.features, out, drop)

    per = []
    for j in range(y.shape[1]):
        per.append({"target": j,
                    "auroc": auroc(y[:, j], scores[:, j]),
                    "auprc": auprc(y[:, j], scores[:, j])})
    acute, disp = list(range(2, 9)), [0, 1]

    def mac(rows, cols_, key):
        return float(np.nanmean([rows[i][key] for i in cols_]))

    metrics = {
        "arm": a.arm, "seed": a.seed, "features": a.features,
        "dropped": list(drop), "data_id": data_id,
        "n_features": len(cols), "feature_names": cols,
        "n_parameters": n_par,
        "per_target": per, "per_target_val": val_per,
        "macro_acute_auroc": mac(per, acute, "auroc"),
        "macro_acute_auprc": mac(per, acute, "auprc"),
        "macro_disp_auroc": mac(per, disp, "auroc"),
        "macro_disp_auprc": mac(per, disp, "auprc"),
        "val_macro_acute_auroc": mac(val_per, acute, "auroc"),
        "val_macro_acute_auprc": mac(val_per, acute, "auprc"),
        "val_macro_disp_auroc": mac(val_per, disp, "auroc"),
        "val_macro_disp_auprc": mac(val_per, disp, "auprc"),
        "gate": gate, "train_log": log,
        # logits_umn_*.npy is misnamed for half the arms: the neural arms are
        # trained with BCEWithLogitsLoss and store the linear predictor, the
        # tree arms store predict_proba. AUROC and AUPRC are rank statistics
        # and do not notice, which is why nobody noticed -- until a logit was
        # fed to a calibration routine expecting a probability and produced a
        # table of plausible, meaningless slopes. Anything that cares about
        # the scale rather than the order must read this field.
        "score_scale": ("probability" if a.arm.startswith("A2") else "logit"),
        "minutes": (time.time() - t0) / 60,
        # The estimator's own settings, not the network's. A tree arm used to
        # record hidden=256 and thirteen experts, which it does not have.
        "hp": (dict(TREE_HP) if a.arm.startswith("A2") else HP),
        "fingerprint": run_fingerprint(),
    }
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=2)

    print(f"  val   macro AUROC acute {metrics['val_macro_acute_auroc']:.4f}"
          f"   AUPRC {metrics['val_macro_acute_auprc']:.4f}")
    print(f"  macro AUROC  acute {metrics['macro_acute_auroc']:.4f}   "
          f"disposition {metrics['macro_disp_auroc']:.4f}")
    print(f"  macro AUPRC  acute {metrics['macro_acute_auprc']:.4f}   "
          f"disposition {metrics['macro_disp_auprc']:.4f}")
    print(f"  {metrics['minutes']:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
