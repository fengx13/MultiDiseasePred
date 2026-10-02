"""Command line: train from a CSV, then score CSVs in batch.

    multidiseasepred train data.csv --targets outcome_ --out myapp
    streamlit run myapp/app.py
    multidiseasepred predict myapp/model new_patients.csv -o predictions.csv

`train` does the whole thing in one call -- variable selection, training,
calibration, evaluation, and writing a dashboard for your outcomes -- because
the three-step version is the same three steps every time and getting one of
them out of order is the kind of mistake that is hard to see afterwards.

The Python API is still there when you want to control the pieces. This is the
front door, not the only door.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _split_columns(df: pd.DataFrame, targets_arg: list[str]) -> tuple[list[str], list[str]]:
    """Target columns are either named outright or given by a prefix.

    A prefix is the common case -- `outcome_sepsis`, `outcome_aki` -- and naming
    twenty columns on the command line is how typos get in.
    """
    if len(targets_arg) == 1 and targets_arg[0].endswith("_"):
        prefix = targets_arg[0]
        targets = [c for c in df.columns if c.startswith(prefix)]
        if not targets:
            raise SystemExit(f"no column starts with {prefix!r}")
    else:
        missing = [t for t in targets_arg if t not in df.columns]
        if missing:
            raise SystemExit(f"columns not in the file: {missing}")
        targets = list(targets_arg)
    variables = [c for c in df.columns if c not in targets]
    return variables, targets


def _check_ready(df: pd.DataFrame, variables, targets):
    """Fail early and say what to fix, rather than deep inside a training loop."""
    problems = []
    nonnum = [c for c in variables + targets
              if not pd.api.types.is_numeric_dtype(df[c])]
    if nonnum:
        problems.append(
            f"non-numeric columns: {nonnum[:8]}"
            f"{' ...' if len(nonnum) > 8 else ''}. Encode categories as numbers "
            f"(one-hot or ordinal) before training.")
    nan_v = [c for c in variables if df[c].isna().any()]
    if nan_v:
        problems.append(
            f"missing values in {len(nan_v)} predictor column(s): {nan_v[:6]}"
            f"{' ...' if len(nan_v) > 6 else ''}. Impute them first; the model "
            f"does not choose an imputation for you, because that choice belongs "
            f"in your methods section.")
    nan_t = [c for c in targets if df[c].isna().any()]
    if nan_t:
        problems.append(f"missing values in outcome column(s): {nan_t}. Rows "
                        f"without a label cannot be used.")
    bad = [c for c in targets if not set(df[c].dropna().unique()) <= {0, 1}]
    if bad:
        problems.append(f"outcome columns must be 0/1: {bad}")
    empty = [c for c in targets if df[c].sum() == 0]
    if empty:
        problems.append(f"outcome column(s) with no positive cases: {empty}. "
                        f"Drop them or get more data; nothing can be learned or "
                        f"scored for an outcome that never happens.")
    if problems:
        raise SystemExit("Your data is not ready:\n  - " + "\n  - ".join(problems))


def cmd_train(a):
    from .estimator import MultiDiseasePred

    df = pd.read_csv(a.data)
    variables, targets = _split_columns(df, a.targets)
    _check_ready(df, variables, targets)

    X = df[variables].to_numpy(np.float32)
    y = df[targets].to_numpy(int)
    print(f"{len(df):,} rows, {len(variables)} candidate variables, "
          f"{len(targets)} outcomes")
    for t in targets:
        print(f"    {t:<34} {df[t].sum():>8,} events  ({df[t].mean():.2%})")

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(X))
    cut = int(round((1 - a.test_frac) * len(X)))
    dev, test = perm[:cut], perm[cut:]

    groups = None
    if a.group:
        groups = {}
        for spec in a.group:
            name, cols = spec.split("=", 1)
            groups[name] = cols.split(",")

    model = MultiDiseasePred(variables=variables, targets=targets, groups=groups,
                             seed=a.seed)

    vote = a.vote or targets
    print(f"\nselecting variables against: {vote}")
    res = model.select_variables(X[dev], y[dev], vote=vote, max_k=a.max_k,
                                 threshold=a.threshold, sample=a.sample)
    print(f"\nkeeping {len(model.selected_)}: {model.selected_}")

    print("\ntraining")
    model.fit(X[dev], y[dev], epochs=a.epochs, n_seeds=a.seeds)
    model.set_prevalence(y[dev])

    out = Path(a.out)
    if len(test):
        print(f"\nevaluating on {len(test):,} held-out rows")
        report = model.evaluate(X[test], y[test], n_boot=a.boot)
        for g, v in report["macro"].items():
            print(f"    macro {g:<22} AUROC {v['AUROC']}   AUPRC {v['AUPRC']}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "evaluation.json").write_text(json.dumps(
            {"macro": {g: {m: vars(e) for m, e in d.items()}
                       for g, d in report["macro"].items()},
             "per_target": {t: {"AUROC": vars(d["AUROC"]),
                                "AUPRC": vars(d["AUPRC"]),
                                "prevalence": d["prevalence"],
                                "events": d["events"]}
                            for t, d in report["per_target"].items()},
             "calibration": report["calibration"],
             "operating_point": report["operating_point"]}, indent=2))

    model.write_dashboard(out, X=X[dev], title=a.title)
    (out / "selection.txt").write_text(res.summary())
    print(f"\nwritten to {out}/")
    print(f"    streamlit run {out}/app.py")
    return 0


def cmd_predict(a):
    from .estimator import MultiDiseasePred

    model = MultiDiseasePred.load(a.model)
    df = pd.read_csv(a.data)
    missing = [v for v in model.active_variables if v not in df.columns]
    if missing:
        raise SystemExit(
            f"the file is missing {len(missing)} column(s) the model needs: "
            f"{missing}")
    X = df[model.active_variables].to_numpy(np.float32)
    if np.isnan(X).any():
        raise SystemExit("missing values in the predictor columns; impute first")

    probs = model.predict(X)
    out = df.copy()
    for i, t in enumerate(model.targets):
        out[f"risk_{t}"] = probs[:, i]
    if getattr(model, "prevalence_", None) is not None:
        for i, t in enumerate(model.targets):
            out[f"lift_{t}"] = probs[:, i] / max(float(model.prevalence_[i]), 1e-9)
    out.to_csv(a.out, index=False)
    print(f"{len(df):,} rows scored -> {a.out}")
    print(f"    added {len(model.targets)} risk_ columns"
          + (" and matching lift_ columns" if getattr(model, "prevalence_", None) is not None else ""))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="multidiseasepred",
        description="Multi-task risk prediction for outcomes that co-occur.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="select variables, train, calibrate, "
                                     "evaluate and write a dashboard")
    t.add_argument("data", help="CSV with one row per subject")
    t.add_argument("--targets", nargs="+", required=True,
                   help="outcome column names, or a single prefix ending in '_'")
    t.add_argument("--out", default="myapp", help="output directory")
    t.add_argument("--vote", nargs="+", default=None,
                   help="outcomes the variable selection is scored against "
                        "(default: all of them)")
    t.add_argument("--group", action="append", default=None,
                   metavar="NAME=col1,col2",
                   help="macro-average these outcomes together; repeatable")
    t.add_argument("--max-k", type=int, default=15)
    t.add_argument("--threshold", type=float, default=0.98)
    t.add_argument("--sample", type=int, default=400_000)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--seeds", type=int, default=3)
    t.add_argument("--boot", type=int, default=1000)
    t.add_argument("--test-frac", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--title", default="MultiDiseasePred")
    t.set_defaults(func=cmd_train)

    q = sub.add_parser("predict", help="score a CSV in batch")
    q.add_argument("model", help="directory written by train (the model/ folder)")
    q.add_argument("data", help="CSV to score")
    q.add_argument("-o", "--out", default="predictions.csv")
    q.set_defaults(func=cmd_predict)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
