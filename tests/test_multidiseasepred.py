"""Tests for multidiseasepred.

    pip install pytest
    pytest tests/ -v

The tests that need PyTorch skip themselves when it is missing, so the
selection, calibration and metrics machinery can be checked in an environment
that only has scikit-learn. Run the whole file on a machine with torch before
you trust the training path.
"""

import numpy as np
import pytest

from multidiseasepred import MultiDiseasePred
from multidiseasepred.calibration import Calibrator
from multidiseasepred.metrics import (Estimate, calibration_metrics, compare,
                                      evaluate, operating_point)
from multidiseasepred.selection import greedy_forward

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:                                            # pragma: no cover
    HAVE_TORCH = False

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="PyTorch not installed")


def make_data(n=3000, seed=0):
    """Two real drivers, one redundant copy, three targets of varying rarity."""
    rng = np.random.default_rng(seed)
    d = 8
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[:, 1] = X[:, 0] + rng.normal(scale=0.05, size=n)      # copy of v0
    X[:, 7] = (rng.random(n) < 0.2).astype(np.float32)      # a binary flag
    lin = np.stack([2.0 * X[:, 0], 2.0 * X[:, 5] - 1.0,
                    1.5 * X[:, 7] - 3.0], axis=1)
    y = (rng.random((n, 3)) < 1 / (1 + np.exp(-lin))).astype(int)
    names = [f"v{i}" for i in range(d)]
    return X, y, names


# ------------------------------------------------------------------ selection
def test_greedy_does_not_spend_the_budget_twice_on_one_signal():
    """v0 and v1 carry the same information; v5 carries the other one.

    Which of the redundant pair is picked first is a coin flip and testing for a
    particular one just bakes in a seed. The property that matters is that the
    second pick goes to the *other* signal rather than to the copy: that is what
    greedy selection buys over a ranking prefix.
    """
    X, y, names = make_data()
    r = greedy_forward(X[:2000], y[:2000], X[2000:], y[2000:],
                       targets=[0, 1], variable_names=names,
                       max_k=5, sample=None, seed=1, verbose=False)
    first_two = set(r.names[:2])
    assert len(first_two & {"v0", "v1"}) == 1, first_two   # one of the pair, not both
    assert "v5" in first_two, first_two                    # and the other signal


def test_selection_curve_and_rule():
    X, y, names = make_data()
    r = greedy_forward(X[:2000], y[:2000], X[2000:], y[2000:],
                       targets=[0, 1], variable_names=names,
                       max_k=5, sample=None, seed=1, verbose=False)
    assert len(r.scores) == 5
    assert r.k_at(0.5) == 1
    assert r.k_at(10.0) is None            # unreachable is a real answer
    assert "greedy forward selection" in r.summary()


def test_selection_rejects_empty_targets():
    X, y, names = make_data(500)
    with pytest.raises(ValueError):
        greedy_forward(X[:400], y[:400], X[400:], y[400:], targets=[],
                       variable_names=names, max_k=2, sample=None, verbose=False)


def test_selection_rejects_name_mismatch():
    X, y, names = make_data(500)
    with pytest.raises(ValueError):
        greedy_forward(X[:400], y[:400], X[400:], y[400:], targets=[0],
                       variable_names=names[:3], max_k=2, sample=None,
                       verbose=False)


# ---------------------------------------------------------------- calibration
def _miscalibrated(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    lin = np.stack([rng.normal(0, 1.5, n), rng.normal(-2, 1.2, n)], 1)
    y = (rng.random((n, 2)) < 1 / (1 + np.exp(-lin))).astype(int)
    p = 1 / (1 + np.exp(-(lin * 0.5 + 1.2)))               # shifted and flattened
    return p, y


def test_calibration_improves_calibration():
    p, y = _miscalibrated()
    c = Calibrator().fit(p[:2000], y[:2000])
    pc = c.transform(p[2000:])
    before = calibration_metrics(y[2000:, 0], p[2000:, 0])
    after = calibration_metrics(y[2000:, 0], pc[:, 0])
    assert after["ece"] < before["ece"]
    assert abs(after["intercept"]) < abs(before["intercept"])


def test_calibration_never_reverses_order():
    p, y = _miscalibrated()
    c = Calibrator().fit(p[:2000], y[:2000])
    assert c.check_monotone(p)


def test_ranking_shift_is_zero_or_negative():
    """Ties can cost a little AUROC; nothing should ever gain it."""
    p, y = _miscalibrated()
    c = Calibrator().fit(p[:2000], y[:2000])
    shift = c.ranking_shift(p[2000:], y[2000:])
    assert all(s <= 1e-12 for s in shift)


def test_calibration_skips_single_class_targets():
    rng = np.random.default_rng(0)
    p = rng.random((500, 2))
    y = np.zeros((500, 2), dtype=int)
    y[:, 0] = (rng.random(500) < 0.3).astype(int)          # target 1 has no events
    c = Calibrator().fit(p, y)
    assert c.skipped_ == [1]
    assert np.allclose(c.transform(p)[:, 1], p[:, 1])      # left untouched


def test_calibrator_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        Calibrator().fit(np.random.random((10, 2)), np.zeros((10, 3), dtype=int))


# -------------------------------------------------------------------- metrics
def test_evaluate_groups_are_kept_separate():
    X, y, _ = make_data()
    p = np.clip(y * 0.7 + np.random.default_rng(0).random(y.shape) * 0.3, 0, 1)
    r = evaluate(y, p, target_names=["a", "b", "c"],
                 groups={"pair": [0, 1], "solo": [2]}, n_boot=50)
    assert set(r["macro"]) == {"pair", "solo"}
    assert isinstance(r["per_target"]["a"]["AUROC"], Estimate)
    assert r["per_target"]["a"]["events"] == int(y[:, 0].sum())


def test_operating_point_reaches_requested_sensitivity():
    X, y, _ = make_data()
    p = np.clip(y[:, 0] * 0.6 + np.random.default_rng(1).random(len(y)) * 0.4, 0, 1)
    op = operating_point(y[:, 0], p, 0.90)
    assert op["sensitivity"] >= 0.90 - 1e-9
    assert 0.0 <= op["specificity"] <= 1.0


def test_operating_point_with_no_events_is_nan_not_a_crash():
    op = operating_point(np.zeros(100, dtype=int), np.random.random(100))
    assert np.isnan(op["sensitivity"])


def test_compare_declines_to_call_overlapping_intervals():
    a = Estimate(0.90, 0.88, 0.92)
    b = Estimate(0.89, 0.87, 0.91)
    c = Estimate(0.80, 0.78, 0.82)
    assert compare(a, b) == "indistinguishable (intervals overlap)"
    assert compare(a, c) == "higher"


# ------------------------------------------------------------------- the model
def test_import_and_construct_without_torch():
    m = MultiDiseasePred(variables=["a", "b"], targets=["x", "y"],
                         groups={"g": ["x"], "h": ["y"]})
    assert m._group_idx() == {"g": [0], "h": [1]}
    assert "not fitted" in repr(m)


def test_predict_before_fit_raises():
    m = MultiDiseasePred(variables=["a"], targets=["x"])
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((2, 1)))


@needs_torch
def test_fit_predict_evaluate_roundtrip(tmp_path):
    X, y, names = make_data(2000)
    m = MultiDiseasePred(variables=names, targets=["t0", "t1", "t2"],
                         groups={"common": ["t0", "t1"], "rare": ["t2"]},
                         n_experts=4, hidden=32)
    m.select_variables(X[:1500], y[:1500], vote=[0, 1], max_k=4, sample=None,
                       verbose=False)
    assert 1 <= len(m.selected_) <= 4
    m.fit(X[:1500], y[:1500], epochs=2, n_seeds=2, verbose=False)
    m.set_prevalence(y[:1500])

    p = m.predict(X[1500:])
    assert p.shape == (500, 3)
    assert ((p >= 0) & (p <= 1)).all()

    r = m.evaluate(X[1500:], y[1500:], n_boot=30)
    assert set(r["macro"]) == {"common", "rare"}

    lift = m.predict_lift(X[1500:])
    assert lift.shape == p.shape

    g = m.gate_usage(X[1500:])
    assert g["effective_experts"].shape == (3,)
    assert (g["effective_experts"] <= g["n_experts"] + 1e-6).all()

    # explain() is what the dashboard calls on every interaction; it must run.
    attr = m.explain(X[1500], X[:200], n_samples=8)
    assert attr.shape == (3, len(m.selected_))
    assert np.isfinite(attr).all()

    m.save(tmp_path / "m")
    loaded = MultiDiseasePred.load(tmp_path / "m")
    assert np.allclose(loaded.predict(X[1500:]), p, atol=1e-5)


@needs_torch
def test_x_may_be_full_or_selected_width():
    X, y, names = make_data(800)
    m = MultiDiseasePred(variables=names, targets=["t0", "t1", "t2"],
                         n_experts=3, hidden=16)
    m.select_variables(X, y, vote=[0], max_k=3, sample=None, verbose=False)
    m.fit(X, y, epochs=1, n_seeds=1, verbose=False)
    cols = [names.index(v) for v in m.selected_]
    assert np.allclose(m.predict(X), m.predict(X[:, cols]))


@needs_torch
def test_wrong_width_is_rejected_clearly():
    X, y, names = make_data(400)
    m = MultiDiseasePred(variables=names, targets=["a", "b", "c"],
                         n_experts=3, hidden=16)
    m.fit(X, y, epochs=1, n_seeds=1, verbose=False)
    with pytest.raises(ValueError, match="columns"):
        m.predict(np.zeros((5, 3)))


@needs_torch
def test_write_dashboard_produces_a_runnable_app(tmp_path):
    import ast
    X, y, names = make_data(600)
    m = MultiDiseasePred(variables=names, targets=["a", "b", "c"],
                         n_experts=3, hidden=16)
    m.fit(X, y, epochs=1, n_seeds=1, verbose=False)
    m.set_prevalence(y)
    out = m.write_dashboard(tmp_path / "app", X=X, title="My Outcomes")

    for f in ("app.py", "variables.json", "requirements.txt", "README.md",
              "background.npy"):
        assert (out / f).exists(), f
    ast.parse((out / "app.py").read_text())
    assert "My Outcomes" in (out / "app.py").read_text()

    import json
    meta = json.loads((out / "variables.json").read_text())
    assert [v["name"] for v in meta["variables"]] == m.active_variables
    assert meta["targets"] == m.targets
    # v7 was built as a 0/1 flag and should become a checkbox
    kinds = {v["name"]: v["kind"] for v in meta["variables"]}
    if "v7" in kinds:
        assert kinds["v7"] == "binary"


# ----------------------------------------------------------------------- CLI
def test_targets_by_prefix_and_by_name():
    import pandas as pd
    from multidiseasepred.cli import _split_columns
    df = pd.DataFrame({"age": [1, 2], "flag": [0, 1],
                       "outcome_a": [0, 1], "outcome_b": [1, 0]})
    v, t = _split_columns(df, ["outcome_"])
    assert t == ["outcome_a", "outcome_b"] and v == ["age", "flag"]
    v, t = _split_columns(df, ["outcome_a"])
    assert t == ["outcome_a"] and v == ["age", "flag", "outcome_b"]


def test_unknown_target_column_is_rejected():
    import pandas as pd
    from multidiseasepred.cli import _split_columns
    df = pd.DataFrame({"a": [1], "outcome_x": [0]})
    with pytest.raises(SystemExit, match="not in the file"):
        _split_columns(df, ["outcome_typo"])


@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d.assign(sex=["M", "F", "M", "F"]), "non-numeric"),
    (lambda d: d.assign(age=[50, None, 70, 80]), "missing values in 1 predictor"),
    (lambda d: d.assign(outcome_a=[0, 1, 2, 1]), "must be 0/1"),
    (lambda d: d.assign(outcome_b=[0, 0, 0, 0]), "no positive cases"),
])
def test_data_readiness_says_what_to_fix(mutate, expect):
    import pandas as pd
    from multidiseasepred.cli import _check_ready, _split_columns
    df = pd.DataFrame({"age": [50, 60, 70, 80], "flag": [0, 1, 0, 1],
                       "outcome_a": [0, 1, 0, 1], "outcome_b": [0, 0, 1, 1]})
    bad = mutate(df)
    v, t = _split_columns(bad, ["outcome_"])
    with pytest.raises(SystemExit, match=expect):
        _check_ready(bad, v, t)


def test_clean_data_passes_readiness():
    import pandas as pd
    from multidiseasepred.cli import _check_ready, _split_columns
    df = pd.DataFrame({"age": [50, 60, 70, 80], "flag": [0, 1, 0, 1],
                       "outcome_a": [0, 1, 0, 1], "outcome_b": [0, 0, 1, 1]})
    v, t = _split_columns(df, ["outcome_"])
    _check_ready(df, v, t)          # must not raise


@needs_torch
def test_cli_train_then_predict_end_to_end(tmp_path):
    import pandas as pd
    from multidiseasepred.cli import main
    X, y, names = make_data(600)
    df = pd.DataFrame(X, columns=names)
    for i, t in enumerate(["outcome_a", "outcome_b", "outcome_c"]):
        df[t] = y[:, i]
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    out = tmp_path / "app"
    main(["train", str(csv), "--targets", "outcome_", "--out", str(out),
          "--max-k", "3", "--epochs", "1", "--seeds", "1", "--boot", "20",
          "--sample", "0"])
    for f in ("app.py", "variables.json", "model", "selection.txt",
              "evaluation.json"):
        assert (out / f).exists(), f

    scored = tmp_path / "scored.csv"
    main(["predict", str(out / "model"), str(csv), "-o", str(scored)])
    got = pd.read_csv(scored)
    assert all(f"risk_{t}" in got.columns
               for t in ["outcome_a", "outcome_b", "outcome_c"])
    assert len(got) == len(df)


@needs_torch
def test_cli_predict_names_the_missing_columns(tmp_path):
    import pandas as pd
    from multidiseasepred.cli import main
    X, y, names = make_data(300)
    df = pd.DataFrame(X, columns=names)
    for i, t in enumerate(["outcome_a", "outcome_b", "outcome_c"]):
        df[t] = y[:, i]
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "app"
    main(["train", str(csv), "--targets", "outcome_", "--out", str(out),
          "--max-k", "2", "--epochs", "1", "--seeds", "1", "--boot", "10",
          "--sample", "0", "--test-frac", "0"])
    # Drop a variable the model actually selected. Dropping names[0] only
    # works if greedy selection happened to pick it, which it need not.
    import json
    kept = json.load(open(out / "variables.json"))["variables"]
    needed = kept[0]["name"] if isinstance(kept[0], dict) else kept[0]
    df.drop(columns=[needed]).to_csv(tmp_path / "short.csv", index=False)
    with pytest.raises(SystemExit, match="missing"):
        main(["predict", str(out / "model"), str(tmp_path / "short.csv"),
              "-o", str(tmp_path / "x.csv")])
