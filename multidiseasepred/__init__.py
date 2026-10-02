"""MultiDiseasePred: one model, many non-mutually-exclusive outcomes.

    from multidiseasepred import MultiDiseasePred

    model = MultiDiseasePred(variables=cols, targets=outcomes)
    model.select_variables(X, y, vote=range(7))    # greedy forward selection
    model.fit(X, y)                                # train + calibrate
    model.evaluate(X_test, y_test)                 # AUROC/AUPRC with intervals
    model.write_dashboard("myapp")                 # a Streamlit app for *your* outcomes

Nothing in the package is specific to emergency care or to the outcomes in the
paper. The dashboard is generated from your variable names, your ranges and your
base rates.

PyTorch is only needed for the model itself. `selection`, `calibration` and
`metrics` work with scikit-learn alone, so you can use the variable-selection and
evaluation machinery without a GPU stack.
"""

from .estimator import MultiDiseasePred

__version__ = "0.1.0"
__all__ = ["MultiDiseasePred", "__version__"]


def __getattr__(name):
    # Submodules are importable without pulling in torch at package import time.
    if name in {"selection", "calibration", "metrics", "model", "dashboard"}:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
