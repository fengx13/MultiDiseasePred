"""Export the Streamlit ED dashboard's model into files a static web page can load.

Usage:
    python scripts/export_web_dashboard.py path/to/dashboard docs/dashboard

`path/to/dashboard` is the Streamlit app folder (it contains app/config.py and
output/best_multidiseasepred.pt, feature_scaler_umn.pkl, calibrator_isotonic.pkl).
Writes model.bin (float32 weights) and model.json (shapes, scaler, calibrators,
feature metadata) into the output folder. Needs torch, joblib and scikit-learn.
"""
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(src))
from app import config  # noqa: E402
from app import model_utils  # noqa: E402,F401  (needed to unpickle the calibrator)

out.mkdir(parents=True, exist_ok=True)
sd = torch.load(src / "output/best_multidiseasepred.pt", map_location="cpu")
blob = joblib.load(src / "output/feature_scaler_umn.pkl")
with open(src / "output/calibrator_isotonic.pkl", "rb") as f:
    cal = pickle.load(f)
assert list(blob["feat_cols"]) == list(config.COL_LIST)

tensors, offset, chunks = {}, 0, []
for name, t in sd.items():
    a = t.detach().cpu().numpy().astype("<f4").ravel()
    tensors[name] = {"offset": offset, "shape": list(t.shape)}
    offset += a.size
    chunks.append(a)
np.concatenate(chunks).tofile(out / "model.bin")

meta = {
    "note": "Exported from the MultiDiseasePred Streamlit dashboard. Weights were trained on synthetic data; probabilities are illustrative.",
    "arch": {"hidden": 256, "experts": 13, "tasks": len(config.OUTCOME_LIST),
             "temperature": 0.2, "alpha": 0.1},
    "tensors": tensors,
    "feat_cols": list(config.COL_LIST),
    "scaler_mean": blob["scaler"].mean_.tolist(),
    "scaler_scale": blob["scaler"].scale_.tolist(),
    "median": [float(blob["median"][c]) for c in config.COL_LIST],
    "calibrators": [{"x": c.X_thresholds_.tolist(), "y": c.y_thresholds_.tolist()}
                    for c in cal.calibrators],
    "outcomes": list(config.OUTCOME_LIST),
    "outcome_display": config.OUTCOME_DISPLAY,
    "reserved": list(config.RESERVED_OUTCOME_SLOTS),
    "core": list(config.CORE_15),
    "feature_meta": config.FEATURE_META,
    "baseline_prevalence": config.BASELINE_PREVALENCE,
    "default_threshold": config.DEFAULT_THRESHOLD,
}
(out / "model.json").write_text(json.dumps(meta, indent=1))
print(f"wrote {offset} weights to {out}")
