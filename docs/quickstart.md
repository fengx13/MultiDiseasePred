# Using the package

`multidiseasepred` is the interface. `pipeline/` is supplementary code kept so
the manuscript's numbers can be checked; you do not need it to use the model.

```bash
pip install "multidiseasepred[dashboard]"
```

## The short version

```bash
multidiseasepred train data.csv --targets outcome_ --out myapp   # everything
streamlit run myapp/app.py                                       # one at a time
multidiseasepred predict myapp/model new.csv -o predictions.csv  # in batch
```

`train` runs variable selection, trains the network, fits the calibrator on
held-out rows, evaluates on a held-out split, and writes the dashboard. It
checks your file first and names the columns to fix if anything is wrong.

## Preparing your data

One CSV, one row per subject:

- outcome columns are 0/1, ideally sharing a prefix so `--targets outcome_`
  finds them;
- every other column is treated as a candidate predictor, so drop identifiers
  and dates first;
- all columns numeric — encode categories yourself;
- no missing values — impute before training;
- every outcome has at least some positive cases.

## A whole workflow in Python

```python
import numpy as np
from multidiseasepred import MultiDiseasePred

model = MultiDiseasePred(
    variables=feature_names,
    targets=["sepsis", "aki", "admission"],
    groups={"conditions": ["sepsis", "aki"], "disposition": ["admission"]},
)

res = model.select_variables(X_dev, y_dev, vote=["sepsis", "aki"], threshold=0.98)
print(res.summary())

model.fit(X_dev, y_dev).set_prevalence(y_dev)
report = model.evaluate(X_test, y_test)
model.write_dashboard("myapp", X=X_dev, title="ICU Outcomes")
```

```bash
cd myapp && streamlit run app.py
```

## What you get in `myapp/`

| File | What it is |
|---|---|
| `app.py` | The Streamlit app. Plain source, yours to edit. |
| `variables.json` | Widget labels, units, defaults and ranges. Edit this rather than the code. |
| `model/` | The saved networks, scaler and calibrator. |
| `background.npy` | Reference rows for attribution. |
| `requirements.txt` | What the app needs to run. |

Widget types are inferred from your training data: a 0/1 column becomes a
checkbox, anything else a number box over the observed range. If a label reads
badly -- `Triage o2sat` rather than `SpO2 (%)` -- fix it in `variables.json`;
the app reads it at startup and `app.py` does not change.

## Things the API makes you decide

**`vote=` has no default.** The variable set you get is optimal for the outcomes
it was selected against and no others. If some of your outcomes sit it out, that
belongs in your methods section.

**`groups=` is worth setting.** Outcomes that are different kinds of thing
should not share a macro-average. The package will average everything together
if you say nothing, and that is usually the wrong number.

**Calibration is fitted on held-out data.** `fit` reserves a slice and uses it
for the calibrator and the checkpoint, nothing else. If you want a locally
refitted calibrator at a new site, build a second `Calibrator` explicitly --
the library will not do it by accident, because doing it silently turns an
external-validation result into a local-refit result.

## Without PyTorch

Selection, calibration and metrics need only scikit-learn:

```python
from multidiseasepred.selection import greedy_forward
from multidiseasepred.calibration import Calibrator
from multidiseasepred.metrics import evaluate, operating_point
```

Useful if you want the variable-selection or evaluation machinery around a model
of your own.
