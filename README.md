# MultiDiseasePred

One model for many outcomes that can happen at the same time.

Most clinical risk tools predict one thing. When a patient could plausibly have
several conditions at once, you end up training and maintaining one model per
condition, running them separately, and reading their outputs side by side with
no way to compare a 2% risk of something rare against a 20% risk of something
common.

MultiDiseasePred trains a single network over a shared pool of experts, gives
each outcome its own weighting of that pool, and returns a calibrated
probability for every outcome in one pass. It then writes you a dashboard for
*your* outcomes.

```bash
pip install "multidiseasepred[dashboard]"    # everything: model + dashboard
pip install multidiseasepred                 # lighter: selection, calibration,
                                             # evaluation, no torch needed
```

Or from a clone: `pip install -r requirements.txt && pip install -e .`

---

## How it works, end to end

```
your CSV  ──►  multidiseasepred train  ──►  myapp/  ──►  streamlit run app.py
                       │                      │
                       │                      └──►  multidiseasepred predict  ──►  predictions.csv
                       │
        selects a parsimonious variable set, trains the multi-task
        network, calibrates it, evaluates on a held-out split, and
        writes a dashboard for your outcomes
```

**One command trains everything and writes the dashboard:**

```bash
multidiseasepred train data.csv --targets outcome_ --out myapp \
    --group "conditions=outcome_sepsis,outcome_aki" \
    --group "disposition=outcome_admission" \
    --vote outcome_sepsis outcome_aki
```

It prints the event count for every outcome, runs greedy forward selection,
reports which variables it kept and why, trains three seeds, calibrates on
held-out rows, and evaluates with bootstrap intervals. You get:

```
myapp/
├── app.py            the dashboard
├── variables.json    widget labels, units and ranges — edit these
├── model/            networks, scaler, calibrator
├── background.npy    reference rows for attribution
├── selection.txt     the whole selection curve, not just the answer
└── evaluation.json   AUROC/AUPRC with intervals, calibration, operating points
```

**One person at a time:**

```bash
cd myapp && streamlit run app.py
```

Type a subject's values in the sidebar, get a calibrated probability for every
outcome, ranked by lift over each outcome's own base rate, with a per-variable
attribution for whichever outcome you select.

**A whole file at once:**

```bash
multidiseasepred predict myapp/model new_patients.csv -o predictions.csv
```

Adds a `risk_<outcome>` column for each outcome, plus `lift_<outcome>` when base
rates were recorded. Your original columns are kept, so the output is a superset
of the input.

---

## Preparing your data

One CSV, one row per subject. Nothing else.

| Requirement | Why |
|---|---|
| **Outcome columns are 0/1** | They are binary events. Give them a shared prefix (`outcome_`) and `--targets outcome_` finds them all. |
| **Every other column is a predictor** | Drop identifiers and dates before you start, or they become candidate variables. |
| **All columns numeric** | Encode categories yourself — one-hot or ordinal. |
| **No missing values** | Impute before training. The package will not silently pick an imputation for you, because that choice belongs in your methods section. |
| **Each outcome has some positive cases** | Nothing can be learned or scored for an outcome that never happens. |

`train` checks all of this before it starts and tells you which columns to fix
rather than failing halfway through a training loop.

```
Your data is not ready:
  - non-numeric columns: ['sex', 'admit_date']. Encode categories as numbers
    (one-hot or ordinal) before training.
  - missing values in 3 predictor column(s): ['spo2', 'dbp', 'hr']. Impute them
    first; the model does not choose an imputation for you, because that choice
    belongs in your methods section.
```

Two things worth deciding before you run it, because they change the result and
belong in your write-up:

- **`--vote`** — which outcomes the variable selection is scored against. The
  set you get is optimal for those and no others. Defaults to all of them.
- **`--group`** — which outcomes get macro-averaged together. Outcomes that are
  different kinds of thing should not share an average.

---

## Five minutes in Python

```python
import numpy as np
from multidiseasepred import MultiDiseasePred

model = MultiDiseasePred(
    variables=feature_names,                 # your column names
    targets=["sepsis", "aki", "admission"],  # your outcomes; they may co-occur
    groups={"conditions": ["sepsis", "aki"], "disposition": ["admission"]},
)

model.select_variables(X_dev, y_dev, vote=["sepsis", "aki"])   # greedy, k by rule
model.fit(X_dev, y_dev)                                        # train + calibrate
model.set_prevalence(y_dev)

model.evaluate(X_test, y_test)          # AUROC/AUPRC with bootstrap intervals
model.write_dashboard("myapp", X=X_dev) # a Streamlit app for your outcomes
```

```bash
cd myapp && streamlit run app.py
```

That last step is the point. The app you get is built from *your* variable
names, *your* observed ranges and *your* base rates. It is plain source in a
folder you own — edit it, deploy it, delete our name from it.

---

## What each piece does, and why it works that way

### `select_variables` — greedy, not a ranking prefix

Ranking variables by importance and taking the top k is the wrong operation when
your candidates are correlated: two variables carrying the same information both
score highly and both get taken, so the budget is spent twice on one signal.

Greedy forward selection asks the marginal question instead — which single
remaining variable most improves held-out performance *given what is already
in*. A redundant variable falls to the bottom on its own.

```python
res = model.select_variables(X, y, vote=["sepsis", "aki"], threshold=0.98)
print(res.summary())        # the whole curve, not just the answer
res.k_at(0.95)              # what the rule would have returned at 95%
```

`vote` has no default. A variable set is only ever optimal for the thing it was
selected against, and that decision belongs in your methods section rather than
in a library's defaults. If some of your outcomes do not get a vote, say so when
you report the result.

### `fit` — calibrated on held-out data, averaged over seeds

The calibrator is fitted on a slice held out from training and applied unchanged
everywhere else. Fit it on the data you then score and you have measured
nothing: the calibration curve is drawn through the same points it is judged on.

Three seeds are trained and their logits averaged, because a single run of a
network this size moves by a few thousandths between initialisations and you
should not have to wonder whether a result is the model or the seed.

### `evaluate` — groups stay separate, intervals do the talking

```python
r = model.evaluate(X_test, y_test)
r["macro"]["conditions"]["AUROC"]     # 0.9000 (0.8966–0.9034)
r["operating_point"]["sepsis"]        # sens/spec/PPV/NPV at 90% sensitivity
r["calibration"]["sepsis"]            # slope, intercept, Brier, ECE
```

`groups` exists because averaging different kinds of outcome into one number
produces a figure that means nothing. Whether a condition is present depends on
the patient; whether they are admitted also depends on whether there is a bed.

`compare` calls a difference a difference only when the intervals do not
overlap. That is conservative in the right direction.

### `write_dashboard` — your outcomes, your app

```python
model.write_dashboard("myapp", X=X_dev, title="ICU Outcomes")
```

Writes `app.py`, `variables.json`, the saved model, a background sample for
attribution, and a `requirements.txt`. Widget types and ranges are inferred from
your training data: a 0/1 column becomes a checkbox, anything else a number box
with the observed range. Edit `variables.json` to fix labels and units without
touching the code.

Risks are ordered by lift over each outcome's base rate, so a rare outcome is
not buried under a common one — with the calibrated probability still shown as
the primary number.

### `explain` and `gate_usage` — check the claims

```python
model.explain(x, background=X_dev[:200])   # (n_targets, n_variables)
model.gate_usage(X_dev)                    # effective experts, pairwise cosine
```

"Task-adaptive" is a claim about behaviour, not architecture. `gate_usage`
measures how differently the outcomes actually weight the experts, so you can
report the number instead of asserting the property.

The gate is dense: every expert gets non-zero weight for every outcome, and
there is no top-k anywhere in the forward pass. What differs between outcomes is
the weighting, not the membership.

---

## Repository layout

```
multidiseasepred/    the package — this is the thing to use
examples/            build_ed_dashboard.py — builds our ED demo *using the package*
pipeline/            supplementary: the exact scripts behind the manuscript
notebook/            an earlier teaching walkthrough (see the warning below)
tests/               pytest suite
docs/                documentation site
```

**`pipeline/` is supplementary.** Those 29 scripts are what produced the numbers
in the paper, kept so the results can be checked. They are not the interface;
`multidiseasepred` is.

**`notebook/` does not reproduce the manuscript.** It predates the final method
and differs from it where it matters — it ranks variables by global SHAP
importance rather than greedy selection, and fits the calibrator inside the
external dataset. It is kept as a readable example, and the tutorial pages say
so at the top.

---

## Tests

```bash
pip install "multidiseasepred[dev]"
pytest tests/ -v
```

Tests needing PyTorch skip themselves when it is missing, so selection,
calibration and metrics can be checked with scikit-learn alone.

---

## The published application

MultiDiseasePred was developed on 2.2 million emergency department encounters
across three health systems, predicting nine outcomes from twelve variables
recorded at triage. Method details are in
[the published pipeline](docs/pipeline.md); `examples/ed_triage_dashboard/` shows
the resulting app.

---

## Data

No clinical data ships with this repository.

- MIMIC-IV-ED — https://physionet.org/content/mimiciv/3.1/
- MC-MED — https://physionet.org/content/mc-med/1.0.0/

both through PhysioNet under their own credentialing and data use agreements.
The University of Minnesota / M Health Fairview data cannot be shared: they
contain protected health information, and the agreement and IRB approval under
which they were obtained permit analysis only inside the University of Minnesota
secure computing environment, from which record-level data cannot be exported.

## Citation

Please cite the associated manuscript once it is available.

## License

MIT.
