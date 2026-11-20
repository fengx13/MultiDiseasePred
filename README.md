# MultiDiseasePred

**Live dashboard:** <https://fengx13.github.io/MultiDiseasePred/dashboard/>

One multi-task model that predicts many outcomes that can occur at the same time
(for example sepsis, AKI and admission), with a calibrated probability for each
outcome and a ready-to-run dashboard.

## Install

```bash
pip install "multidiseasepred[dashboard] @ git+https://github.com/fengx13/MultiDiseasePred"
```

Or from a clone: `pip install -r requirements.txt && pip install -e .`

## Prepare your data

One CSV, one row per subject, all columns numeric, no missing values.
Outcome columns are 0/1 and share a prefix (e.g. `outcome_`); every other
column is treated as a predictor, so drop IDs and dates first.

## Use

Train (selects variables, trains, calibrates, evaluates and writes an app):

```bash
multidiseasepred train data.csv --targets outcome_ --out myapp
```

Open the dashboard for one patient at a time:

```bash
cd myapp && streamlit run app.py
```

Score a whole file:

```bash
multidiseasepred predict myapp/model new_patients.csv -o predictions.csv
```

## In Python

```python
from multidiseasepred import MultiDiseasePred

model = MultiDiseasePred(variables=feature_names, targets=["sepsis", "aki", "admission"])
model.select_variables(X_dev, y_dev, vote=["sepsis", "aki"])
model.fit(X_dev, y_dev)
model.evaluate(X_test, y_test)
model.write_dashboard("myapp", X=X_dev)
```

Full documentation: <https://fengx13.github.io/MultiDiseasePred/>

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
