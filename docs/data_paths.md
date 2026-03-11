# Data Paths & Output Configuration

This page explains how to configure dataset paths and output artifact locations used in the **MultiDiseasePred** pipeline.

All dataset and artifact paths are defined near the beginning of the main notebook:

MultiDiseasePred_Train_Validation.ipynb

Users should modify these paths according to their local environment before running the pipeline.

---

# 1. Input Dataset Paths

The notebook expects CSV files containing feature columns and outcome labels.

Example configuration:

```python
UMN_TRAIN_CSV = "data/umn/train.csv"
UMN_TEST_CSV  = "data/umn/test.csv"

MIMIC_CSV     = "data/master_dataset.csv"
STANFORD_CSV  = "data/stanford/stanford_features_labels.csv"
```

## 1.1 Training Dataset

```python
UMN_TRAIN_CSV
```
This dataset is used to:  

- train the MultiDiseasePred model  

- compute preprocessing statistics  

- fit the feature scaler  

- generate model checkpoints  

The training dataset should contain:  

- all required feature columns  

- outcome label columns

## 1.2 Internal Test Dataset  

```python
UMN_TEST_CSV
```

This dataset is used for:  

- internal evaluation  

- applying the fitted calibration model  

- computing AUROC / AUPRC metrics  

- generating bootstrap confidence intervals  

## 1.3 External Validation Datasets

External datasets can be evaluated by updating the corresponding path.  

Example datasets include:  

```python
MIMIC_CSV
STANFORD_CSV
```

These datasets allow evaluation of model generalization across different clinical cohorts.  

External datasets may not contain all training features. Missing columns must be restored before evaluation.  

---

# 2. Expected CSV Structure

Each dataset should contain the following types of columns.  

## 2.1 Feature Columns

Structured tabular variables such as:  

- demographics  

- triage vital signs  

- comorbidity indicators  

- utilization history  

- other engineered features  

Example feature names:  

```bash
age
triage_heartrate
triage_resprate
triage_temperature
triage_o2sat
cci_score
n_ed_90d
```

## 2.2 Outcome Columns

Binary outcome labels prefixed with:  

```bash
outcome_
```
Example outcome columns: 

```bash
outcome_hospitalization
outcome_critical
outcome_sepsis
outcome_aki
outcome_pe
```

Each outcome column should contain:  

```bash
0 = negative
1 = positive
```
---

# 3. Output Directory

The notebook saves all generated artifacts under a user-defined output directory.  

Example configuration:  

```python
OUTPUT_DIR = "output/"
```

If the directory does not exist, the notebook will create it automatically.  

---

# 4. Saved Artifacts

The pipeline produces several reusable artifacts during training.  

## 4.1 Model Checkpoint

```python
MODEL_CKPT_PATH = OUTPUT_DIR + "best_multidiseasepred.pt"
```

This file stores the trained PyTorch model weights selected using validation performance.  

The checkpoint is reused for:  

- calibration    

- internal evaluation  

- external validation  

## 4.2 Feature Scaler

```python
FEATURE_SCALER_PATH = OUTPUT_DIR + "feature_scaler_umn.pkl"
```

This file stores preprocessing information learned from the training dataset, including:  

- fitted feature scaler  

- feature medians  

- feature column order  

The scaler must be reused for:

- validation  

- test evaluation  

- external validation  

The scaler should **never be refit on external datasets**.  

## 4.3 Calibration Model

```python
CALIBRATOR_PATH = OUTPUT_DIR + "calibrator_isotonic.pkl"
```

This file stores task-wise isotonic regression models used to calibrate prediction scores.  

Calibration is fitted on validation outputs and then applied to:  

- internal test predictions  

- external validation predictions

---

# 5. Feature Alignment

External datasets may not contain all training features.  

To ensure consistency, the notebook uses a helper function:  

```python
ensure_feature_columns(df, required_cols)
```
This function:  

- adds missing feature columns  

- fills missing columns with **NaN**  

- restores the correct training feature order  

This step is critical for preventing shape mismatches during evaluation.  

---

# 6. Important Consistency Rules

To obtain valid results, the following rules must be respected.  

## Feature Order Consistency

The feature order used during training must be preserved during:  

- validation  

- testing  

- external evaluation

## Preprocessing Consistency

Only the training dataset should be used to compute preprocessing statistics.  

External datasets must reuse the saved scaler.  

## Calibration Consistency

Calibration input scale must remain consistent.  

If calibrators are fitted on logits:  

```bash
apply calibrator to logits
```
If calibrators are fitted on probabilities:  

```bash
apply calibrator to probabilities
```
Mixing scales may produce incorrect performance metrics.   

---

# 7. Typical Directory Layout

A typical project structure may look like:  

```bash
MultiDiseasePred/
│
├── data/
│   ├── umn/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   ├── master_dataset.csv
│   └── stanford/
│       └── stanford_features_labels.csv
│
├── output/
│   ├── best_multidiseasepred.pt
│   ├── feature_scaler_umn.pkl
│   └── calibrator_isotonic.pkl
│
└── MultiDiseasePred_Train_Validation.ipynb
```

---

# 8. Summary

Before running the pipeline, users should confirm:  

- dataset paths are correctly configured  

- required feature columns exist  

- outcome labels are properly formatted  

- output directory is writable  

Once configured, the notebook can be executed sequentially to run the complete MultiDiseasePred workflow.  

