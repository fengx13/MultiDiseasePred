# External Validation

This tutorial explains how **external validation** is performed in the MultiDiseasePred pipeline.

External validation evaluates how well a model trained on the development dataset generalizes to **independent datasets** such as MIMIC or other hospital cohorts.

The workflow implemented in the notebook is:

1. Load the external dataset
2. Align features with the training feature schema
3. Apply the saved preprocessing pipeline
4. Generate prediction logits from the trained model
5. Fit a task-wise calibrator using prediction logits and targets
6. Apply the calibrator to obtain calibrated probabilities
7. Compute evaluation metrics

All steps are implemented in the notebook:

MultiDiseasePred_Train_Validation.ipynb

---

# 1. Load the External Dataset

The external dataset is loaded using the configured path.

```python
df_external = pd.read_csv(str(MIMIC_CSV))
```

Users may replace this with other external datasets, for example:

```python
STANFORD_CSV
```

The dataset should contain:

- the same feature columns used during training
- outcome label columns
- a patient or encounter identifier column

---

# 2. Configure External Evaluation

External evaluation is controlled using a configuration dictionary.

```python
cfg = {
    "label_cols": outcome_list,
    "id_col": "subject_id",
    "val_frac": 0.2,
    "seed": 42,
    "batch_size": 256,
    "feature_scaler_path": str(FEATURE_SCALER_PATH),
    "stratify_mode": "any_positive",
    "preprocess_fn": preprocess_other,
}
```

Important fields:

- label_cols: List of outcome columns predicted by the model.  

- id_col: Patient or encounter identifier used for group-based splitting.  

- feature_scaler_path: Path to the saved feature scaler fitted on the training dataset.  

- preprocess_fn: Preprocessing function used to ensure consistent feature transformation.  

---

# 3. Align Features with Training Schema

External datasets may not contain exactly the same features as the training dataset.

Before evaluation, the pipeline ensures that:

- all required feature columns exist
- missing columns are restored
- feature order matches training

This step is essential for preventing shape mismatches during inference.

---

# 4. Build External Dataloaders

The external dataset is converted into PyTorch dataloaders.

```python
ext_test_dl = build_dataloader_from_df_generic(
    ext_test_df,
    label_cols=cfg["label_cols"],
    feature_scaler_path=cfg["feature_scaler_path"],
    batch_size=cfg["batch_size"],
    preprocess_fn=cfg["preprocess_fn"],
)
```

These dataloaders ensure that:

- the training-time scaler is reused
- preprocessing remains consistent
- features are correctly ordered

---

# 5. Load the Trained Model

The trained model checkpoint is loaded before external inference.

```python
model.load_state_dict(
    torch.load(
        str(MODEL_CKPT_PATH),  # Change to your saved model path
        weights_only=True,
        map_location=torch.device("cpu")
    )
```

The model is then switched to evaluation mode:

```python
model.eval()
```

---

# 6. Generate External Predictions

The trained model is applied to the external dataset to generate **prediction logits**.

During inference:

- logits are collected for each batch
- labels are collected simultaneously

```python
test_logits.append(logits)
test_targets.append(y_batch)
```

After inference, predictions are aggregated:

```python
test_logits = np.concatenate(test_logits, axis=0)
test_targets = np.concatenate(test_targets, axis=0)
```

The resulting arrays have shape:

```bash
(N, T)
```

where:

N = number of samples  
T = number of prediction tasks

---

# 7. Fit the Calibration Model

Calibration is fitted directly using the predicted logits and the ground truth labels.

For each prediction task:

1. extract logits for that task
2. fit an isotonic regression model
3. store the calibrator

Conceptually:

fit isotonic regression using  
```test_logits[:, task]``` and ```test_targets[:, task]```

Isotonic regression learns a **monotonic mapping from prediction scores to calibrated probabilities** and is commonly used for probability calibration in classification models.

---

# 8. Apply the Calibrator

After fitting the calibrator, it is applied to the logits to obtain calibrated probabilities.

Example logic:

```python
scores = test_logits[:, t_idx]

test_probs[:, t_idx] = calibrator[task_name].transform(scores)
```

If a task does not have a valid calibrator (for example, due to only one class in the dataset), the pipeline falls back to the raw sigmoid probabilities.

```python
raw_probs = expit(test_logits)
```

---

# 9. Compute Evaluation Metrics

Evaluation metrics are computed using the calibrated probabilities.

Typical metrics include:

1. AUROC  
Area under the receiver operating characteristic curve.

2. AUPRC  
Area under the precision-recall curve.

3. Normalized PRC  
AUPRC normalized by outcome prevalence.

4. Sensitivity / Specificity  
Threshold-based classification metrics.

5. Bootstrap Confidence Intervals  
Confidence intervals estimated using bootstrap resampling.

output:

```python
Task 00 (outcome_hospitalization)
AUROC = 0.85
AUPRC = 0.60
Sensitivity = 0.79
Specificity = 0.76
```

---

# 10. Important Notes

Calibration must always use the same input scale for fitting and inference.

In this pipeline:

- fit calibrator using logits   
- apply calibrator to logits

Mixing logits and probabilities during calibration may produce incorrect results.

---

# 11. Summary

The external validation workflow is:

load external dataset  
↓  
align feature columns  
↓  
apply training scaler  
↓  
run model to obtain logits  
↓  
fit calibrator using logits and targets  
↓  
transform logits to calibrated probabilities  
↓  
compute evaluation metrics

This workflow evaluates how well the trained MultiDiseasePred model generalizes to independent clinical datasets.
