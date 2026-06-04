# Calibration

This tutorial explains how calibration is performed in the **external validation pipeline** of MultiDiseasePred.

Unlike a standard internal validation workflow, the calibrator in this notebook is **not fitted on the original model validation set from training**.  
Instead, calibration is performed **within the external dataset**.

More specifically, the external dataset is first split into:

- an **external validation subset**
- an **external test subset**

The calibrator is then fitted on the **external validation subset** and applied to the **external test subset**.

---

# 1. Why Calibration Is Needed

Deep learning models are often optimized for classification performance, but their raw outputs are not always well-calibrated as probabilities.

This means:

- predicted scores may rank patients correctly
- but the score values themselves may not reflect true event probabilities

Calibration is used to improve the correspondence between:

- predicted risk
- observed event frequency

---

# 2. Calibration Strategy Used in This Notebook

The MultiDiseasePred pipeline uses **task-wise isotonic regression** for calibration.

However, in this notebook, calibration is **not based on the internal training validation set**.

Instead, the workflow is:

1. Load the external dataset
2. Split the external dataset into:
   - external validation subset
   - external test subset
3. Run the trained model on the external validation subset
4. Collect external validation logits and targets
5. Fit one isotonic calibrator per task using the external validation outputs
6. Apply the fitted calibrator to the external test outputs

This design makes the calibrator specific to the external validation setting.

---

# 3. External Calibration Dataset Generation

The external dataset is loaded first:

```python
df_external = pd.read_csv(str(MIMIC_CSV))
```

A configuration dictionary is then defined, including:  

- outcome label columns  

- group ID column  

- validation fraction  

- random seed  

- batch size  

- feature scaler path  

- preprocessing function

Example structure:  

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

The external dataset is then split into:  

- ```ext_val_df```  

- ```ext_test_df```

using group-based splitting:  

```python
ext_val_df, ext_test_df = split_external_val_test_by_group(
    df_external,
    group_col=cfg["id_col"],
    val_frac=cfg["val_frac"],
    seed=cfg["seed"],
    outcome_cols=cfg["label_cols"],
    stratify_mode=cfg["stratify_mode"],
)
```
This ensures that the same group (for example, the same patient) does not appear in both subsets.  

---

# 4. Build External Validation and Test Dataloaders

After splitting, two separate dataloaders are built:  

- ```ext_val_dl```  

- ```ext_test_dl```  

These dataloaders reuse:  

- the same training-time feature scaler  

- the same preprocessing logic  

- the same feature alignment rules  

This ensures consistency between:  

- model training  

- external calibration  

- external testing

---

# 5. Fit the Calibrator on External Validation Outputs

The trained MultiDiseasePred model is first loaded from the saved checkpoint.  

Then the model is run on the external validation dataloader to obtain:  

- external validation logits  

- external validation targets  

These outputs are used to fit one isotonic regression model per task.  

Conceptually:  

```python
    for t_idx, task_name in enumerate(outcome_list):
        y_true = val_targets[:, t_idx]
        y_score = val_logits[:, t_idx]   # can use logits or sigmoid probabilities

        # Only perform calibration if both classes (0/1) are present
        if len(np.unique(y_true)) < 2:
            print(f"Task {task_name} only has one class, skip isotonic.")
            continue

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_score, y_true)
        calibrator[task_name] = iso
```
This means the calibrator is learned from the external validation subset, not from the internal validation data used during training.  

---

# 6. Apply the Calibrator to External Test Outputs

After calibration models are fitted on the external validation subset, they are applied to the outputs from the external test subset.  

The workflow is:  

1. Run the trained model on ```ext_test_dl```  

2. Collect external test logits  

3. Apply the task-wise isotonic calibrators  

4. Compute calibrated probabilities  

5. Evaluate performance on the external test subset  

This makes calibration and evaluation fully separated within the external dataset.  

---

# 7. Important Rule: Score Scale Must Remain Consistent

Calibration must always use the same input scale for both fitting and application.  

If the isotonic calibrator is fitted on:  

- logits → it must also be applied to logits  

- probabilities → it must also be applied to probabilities  

In this notebook, the calibration pipeline should remain consistent throughout the external validation workflow.  

Mixing scales may lead to incorrect calibration and misleading evaluation results.  

---

# 8. Why This Is Different from Standard Internal Calibration

In many machine learning pipelines, calibration is fitted using the internal validation set from model development.  

That is **not** the case here.  

In this notebook:  

- the model is trained on the original development dataset  

- the external dataset is then split into external validation and external test subsets  

- calibration is fitted inside the external dataset  

- evaluation is reported on the external test subset  

This makes the calibration step **domain-specific to the external dataset.**

---

# 9. Edge Cases

Some outcome tasks may contain only one class in the external validation subset.  

For example:  

- all labels are 0  

- all labels are 1  

In such cases, isotonic regression cannot be fitted for that task.  

The pipeline should then skip calibration for that task and fall back to the uncalibrated prediction scores.  

---

# 10. Output Artifacts

The external calibration workflow may generate:  

- saved task-wise isotonic calibrator  

- external validation metrics  

- external test metrics  

Example artifact paths include:

```bash
output/
├── best_multidiseasepred.pt
├── feature_scaler_umn.pkl
└── calibrator_isotonic.pkl
```

---

# 11. Summary

In this notebook, calibration is performed as part of the **external validation workflow**.  

The actual logic is:  

1. train the model on the development dataset  

2. load the external dataset  

3. split external data into external validation and external test subsets  

4. fit task-wise calibrators on external validation logits and targets  

5. apply the calibrator to external test outputs  

6. evaluate performance on the calibrated external test predictions  

This is different from a standard internal-validation-based calibration pipeline and should be interpreted as **external-domain-specific calibration**.

---

# 12. Next Step

After calibration, proceed to the external validation tutorial:  

[External Validation](https://github.com/fengx13/MultiDiseasePred/blob/main/docs/tutorial/03_external_validation.md) 
