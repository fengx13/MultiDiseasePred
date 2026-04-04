## Parsimonious Feature Selection

This tutorial explains how the notebook performs **SHAP-based global feature ranking** and constructs parsimonious feature subsets for reduced-feature experiments.

All steps are implemented at the end of:

MultiDiseasePred_Train_Validation.ipynb

---

### What this module does

After training the full MultiDiseasePred model, the notebook:

1. wraps the multitask model so that outputs are consistently formatted as `(B, T)`
2. computes SHAP values using `shap.DeepExplainer`
3. converts SHAP outputs into a unified tensor with shape `(T, N, D)`, where:
   - `T` = number of tasks
   - `N` = number of explained samples
   - `D` = number of input features
4. computes **global feature importance** using:

   `mean(|SHAP|)` across tasks and samples

5. ranks all features from most important to least important
6. extracts:
   - **Top-10 features**
   - **Top-15 features**
7. saves the full ranking table and ranking plots

### Ranking outputs

The notebook generates:

- `global_shap_feature_ranking_all_63.csv`
- `global_shap_feature_ranking_top10.png`
- `global_shap_feature_ranking_top10.pdf`
- `global_shap_feature_ranking_top15.png`
- `global_shap_feature_ranking_top15.pdf`

### Purpose

These rankings can be used to define **parsimonious feature subsets** for follow-up experiments, such as:

- retraining the model using only the Top-10 features
- retraining the model using only the Top-15 features
- comparing reduced-feature performance against the full 63-feature model

### Current implementation note

The current notebook computes and exports the ranking results, including the Top-10 and Top-15 feature lists. These selected feature subsets can then be reused in downstream training experiments.

---

# 1. Goal

The purpose of this module is to identify a compact subset of important features after training the full MultiDiseasePred model.

Instead of manually selecting variables, the notebook uses **global SHAP importance** to rank all input features and extract:

- Top-10 features
- Top-15 features

These ranked subsets can then be used for parsimonious retraining and model comparison.

---

# 2. Wrap the Multi-Task Model Output

Because the model may return outputs in different formats (dictionary, tuple/list, or tensor), the notebook first defines a wrapper class:

```python
class MultiOut(nn.Module):
```

This wrapper converts the model output into a consistent tensor of shape:

`(B, T)`

where:

`B` = batch size  
`T` = number of tasks  

This step is required so that SHAP can correctly interpret the multi-task model. 

---

# 3. Prepare Background and Explanation Samples

The notebook then prepares:

- a background set for `shap.DeepExplainer`
- a separate explanation set for SHAP computation

Example configuration:

```python
bg_size = 256
exp_size = 1024
```

The background set defines the SHAP baseline, while the explanation set is used to estimate feature contributions.

---

# 4. Compute SHAP Values

SHAP values are computed using:

```python
explainer = shap.DeepExplainer(wrapped, bg_data)
shap_vals = explainer.shap_values(exp_data, check_additivity=False)
```

Depending on the SHAP output format, the notebook standardizes the result into a tensor with shape:

`(T, N, D)`

where:

- `T` = number of prediction tasks
- `N` = number of explained samples
- `D` = number of features

This standardized tensor is stored as:

`shap_TND`

---

# 5. Compute Global Feature Importance

Once `shap_TND` is available, the notebook computes global importance using:

```python
global_mean_abs = np.mean(np.abs(shap_TND), axis=(0, 1))
```

This computes:

mean(SHAP) across all tasks and all samples, which gives one global importance score per feature.

The notebook also computes task-specific SHAP summaries for reference:

```python
task_mean_abs = np.mean(np.abs(shap_TND), axis=1)
```
---

# 6. Build the Global Ranking Table

The notebook constructs a ranking table containing:

- raw feature name
- pretty-formatted feature name
- global mean absolute SHAP value
- global rank

Example:

```python
rank_df = pd.DataFrame({
    "feature": feature_names,
    "feature_pretty": [format_feature_name(f) for f in feature_names],
    "global_mean_abs_shap": global_mean_abs,
})
```

The table is then sorted in descending order of SHAP importance.

---

# 7. Extract Top-10 and Top-15 Features

After sorting, the notebook extracts:

```python
top10_features = rank_df.loc[:9, "feature"].tolist()
top15_features = rank_df.loc[:14, "feature"].tolist()
```

It also stores pretty-formatted versions for presentation:

```python
top10_pretty
top15_pretty
```

These lists are printed directly in the notebook output.

---

8. Save the Full Ranking Table

The full ranking table for all 63 features is saved as:

```python
global_shap_feature_ranking_all_63.csv
```

This file contains the complete SHAP-based global ranking and can be used for:

- documentation
- reporting
- follow-up parsimonious training experiments

---

# 9. Plot the Global Ranking

The notebook defines a plotting helper:

```python
def plot_global_ranking(rank_df, top_k=10, save_prefix="global_shap_feature_ranking"):
...
```

This function generates horizontal bar plots for the most important features.

The notebook currently generates:

- Top-10 ranking plot
- Top-15 ranking plot

---

# 10. Outputs Produced by This Module

After running the parsimonious feature selection cells, the notebook produces:

- a unified SHAP tensor: `shap_TND`
- a global ranking table: `rank_df`
- Top-10 feature list
- Top-15 feature list
- full ranking CSV
- Top-10 / Top-15 ranking figures

---

# 11. How to Use the Selected Features

The current notebook computes and exports the feature rankings.
To run a parsimonious experiment, users can manually replace the full feature set with one of the selected subsets, for example:

```python
selected_features = top10_features
```

and then rerun the training / calibration / evaluation pipeline using only those selected columns.

This enables direct comparison between:

- full 63-feature model
- Top-10 parsimonious model
- Top-15 parsimonious model

---

# 12. Summary

The parsimonious feature selection workflow is:

train full model  
↓  
wrap model outputs into (B, T)  
↓  
compute SHAP values  
↓  
standardize SHAP tensor to (T, N, D)  
↓  
compute global mean(SHAP)  
↓  
rank all 63 features  
↓  
extract Top-10 and Top-15 feature sets  
↓  
save ranking table and plots  

This module provides an interpretable and reproducible way to define reduced-feature versions of MultiDiseasePred.
