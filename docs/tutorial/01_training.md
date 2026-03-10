# Training Pipeline

This tutorial explains how to run the **model training stage** of the MultiDiseasePred pipeline.

All steps are implemented in the notebook:

MultiDiseasePred_Train_Validation.ipynb

The notebook contains the complete workflow for data preprocessing, model training, and checkpoint selection.

---

# 1. Open the Training Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Then open the notebook:  

MultiDiseasePred_Train_Validation.ipynb  

Run cells **sequentially from top to bottom**.  

# 2. Configure Dataset Paths

Before running the training cells, update the dataset paths defined near the beginning of the notebook.  

Example configuration:  

```python
UMN_TRAIN_CSV = "data/umn/train.csv"
UMN_TEST_CSV  = "data/umn/test.csv"
```

The training dataset will be used to:  

- train the MultiDiseasePred model  

- compute preprocessing statistics  

- fit the feature scaler  

- generate model checkpoints

# 3. Load the Dataset

The training CSV file is loaded into a pandas DataFrame.  

Typical steps include:  

- reading the CSV file  

- verifying feature columns  

- verifying outcome label columns  

- separating feature variables and target labels  

Example workflow:  

```python
df_train = pd.read_csv(UMN_TRAIN_CSV)
```
# 4. Define Feature and Outcome Columns

The pipeline requires explicit definitions for:  

- feature columns  

- outcome columns

Example:  

```python
feature_cols = [...]
outcome_list = [...]
```
Outcome columns should follow the naming pattern:  
```bash
outcome_*
```

# 5. Align Feature Columns

External datasets may not contain every feature used during training.  

To ensure consistency, the notebook applies a feature alignment function:  

```python
ensure_feature_columns(df, required_cols)
```

This prevents shape mismatches during training and evaluation.  

# 6. Feature Preprocessing

After feature alignment, preprocessing is applied.  

Typical preprocessing steps include:  

- numeric type conversion  

- missing value handling  

- feature scaling  

A training-time feature scaler is fitted and saved for reuse.  

Example:  

```python
scaler = StandardScaler()
X_np = scaler.fit_transform(feat_df.to_numpy(dtype=float))

# Save scaler & preprocessing metadata for reuse
# (e.g., other notebooks or external validation)
joblib.dump(
    {
        "scaler": scaler,
        "median": feat_median,
        "feat_cols": feat_cols
    },
    str(FEATURE_SCALER_PATH)
)
```

# 7. Create PyTorch Dataloaders

After preprocessing, the dataset is converted to PyTorch tensors.  

Dataloaders are created for training and validation.  

Each batch contains:  

- feature tensor ```(batch_size, num_features)```

- label tensor ```(batch_size, num_tasks)```


# 8. Initialize the Model

The MultiDiseasePred model is then initialized.  

Typical configuration parameters include:  

- input feature dimension  

- hidden layer dimension  

- number of prediction tasks

- number of experts

- optional regularization settings
 
The model produces **task-specific logits**:

```bash
logits: (batch_size, num_tasks)
```
Each column corresponds to a predicted task.  

# 9. Train the Model

Training is performed using the helper function:  

```python
    # Run one training epoch
    tr_loss, tr_aucs, tr_prcs = run_epoch(
        train_dl,
        model,
        optimizer,
        criterion,
        "train",
        l1_lambda=1e-4,
        aux_lambda=1.0
    )

    # Run one validation epoch
    vl_loss, vl_aucs, vl_prcs = run_epoch(
        val_dl,
        model,
        optimizer,
        criterion,
        "val",
        l1_lambda=1e-4,
        aux_lambda=1.0
    )
```

This function handles:  

- forward pass  

- loss computation  

- gradient updates  

- validation evaluation  

- prediction collection


The loss function is typically ```BCEWithLogitsLoss``` which directly operates on logits.

# 10. Monitor Validation Performance

During training, validation metrics are tracked to identify the best model checkpoint.  

Typical metrics include:  

- AUROC  

- AUPRC  

- task-wise prediction performance  

The model achieving the best validation performance is saved.  

# 11. Save the Best Model

The best checkpoint is saved to the configured output directory.  

Example:  
```python
save_path = str(MODEL_CKPT_PATH)  # where to save best checkpoint
```
This checkpoint will be reused for:  

- calibration  

- internal evaluation  

- external validation

# 12. Training Output

After completing the training pipeline, the following artifacts will be generated:  
```bash
output/
├── best_multidiseasepred.pt
├── feature_scaler_umn.pkl
```
These artifacts are required for downstream steps in the pipeline.

# 13. Next Step

Once training is complete, proceed to the calibration tutorial:  

Calibration  
