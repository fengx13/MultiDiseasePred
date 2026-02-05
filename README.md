# MultiDiseasePred: Multi-Task Emergency Outcome Prediction

This repository provides an end-to-end implementation of MultiDiseasePred, a multi-task deep learning framework for predicting multiple acute clinical outcomes from emergency department (ED) data.

The pipeline covers:

- Data loading and preprocessing

- Multi-task model training

- Isotonic calibration

- Reproducible internal/external validation evaluation

All core functionality is demonstrated in a single executable notebook.

## Environment Setup
Python Requirements:
- Python ≥ 3.9
- PyTorch ≥ 1.12
- NumPy, Pandas
- scikit-learn
- joblib
- scipy
- os
- matplotlib/seaborn (optional, for plots)<br>

Install dependencies:<br>
```bash
pip install torch numpy pandas scikit-learn joblib scipy os matplotlib seaborn
```

## Step-by-Step Pipeline
Important: Run the notebook from top to bottom without skipping cells.
### Step 1: Import Libraries & Load and Preprocess Data
Notebook section: Libs Imports
- Import PyTorch, NumPy, Pandas, sklearn, scipy
- Set random seeds for reproducibility
- Define device (CPU / GPU)

Notebook section: Data Loading<br>
Configure the data and artifacts paths for training and evaluation：
- 'UMN_TRAIN_CSV': the path for model training data csv.
- 'UMN_TEST_CSV' : the path for internal validation data csv.
- 'MIMIC_CSV'& 'STANFORD_CSV': the paths for external validation data csv.
- 'OUTPUT_DIR': the path for saving output artifacts. If the directory does not exist, it will be created automatically.
- 'MODEL_CKPT_PATH': the path for PyTorch model weights checkpoint. 
- 'FEATURE_SCALER_PATH': the path for feature scaler. It should be fitted only on the training set.
- 'CALIBRATOR_PATH': the path for Isotonic Regression calibrator. If the calibrator file exists, it will be loaded automatically. Otherwise, it will be fitted and saved during the validation stage.

Each CSV file should contain:
- Feature columns (numeric, aligned to training features)
- Outcome columns prefixed with outcome_ or other type (binary labels)

### Step 2: Create Training Dataloaders
Notebook section: Training Data Preparation
- Define feature and outcome lists through 'col_list' and 'outcome_list'.
- Convert NumPy arrays to tensors and build DataLoader objects for training and validation.
- 'df = pd.read_csv(str(UMN_TRAIN_CSV))': load your training csv file by changing UMN_TRAIN_CSV.
- 'joblib.dump(..., str(FEATURE_SCALER_PATH))' : the fitted standard scaler was saved in FEATURE_SCALER_PATH.
- Each batch yields: X_batch (batch_size, num_features), y_batch (batch_size, num_tasks)

### Step 3: Train the Model
Notebook section: Model Parameters Setup & Training Loop
- Define model hyperparameters, optimizer, loss function, schduler and number of epoch
- 'save_path = str(MODEL_CKPT_PATH)': define the path to save model checkpoint via MODEL_CKPT_PATH
- Model output: logits (batch, num_tasks).

### Step 4: Create Exteranl Validation Dataloaders & Fit Isotonic Calibrator
Notebook section: External dataset generation & Generarte Calibrator
- 'df_external = pd.read_csv(str(MIMIC_CSV))' : load the external/internal test data csv via MIMIC_CSV or other strings mentioned in Step 1.
- 'ext_val_df' should be used to fit isotonic calibrator. It contains 20% of test data. (Change the spilt ratio via 'val_frac')
- 'ext_test_df' is the final test set used for external validation
- 'calibrator_path = str(CALIBRATOR_PATH)' : save and load the fitted calibrator via CALIBRATOR_PATH. It will automatically created the calibrator pkl file.

### Step 5: Apply Calibration to Test set & Evaluation with Confidence Intervals
Notebook section: Apply isotonic calibration & Final Evaluation
- 'calibrator_path = str(CALIBRATOR_PATH)': load saved calibrator pkl file.
- 'test_logits' & 'test_targets': raw logits and ground-truth labels collected.
- 'test_probs' : the output probabilities after calibration.

Final evaluation metrics for each task:
- AUROC + 95% CI (bootstrap)
- AUPRC + 95% CI
- Normalized PRC (nPRC)
- Sensitivity / Specificity (Youden J)
- Optimal threshold
Macro-level metrics: Macro AUROC & Macro AUPRC


