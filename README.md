# MultiDiseasePred

MultiDiseasePred is a multi-task learning framework for clinical outcome prediction using structured tabular clinical data. The repository provides a reproducible pipeline for training and evaluating multi-task models on healthcare datasets, with support for probability calibration and external validation.

The implementation is centered around a Jupyter notebook workflow that integrates model training, evaluation, and calibration within a single reproducible pipeline.

---

## Overview

![Framework](https://github.com/fengx13/MultiDiseasePred/blob/main/framework_overview.png)

MultiDiseasePred predicts multiple clinical outcomes simultaneously using a shared representation learned from structured patient features. The framework is designed for clinical prediction tasks such as emergency department risk prediction and hospital outcome forecasting.

The pipeline includes:

- structured feature preprocessing
- multi-task neural network training
- validation-based model checkpoint selection
- probability calibration using isotonic regression
- external validation on independent datasets
- task-wise performance evaluation

The main pipeline is implemented in the notebook:  
MultiDiseasePred_Train_Validation.ipynb  

---

## Pipeline Workflow

The typical workflow for running the MultiDiseasePred pipeline is:

data loading  
↓  
feature preprocessing  
↓  
model training  
↓  
checkpoint selection  
↓  
external dataset loading  
↓  
prediction logits generation  
↓  
probability calibration  
↓  
performance evaluation  
↓  
parsimounious feature selection


Each stage of the workflow is documented in detail in the repository tutorials, with step-by-step instructions and runnable examples provided for each module.

Users can follow the tutorials in order to reproduce the full pipeline:

- [Data loading and preprocessing](https://github.com/fengx13/MultiDiseasePred/blob/main/docs/data_paths.md)
- [Model training](https://github.com/fengx13/MultiDiseasePred/blob/main/docs/tutorial/01_training.md)
- [Calibration](https://github.com/fengx13/MultiDiseasePred/blob/main/docs/tutorial/02_calibration.md)  
- [External validation](https://github.com/fengx13/MultiDiseasePred/blob/main/docs/tutorial/03_external_validation.md) 
- [SHAP-based parsimonious feature selection](https://github.com/fengx13/MultiDiseasePred/blob/main/docs/tutorial/04_parsimonious_feature_selection.md) 

---

## Installation

Clone the repository:

```bash
git clone https://github.com/fengx13/MultiDiseasePred.git
cd MultiDiseasePred
```

Create a Python environment and install dependencies:  

```bash
pip install torch numpy pandas scikit-learn scipy joblib matplotlib jupyter
```

Launch Jupyter:  

```bash
jupyter notebook
```

Then open: ```MultiDiseasePred_Train_Validation.ipynb```

Run all cells sequentially to execute the full pipeline.  

---

# Dataset Configuration

Before running the notebook, update dataset paths in the configuration section.   

Example configuration:  

```python
UMN_TRAIN_CSV = "data/umn/train.csv"
UMN_TEST_CSV  = "data/umn/test.csv"

MIMIC_CSV     = "data/master_dataset.csv"
STANFORD_CSV  = "data/stanford/stanford_features_labels.csv"
```

Each dataset should include:  

- feature columns used during training  

- outcome label columns (prefixed with ```outcome_```)  

- an identifier column for group-based splitting  

Example outcome labels:  

```bash
outcome_hospitalization
outcome_critical
outcome_sepsis
outcome_aki
outcome_pe
```

---

# Output Artifacts

The pipeline saves trained models and preprocessing artifacts to the configured output directory.  

Example configuration:  

```python
OUTPUT_DIR = "output/"

MODEL_CKPT_PATH     = OUTPUT_DIR + "best_multidiseasepred.pt"
FEATURE_SCALER_PATH = OUTPUT_DIR + "feature_scaler_umn.pkl"
CALIBRATOR_PATH     = OUTPUT_DIR + "calibrator_isotonic.pkl"
```
After running the notebook, the following files are typically generated:

```bash
output/
├── best_multidiseasepred.pt
├── feature_scaler_umn.pkl
└── calibrator_isotonic.pkl
```

These artifacts allow the trained model to be reused for future evaluation and deployment.  

---

# External Validation

The repository supports evaluation on external datasets such as MIMIC or other hospital cohorts.  

External validation involves:  

1. loading the external dataset  

2. aligning feature columns with the training schema  

3. applying the saved preprocessing scaler  

4. generating prediction logits using the trained model  

5. fitting a calibration model  

6. converting logits to calibrated probabilities  

7. computing evaluation metrics  

Typical evaluation metrics include:  

- AUROC  

- AUPRC  

- normalized PRC  

- sensitivity  

- specificity  

- bootstrap confidence intervals

---

# Documentation

Detailed documentation for the full pipeline is available at:
```bash
https://fengx13.github.io/MultiDiseasePred/
```

The documentation includes:  

- installation instructions  

- dataset configuration  

- training tutorial  

- calibration workflow  

- external validation procedure  

- demo instructions

---

# Repository Structure

```bash
MultiDiseasePred/
│
├── MultiDiseasePred_Train_Validation.ipynb
├── docs/
│   ├── index.md
│   ├── installation.md
│   ├── data_paths.md
│   ├── demo.md
│   └── tutorial/
│       ├── 01_training.md
│       ├── 02_calibration.md
│       └── 03_external_validation.md
│
├── mkdocs.yml
└── README.md
```

---

# Notes

This repository does not include clinical datasets. Users must provide their own datasets with the required feature and outcome columns.  

To ensure reproducibility:  

- maintain consistent feature ordering between training and evaluation  

- reuse the saved feature scaler for all evaluation datasets  

- ensure calibration uses the same score scale as the model outputs

---

# Citation

If you use this repository in your research, please cite the corresponding work once the associated manuscript or preprint is available.  

# License

This project is intended for research and academic use.  
