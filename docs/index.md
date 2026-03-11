# MultiDiseasePred

MultiDiseasePred is a multi-task learning framework for clinical outcome prediction using structured tabular data from emergency department and hospital cohorts.

This repository provides a complete pipeline for:

- feature preprocessing and alignment
- multi-task model training
- validation and checkpoint selection
- isotonic probability calibration
- internal and external validation

The current implementation is centered around a reproducible notebook workflow.

---

## Repository Overview

The full pipeline is implemented in the notebook:
MultiDiseasePred_Train_Validation.ipynb

This notebook includes the entire workflow:

1. Data loading
2. Feature preprocessing
3. Multi-task model training
4. Model checkpoint selection
5. Calibration using validation data
6. External validation

---

## Pipeline Structure

The typical workflow for using this repository is shown below.

Data preparation  
↓  
Feature preprocessing  
↓  
Model training  
↓  
Validation evaluation  
↓  
Isotonic calibration  
↓  
Internal test evaluation  
↓  
External validation  


Each step is explained in detail in the tutorial section of this documentation.

---

## Getting Started

If you are new to the repository, follow this order:

1. **Installation**  
   Set up the Python environment and install dependencies.

2. **Data Paths & Outputs**  
   Configure dataset paths and output artifact locations.

3. **Training Pipeline**  
   Train the MultiDiseasePred model using your dataset.

4. **Calibration**  
   Fit task-wise isotonic regression calibrators using validation outputs.

5. **External Validation**  
   Evaluate the trained model on external datasets such as MIMIC.
   
---

## Run the Notebook Demo

The easiest way to reproduce the full pipeline is to run the notebook directly.

Launch Jupyter:

```bash
jupyter notebook
```

Open the notebook:

```bash
MultiDiseasePred_Train_Validation.ipynb
```

Then execute all cells sequentially from top to bottom.

---

## Expected Outputs

After running the notebook, the pipeline will produce:  

- trained model checkpoint  

- saved feature scaler  

- task-wise isotonic calibrator  

- external validation metrics

All artifacts are saved under the configured output directory.

---

## Important Notes

- This repository does not include **clinical datasets**.  

- Users must provide their own training and evaluation CSV files.  

- Feature order must remain consistent between training and evaluation.  

- The same scaler and calibrator must be reused for validation and external testing.

---

## Citation
If you use this repository in your research, please cite the corresponding MultiDiseasePred work.  

(Reference information can be added here once the manuscript or preprint is available.)  


