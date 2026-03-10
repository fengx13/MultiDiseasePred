# Installation

This page explains how to set up the environment required to run the **MultiDiseasePred** pipeline.

The current implementation is centered around a reproducible Jupyter notebook:

MultiDiseasePred_Train_Validation.ipynb

---

## 1. Clone the Repository

First, clone the GitHub repository to your local machine.

```bash
git clone https://github.com/fengx13/MultiDiseasePred.git
cd MultiDiseasePred
```
## 2. Create a Python Environment

It is recommended to create a clean Python environment before installing dependencies.  

**Using** ```venv```

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

## 3. Install Required Packages

Install the main dependencies used in the notebook pipeline.  

```bash
pip install --upgrade pip

pip install torch \
numpy \
pandas \
scikit-learn \
scipy \
joblib \
matplotlib \
shap \
jupyter \
notebook
```
These libraries support:  

- model training (PyTorch)  

- data processing (NumPy, Pandas)  

- evaluation metrics (scikit-learn)  

- calibration and utilities (SciPy, Joblib)  

- visualization and interpretability (Matplotlib, SHAP)  

- notebook execution (Jupyter)  


## 4. Launch Jupyter Notebook

Start the Jupyter environment:   

```bash
jupyter notebook
```
Then open the main pipeline notebook:  

MultiDiseasePred_Train_Validation.ipynb  

Run all cells sequentially from top to bottom to execute the full workflow.  
