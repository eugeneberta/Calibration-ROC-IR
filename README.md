# Classifier Calibration with ROC-Regularized Isotonic Regression
Code for experiments and figures of the paper "Classifier Calibration with ROC-Regularized Isotonic Regression", published at AISTATS 2024.

## Files
- `covtypesmall.csv`: Dataset, exctracted from the UCI Covertype dataset.
- `classifiers.py`: Train logistic regression classifiers and save predictions on test and calibration set in the `predictions/` folder.
- `IRPexperiments.jl`: Calibrate logistic regression with IRP and recursive binning for `K=2,3,4`. Log cross-entropy and AUC for calibration and test sets. Save figures in the `figures/` folder.
- `utils.jl`: Julia functions to run our experiments.

## Dependencies
- **Python**: Numpy, Pandas, Sklearn
- **Julia**: Random, Plots, PlotlyJS, Measures, LaTeXStrings, Polyhedra, GLPK, NPZ, Printf
