Files:
covtypesmall.csv --> Dataset, exctracted from the UCI Covertype dataset.
classifiers.py --> Train logistic regression classifiers and save predictions on test and calibration set in the predictions/ folder.
IRPexperiments.jl --> Calibrate logistic regression with IRP and recursive binning for K=2,3,4, log calibration/test cross-entropy/AUC and save figures in figures/ folder.
utils.jl --> Julia functions to run our experiments.

Dependencies:
Python --> numpy, pandas, sklearn
Julia --> Random, Plots, PlotlyJS, Measures, LaTeXStrings, Polyhedra, GLPK, NPZ, Printf
