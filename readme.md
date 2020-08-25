# Parking imputation modelling

## What is this?
This repository contains the code used for training and evaluating imputation models in my Master's Thesis (which can be
 found here: PENDING).

## What is it useful for?
The code can be used for training and evaluating RandomForest and GradientBoostingMachine regressors on a dataset. The 
results are exported as a CSV and can then be compared to understand the performance of different data and modelling 
choices.

## How to run?
Begin with installing the requirements found in `requirements.txt`. Doing so in a virtual environment is probably the best
choice.

The repository has a single entry point at `run.py`, which should be run with an argument to indicate the path to a 
valud config file in this fashion:

`python3 run.py --config_file_path /path/to/config.json`

Before running, the following must be prepared:
 * A config file which must be in JSON format. A template can be found in the `config_example.json`
 file.
 * One or more experiment definition files, in JSON format. A template can be found in the `experiment_example.json`.




