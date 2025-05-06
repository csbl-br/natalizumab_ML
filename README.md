# Natalizumab ML Analysis

This repository contains a reproducible set of scripts to analyze biomarker data and distinguish responders from non-responders to natalizumab treatment in multiple sclerosis. The analysis is based on two patient cohorts and aims to identify the most relevant features for predicting treatment response.

The approach uses machine learning techniques, specifically Random Forest classifiers, to evaluate the importance of features and their combinations. We already have a set of features that are known to be important for the analysis, and we will use these features to train our models. The scripts are designed to run sequentially, covering hyperparameter tuning, feature selection, and evaluation of feature combinations.

## Overview

The analysis is organized into three scripts that should be run in sequence:

1. **grid_search_tuning.py**  
   Performs hyperparameter tuning with `GridSearchCV` on a RandomForest classifier and saves the best parameters in the file `parameters/selected_grid_search_parameters.tsv`.

2. **feature_selection.py**  
   Selects the top 20 most important features using the best model parameters and saves them to `results/new_features_selected.txt`.

3. **machine_learning.py**  
   Loads the selected features, tests all possible combinations, evaluates them on multiple metrics, and outputs a full report.

## Folder Structure

- `data/` — Contains input cohort files (`first_cohort.tsv`, `second_cohort.tsv`).
- `parameters/` — Stores selected hyperparameters (`selected_grid_search_parameters.tsv`).
- `results/` — Saves feature importance, selected features, and combination results.

## Requirements

All dependencies are listed in `environment.yml`.  
To set up the environment:

```bash
conda env create -f environment.yml
conda activate natalizumab_ml
```

## Running the scripts

1. Run grid search tuning:
```bash
python 01_grid_search_tuning.py
```

2. Run feature selection:
```bash
python 02_feature_selection.py
```

3. Run machine learning combination evaluation:
```bash
python 03_machine_learning.py
```

## Outputs

- `parameters/selected_grid_search_parameters.tsv` — Best hyperparameters.
- `results/feature_importances.tsv` — Feature importance scores.
- `results/new_features_selected.txt` — Top N selected features.
- `results/report_combination_metrics.tsv` — Performance metrics for all feature combinations.

## Notes

- You can switch datasets (`first_cohort.tsv` or `second_cohort.tsv`) by editing the dataset loading line in each script.
- Make sure `selected_features.txt` in `results/` matches the feature file you want to test (you can rename `new_features_selected.txt` if needed).

## Contact

For questions or collaboration, please contact the repository owner.
