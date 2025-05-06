import os
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Parameters
target_variable = "Response"
case_group = "R"
path_results = "results/discovery_cohort"
# path_results = "results/validation_cohort"
selected_features_file = f"results/selected_features.txt"
best_params_file = f"parameters/selected_grid_search_parameters.tsv"
os.makedirs(path_results, exist_ok=True)

# Load dataset
dataset = pd.read_csv("./data/first_cohort.tsv", sep="\t")
# dataset = pd.read_csv("./data/second_cohort.tsv", sep="\t")

# Prepare data
labels = dataset[target_variable].unique()
control_group = [l for l in labels if l != case_group][0]
dataset.index = dataset[target_variable]
dataset.drop(columns=target_variable, inplace=True)
subset_labels = [control_group, case_group]
dataset = dataset.loc[subset_labels]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    dataset.reset_index(drop=True), dataset.index, stratify=dataset.index, test_size=0.3, random_state=30
)

# Imputation
imputer = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), selector(dtype_exclude=object)),
    ('cat', SimpleImputer(strategy='most_frequent'), selector(dtype_include=object)),
], verbose_feature_names_out=False)
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns).astype(X_train.dtypes.to_dict())
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns).astype(X_test.dtypes.to_dict())

# Load selected features
if not os.path.exists(selected_features_file):
    raise FileNotFoundError(f"{selected_features_file} not found. Run feature_selection.py first.")
with open(selected_features_file, 'r') as f:
    selected_features = [line.strip() for line in f.readlines()]

# Check best parameters
best_params_dict = None
if os.path.exists(best_params_file):
    try:
        best_params = pd.read_csv(best_params_file, sep='\t')
        if not best_params.empty:
            best_params_dict = best_params.iloc[0].to_dict()
    except Exception as e:
        print(f"Failed to load best_parameters.tsv: {e}")

# Preprocessing pipeline
if best_params_dict:
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(with_mean=bool(best_params_dict.get('preprocessor__num__with_mean', True))),
         selector(dtype_exclude=object)),
        ('cat', OneHotEncoder(handle_unknown='ignore'), selector(dtype_include=object)),
    ], verbose_feature_names_out=False)
else:
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), selector(dtype_exclude=object)),
        ('cat', OneHotEncoder(handle_unknown='ignore'), selector(dtype_include=object)),
    ], verbose_feature_names_out=False)

# Fit preprocessing
preprocessor.fit(X_train)
X_train_scaled = pd.DataFrame(preprocessor.transform(X_train), columns=preprocessor.get_feature_names_out())
X_test_scaled = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out())

# Generate combinations
combos = [list(c) for i in range(1, len(selected_features) + 1) for c in combinations(selected_features, i)]

# Evaluate combinations
results = []
total_combos = len(combos)
for idx, combo in enumerate(combos, 1):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_scaled[combo], y_train)
    y_pred = clf.predict(X_test_scaled[combo])
    y_proba = clf.predict_proba(X_test_scaled[combo])[:, 1]
    results.append({
        'Classifier': clf.__class__.__name__,
        'N_of_Features': len(combo),
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred, pos_label=case_group),
        'Precision': precision_score(y_test, y_pred, pos_label=case_group, zero_division=0),
        'Recall': recall_score(y_test, y_pred, pos_label=case_group),
        'AUC': roc_auc_score(y_test, y_proba),
        'Combination': ', '.join(combo),
    })
    print(f"Evaluated combination {idx} out of {total_combos}: {combo}")

# Save results
results_df = pd.DataFrame(results).sort_values(by=['N_of_Features', 'Combination'])
results_df.to_csv(f"{path_results}/report_combination_metrics.tsv", sep='\t', index=False)
print("Combination analysis complete!")
