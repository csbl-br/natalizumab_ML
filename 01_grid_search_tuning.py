import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Parameters
target_variable = "Response"
case_group = "R"
cv_splits = 10
path_results = "parameters"
best_params_file = f"{path_results}/selected_grid_search_parameters.tsv"
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

# Define pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), selector(dtype_exclude=object)),
    ('cat', OneHotEncoder(handle_unknown='ignore'), selector(dtype_include=object)),
], verbose_feature_names_out=False)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define parameter grid
param_grid = {
    'classifier__n_estimators': [200, 500],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__max_depth': [4, 5, 6, 7, 8],
    'classifier__criterion': ['gini', 'entropy'],
    'preprocessor__num__with_mean': [True, False],
}

# Run GridSearchCV
grid = GridSearchCV(pipeline, param_grid, cv=cv_splits, n_jobs=-1)
grid.fit(X_train, y_train)

# Save best parameters
best_params_dict = grid.best_params_
best_params_df = pd.DataFrame({k: [v] for k, v in best_params_dict.items()})
best_params_df.to_csv(best_params_file, sep='\t', index=False)

print(f"Best parameters saved to {best_params_file}")
