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
top_n = 20
cv_splits = 10
path_results = "results/discovery_cohort"
# path_results = "results/validation_cohort"
best_params_file = f"parameters/selected_grid_search_parameters.tsv"
selected_features_file = f"results/new_features_selected.txt"
os.makedirs(path_results, exist_ok=True)

# Load dataset
dataset = pd.read_csv("./data/first_cohort.tsv", sep="\t")

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

# Check if best_parameters.tsv exists and load
best_params_dict = None
if os.path.exists(best_params_file):
    try:
        best_params = pd.read_csv(best_params_file, sep='\t')
        if not best_params.empty:
            best_params_dict = best_params.iloc[0].to_dict()
    except Exception as e:
        print(f"Failed to load best_parameters.tsv: {e}")

# Model tuning (if needed)
if best_params_dict:
    print("Using loaded best parameters.")
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(with_mean=bool(best_params_dict.get('preprocessor__num__with_mean', True))),
         selector(dtype_exclude=object)),
        ('cat', OneHotEncoder(handle_unknown='ignore'), selector(dtype_include=object)),
    ], verbose_feature_names_out=False)
    classifier = RandomForestClassifier(
        n_estimators=int(best_params_dict.get('classifier__n_estimators', 200)),
        max_features=best_params_dict.get('classifier__max_features', 'sqrt'),
        max_depth=int(best_params_dict.get('classifier__max_depth', 5)),
        criterion=best_params_dict.get('classifier__criterion', 'gini'),
        random_state=42
    )
else:
    print("Using default parameters.")
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), selector(dtype_exclude=object)),
        ('cat', OneHotEncoder(handle_unknown='ignore'), selector(dtype_include=object)),
    ], verbose_feature_names_out=False)
    classifier = RandomForestClassifier(random_state=42)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])
pipeline.fit(X_train, y_train)

# Feature importance
features_out = pipeline.named_steps['preprocessor'].get_feature_names_out()
importances = pd.Series(
    pipeline.named_steps['classifier'].feature_importances_,
    index=features_out
).sort_values(ascending=False)
importances.to_csv(f"{path_results}/feature_importances.tsv", sep='\t')

# Save top N features to text file
top_features = importances.head(top_n).index
with open(selected_features_file, 'w') as f:
    f.writelines('\n'.join(top_features))
print(f"✅ Saved top {top_n} selected features to {selected_features_file}")
