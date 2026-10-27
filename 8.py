import pandas as pd
from dataset_utils import load_dataset
import numpy as np

from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# 1. Load dataset
df = load_dataset("used_cars.csv")

# 2. Dataset exploring
print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

# 3. Clean dataset
df = df[["year", "km_driven", "transmission",
         "owner", "selling_price"]].copy()

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["km_driven"] = pd.to_numeric(df["km_driven"], errors="coerce")
df["selling_price"] = pd.to_numeric(
    df["selling_price"], errors="coerce"
)

df = df.dropna()
df = df.drop_duplicates()

# 4. Define features and target
X = df[["year", "km_driven", "transmission", "owner"]]
y = df["selling_price"]

# 5. Encode categorical features
X = pd.get_dummies(
    X, columns=["transmission", "owner"], drop_first=True
)

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Baseline Decision Tree
baseline_model = DecisionTreeRegressor(random_state=42)

baseline_cv_scores = cross_val_score(
    baseline_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

baseline_cv_rmse = -baseline_cv_scores.mean()

baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)

baseline_r2 = r2_score(y_test, y_pred_baseline)

# 8. Define hyperparameter grid
param_grid = {
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# 9. Apply GridSearchCV
grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=1
)

grid_search.fit(X_train, y_train)

# 10. Get the best model
tuned_model = grid_search.best_estimator_

tuned_cv_rmse = -grid_search.best_score_

y_pred_tuned = tuned_model.predict(X_test)

tuned_r2 = r2_score(y_test, y_pred_tuned)

print("\nBest Parameters:")
print(grid_search.best_params_)

# 11. Compare baseline and tuned models
results = pd.DataFrame({
    "Model": ["Baseline", "Tuned Decision Tree"],
    "Cross-validation RMSE": [
        baseline_cv_rmse, tuned_cv_rmse
    ],
    "Test R2 Score": [baseline_r2, tuned_r2]
})

print("\nModel Comparison:")
print(results.round(3).to_string(index=False))
