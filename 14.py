
import pandas as pd
from dataset_utils import load_dataset
import numpy as np
import re

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


df = load_dataset("Social_Network_Ads.csv")


def clean_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


df.columns = [clean_name(col) for col in df.columns]

df = df.rename(columns={
    "purchase": "purchased",
    "buy": "purchased",
    "target": "purchased",
    "estimated_salary": "estimated_salary",
    "salary": "estimated_salary",
    "estimatedsalary": "estimated_salary"
})

target = "purchased"

if target not in df.columns:
    raise ValueError(
        f"Target '{target}' not found. Columns: {df.columns.tolist()}"
    )

# Exclude identifier columns
id_cols = [
    col for col in df.columns
    if col in ["id", "user_id", "customer_id", "unnamed_0"]
]

df = df.drop(columns=id_cols)
df = df.dropna(subset=[target]).drop_duplicates()

X = df.drop(columns=[target])
y = df[target]

if y.nunique() < 2:
    raise ValueError("Classification requires at least two target classes.")

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [col for col in X.columns if col not in numeric_cols]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Default SVM
default_model = Pipeline([
    ("preprocessor", preprocessor),
    ("svm", SVC(kernel="rbf"))
])

default_cv = cross_val_score(
    default_model, X_train, y_train,
    cv=cv, scoring="accuracy"
).mean()

default_model.fit(X_train, y_train)
default_pred = default_model.predict(X_test)

# Hyperparameter tuning
tuned_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("svm", SVC(kernel="rbf"))
])

param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": [0.001, 0.01, 0.1, 1]
}

grid = GridSearchCV(
    tuned_pipeline,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

tuned_pred = grid.predict(X_test)

# Weighted F1 works for binary and multiclass targets
results = pd.DataFrame([
    {
        "Model": "Default SVM",
        "CV Accuracy": default_cv,
        "Test Accuracy": accuracy_score(y_test, default_pred),
        "F1 Score": f1_score(
            y_test, default_pred,
            average="weighted", zero_division=0
        )
    },
    {
        "Model": "Tuned SVM",
        "CV Accuracy": grid.best_score_,
        "Test Accuracy": accuracy_score(y_test, tuned_pred),
        "F1 Score": f1_score(
            y_test, tuned_pred,
            average="weighted", zero_division=0
        )
    }
])

print("\nBest Parameters:", grid.best_params_)
print("\nComparison:")
print(results.round(4).to_string(index=False))
