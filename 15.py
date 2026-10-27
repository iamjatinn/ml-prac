import pandas as pd
from dataset_utils import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Load dataset
df = load_dataset("Social_Network_Ads.csv")

# 2. Dataset exploring
print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df[["Age", "EstimatedSalary", "Purchased"]].isnull().sum())

# 3. Clean dataset
df = df[["Age", "EstimatedSalary", "Purchased"]].copy()
df = df.dropna()

# 4. Define features and target
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Standardize features
scaler_age = StandardScaler()
scaler_salary = StandardScaler()
scaler_both = StandardScaler()

X_train_age = scaler_age.fit_transform(X_train[["Age"]])
X_test_age = scaler_age.transform(X_test[["Age"]])

X_train_salary = scaler_salary.fit_transform(
    X_train[["EstimatedSalary"]]
)
X_test_salary = scaler_salary.transform(
    X_test[["EstimatedSalary"]]
)

X_train_both = scaler_both.fit_transform(X_train)
X_test_both = scaler_both.transform(X_test)

# 7. SVM using Age only
model_age = SVC(kernel="rbf")

model_age.fit(X_train_age, y_train)

y_pred_age = model_age.predict(X_test_age)

accuracy_age = accuracy_score(y_test, y_pred_age)
precision_age = precision_score(
    y_test, y_pred_age, zero_division=0
)
recall_age = recall_score(
    y_test, y_pred_age, zero_division=0
)
f1_age = f1_score(
    y_test, y_pred_age, zero_division=0
)

# 8. SVM using Estimated Salary only
model_salary = SVC(kernel="rbf")

model_salary.fit(X_train_salary, y_train)

y_pred_salary = model_salary.predict(X_test_salary)

accuracy_salary = accuracy_score(y_test, y_pred_salary)
precision_salary = precision_score(
    y_test, y_pred_salary, zero_division=0
)
recall_salary = recall_score(
    y_test, y_pred_salary, zero_division=0
)
f1_salary = f1_score(
    y_test, y_pred_salary, zero_division=0
)

# 9. SVM using Age and Estimated Salary
model_both = SVC(kernel="rbf")

model_both.fit(X_train_both, y_train)

y_pred_both = model_both.predict(X_test_both)

accuracy_both = accuracy_score(y_test, y_pred_both)
precision_both = precision_score(
    y_test, y_pred_both, zero_division=0
)
recall_both = recall_score(
    y_test, y_pred_both, zero_division=0
)
f1_both = f1_score(
    y_test, y_pred_both, zero_division=0
)

# 10. Compare models
results = pd.DataFrame({
    "Features": [
        "Age Only",
        "Salary Only",
        "Age + Salary"
    ],
    "Accuracy": [
        accuracy_age, accuracy_salary, accuracy_both
    ],
    "Precision": [
        precision_age, precision_salary, precision_both
    ],
    "Recall": [
        recall_age, recall_salary, recall_both
    ],
    "F1 Score": [
        f1_age, f1_salary, f1_both
    ]
})

print("\nModel Comparison:")
print(results.round(3).to_string(index=False))

# 11. Identify the best model based on F1 Score
best_model = results.loc[results["F1 Score"].idxmax()]

print("\nBest Model based on F1 Score:")
print(best_model)
