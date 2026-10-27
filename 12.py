import pandas as pd
from dataset_utils import load_dataset
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
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
    X, y, test_size=0.2, random_state=42
)

# 6. Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Linear Kernel SVM
linear_model = SVC(kernel="linear")

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

cm_linear = confusion_matrix(y_test, y_pred_linear)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
precision_linear = precision_score(
    y_test, y_pred_linear, zero_division=0
)
recall_linear = recall_score(
    y_test, y_pred_linear, zero_division=0
)
f1_linear = f1_score(
    y_test, y_pred_linear, zero_division=0
)

print("\nLinear SVM Confusion Matrix:")
print(cm_linear)

# 8. RBF Kernel SVM
rbf_model = SVC(kernel="rbf")

rbf_model.fit(X_train, y_train)

y_pred_rbf = rbf_model.predict(X_test)

cm_rbf = confusion_matrix(y_test, y_pred_rbf)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
precision_rbf = precision_score(
    y_test, y_pred_rbf, zero_division=0
)
recall_rbf = recall_score(
    y_test, y_pred_rbf, zero_division=0
)
f1_rbf = f1_score(
    y_test, y_pred_rbf, zero_division=0
)

print("\nRBF SVM Confusion Matrix:")
print(cm_rbf)

# 9. Compare models
results = pd.DataFrame({
    "Model": ["Linear SVM", "RBF SVM"],
    "Accuracy": [accuracy_linear, accuracy_rbf],
    "Precision": [precision_linear, precision_rbf],
    "Recall": [recall_linear, recall_rbf],
    "F1 Score": [f1_linear, f1_rbf]
})

print("\nModel Comparison:")
print(results.round(3).to_string(index=False))

# 10. Identify the better model based on F1 Score
best_model = results.loc[results["F1 Score"].idxmax()]

print("\nBest Model based on F1 Score:")
print(best_model)
