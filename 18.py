import pandas as pd
from dataset_utils import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# 7. Logistic Regression
lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

cm_lr = confusion_matrix(y_test, y_pred_lr)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(
    y_test, y_pred_lr, zero_division=0
)
recall_lr = recall_score(
    y_test, y_pred_lr, zero_division=0
)
f1_lr = f1_score(
    y_test, y_pred_lr, zero_division=0
)

print("\nLogistic Regression Confusion Matrix:")
print(cm_lr)

# 8. RBF Kernel SVM
svm_model = SVC(kernel="rbf")

svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

cm_svm = confusion_matrix(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(
    y_test, y_pred_svm, zero_division=0
)
recall_svm = recall_score(
    y_test, y_pred_svm, zero_division=0
)
f1_svm = f1_score(
    y_test, y_pred_svm, zero_division=0
)

print("\nRBF SVM Confusion Matrix:")
print(cm_svm)

# 9. Compare models
results = pd.DataFrame({
    "Model": ["Logistic Regression", "RBF SVM"],
    "Accuracy": [accuracy_lr, accuracy_svm],
    "Precision": [precision_lr, precision_svm],
    "Recall": [recall_lr, recall_svm],
    "F1 Score": [f1_lr, f1_svm]
})

print("\nModel Comparison:")
print(results.round(3).to_string(index=False))

# 10. Identify the better model based on F1 Score
best_model = results.loc[results["F1 Score"].idxmax()]

print("\nBest Model based on F1 Score:")
print(best_model)
