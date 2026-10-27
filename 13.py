import pandas as pd
from dataset_utils import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score
)

# 1. Load dataset
df = load_dataset("Social_Network_Ads.csv")

# 2. Dataset exploring
print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

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

# 6. RBF SVM using unscaled features
model_unscaled = SVC(kernel="rbf")

model_unscaled.fit(X_train, y_train)

y_pred_unscaled = model_unscaled.predict(X_test)

cm_unscaled = confusion_matrix(y_test, y_pred_unscaled)
accuracy_unscaled = accuracy_score(y_test, y_pred_unscaled)
f1_unscaled = f1_score(
    y_test, y_pred_unscaled, zero_division=0
)

print("\nUnscaled SVM Confusion Matrix:")
print(cm_unscaled)

# 7. Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. RBF SVM using scaled features
model_scaled = SVC(kernel="rbf")

model_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = model_scaled.predict(X_test_scaled)

cm_scaled = confusion_matrix(y_test, y_pred_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
f1_scaled = f1_score(
    y_test, y_pred_scaled, zero_division=0
)

print("\nScaled SVM Confusion Matrix:")
print(cm_scaled)

# 9. Compare models
results = pd.DataFrame({
    "Model": ["Unscaled RBF SVM", "Scaled RBF SVM"],
    "Accuracy": [accuracy_unscaled, accuracy_scaled],
    "F1 Score": [f1_unscaled, f1_scaled]
})

print("\nModel Comparison:")
print(results.round(3).to_string(index=False))
