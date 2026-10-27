import pandas as pd
from dataset_utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

accuracy_linear = accuracy_score(y_test, y_pred_linear)

# 8. RBF Kernel SVM
rbf_model = SVC(kernel="rbf")

rbf_model.fit(X_train, y_train)

y_pred_rbf = rbf_model.predict(X_test)

accuracy_rbf = accuracy_score(y_test, y_pred_rbf)

# 9. Compare models
results = pd.DataFrame({
    "Model": ["Linear SVM", "RBF SVM"],
    "Accuracy": [accuracy_linear, accuracy_rbf]
})

print("\nModel Comparison:")
print(results.round(3).to_string(index=False))

# 10. Plot decision boundaries
X_plot = np.vstack((X_train, X_test))

x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

for model, title in [
    (linear_model, "Linear Kernel SVM"),
    (rbf_model, "RBF Kernel SVM")
]:
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(
        X_test[:, 0], X_test[:, 1],
        c=y_test, edgecolors="k"
    )

    plt.xlabel("Age (Standardized)")
    plt.ylabel("Estimated Salary (Standardized)")
    plt.title(title + " Decision Boundary")
    plt.show()
