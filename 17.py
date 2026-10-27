import pandas as pd
from dataset_utils import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score

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

# 7. Train RBF Kernel SVM
model = SVC(kernel="rbf")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 8. Identify false positives and false negatives
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
}, index=y_test.index)

false_positives = results[
    (results["Actual"] == 0) & (results["Predicted"] == 1)
]

false_negatives = results[
    (results["Actual"] == 1) & (results["Predicted"] == 0)
]

print("\nFalse Positives:")
print(false_positives)

print("\nFalse Negatives:")
print(false_negatives)

# 9. Calculate precision and recall
precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

print("\nPrecision:", round(precision, 3))
print("Recall:", round(recall, 3))

# 10. Discuss classification errors
print("\nPractical Implications:")
print("False Positives: Customers predicted to purchase but who do not.")
print("False Negatives: Customers predicted not to purchase but who do.")
print("Low precision indicates more false positives.")
print("Low recall indicates more false negatives.")
