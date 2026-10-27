import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. Dataset exploring
print("First 5 rows:")
print(X.head())

print("\nDataset shape:", X.shape)

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Define tree depths
depths = [2, 3, 4, 5]
results = []

# 5. Train and evaluate Decision Tree models
for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    results.append({
        "Max Depth": depth,
        "Training Accuracy": train_accuracy,
        "Testing Accuracy": test_accuracy
    })

# 6. Compare results
results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.round(3).to_string(index=False))

# 7. Plot training and testing accuracy
plt.figure(figsize=(8, 6))

plt.plot(
    results_df["Max Depth"],
    results_df["Training Accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    results_df["Max Depth"],
    results_df["Testing Accuracy"],
    marker="o",
    label="Testing Accuracy"
)

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Effect of Decision Tree Depth on Iris Classification")
plt.xticks(depths)
plt.legend()
plt.grid(True)
plt.show()

# 8. Identify the best depth based on testing accuracy
best_model = results_df.loc[
    results_df["Testing Accuracy"].idxmax()
]

print("\nBest Depth based on Testing Accuracy:")
print(best_model)
