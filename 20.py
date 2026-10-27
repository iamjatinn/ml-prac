import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

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

# 4. Decision Tree using Gini
gini_model = DecisionTreeClassifier(
    criterion="gini", max_depth=3, random_state=42
)

gini_model.fit(X_train, y_train)

y_pred_gini = gini_model.predict(X_test)

cm_gini = confusion_matrix(y_test, y_pred_gini)
accuracy_gini = accuracy_score(y_test, y_pred_gini)

print("\nGini Confusion Matrix:")
print(cm_gini)

print("\nGini Accuracy:", round(accuracy_gini, 3))

# 5. Decision Tree using Entropy
entropy_model = DecisionTreeClassifier(
    criterion="entropy", max_depth=3, random_state=42
)

entropy_model.fit(X_train, y_train)

y_pred_entropy = entropy_model.predict(X_test)

cm_entropy = confusion_matrix(y_test, y_pred_entropy)
accuracy_entropy = accuracy_score(y_test, y_pred_entropy)

print("\nEntropy Confusion Matrix:")
print(cm_entropy)

print("\nEntropy Accuracy:", round(accuracy_entropy, 3))

# 6. Compare models
results = pd.DataFrame({
    "Criterion": ["Gini", "Entropy"],
    "Accuracy": [accuracy_gini, accuracy_entropy]
})

print("\nModel Comparison:")
print(results.round(3).to_string(index=False))

# 7. Visualize Decision Trees
plt.figure(figsize=(12, 6))

plot_tree(
    gini_model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title("Decision Tree using Gini")
plt.show()

plt.figure(figsize=(12, 6))

plot_tree(
    entropy_model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title("Decision Tree using Entropy")
plt.show()

# 8. Compare splitting decisions
print("\nGini and Entropy may select different features and thresholds.")
print("Compare the tree plots to observe their splitting decisions.")
