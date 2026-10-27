import pandas as pd
from dataset_utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load dataset
df = load_dataset("Advertising.csv")

# 2. Dataset exploring
print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

# 3. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()

# 4. Define features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Multiple Linear Regression model
model = LinearRegression()

model.fit(X_train, y_train)

# 7. Predict sales
y_pred = model.predict(X_test)

# 8. Calculate residuals
results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
}, index=y_test.index)

results["Residual"] = (
    results["Actual Sales"] - results["Predicted Sales"]
)

results["Absolute Residual"] = results["Residual"].abs()

print("\nAll Residuals:")
print(results)

# 9. Identify the five largest absolute residuals
top_5 = results.sort_values(
    by="Absolute Residual", ascending=False
).head(5)

print("\nTop 5 Largest Prediction Errors:")
print(top_5.round(3))

# 10. Plot actual versus predicted sales
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    color="red"
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()
