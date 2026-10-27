import pandas as pd
from dataset_utils import load_dataset
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = load_dataset("used_cars.csv")

# 2. Dataset exploring
print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

# 3. Clean dataset
df = df[["year", "km_driven", "transmission",
         "owner", "selling_price"]].copy()

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["km_driven"] = pd.to_numeric(df["km_driven"], errors="coerce")
df["selling_price"] = pd.to_numeric(
    df["selling_price"], errors="coerce"
)

df = df.dropna()
df = df.drop_duplicates()

# 4. Define features and target
X = df[["year", "km_driven", "transmission", "owner"]]
y = df["selling_price"]

# 5. Encode categorical features
X = pd.get_dummies(
    X, columns=["transmission", "owner"], drop_first=True
)

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train model on original training data
model_original = LinearRegression()

model_original.fit(X_train, y_train)

y_pred_original = model_original.predict(X_test)

rmse_original = np.sqrt(
    mean_squared_error(y_test, y_pred_original)
)

r2_original = r2_score(y_test, y_pred_original)

# 8. Identify outliers using IQR
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = (y_train < lower) | (y_train > upper)

print("\nLower Bound:", lower)
print("Upper Bound:", upper)
print("Number of Outliers:", outliers.sum())

# 9. Remove outliers from training data only
X_train_clean = X_train.loc[~outliers]
y_train_clean = y_train.loc[~outliers]

print("Training rows before:", len(y_train))
print("Training rows after:", len(y_train_clean))

# 10. Train model on cleaned training data
model_clean = LinearRegression()

model_clean.fit(X_train_clean, y_train_clean)

y_pred_clean = model_clean.predict(X_test)

rmse_clean = np.sqrt(
    mean_squared_error(y_test, y_pred_clean)
)

r2_clean = r2_score(y_test, y_pred_clean)

# 11. Compare both models
results = pd.DataFrame({
    "Model": ["Original Data", "After Outlier Removal"],
    "RMSE": [rmse_original, rmse_clean],
    "R2 Score": [r2_original, r2_clean]
})

print("\nModel Comparison:")
print(results.round(3).to_string(index=False))
