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

# 7. Numerical-features-only baseline
model_num = LinearRegression()

model_num.fit(
    X_train[["year", "km_driven"]], y_train
)

y_pred_num = model_num.predict(
    X_test[["year", "km_driven"]]
)

rmse_num = np.sqrt(mean_squared_error(y_test, y_pred_num))
r2_num = r2_score(y_test, y_pred_num)

# 8. Model using all encoded features
model_encoded = LinearRegression()

model_encoded.fit(X_train, y_train)

y_pred_encoded = model_encoded.predict(X_test)

rmse_encoded = np.sqrt(
    mean_squared_error(y_test, y_pred_encoded)
)

r2_encoded = r2_score(y_test, y_pred_encoded)

# 9. Compare both models
results = pd.DataFrame({
    "Model": [
        "Numerical Features Only",
        "Encoded Features"
    ],
    "RMSE": [rmse_num, rmse_encoded],
    "R2 Score": [r2_num, r2_encoded]
})

print("\nModel Comparison:")
print(results.round(3).to_string(index=False))

# 10. Identify the better models
best_rmse = results.loc[results["RMSE"].idxmin()]
best_r2 = results.loc[results["R2 Score"].idxmax()]

print("\nBest model based on RMSE:")
print(best_rmse)

print("\nBest model based on R2 Score:")
print(best_r2)
