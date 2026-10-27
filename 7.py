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
df = df[["year", "km_driven", "selling_price"]].copy()

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["km_driven"] = pd.to_numeric(df["km_driven"], errors="coerce")
df["selling_price"] = pd.to_numeric(
    df["selling_price"], errors="coerce"
)

df = df.dropna()

print("\nMissing values:")
print(df.isnull().sum())

# 4. Calculate vehicle age
reference_year = 2026

df["vehicle_age"] = reference_year - df["year"]

print("\nDataset with Vehicle Age:")
print(df.head())

# 5. Define features and target
X = df[["km_driven", "vehicle_age"]]
y = df["selling_price"]

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Model using kilometres driven only
model_km = LinearRegression()

model_km.fit(X_train[["km_driven"]], y_train)

y_pred_km = model_km.predict(X_test[["km_driven"]])

rmse_km = np.sqrt(mean_squared_error(y_test, y_pred_km))
r2_km = r2_score(y_test, y_pred_km)

# 8. Model using age and kilometres driven
model_both = LinearRegression()

model_both.fit(X_train, y_train)

y_pred_both = model_both.predict(X_test)

rmse_both = np.sqrt(mean_squared_error(y_test, y_pred_both))
r2_both = r2_score(y_test, y_pred_both)

# 9. Compare both models
results = pd.DataFrame({
    "Model": [
        "Kilometres Only",
        "Age + Kilometres"
    ],
    "RMSE": [rmse_km, rmse_both],
    "R2 Score": [r2_km, r2_both]
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
