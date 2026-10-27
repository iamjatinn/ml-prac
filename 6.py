import pandas as pd
from dataset_utils import load_dataset
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = load_dataset("used_cars.csv")

# 2. Dataset exploring
print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df[["year", "km_driven", "transmission",
          "owner", "selling_price"]].isnull().sum())

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

# 7. Linear Regression
lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# 8. Decision Tree Regression
dt_model = DecisionTreeRegressor(random_state=42)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)

# 9. Random Forest Regression
rf_model = RandomForestRegressor(
    n_estimators=100, random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# 10. Compare models
results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree Regression",
        "Random Forest Regression"
    ],
    "RMSE": [rmse_lr, rmse_dt, rmse_rf],
    "R2 Score": [r2_lr, r2_dt, r2_rf]
})

print("\nModel Comparison:")
print(results.round(3).to_string(index=False))

# 11. Identify the best models
best_rmse = results.loc[results["RMSE"].idxmin()]
best_r2 = results.loc[results["R2 Score"].idxmax()]

print("\nBest model based on RMSE:")
print(best_rmse)

print("\nBest model based on R2 Score:")
print(best_r2)
