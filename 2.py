import pandas as pd
from dataset_utils import load_dataset
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = load_dataset("Advertising.csv")

# 2. Dataset exploring
print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# 3. Handle missing values
df = df.dropna()

# 4. Define features and target
X = df[["TV", "Radio"]]
y = df["Sales"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Linear Regression
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

rmse_linear = np.sqrt(
    mean_squared_error(y_test, y_pred_linear)
)

r2_linear = r2_score(y_test, y_pred_linear)

# 7. Polynomial Regression
poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

rmse_poly = np.sqrt(
    mean_squared_error(y_test, y_pred_poly)
)

r2_poly = r2_score(y_test, y_pred_poly)

# 8. Compare both models
results = pd.DataFrame({
    "Model": ["Linear Regression", "Polynomial Regression"],
    "RMSE": [rmse_linear, rmse_poly],
    "R2 Score": [r2_linear, r2_poly]
})

print("\nModel Comparison:")
print(results.round(3).to_string(index=False))

# 9. Identify the better model based on RMSE
if rmse_poly < rmse_linear:
    print("\nPolynomial Regression performs better based on RMSE.")
else:
    print("\nLinear Regression performs better or ties based on RMSE.")
