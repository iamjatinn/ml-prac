import pandas as pd
from dataset_utils import load_dataset
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = load_dataset("Advertising.csv")

# 2. Dataset exploring
print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nDataset information:")
df.info()

# 3. Check and handle missing values
print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()

# 4. Define feature combinations
feature_combinations = [
    ["TV"],
    ["Radio"],
    ["Newspaper"],
    ["TV", "Radio"],
    ["TV", "Newspaper"],
    ["Radio", "Newspaper"],
    ["TV", "Radio", "Newspaper"]
]

y = df["Sales"]
results = []

# 5. Train and evaluate regression models
for features in feature_combinations:

    X = df[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Features": " + ".join(features),
        "RMSE": rmse,
        "R2 Score": r2
    })

# 6. Compare all models
results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.round(3).to_string(index=False))

# 7. Identify the best models
best_rmse = results_df.loc[results_df["RMSE"].idxmin()]
best_r2 = results_df.loc[results_df["R2 Score"].idxmax()]

print("\nBest model based on RMSE:")
print(best_rmse)

print("\nBest model based on R2 Score:")
print(best_r2)
