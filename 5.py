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

# 3. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()

# 4. Define features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# 5. Create a fixed test set
X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Shuffle the training pool once
X_train_pool = X_train_pool.sample(frac=1, random_state=42)

# 6. Define training sizes
train_sizes = [0.5, 0.7, 0.8]
results = []

# 7. Train and evaluate models
for size in train_sizes:

    n_train = int(size * len(df))

    X_train = X_train_pool.iloc[:n_train]
    y_train = y_train_pool.loc[X_train.index]

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Training Size (%)": int(size * 100),
        "Training Rows": n_train,
        "RMSE": rmse,
        "R2 Score": r2
    })

# 8. Compare results
results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.round(3).to_string(index=False))

# 9. Identify the best models
best_rmse = results_df.loc[results_df["RMSE"].idxmin()]
best_r2 = results_df.loc[results_df["R2 Score"].idxmax()]

print("\nBest model based on RMSE:")
print(best_rmse)

print("\nBest model based on R2 Score:")
print(best_r2)
