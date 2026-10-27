
import pandas as pd
from dataset_utils import load_dataset
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
df = load_dataset("Advertising.csv")

# Standardize column names
def clean_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

df.columns = [clean_name(col) for col in df.columns]

# Handle common alternative column names
aliases = {
    "television": "tv",
    "tv_advertising": "tv",
    "radio_advertising": "radio",
    "newspaper_advertising": "newspaper",
    "sale": "sales",
    "total_sales": "sales"
}
df = df.rename(columns=aliases)

required = ["tv", "radio", "newspaper", "sales"]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns: {missing}\nAvailable: {df.columns.tolist()}"
    )

df = df[required].copy()

for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().drop_duplicates()

X = df[["tv", "radio", "newspaper"]]
y = df["sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def evaluate(features, model_name):
    model = LinearRegression()

    model.fit(X_train[features], y_train)

    train_pred = model.predict(X_train[features])
    test_pred = model.predict(X_test[features])

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    n = len(y_train)
    p = len(features)

    adjusted_r2 = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)

    rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    return {
        "Model": model_name,
        "Test RMSE": rmse,
        "Test R2": test_r2,
        "Train Adjusted R2": adjusted_r2
    }


results = [
    evaluate(["tv", "radio", "newspaper"], "All Features"),
    evaluate(["tv", "radio"], "Without Newspaper")
]

print(pd.DataFrame(results).round(4).to_string(index=False))
