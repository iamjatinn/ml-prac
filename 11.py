
import pandas as pd
from dataset_utils import load_dataset
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


df = load_dataset("used_cars.csv")


def clean_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


df.columns = [clean_name(col) for col in df.columns]

# Alternative names for target
df = df.rename(columns={
    "price": "selling_price",
    "sellingprice": "selling_price",
    "sale_price": "selling_price",
    "resale_price": "selling_price",
    "kms_driven": "km_driven",
    "kilometers_driven": "km_driven"
})

target = "selling_price"

if target not in df.columns:
    raise ValueError(
        f"Target '{target}' not found. Columns: {df.columns.tolist()}"
    )

# Remove ID-like columns
id_columns = [
    col for col in df.columns
    if col in ["id", "car_id", "index", "unnamed_0"]
]

df = df.drop(columns=id_columns)

# Extract car brand instead of one-hot encoding every car model
if "name" in df.columns:
    df["brand"] = df["name"].astype(str).str.split().str[0]
    df = df.drop(columns=["name"])

df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target]).drop_duplicates()

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def make_model(feature_names):
    numeric_cols = X_train[feature_names].select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = [
        col for col in feature_names if col not in numeric_cols
    ]

    from sklearn.impute import SimpleImputer

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("forest", RandomForestRegressor(
            n_estimators=100,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ))
    ])

    return model, numeric_cols, categorical_cols


all_features = X.columns.tolist()

full_model, numeric_cols, categorical_cols = make_model(all_features)

full_model.fit(X_train, y_train)

# Calculate importance of each original feature
encoder = full_model.named_steps["preprocessor"]
forest = full_model.named_steps["forest"]

importances = forest.feature_importances_

importance_rows = []
position = 0

# Numeric columns: one importance value each
for col in numeric_cols:
    importance_rows.append({
        "Feature": col,
        "Importance": importances[position]
    })
    position += 1

# Categorical columns: sum importance across encoded categories
cat_encoder = encoder.named_transformers_["cat"].named_steps["encoder"]

for col, categories in zip(categorical_cols, cat_encoder.categories_):
    count = len(categories)

    importance_rows.append({
        "Feature": col,
        "Importance": importances[position:position + count].sum()
    })

    position += count

importance_df = pd.DataFrame(importance_rows).sort_values(
    "Importance", ascending=False
)

print("\nFeature Importance:")
print(importance_df.round(4).to_string(index=False))

top3 = importance_df.head(3)["Feature"].tolist()

print("\nTop 3 Features:", top3)

top3_model, _, _ = make_model(top3)
top3_model.fit(X_train[top3], y_train)


def evaluate(model, features, name):
    predictions = model.predict(X_test[features])

    return {
        "Model": name,
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    }


results = [
    evaluate(full_model, all_features, "All Features"),
    evaluate(top3_model, top3, "Top 3 Features")
]

print("\nComparison:")
print(pd.DataFrame(results).round(4).to_string(index=False))
