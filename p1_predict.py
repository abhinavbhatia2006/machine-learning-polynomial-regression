import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline


# FILE PATHS

FOLDER = Path(r"C:\Users\Abhinav Bhatia\Desktop\sem5\ML\Assignment 1")

TRAIN_BASENAME = "BT2024156_train_var1"
TEST_BASENAME = "BT2024156_test_var1"
OUTPUT_BASENAME = "BT2024156_pred_var1"


# HELPER: FIND CSV/XLSX FILE

def find_data_file(folder, basename):
    candidates = [
        folder / f"{basename}.xlsx",
        folder / f"{basename}.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find {basename}.xlsx or {basename}.csv in:\n{folder}"
    )


def read_data(path):
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


# 1. LOAD TRAINING AND TEST DATA

train_path = find_data_file(FOLDER, TRAIN_BASENAME)
test_path = find_data_file(FOLDER, TEST_BASENAME)

train_df = read_data(train_path)
test_df = read_data(test_path)

features = ["x1", "x2", "x3", "x4", "x5", "x6"]

X_train = train_df[features].values
y_train = train_df["y"].values

X_test = test_df[features].values


# 2. BEST MODEL FROM PART 1 RESULTS
#
# Best model:
#   Polynomial degree = 5
#   L1 / Lasso
#   alpha = 0.01
#
# This reproduces the model selected during Part 1.

best_model = Pipeline([
    (
        "input_scaler",
        StandardScaler()
    ),

    (
        "polynomial",
        PolynomialFeatures(
            degree=5,
            include_bias=False
        )
    ),

    (
        "poly_scaler",
        StandardScaler()
    ),

    (
        "regression",
        Lasso(
            alpha=0.01,
            max_iter=200000,
            tol=1e-4,
            selection="random",
            random_state=42
        )
    )
])


# 3. TRAIN ON COMPLETE TRAINING DATA

best_model.fit(X_train, y_train)


# 4. PREDICT TEST DATA

predictions = best_model.predict(X_test)


# 5. CREATE SUBMISSION FILE
#
# The sample submission contains exactly one column:
#     y

output_path = FOLDER / f"{OUTPUT_BASENAME}.csv"

prediction_df = pd.DataFrame({
    "y": predictions
})

prediction_df.to_csv(
    output_path,
    index=False
)


# 6. VERIFY OUTPUT

print("=" * 70)
print("PART 1 PREDICTION COMPLETE")
print("=" * 70)

print(f"Training file : {train_path}")
print(f"Test file     : {test_path}")
print(f"Output file   : {output_path}")

print()
print("Model:")
print("  Degree      : 5")
print("  Regression  : Lasso (L1)")
print("  Alpha       : 0.01")

print()
print(f"Test samples  : {len(predictions)}")
print(f"Output shape  : {prediction_df.shape}")
print()

print("First 5 predictions:")
print(prediction_df.head())

print()
print(f"Saved to: {output_path}")
print("=" * 70)
