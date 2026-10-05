import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate

from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

FILE_PATH = r"C:\Users\Abhinav Bhatia\Desktop\sem5\ML\Assignment 1\BT2024156_train_var2.xlsx"

df = pd.read_excel(FILE_PATH)

# Input features
X = df[["x1", "x2", "x3"]].values

# Target variable
y = df["y"].values

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)
print()


# ============================================================
# 2. 5-FOLD CROSS VALIDATION
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 3. HYPERPARAMETERS
# ============================================================

# Regularization strengths
ALPHAS = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
]

# Elastic Net L1/L2 mixing ratios
L1_RATIOS = [
    0.1,
    0.25,
    0.5,
    0.75,
    0.9
]


# ============================================================
# 4. FUNCTION TO EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    degree,
    regularization,
    alpha=None,
    l1_ratio=None
):

    # --------------------------------------------------------
    # Select regression model
    # --------------------------------------------------------

    if regularization == "OLS":

        regression_model = LinearRegression()

    elif regularization == "L2":

        regression_model = Ridge(
            alpha=alpha,
            max_iter=10000
        )

    elif regularization == "L1":

        regression_model = Lasso(
            alpha=alpha,
            max_iter=200000,
            tol=1e-4,
            selection="random",
            random_state=42
        )

    elif regularization == "ElasticNet":

        regression_model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=200000,
            tol=1e-4,
            selection="random",
            random_state=42
        )

    else:

        raise ValueError("Invalid regularization type")


    # --------------------------------------------------------
    # Pipeline
    #
    # 1. Standardize original input features
    # 2. Generate polynomial features
    # 3. Standardize polynomial features
    # 4. Apply regression model
    # --------------------------------------------------------

    pipeline = Pipeline([
        (
            "input_scaler",
            StandardScaler()
        ),

        (
            "polynomial",
            PolynomialFeatures(
                degree=degree,
                include_bias=False
            )
        ),

        (
            "poly_scaler",
            StandardScaler()
        ),

        (
            "regression",
            regression_model
        )
    ])


    # --------------------------------------------------------
    # 5-FOLD CROSS VALIDATION
    #
    # MSE:
    #   neg_mean_squared_error
    #
    # R2:
    #   r2
    #
    # Both train and validation scores are returned.
    # --------------------------------------------------------

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=kf,

        scoring={
            "mse": "neg_mean_squared_error",
            "r2": "r2"
        },

        return_train_score=True,
        n_jobs=1
    )


    # --------------------------------------------------------
    # Average Train MSE
    # --------------------------------------------------------

    avg_train_mse = -cv_results["train_mse"].mean()


    # --------------------------------------------------------
    # Average Validation MSE
    # --------------------------------------------------------

    avg_validation_mse = -cv_results["test_mse"].mean()


    # --------------------------------------------------------
    # Average Train R2
    # --------------------------------------------------------

    avg_train_r2 = cv_results["train_r2"].mean()


    # --------------------------------------------------------
    # Average Validation R2
    # --------------------------------------------------------

    avg_validation_r2 = cv_results["test_r2"].mean()


    return {
        "degree": degree,
        "regularization": regularization,
        "alpha": alpha,
        "l1_ratio": l1_ratio,

        "train_mse": avg_train_mse,
        "validation_mse": avg_validation_mse,

        "train_r2": avg_train_r2,
        "validation_r2": avg_validation_r2
    }


# ============================================================
# 5. CREATE ALL MODEL CONFIGURATIONS
# ============================================================

tasks = []

for degree in range(1, 21):

    # --------------------------------------------------------
    # OLS
    # --------------------------------------------------------

    tasks.append(
        (
            degree,
            "OLS",
            None,
            None
        )
    )


    # --------------------------------------------------------
    # Ridge / L2
    # --------------------------------------------------------

    for alpha in ALPHAS:

        tasks.append(
            (
                degree,
                "L2",
                alpha,
                None
            )
        )


    # --------------------------------------------------------
    # Lasso / L1
    # --------------------------------------------------------

    for alpha in ALPHAS:

        tasks.append(
            (
                degree,
                "L1",
                alpha,
                None
            )
        )


    # --------------------------------------------------------
    # Elastic Net
    # --------------------------------------------------------

    for alpha in ALPHAS:

        for l1_ratio in L1_RATIOS:

            tasks.append(
                (
                    degree,
                    "ElasticNet",
                    alpha,
                    l1_ratio
                )
            )


# ============================================================
# 6. PRINT EXPERIMENT SETUP
# ============================================================

print("=" * 120)
print("QUESTION 2 - EXPERIMENT SETUP")
print("=" * 120)

print("Polynomial degrees     : 1 to 20")

print(
    f"OLS models             : "
    f"{20}"
)

print(
    f"Ridge models           : "
    f"{20 * len(ALPHAS)}"
)

print(
    f"Lasso models           : "
    f"{20 * len(ALPHAS)}"
)

print(
    f"Elastic Net models     : "
    f"{20 * len(ALPHAS) * len(L1_RATIOS)}"
)

print(
    f"Total models tested    : "
    f"{len(tasks)}"
)

print("Cross-validation       : 5-Fold")
print("Number of threads      : 5")
print("Lasso/ElasticNet       : Random coordinate selection")
print("Random state            : 42")

print("=" * 120)
print()
print("MODEL RESULTS")
print()


# ============================================================
# 7. RUN ALL MODELS USING 5 THREADS
# ============================================================

results = []

with ThreadPoolExecutor(max_workers=5) as executor:

    futures = []

    for degree, regularization, alpha, l1_ratio in tasks:

        future = executor.submit(
            evaluate_model,
            degree,
            regularization,
            alpha,
            l1_ratio
        )

        futures.append(future)


    # --------------------------------------------------------
    # Collect results as models finish
    # --------------------------------------------------------

    for future in as_completed(futures):

        result = future.result()

        results.append(result)


        # ----------------------------------------------------
        # OLS
        # ----------------------------------------------------

        if result["regularization"] == "OLS":

            print(
                f"Degree = {result['degree']:2d} | "
                f"Model = OLS         | "
                f"Train MSE = {result['train_mse']:.8f} | "
                f"Validation MSE = {result['validation_mse']:.8f} | "
                f"Train R2 = {result['train_r2']:.8f} | "
                f"Validation R2 = {result['validation_r2']:.8f}"
            )


        # ----------------------------------------------------
        # Ridge
        # ----------------------------------------------------

        elif result["regularization"] == "L2":

            print(
                f"Degree = {result['degree']:2d} | "
                f"Model = Ridge       | "
                f"Alpha = {result['alpha']:7g} | "
                f"Train MSE = {result['train_mse']:.8f} | "
                f"Validation MSE = {result['validation_mse']:.8f} | "
                f"Train R2 = {result['train_r2']:.8f} | "
                f"Validation R2 = {result['validation_r2']:.8f}"
            )


        # ----------------------------------------------------
        # Lasso
        # ----------------------------------------------------

        elif result["regularization"] == "L1":

            print(
                f"Degree = {result['degree']:2d} | "
                f"Model = Lasso       | "
                f"Alpha = {result['alpha']:7g} | "
                f"Train MSE = {result['train_mse']:.8f} | "
                f"Validation MSE = {result['validation_mse']:.8f} | "
                f"Train R2 = {result['train_r2']:.8f} | "
                f"Validation R2 = {result['validation_r2']:.8f}"
            )


        # ----------------------------------------------------
        # Elastic Net
        # ----------------------------------------------------

        else:

            print(
                f"Degree = {result['degree']:2d} | "
                f"Model = ElasticNet  | "
                f"Alpha = {result['alpha']:7g} | "
                f"L1 Ratio = {result['l1_ratio']:.2f} | "
                f"Train MSE = {result['train_mse']:.8f} | "
                f"Validation MSE = {result['validation_mse']:.8f} | "
                f"Train R2 = {result['train_r2']:.8f} | "
                f"Validation R2 = {result['validation_r2']:.8f}"
            )


# ============================================================
# 8. STORE RESULTS IN DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 9. SELECT BEST MODEL
#
# ONLY validation MSE is used.
# ============================================================

results_df = results_df.sort_values(
    by="validation_mse",
    ascending=True
).reset_index(drop=True)

best = results_df.iloc[0]


# ============================================================
# 10. PRINT FINAL BEST MODEL
# ============================================================

print()
print()
print("=" * 120)
print("FINAL BEST MODEL")
print("=" * 120)

print(
    f"Number of models tested : "
    f"{len(tasks)}"
)

print(
    f"Best polynomial degree  : "
    f"{int(best['degree'])}"
)

print(
    f"Best model              : "
    f"{best['regularization']}"
)

if best["regularization"] != "OLS":

    print(
        f"Best alpha              : "
        f"{best['alpha']}"
    )

if best["regularization"] == "ElasticNet":

    print(
        f"Best L1 ratio           : "
        f"{best['l1_ratio']}"
    )

print(
    f"Average Train MSE       : "
    f"{best['train_mse']:.10f}"
)

print(
    f"Average Validation MSE  : "
    f"{best['validation_mse']:.10f}"
)

print(
    f"Average Train R2        : "
    f"{best['train_r2']:.10f}"
)

print(
    f"Average Validation R2   : "
    f"{best['validation_r2']:.10f}"
)

print("=" * 120)


# ============================================================
# 11. TRAIN BEST MODEL ON COMPLETE TRAINING DATA
# ============================================================

best_degree = int(best["degree"])
best_regularization = best["regularization"]
best_alpha = best["alpha"]
best_l1_ratio = best["l1_ratio"]


# ------------------------------------------------------------
# Create final regression model
# ------------------------------------------------------------

if best_regularization == "OLS":

    best_regression = LinearRegression()

elif best_regularization == "L2":

    best_regression = Ridge(
        alpha=best_alpha,
        max_iter=10000
    )

elif best_regularization == "L1":

    best_regression = Lasso(
        alpha=best_alpha,
        max_iter=200000,
        tol=1e-4,
        selection="random",
        random_state=42
    )

elif best_regularization == "ElasticNet":

    best_regression = ElasticNet(
        alpha=best_alpha,
        l1_ratio=best_l1_ratio,
        max_iter=200000,
        tol=1e-4,
        selection="random",
        random_state=42
    )


# ------------------------------------------------------------
# Final pipeline
# ------------------------------------------------------------

best_model = Pipeline([
    (
        "input_scaler",
        StandardScaler()
    ),

    (
        "polynomial",
        PolynomialFeatures(
            degree=best_degree,
            include_bias=False
        )
    ),

    (
        "poly_scaler",
        StandardScaler()
    ),

    (
        "regression",
        best_regression
    )
])


# ============================================================
# 12. TRAIN ON COMPLETE TRAINING DATA
# ============================================================

best_model.fit(X, y)

print()
print("Best model trained on the complete training dataset.")
print("Variable 'best_model' is ready for prediction.")