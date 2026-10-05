# Machine Learning — Polynomial Regression

This repository contains the implementation for **Machine Learning Assignment 1: Polynomial Regression**.

The assignment consists of two personalized regression problems using separate datasets. The objective is to build polynomial regression models for predicting a continuous target variable `y`, select an appropriate polynomial degree and regression technique using cross-validation, and use the trained model for prediction on the corresponding test dataset.

The assignment explicitly requires **polynomial regression**, with the polynomial degree allowed to go up to 10 for Part 1 and up to 20 for Part 2. The primary evaluation metrics are **Mean Squared Error (MSE)** and **R² Score**.   

---

## Repository Structure

```text
machine-learning-polynomial-regression/
│
├── p1.py
├── p2.py
└── README.md
```

### `p1.py`

Implementation for **Part 1 — Power Plant Steam Turbine Optimization (var1)**.

### `p2.py`

Implementation for **Part 2 — Subterranean Thermal Reservoir Mapping (var2)**.

---

# 1. Problem Statement

## Part 1 — Power Plant Steam Turbine Optimization

The first problem models the Net Power Score of a geothermal power plant.

The plant's turbine efficiency is affected by six operational parameters:

| Feature | Description |
|---|---|
| `x1` | High-pressure steam valve adjustment |
| `x2` | Condenser coolant flow rate adjustment |
| `x3` | Re-injection pump hydraulic pressure |
| `x4` | Turbine blade pitch angle |
| `x5` | Non-condensable gas exhaust valve rate |
| `x6` | Steam inlet pressure adjustment |

The target variable `y` represents the **Net Power Score**.

The task is to use historical plant calibration data to construct a polynomial regression model that predicts `y` from the six input variables. The polynomial degree can be selected up to degree 10.

### Part 1 input

```text
x1, x2, x3, x4, x5, x6
```

### Target

```text
y
```

---

# 2. Part 2 — Subterranean Thermal Reservoir Mapping

The second problem models the Thermal Anomaly Score measured by seismic and thermal sensors across a 3D geological survey region.

The three spatial variables are:

| Feature | Description |
|---|---|
| `x1` | East-West coordinate offset |
| `x2` | North-South coordinate offset |
| `x3` | Vertical depth offset relative to the basecamp |

The target variable `y` is the **Thermal Anomaly Score**.

The task is to use the randomly scattered training core-sample locations to build a polynomial regression model that predicts the thermal anomaly score at unseen coordinates. The polynomial degree can be selected up to degree 20.

### Part 2 input

```text
x1, x2, x3
```

### Target

```text
y
```

---

# 3. Dataset Format

The assignment provides separate training and testing datasets for the two problems.

The general naming convention is:

```text
<ROLLNO>_train_var1.csv
<ROLLNO>_test_var1.csv

<ROLLNO>_train_var2.csv
<ROLLNO>_test_var2.csv
```

The target variable is `y`, while the remaining columns contain the input features. The training and testing datasets are treated as separate datasets.

For this implementation, the training datasets are read from Excel files:

```text
BT2024156_train_var1.xlsx
BT2024156_train_var2.xlsx
```

The paths in the Python scripts currently point to the local assignment directory and may need to be changed when running on another machine.

---

# 4. Approach

The same general modelling strategy is used for both problems.

For every polynomial degree in the permitted range, multiple regression techniques are evaluated using **5-fold cross-validation**.

The tested regression approaches are:

1. **Ordinary Least Squares (OLS)**
2. **Ridge Regression (L2 regularization)**
3. **Lasso Regression (L1 regularization)**
4. **Elastic Net Regression (L1 + L2 regularization)**

The model with the **lowest average validation MSE** is selected as the final best model.

Train MSE and validation MSE are also recorded for analysis, while R² is reported as an additional performance metric.

---

# 5. Polynomial Feature Generation

Polynomial features are generated using:

```python
PolynomialFeatures(
    degree=degree,
    include_bias=False
)
```

A polynomial of degree `d` includes terms for which the sum of the powers of the input variables is at most `d`.

For example, with two variables and degree 2, polynomial features include terms such as:

```text
x1
x2
x1²
x1*x2
x2²
```

For higher degrees, the number of generated terms increases rapidly.

---

# 6. Feature Scaling

The implementation uses two `StandardScaler` transformations:

```text
Original input features
        ↓
StandardScaler
        ↓
PolynomialFeatures
        ↓
StandardScaler
        ↓
Regression model
```

The first scaler standardizes the original input variables before polynomial expansion.

The second scaler standardizes the generated polynomial features before applying the regression model.

Both scalers are part of the scikit-learn `Pipeline`, so preprocessing is performed correctly within the cross-validation procedure.

---

# 7. Regression Models

## Ordinary Least Squares

OLS provides the basic unregularized polynomial regression baseline.

```python
LinearRegression()
```

For each polynomial degree, one OLS model is evaluated.

---

## Ridge Regression — L2

Ridge applies L2 regularization to reduce the effect of excessively large coefficients.

```python
Ridge(
    alpha=alpha,
    max_iter=10000
)
```

The alpha values tested are:

```text
0.001
0.01
0.1
1
10
100
```

---

## Lasso Regression — L1

Lasso applies L1 regularization and can shrink polynomial coefficients toward zero.

```python
Lasso(
    alpha=alpha,
    max_iter=200000,
    tol=1e-4,
    selection="random",
    random_state=42
)
```

The same six alpha values are tested.

---

## Elastic Net

Elastic Net combines L1 and L2 regularization.

```python
ElasticNet(
    alpha=alpha,
    l1_ratio=l1_ratio,
    max_iter=200000,
    tol=1e-4,
    selection="random",
    random_state=42
)
```

The alpha values are:

```text
0.001
0.01
0.1
1
10
100
```

The following L1 ratios are tested:

```text
0.1
0.25
0.5
0.75
0.9
```

Thus the experiment evaluates different combinations of overall regularization strength and the balance between L1 and L2 regularization.

---

# 8. Cross-Validation

A **5-fold K-Fold cross-validation** strategy is used:

```python
KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The training dataset is divided into five folds.

For each iteration:

```text
4 folds → training
1 fold  → validation
```

This process is repeated five times so that every data point is used as validation data exactly once.

For each model, the validation MSE is calculated for all five folds and then averaged:

```text
Average Validation MSE
=
(MSE₁ + MSE₂ + MSE₃ + MSE₄ + MSE₅) / 5
```

The same process is used to calculate the average training MSE.

---

# 9. MSE

The primary model-selection metric is Mean Squared Error:

```text
MSE = average((y_actual - y_predicted)^2)
```

For a validation fold:

\[
MSE =
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
\]

A lower MSE indicates smaller prediction errors.

The **best model is always selected using the lowest average validation MSE**.

Train MSE is reported separately but is not used to determine the winning model.

---

# 10. R² Score

R², or the coefficient of determination, is also calculated.

\[
R^2 =
1-
\frac{\sum(y_i-\hat{y}_i)^2}
{\sum(y_i-\bar{y})^2}
\]

R² measures how well the model explains the variation in the target variable.

Higher R² indicates better explanatory performance.

Both training and validation R² are reported.

---

# 11. Parallel Processing

The different model configurations are evaluated concurrently using:

```python
ThreadPoolExecutor(max_workers=5)
```

Therefore, up to five model evaluations are processed concurrently.

Inside each individual cross-validation evaluation:

```python
n_jobs=1
```

is used so that the outer thread pool controls the parallelism.

This provides a maximum of five concurrent model-evaluation tasks without unnecessarily creating additional threads inside each task.

---

# 12. Number of Models Tested

## Part 1

Polynomial degrees:

```text
1 through 10
```

### OLS

```text
10 models
```

### Ridge

```text
10 degrees × 6 alpha values
= 60 models
```

### Lasso

```text
10 degrees × 6 alpha values
= 60 models
```

### Elastic Net

```text
10 degrees × 6 alpha values × 5 L1 ratios
= 300 models
```

### Total

```text
10 + 60 + 60 + 300 = 430 models
```

Therefore:

```text
Part 1 → 430 models
```

Each model is evaluated using 5-fold cross-validation.

---

## Part 2

Polynomial degrees:

```text
1 through 20
```

### OLS

```text
20 models
```

### Ridge

```text
20 × 6 = 120 models
```

### Lasso

```text
20 × 6 = 120 models
```

### Elastic Net

```text
20 × 6 × 5 = 600 models
```

### Total

```text
20 + 120 + 120 + 600 = 860 models
```

Therefore:

```text
Part 2 → 860 models
```

---

# 13. Model Selection Strategy

The final model is selected using only:

```text
Lowest Average Validation MSE
```

The procedure is:

```text
Generate polynomial features
          ↓
Train candidate model
          ↓
5-fold cross-validation
          ↓
Calculate average validation MSE
          ↓
Compare all candidate models
          ↓
Choose minimum validation MSE
          ↓
Train selected model on complete training dataset
```

No train-validation gap or percentage threshold is used for final model selection.

---

# 14. Results

## Part 1 — var1

### Final Best Model

```text
Number of models tested : 430
Best polynomial degree  : 5
Best model              : L1
Best alpha              : 0.01

Average Train MSE       : 0.2271912594
Average Validation MSE  : 0.3037583528

Average Train R2        : 0.9779475432
Average Validation R2   : 0.9701815630
```

### Summary

| Metric | Result |
|---|---:|
| Models tested | 430 |
| Best degree | 5 |
| Best model | Lasso / L1 |
| Alpha | 0.01 |
| Average Train MSE | 0.2271912594 |
| Average Validation MSE | 0.3037583528 |
| Average Train R² | 0.9779475432 |
| Average Validation R² | 0.9701815630 |

The selected Part 1 model therefore uses a **degree-5 polynomial with L1 regularization and α = 0.01**.

---

# 15. Part 2 — var2

### Final Best Model

```text
Number of models tested : 860
Best polynomial degree  : 11
Best model              : ElasticNet
Best alpha              : 0.001
Best L1 ratio            : 0.1

Average Train MSE       : 0.1655382392
Average Validation MSE  : 0.2644830804

Average Train R2        : 0.9964517668
Average Validation R2   : 0.9942217064
```

### Summary

| Metric | Result |
|---|---:|
| Models tested | 860 |
| Best degree | 11 |
| Best model | Elastic Net |
| Alpha | 0.001 |
| L1 ratio | 0.1 |
| Average Train MSE | 0.1655382392 |
| Average Validation MSE | 0.2644830804 |
| Average Train R² | 0.9964517668 |
| Average Validation R² | 0.9942217064 |

The selected Part 2 model therefore uses a **degree-11 polynomial with Elastic Net regularization, α = 0.001 and L1 ratio = 0.1**.

---

# 16. Running the Code

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas scikit-learn openpyxl
```

`openpyxl` is required because the training datasets are stored as `.xlsx` files.

---

## Part 1

Before running `p1.py`, make sure the training file exists at the path specified in the script:

```python
FILE_PATH = r"C:\Users\Abhinav Bhatia\Desktop\sem5\ML\Assignment 1\BT2024156_train_var1.xlsx"
```

Run:

```bash
python p1.py
```

The program evaluates all 430 configurations and prints the MSE and R² results for every model.

At the end, it prints the best model and its parameters.

---

## Part 2

The Part 2 script uses:

```python
FILE_PATH = r"C:\Users\Abhinav Bhatia\Desktop\sem5\ML\Assignment 1\BT2024156_train_var2.xlsx"
```

Run:

```bash
python p2.py
```

The program evaluates all 860 configurations and prints their results.

At the end, it reports the model with the lowest validation MSE.

---

# 17. Prediction / Inference

After model selection, the best pipeline is fitted using the complete training dataset.

The resulting object is stored as:

```python
best_model
```

It can then be used to predict the target for unseen test coordinates.

Example:

```python
test_df = pd.read_excel(TEST_FILE_PATH)

X_test = test_df[["x1", "x2", "x3"]].values

predictions = best_model.predict(X_test)
```

For Part 1, the feature list should be:

```python
["x1", "x2", "x3", "x4", "x5", "x6"]
```

For Part 2:

```python
["x1", "x2", "x3"]
```

The resulting predictions should be saved in the required submission format specified by the assignment/sample submission.

The assignment requires two completed prediction files, one for each problem.

---

# 18. Key Design Choices

### Why polynomial regression?

The assignment specifies that the target relationship can be represented by polynomial functions, with degrees up to 10 and 20 for the two problems. 

### Why regularization?

High-degree polynomial expansion produces many correlated features and can lead to overfitting. Ridge, Lasso and Elastic Net provide different forms of regularization to control model complexity.

### Why 5-fold cross-validation?

It allows every training observation to participate in validation once while keeping a large portion of the data available for training in every fold.

### Why use validation MSE for model selection?

The objective is to obtain accurate predictions on unseen test data. Validation MSE therefore provides the primary estimate used to compare candidate models.

### Why report R²?

The assignment explicitly lists R² as an evaluation metric in addition to MSE.

---

# 19. Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- OpenPyXL
- Python `concurrent.futures`

### Main Scikit-learn components

```python
PolynomialFeatures
StandardScaler

LinearRegression
Ridge
Lasso
ElasticNet

Pipeline
KFold
cross_validate
```

---

# 20. Final Results at a Glance

| Problem | Best Degree | Best Model | Alpha | L1 Ratio | Validation MSE | Validation R² |
|---|---:|---|---:|---:|---:|---:|
| Part 1 — var1 | 5 | Lasso | 0.01 | — | 0.3037583528 | 0.9701815630 |
| Part 2 — var2 | 11 | Elastic Net | 0.001 | 0.1 | 0.2644830804 | 0.9942217064 |

---

# 21. Conclusion

A systematic polynomial regression search was performed for both personalized datasets.

For **Part 1**, the best-performing configuration among 430 tested models was a **degree-5 Lasso model with α = 0.01**, achieving an average validation MSE of **0.3037583528** and validation R² of **0.9701815630**.

For **Part 2**, the best-performing configuration among 860 tested models was a **degree-11 Elastic Net model with α = 0.001 and L1 ratio = 0.1**, achieving an average validation MSE of **0.2644830804** and validation R² of **0.9942217064**.

The complete modelling process is implemented in `p1.py` and `p2.py`, with cross-validation, polynomial feature generation, regularization, scaling, parallel model evaluation, and final model selection based on validation MSE.
