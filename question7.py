import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

SALES_DATA_FILE_PATH = r"data\techgear_sales_data_monthly.xlsx"

# 1. Load data
df = pd.read_excel(SALES_DATA_FILE_PATH).dropna()
df.name = "sales"

# 2. Initialize the model
x = df[["Ad_Spend_Facebook", "Ad_Spend_Instagram", "Discount_Rate"]]
y = df["Sales"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

mlr_model = LinearRegression()
mlr_model.fit(x_train, y_train)

# 3. Define the K-Fold Cross-Validation strategy
# n_splits=5 for 5-fold cross-validation
# shuffle=True to randomly shuffle the data before splitting
# random_state for reproducibility
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. Perform cross-validation
# 'neg_mean_squared_error' is used because cross_val_score maximizes the score.
# For error metrics like MSE, we want to minimize them, so we use the negative value.
mlr_scores = cross_val_score(mlr_model, x, y, cv=kf, scoring="neg_mean_squared_error")

# 5. Analyze the results
# Convert negative MSE to positive MSE and then calculate RMSE
mlr_rmse_scores = (-mlr_scores) ** 0.5

print(f"Individual RMSE scores for each fold: {mlr_rmse_scores}")
print(f"Average RMSE across all folds: {np.mean(mlr_rmse_scores):.4f}")
print(f"Standard deviation of RMSE scores: {np.std(mlr_rmse_scores):.4f}")

# 6. Decision Tree Regression Model
print("Decision Tree Regression with 5-Fold Cross-Validation")

dt_model = DecisionTreeRegressor(random_state=42)
dt_scores = cross_val_score(dt_model, x, y, cv=kf, scoring="neg_mean_squared_error")
dt_rmse_scores = (-dt_scores) ** 0.5
print(f"Invidual RMSE scores for each fold: {dt_rmse_scores}")
print(f"Mean RMSE for Decision Tree Regression: {np.mean(dt_rmse_scores):.4f}")
print(f"Standard deviation of RMSE scores: {np.std(dt_rmse_scores):.4f}")