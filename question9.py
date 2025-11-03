import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
import pulp

SALES_DATA_FILE_PATH = r"data\techgear_sales_data_monthly.xlsx"
df = pd.read_excel(SALES_DATA_FILE_PATH).dropna()
df.name = "sales"

# 1. Define the problem
prob = pulp.LpProblem("AdBudgetOptimization", pulp.LpMaximize)

# 2. Define Decision Variables
facebook_spend = pulp.LpVariable("FacebookSpend", lowBound=0, cat="Continuous")
instagram_spend = pulp.LpVariable("InstagramSpend", lowBound=0, cat="Continuous")

# 3. Define Objective Function
SALES_PER_DOLLAR_FACEBOOK = 5
SALES_PER_DOLLAR_INSTAGRAM = 3
prob += (SALES_PER_DOLLAR_FACEBOOK * facebook_spend) + (SALES_PER_DOLLAR_INSTAGRAM * instagram_spend), "TotalSales"

# 4. Define Constraints
MONTHLY_BUDGET = 10000
MIN_FACEBOOK_SPEND = 2000
MIN_INSTAGRAM_SPEND = 1000
MAX_INSTAGRAM_SPEND = 7000

prob += facebook_spend + instagram_spend <= MONTHLY_BUDGET, "TotalBudgetConstraint"
prob += facebook_spend >= MIN_FACEBOOK_SPEND, "MinFacebookSpend"
prob += instagram_spend >= MIN_INSTAGRAM_SPEND, "MinInstagramSpend"
prob += instagram_spend <= MAX_INSTAGRAM_SPEND, "MaxInstagramSpend"
prob += facebook_spend * 0.5 >= instagram_spend, "InstagramSpendHalfOrMoreOfFacebookSpend"

# 5. Solve the problem
prob.solve()

# 6. Print the results
print(f"Status: {pulp.LpStatus[prob.status]}")
print(f"Optimal Facebook Spend: ${facebook_spend.varValue:,.2f}")
print(f"Optimal Instagram Spend: ${instagram_spend.varValue:,.2f}")
print(f"Maximum Sales: ${pulp.value(prob.objective):,.2f}")
