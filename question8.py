import numpy as np
import pandas as pd

NUMBER_OF_SIMULATIONS = 1000
SALES_DATA_FILE_PATH = r"data\techgear_sales_data_monthly.xlsx"

df = pd.read_excel(SALES_DATA_FILE_PATH).dropna()
df.name = "sales"

min_sales = np.min(df["Sales"])
max_sales = np.max(df["Sales"])
print(min_sales, max_sales)

np.random.seed(42)
simulated_sales = np.random.uniform(min_sales, max_sales, NUMBER_OF_SIMULATIONS)

average_sales = np.mean(simulated_sales)
median_sales = np.median(simulated_sales)
std_dev_sales = np.std(simulated_sales)

print(f"Estimated Average Monthly Sales: ${average_sales:,.2f}")
print(f"Estimated Median Monthly Sales: ${median_sales:,.2f}")
print(f"Estimated Standard Deviation of Monthly Sales: ${std_dev_sales:,.2f}")