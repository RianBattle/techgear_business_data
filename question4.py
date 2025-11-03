import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm

if __name__ == "__main__":
    sales_data_file_path = r"data\techgear_sales_data.xlsx"
    df = pd.read_excel(sales_data_file_path)
    df.name = "sales"

    # separate independent and dependent variables
    x = df[["Sales"]]
    y = df["Ad_Spend_Facebook"]

    # split data into training and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # create and train linear regression model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # fit a linear regression model
    X = sm.add_constant(df["Sales"])
    model = sm.OLS(df["Ad_Spend_Facebook"], X).fit()

    # get fitted values and residuals
    fitted_values = model.fittedvalues
    residuals = model.resid

    # create the residuals vs fitted plot
    plt.figure(figsize=(8, 6))
    sns.residplot(x=fitted_values, y=residuals, lowess=True, line_kws={"color": "red", "lw": 1, "alpha": 0.8})
    plt.title("Residuals vs Fitted")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.show()

    # create a Q-Q plot of residuals
    plt.figure(figsize=(8, 6))
    sm.qqplot(residuals, line="s", ax=plt.gca())
    plt.title("Normal Q-Q Plot of Residuals")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantities (Residuals)")
    plt.grid(True)
    plt.show()