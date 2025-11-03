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
    x = df[["Ad_Spend_Facebook", "Ad_Spend_Instagram", "Discount_Rate"]]
    y = df["Sales"]

    # split data into training and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # create and train linear regression model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # make predictions
    predictions = model.predict(x_test)

    # evaluate model performance
    print("Coefficient/Slope:", model.coef_[0])
    print("Intercept:", model.intercept_)
    print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, predictions))
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, predictions))
    print("R-squared:", metrics.r2_score(y_test, predictions))

    # visualize predictions
    plt.scatter(y_test, predictions)
    plt.title("Multiple Linear Regression")
    plt.xlabel("Ad Spend Facebook/Instagram w/ Discount Rates")
    plt.ylabel("Sales")
    plt.show()