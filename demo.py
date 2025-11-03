import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

sales_data_file_path = r"data\techgear_sales_data.xlsx"
df = pd.read_excel(sales_data_file_path)
df.name = "sales"

# # print out list of column headers
# print(df.columns)
# # print first 5 rows of data
# print(df.head())
print(df.describe())

# explore and visualize data
# creates scatter plots to visualize the relationship between
# independent variables and a dependent variable/target
sns.scatterplot(x="Date", y="Sales", data=df)
plt.title("Scatter Plot of X vs Y")
plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (Y)")
plt.show()

# separate independent and dependent variables
X = df[["Date"]].values.astype("float64")
Y = df["Sales"].values.astype("float64")

# split data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# create and train linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# make predictions
predictions = model.predict(X_test)

# evaluate model performance
print("Mean Absolute Error:", metrics.mean_absolute_error(Y_test, predictions))
print("Mean Squared Error:", metrics.mean_squared_error(Y_test, predictions))
print("R-squared:", metrics.r2_score(Y_test, predictions))

# visualize predictions
plt.scatter(Y_test, predictions)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()