import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def create_facebook_scatter_plot(df):
    sns.scatterplot(x="Sales", y="Ad_Spend_Facebook", data=df)
    plt.title("Scatter Plot of Sales vs Facebook Ads")
    plt.xlabel("Sales")
    plt.ylabel("Facebook Ads")
    plt.show()

def create_instagram_scatter_plot(df):
    sns.scatterplot(x="Sales", y="Ad_Spend_Instagram", data=df)
    plt.title("Scatter Plot of Sales vs Instagram Ads")
    plt.xlabel("Sales")
    plt.ylabel("Instagram Ads")
    plt.show()

def create_discount_scatter_plot(df):
    sns.scatterplot(x="Sales", y="Discount_Rate", data=df)
    plt.title("Scatter Plot of Sales vs Discount Rates")
    plt.xlabel("Sales")
    plt.ylabel("Discount Rates")
    plt.show()

if __name__ == "__main__":
    sales_data_file_path = r"data\techgear_sales_data.xlsx"
    df = pd.read_excel(sales_data_file_path)
    df.name = "sales"

    create_facebook_scatter_plot(df)
    create_instagram_scatter_plot(df)
    create_discount_scatter_plot(df)
