import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

def create_scatter_plot(df: DataFrame, x: str, y: str, title: str, x_label: str, y_label: str):
    sns.scatterplot(x=x, y=y, data=df)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

if __name__ == "__main__":
    sales_data_file_path = r"data\techgear_sales_data.xlsx"
    df = pd.read_excel(sales_data_file_path)
    df.name = "sales"

    create_scatter_plot(df, "Sales", "Ad_Spend_Facebook", "Scatter Plot of Sales vs Facebook Ads", "Sales", "Facebook Ads")
    create_scatter_plot(df, "Sales", "Ad_Spend_Instagram", "Scatter Plot of Sales vs Instagram Ads", "Sales", "Instagram Ads")
    create_scatter_plot(df, "Sales", "Discount_Rate", "Scatter Plot of Sales vs Discount Rates", "Sales", "Discount Rates")
