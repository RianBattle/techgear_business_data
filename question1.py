import pandas as pd

if __name__ == "__main__":
    sales_data_file_path = r"data\techgear_sales_data.xlsx"
    df = pd.read_excel(sales_data_file_path)
    df.name = "sales"

    print(df.describe())
