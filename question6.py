import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

SALES_DATA_FILE_PATH = r"data\techgear_sales_data_monthly.xlsx"

def create_moving_average_forecast():
    df = pd.read_excel(SALES_DATA_FILE_PATH).dropna()
    df.name = "sales"
    df.set_index("Date", inplace=True)

    # calculate the 3-month moving average
    df["3_month_moving_avg"] = df["Sales"].rolling(window=3).mean()

    # to forecast the next 3 months, you would take the last calculated average
    # as the forecast for the next period, then update it as new data becomes available.
    # for a simple forecast, you could project the last calculated moving average forward.
    last_moving_avg = df["3_month_moving_avg"].iloc[-2]
    print(f"Forecast for the next month (based on 3-month moving average): {last_moving_avg:.2f}")

def create_exponential_smoothing_forecast():
    df = pd.read_excel(SALES_DATA_FILE_PATH).dropna()
    df.name = "sales"
    df.set_index("Date", inplace=True)

    data = df["Sales"]
    index = pd.to_datetime(df.index)
    series = pd.Series(data, index=index)

    model = SimpleExpSmoothing(series, initialization_method="estimated").fit(smoothing_level=0.80)

    forecast = model.forecast(1)
    print("Original Series:", series)
    print("Forecasted Value(s):", forecast)

if __name__ == "__main__":
    create_moving_average_forecast()
    create_exponential_smoothing_forecast()