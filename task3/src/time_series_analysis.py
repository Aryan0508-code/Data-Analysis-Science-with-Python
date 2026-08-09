import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


# ============================================================
# TASK 3: TIME SERIES ANALYSIS
# ============================================================

# Project paths
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "airline-passengers.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

print("=" * 60)
print("TASK 3: TIME SERIES ANALYSIS")
print("=" * 60)

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["Month"],
    index_col="Month"
)

df = df.sort_index()

print("\nData Head:")
print(df.head())

print("\nData Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 2. BASIC TIME SERIES INSPECTION
# ============================================================

print("\n" + "=" * 60)
print("2. BASIC TIME SERIES INSPECTION")
print("=" * 60)

print("\nStart Date:", df.index.min())
print("End Date:", df.index.max())
print("Number of Observations:", len(df))

print("\nSummary Statistics:")
print(df["Passengers"].describe())


# ============================================================
# 3. RAW TIME SERIES PLOT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df.index,
    df["Passengers"],
    label="Passengers"
)

plt.title("Monthly Airline Passengers")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_raw_time_series.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. QUARTERLY RESAMPLING
# ============================================================

print("\n" + "=" * 60)
print("4. QUARTERLY RESAMPLING")
print("=" * 60)

quarterly = (
    df["Passengers"]
    .resample("QE")
    .mean()
)

print("\nQuarterly Mean:")
print(quarterly.head(10))

quarterly.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "quarterly_passengers.csv"
    )
)


# ============================================================
# 5. TREND / SEASONALITY DECOMPOSITION
# ============================================================

print("\n" + "=" * 60)
print("5. TREND / SEASONALITY DECOMPOSITION")
print("=" * 60)

decomposition = seasonal_decompose(
    df["Passengers"],
    model="multiplicative",
    period=12
)

fig = decomposition.plot()

fig.set_size_inches(12, 8)

fig.suptitle(
    "Multiplicative Time Series Decomposition",
    y=1.02
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_decomposition.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Decomposition completed using "
    "12-month seasonality."
)


# ============================================================
# 6. MOVING AVERAGES
# ============================================================

print("\n" + "=" * 60)
print("6. MOVING AVERAGES")
print("=" * 60)

df["MA_6"] = (
    df["Passengers"]
    .rolling(window=6)
    .mean()
)

df["MA_12"] = (
    df["Passengers"]
    .rolling(window=12)
    .mean()
)

print("\nMoving Average Data:")
print(
    df[
        [
            "Passengers",
            "MA_6",
            "MA_12"
        ]
    ].tail(10)
)


# Moving Average Plot

plt.figure(figsize=(12, 6))

plt.plot(
    df.index,
    df["Passengers"],
    label="Actual"
)

plt.plot(
    df.index,
    df["MA_6"],
    label="6-Month MA",
    linestyle="--"
)

plt.plot(
    df.index,
    df["MA_12"],
    label="12-Month MA",
    linestyle=":"
)

plt.title("Moving Averages Smoothing")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_moving_averages.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("7. TRAIN / TEST SPLIT")
print("=" * 60)

# Last 12 months are used for testing.

train = df["Passengers"].iloc[:-12]

test = df["Passengers"].iloc[-12:]

print(
    "Training Observations:",
    len(train)
)

print(
    "Testing Observations:",
    len(test)
)

print(
    "\nTraining Period:",
    train.index.min(),
    "to",
    train.index.max()
)

print(
    "Testing Period:",
    test.index.min(),
    "to",
    test.index.max()
)


# ============================================================
# 8. SARIMA MODEL
# ============================================================

print("\n" + "=" * 60)
print("8. SARIMA FORECASTING")
print("=" * 60)

# Parameters specified in the task:
#
# ARIMA:
# (p,d,q) = (2,1,1)
#
# Seasonal:
# (P,D,Q,s) = (1,1,1,12)

model = ARIMA(
    train,
    order=(2, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)

result = model.fit()

print("\nModel Summary:")
print(result.summary())


# ============================================================
# 9. FORECAST
# ============================================================

forecast = result.forecast(
    steps=12
)

# Make sure forecast has same index as test data

forecast.index = test.index

print("\nForecast:")
print(forecast)


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        test,
        forecast
    )
)

test_mean = test.mean()

rmse_percentage = (
    rmse / test_mean
) * 100

print("\n" + "=" * 60)
print("10. MODEL EVALUATION")
print("=" * 60)

print(
    f"RMSE: {rmse:.2f} passengers"
)

print(
    f"RMSE as percentage of "
    f"test average: {rmse_percentage:.2f}%"
)


# ============================================================
# 11. FORECAST VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    train.index,
    train,
    label="Training Data"
)

plt.plot(
    test.index,
    test,
    label="Actual"
)

plt.plot(
    forecast.index,
    forecast,
    label="Forecast",
    linestyle="--"
)

plt.title(
    f"SARIMA Forecast "
    f"(RMSE = {rmse:.2f})"
)

plt.xlabel("Month")
plt.ylabel("Passengers")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_sarima_forecast.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. FORECAST RESULTS
# ============================================================

forecast_results = pd.DataFrame({
    "Actual": test,
    "Forecast": forecast
})

forecast_results["Error"] = (
    forecast_results["Actual"]
    -
    forecast_results["Forecast"]
)

forecast_results["Absolute_Error"] = (
    forecast_results["Error"]
    .abs()
)

print("\nForecast Results:")
print(forecast_results)

forecast_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "forecast_results.csv"
    )
)


# ============================================================
# 13. TREND ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("13. TREND ANALYSIS")
print("=" * 60)

yearly_average = (
    df["Passengers"]
    .resample("YE")
    .mean()
)

print("\nYearly Average Passengers:")

print(
    yearly_average.round(2)
)


# Year-over-year growth

yearly_growth = (
    yearly_average
    .pct_change()
    * 100
)

print("\nYear-over-Year Growth (%):")

print(
    yearly_growth.round(2)
)


# Overall growth

overall_growth = (
    (
        yearly_average.iloc[-1]
        -
        yearly_average.iloc[0]
    )
    /
    yearly_average.iloc[0]
) * 100

print(
    f"\nOverall Growth: "
    f"{overall_growth:.2f}%"
)


# ============================================================
# 14. SEASONALITY ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("14. SEASONALITY ANALYSIS")
print("=" * 60)

monthly_average = (
    df["Passengers"]
    .groupby(
        df.index.month
    )
    .mean()
)

monthly_average.index = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"
]

print(
    "\nAverage Passengers by Month:"
)

print(
    monthly_average.round(2)
)


# Peak month

peak_month = (
    monthly_average.idxmax()
)

# Lowest month

lowest_month = (
    monthly_average.idxmin()
)

annual_average = (
    df["Passengers"].mean()
)

peak_difference = (
    (
        monthly_average[peak_month]
        -
        annual_average
    )
    /
    annual_average
) * 100

print(
    f"\nPeak Month: {peak_month}"
)

print(
    f"Lowest Month: {lowest_month}"
)

print(
    f"Peak month is "
    f"{peak_difference:.2f}% "
    f"above the overall average."
)


# ============================================================
# 15. MONTHLY SEASONALITY PLOT
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_average.index,
    monthly_average.values,
    marker="o"
)

plt.title(
    "Average Airline Passengers by Month"
)

plt.xlabel("Month")
plt.ylabel("Average Passengers")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_monthly_seasonality.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. RESIDUAL ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("16. RESIDUAL ANALYSIS")
print("=" * 60)

residuals = (
    decomposition
    .resid
    .dropna()
)

print("\nResidual Statistics:")

print(
    residuals.describe()
)


# Residual plot

plt.figure(figsize=(12, 5))

plt.plot(
    residuals.index,
    residuals
)

plt.axhline(
    1,
    linestyle="--"
)

plt.title(
    "Decomposition Residuals"
)

plt.xlabel("Month")
plt.ylabel("Residual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_residual_analysis.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Number of Observations",
        "Training Observations",
        "Testing Observations",
        "RMSE",
        "RMSE Percentage",
        "Peak Month",
        "Lowest Month",
        "Overall Growth (%)"
    ],

    "Value": [
        len(df),
        len(train),
        len(test),
        round(rmse, 2),
        round(rmse_percentage, 2),
        peak_month,
        lowest_month,
        round(overall_growth, 2)
    ]
})

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "time_series_summary.csv"
    ),
    index=False
)


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("TASK 3 COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "\nAll graphs and analysis files "
    "have been saved inside the outputs folder."
)