import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX


# =========================
# 1) Load + Basic Cleaning
# =========================
df = pd.read_csv("dengue dataset.csv")

df["calendar_start_date"] = pd.to_datetime(
    df["calendar_start_date"], errors="coerce"
)
df = df.dropna(subset=["calendar_start_date"]).sort_values(
    "calendar_start_date"
)

# =========================
# 2) Keep Last 15 Years
# =========================
cutoff = df["calendar_start_date"].max() - pd.DateOffset(years=15)
df = df[df["calendar_start_date"] >= cutoff].copy()

# =========================
# 3) Target Transform
# =========================
df["cases"] = df["dengue_total"].astype(float)
df["log_cases"] = np.log1p(df["cases"])

# =========================
# 4) Seasonality Features
# =========================
df["month"] = df["calendar_start_date"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# =========================
# 5) Lag Features
# =========================
df["lag1_log_cases"] = df["log_cases"].shift(1)
df["lag2_log_cases"] = df["log_cases"].shift(2)

feature_cols = [
    "log_cases",
    "month_sin",
    "month_cos",
    "lag1_log_cases",
    "lag2_log_cases",
]

df_model = df.dropna(subset=feature_cols).copy()

print("Model-ready rows (after lags):", len(df_model))

# =========================
# 6) Exploratory Plots
# =========================

plt.figure(figsize=(12, 6))
plt.plot(df["calendar_start_date"], df["log_cases"])
plt.title("Bangladesh Dengue (log1p) — Last 15 Years")
plt.xlabel("Date")
plt.ylabel("log(1 + cases)")
plt.grid(True, alpha=0.3)
plt.show()

seasonality = df.groupby("month")["log_cases"].mean()

plt.figure(figsize=(10, 4))
plt.plot(seasonality.index, seasonality.values, marker="o")
plt.title("Seasonality: Average log(1+cases) by Month")
plt.xlabel("Month")
plt.ylabel("Avg log(1 + cases)")
plt.grid(True, alpha=0.3)
plt.show()

# =========================
# 7) Pull Climate Data (NASA POWER API)
# =========================

lat = 23.6850
lon = 90.3563

start_year = df["calendar_start_date"].min().year
end_year = df["calendar_start_date"].max().year

url = (
    "https://power.larc.nasa.gov/api/temporal/monthly/point?"
    f"parameters=T2M,PRECTOTCORR,RH2M"
    f"&community=AG"
    f"&longitude={lon}"
    f"&latitude={lat}"
    f"&start={start_year}"
    f"&end={end_year}"
    "&format=JSON"
)

response = requests.get(url)
climate_json = response.json()
climate_data = climate_json["properties"]["parameter"]

climate_df = pd.DataFrame({
    "date": climate_data["T2M"].keys(),
    "temperature": climate_data["T2M"].values(),
    "rainfall": climate_data["PRECTOTCORR"].values(),
    "humidity": climate_data["RH2M"].values()
})

climate_df["date"] = climate_df["date"].astype(str).str.zfill(6)
climate_df["calendar_start_date"] = pd.to_datetime(
    climate_df["date"], format="%Y%m", errors="coerce"
)

climate_df = climate_df.dropna(subset=["calendar_start_date"])
climate_df = climate_df.drop(columns=["date"])

# =========================
# 8) Merge Climate
# =========================
df_merged = pd.merge(
    df_model,
    climate_df,
    on="calendar_start_date",
    how="inner"
)

df_ts = df_merged.sort_values("calendar_start_date").copy()

df_ts["rain_lag1"] = df_ts["rainfall"].shift(1)
df_ts["rain_lag2"] = df_ts["rainfall"].shift(2)
df_ts["temp_lag1"] = df_ts["temperature"].shift(1)
df_ts["hum_lag1"] = df_ts["humidity"].shift(1)

feature_cols = [
    "month_sin",
    "month_cos",
    "lag1_log_cases",
    "lag2_log_cases",
    "rain_lag1",
    "rain_lag2",
    "temp_lag1",
    "hum_lag1",
]

df_ts = df_ts.dropna(subset=feature_cols + ["log_cases"]).copy()

# =========================
# 9) OLS Inference
# =========================

X_ols = sm.add_constant(df_ts[feature_cols])
y_ols = df_ts["log_cases"]

ols_fit = sm.OLS(y_ols, X_ols).fit(cov_type="HC3")

print("\n=== OLS SUMMARY ===")
print(ols_fit.summary())

resid = ols_fit.resid

print("\n=== ADF TEST ===")
print("ADF p-value (log_cases):", adfuller(df_ts["log_cases"])[1])
print("ADF p-value (diff log_cases):", adfuller(
    df_ts["log_cases"].diff().dropna())[1])

print("\n=== LJUNG-BOX ===")
print(acorr_ljungbox(resid, lags=[12, 24], return_df=True))

print("\n=== BREUSCH-PAGAN ===")
print("BP p-value:", het_breuschpagan(resid, X_ols)[1])

sm.graphics.tsa.plot_acf(resid, lags=24)
plt.title("Residual ACF (OLS)")
plt.show()

sm.qqplot(resid, line="45")
plt.title("Residual Q-Q Plot (OLS)")
plt.show()

# =========================
# 10) SARIMA Benchmark
# =========================

ts = df_ts.set_index("calendar_start_date")["log_cases"]

sarima_preds = []
sarima_lower = []
sarima_upper = []
sarima_true = []
sarima_dates = []

min_train_size = 60

for i in range(min_train_size, len(ts)):
    train = ts.iloc[:i]
    test = ts.iloc[i]

    model = SARIMAX(
        train,
        order=(1,1,1),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fit = model.fit(disp=False)
    forecast_obj = fit.get_forecast(steps=1)

    mean = forecast_obj.predicted_mean.iloc[0]
    conf_int = forecast_obj.conf_int(alpha=0.05)

    sarima_preds.append(mean)
    sarima_lower.append(conf_int.iloc[0, 0])
    sarima_upper.append(conf_int.iloc[0, 1])
    sarima_true.append(test)
    sarima_dates.append(ts.index[i])

sarima_preds = pd.Series(sarima_preds, index=sarima_dates)
sarima_lower = pd.Series(sarima_lower, index=sarima_dates)
sarima_upper = pd.Series(sarima_upper, index=sarima_dates)
sarima_true = pd.Series(sarima_true, index=sarima_dates)

print("\n=== SARIMA Walk-Forward ===")
print("RMSE:", np.sqrt(mean_squared_error(sarima_true, sarima_preds)))
print("MAE :", mean_absolute_error(sarima_true, sarima_preds))

coverage = ((sarima_true >= sarima_lower) &
            (sarima_true <= sarima_upper)).mean()

print("95% Interval Coverage:", coverage)

plt.figure(figsize=(12,6))
plt.plot(sarima_true.index, sarima_true, label="Actual", color="black")
plt.plot(sarima_preds.index, sarima_preds,
         label="SARIMA Forecast", color="blue")
plt.fill_between(
    sarima_preds.index,
    sarima_lower,
    sarima_upper,
    color="blue",
    alpha=0.2,
    label="95% Prediction Interval"
)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("SARIMA Walk-Forward Forecast")
plt.show()

# =========================
# 11) Linear Walk-Forward
# =========================

y_true = []
y_pred = []
y_pred_naive = []
pred_dates = []

for i in range(min_train_size, len(df_ts)):
    train = df_ts.iloc[:i]
    test_point = df_ts.iloc[i:i+1]

    X_train = train[feature_cols]
    y_train = train["log_cases"]
    X_test = test_point[feature_cols]
    y_test = test_point["log_cases"].values[0]

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)[0]
    naive_pred = y_train.iloc[-1]

    y_true.append(y_test)
    y_pred.append(pred)
    y_pred_naive.append(naive_pred)
    pred_dates.append(test_point["calendar_start_date"].values[0])

y_true = pd.Series(y_true, index=pd.to_datetime(pred_dates))
y_pred = pd.Series(y_pred, index=pd.to_datetime(pred_dates))
y_pred_naive = pd.Series(y_pred_naive, index=pd.to_datetime(pred_dates))

print("\n=== LINEAR WALK-FORWARD ===")
print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
print("MAE :", mean_absolute_error(y_true, y_pred))

plt.figure(figsize=(12,6))
plt.plot(y_true.index, y_true, label="Actual")
plt.plot(y_pred.index, y_pred, label="Linear")
plt.plot(y_pred_naive.index, y_pred_naive,
         label="Naive", linestyle="--")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Walk-Forward Validation (Log Scale)")
plt.show()
