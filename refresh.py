# REFRESH: download new data and rebuild the files the dashboard reads
# Run this whenever you want the dashboard to show up-to-date data:
#     python refresh.py
# Then start the dashboard again:
#     python -m streamlit run app.py
#
# What it does:
#   1. downloads the last 92 days of hourly data (up to now)
#   2. cleans it and adds the same features as analysis.ipynb  -> clean_data.csv
#   3. flags spikes with Isolation Forest and a simple rule      -> spike_flags.csv

import os
import shutil
import requests
import pandas as pd
from sklearn.ensemble import IsolationForest

# ---------- step 1: download fresh data ----------
print("Step 1: downloading fresh data...")

air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
air_settings = {
    "latitude": 24.86,
    "longitude": 67.01,
    "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone",
    "past_days": 92,
    "forecast_days": 1,
    "timezone": "Asia/Karachi",
}
air_data = requests.get(air_url, params=air_settings).json()
air_table = pd.DataFrame(air_data["hourly"])

weather_url = "https://api.open-meteo.com/v1/forecast"
weather_settings = {
    "latitude": 24.86,
    "longitude": 67.01,
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
    "past_days": 92,
    "forecast_days": 1,
    "timezone": "Asia/Karachi",
}
weather_data = requests.get(weather_url, params=weather_settings).json()
weather_table = pd.DataFrame(weather_data["hourly"])

table = pd.merge(air_table, weather_table, on="time")

# remove future hours (they are only forecasts)
table["time"] = pd.to_datetime(table["time"])
now = pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None)
table = table[table["time"] <= now]

# keep a copy of the old history file before replacing it
if os.path.exists("karachi_aq_history.csv"):
    shutil.copy("karachi_aq_history.csv", "karachi_aq_history_old.csv")

table.to_csv("karachi_aq_history.csv", index=False)
print("Your computer clock says:", now)
print("Downloaded rows:", len(table))
print("Latest hour:", table["time"].max())

# ---------- step 2: clean and add features ----------
print("Step 2: cleaning...")

table = table.sort_values("time")
table = table.drop_duplicates(subset="time")

# fill small gaps using the values before and after
table = table.set_index("time")
table = table.interpolate(method="time")
table = table.reset_index()

# time columns
table["hour"] = table["time"].dt.hour
table["day_of_week"] = table["time"].dt.dayofweek
table["month"] = table["time"].dt.month

table["is_weekend"] = 0
table.loc[table["day_of_week"] >= 5, "is_weekend"] = 1

table["is_night"] = 0
table.loc[(table["hour"] >= 22) | (table["hour"] <= 5), "is_night"] = 1

table["is_rush_hour"] = 0
table.loc[(table["hour"] >= 7) & (table["hour"] <= 10), "is_rush_hour"] = 1
table.loc[(table["hour"] >= 17) & (table["hour"] <= 20), "is_rush_hour"] = 1

# past values (lags) and rolling averages
table["pm2_5_lag1"] = table["pm2_5"].shift(1)
table["pm2_5_lag3"] = table["pm2_5"].shift(3)
table["pm2_5_lag24"] = table["pm2_5"].shift(24)

past_pm = table["pm2_5"].shift(1)
table["pm2_5_avg3"] = past_pm.rolling(3).mean()
table["pm2_5_avg24"] = past_pm.rolling(24).mean()

table = table.dropna()
table = table.reset_index(drop=True)

table.to_csv("clean_data.csv", index=False)
print("Saved clean_data.csv with", len(table), "rows")

# ---------- step 3: flag spikes ----------
print("Step 3: flagging spikes...")

test_start = int(len(table) * 0.8)
train_table = table[:test_start]

# normal PM2.5 for each hour of day (from training hours only)
hour_normal = train_table.groupby("hour")["pm2_5"].mean()
table["hour_average"] = table["hour"].map(hour_normal)
table["pm_vs_hour"] = table["pm2_5"] - table["hour_average"]

features = ["pm2_5", "pm10", "wind_speed_10m", "relative_humidity_2m", "pm_vs_hour"]
train_table = table[:test_start]

iso_model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
iso_model.fit(train_table[features])
answer = iso_model.predict(table[features])

table["if_spike"] = 0
table.loc[answer == -1, "if_spike"] = 1

limit = train_table["pm2_5"].quantile(0.95)
table["rule_spike"] = 0
table.loc[table["pm2_5"] > limit, "rule_spike"] = 1

table[["time", "pm2_5", "if_spike", "rule_spike"]].to_csv("spike_flags.csv", index=False)
print("Saved spike_flags.csv")

print("")
print("Done. Latest hour in the dashboard will be:", table["time"].max())
