# PHASE 1 - Step 1: download old data (about 92 days) for Karachi
# Run this ONE time. It makes the file karachi_aq_history.csv

import requests
import pandas as pd

# ---------- air quality data ----------
air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

air_settings = {
    "latitude": 24.86,
    "longitude": 67.01,
    "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone",
    "past_days": 92,
    "forecast_days": 1,
    "timezone": "Asia/Karachi",
}

air_answer = requests.get(air_url, params=air_settings)
air_data = air_answer.json()
air_table = pd.DataFrame(air_data["hourly"])

# ---------- weather data ----------
weather_url = "https://api.open-meteo.com/v1/forecast"

weather_settings = {
    "latitude": 24.86,
    "longitude": 67.01,
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
    "past_days": 92,
    "forecast_days": 1,
    "timezone": "Asia/Karachi",
}

weather_answer = requests.get(weather_url, params=weather_settings)
weather_data = weather_answer.json()
weather_table = pd.DataFrame(weather_data["hourly"])

# ---------- join the two tables using the time column ----------
table = pd.merge(air_table, weather_table, on="time")

# ---------- remove future hours (they are only forecasts, not real readings) ----------
table["time"] = pd.to_datetime(table["time"])
now = pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None)
table = table[table["time"] <= now]

# ---------- save the table ----------
table.to_csv("karachi_aq_history.csv", index=False)

print("Rows and columns:", table.shape)
print(table.head())
print(table.tail())
