# Take ONE reading now and save it. Used by collector.py and by the GitHub Action.
#   live_data.csv     -> the current air quality and weather
#   forecast_log.csv  -> Open-Meteo's own forecast for 24 hours from now
#                        (the notebook compares this with our model later)

import os
import requests
import pandas as pd

air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
weather_url = "https://api.open-meteo.com/v1/forecast"

live_file = "live_data.csv"
log_file = "forecast_log.csv"


def save_row(row, file_name):
    # add one row to a csv file (write the column names only the first time)
    new_table = pd.DataFrame([row])
    if os.path.exists(file_name):
        new_table.to_csv(file_name, mode="a", header=False, index=False)
    else:
        new_table.to_csv(file_name, index=False)


def collect_one_reading():
    # ask for the latest air quality
    air_settings = {
        "latitude": 24.86,
        "longitude": 67.01,
        "current": "pm2_5,pm10,nitrogen_dioxide,ozone",
        "timezone": "Asia/Karachi",
    }
    air_answer = requests.get(air_url, params=air_settings, timeout=60)
    air_now = air_answer.json()["current"]

    # ask for the latest weather
    weather_settings = {
        "latitude": 24.86,
        "longitude": 67.01,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "Asia/Karachi",
    }
    weather_answer = requests.get(weather_url, params=weather_settings, timeout=60)
    weather_now = weather_answer.json()["current"]

    # one row for live_data.csv (same column names as the history file)
    row = {
        "time": air_now["time"],
        "pm2_5": air_now["pm2_5"],
        "pm10": air_now["pm10"],
        "nitrogen_dioxide": air_now["nitrogen_dioxide"],
        "ozone": air_now["ozone"],
        "temperature_2m": weather_now["temperature_2m"],
        "relative_humidity_2m": weather_now["relative_humidity_2m"],
        "wind_speed_10m": weather_now["wind_speed_10m"],
    }
    save_row(row, live_file)
    print("Saved reading:", row["time"], "PM2.5 =", row["pm2_5"])

    # Open-Meteo's own forecast: what does it say about 24 hours from now?
    forecast_settings = {
        "latitude": 24.86,
        "longitude": 67.01,
        "hourly": "pm2_5",
        "forecast_days": 3,
        "timezone": "Asia/Karachi",
    }
    forecast_answer = requests.get(air_url, params=forecast_settings, timeout=60)
    forecast_table = pd.DataFrame(forecast_answer.json()["hourly"])
    forecast_table["time"] = pd.to_datetime(forecast_table["time"])

    made_at = pd.to_datetime(air_now["time"]).floor("h")
    target_time = made_at + pd.Timedelta(hours=24)

    match = forecast_table[forecast_table["time"] == target_time]

    if len(match) == 1:
        log_row = {
            "made_at": made_at,
            "target_time": target_time,
            "openmeteo_pm2_5_forecast": match["pm2_5"].values[0],
        }
        save_row(log_row, log_file)
        print("Saved Open-Meteo forecast for", target_time)


# this part runs only when you start this file directly (python collect_once.py)
if __name__ == "__main__":
    collect_one_reading()
