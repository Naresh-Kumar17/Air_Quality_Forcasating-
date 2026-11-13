# PHASE 1 - Step 2: collect NEW data every hour
# Keep this running (laptop on) until week 14.
# It adds one new row each hour to the file live_data.csv
# To stop it: press Ctrl + C

import os
import time
import requests
import pandas as pd

air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
weather_url = "https://api.open-meteo.com/v1/forecast"

file_name = "live_data.csv"

while True:
    try:
        # ask for the latest air quality
        air_settings = {
            "latitude": 24.86,
            "longitude": 67.01,
            "current": "pm2_5,pm10,nitrogen_dioxide,ozone",
            "timezone": "Asia/Karachi",
        }
        air_answer = requests.get(air_url, params=air_settings)
        air_now = air_answer.json()["current"]

        # ask for the latest weather
        weather_settings = {
            "latitude": 24.86,
            "longitude": 67.01,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": "Asia/Karachi",
        }
        weather_answer = requests.get(weather_url, params=weather_settings)
        weather_now = weather_answer.json()["current"]

        # make one row (same column names as the history file)
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
        new_table = pd.DataFrame([row])

        # add the row to the file (write column names only the first time)
        if os.path.exists(file_name):
            new_table.to_csv(file_name, mode="a", header=False, index=False)
        else:
            new_table.to_csv(file_name, index=False)

        print("Saved:", row["time"], "PM2.5 =", row["pm2_5"])

    except Exception as error:
        # if the internet or API fails, show the problem and try again next time
        print("Problem this time:", error)

    # wait one hour (3600 seconds)
    time.sleep(3600)
