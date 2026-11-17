# Shared download code for backfill.py and refresh.py
# get_history(history_days) returns one table with hourly air quality + weather for Karachi

import time
import requests
import pandas as pd

air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
weather_forecast_url = "https://api.open-meteo.com/v1/forecast"
weather_archive_url = "https://archive-api.open-meteo.com/v1/archive"

latitude = 24.86
longitude = 67.01

air_columns = "pm2_5,pm10,nitrogen_dioxide,ozone"
weather_columns = "temperature_2m,relative_humidity_2m,wind_speed_10m"

# weather FORECASTS that were made one day earlier (Open-Meteo "Previous Runs" service)
# "_previous_day1" = what the weather model predicted for that hour 24 hours before it
previous_runs_url = "https://previous-runs-api.open-meteo.com/v1/forecast"
forecast_columns = ("temperature_2m_previous_day1,relative_humidity_2m_previous_day1,"
                    "wind_speed_10m_previous_day1,wind_direction_10m_previous_day1,"
                    "surface_pressure_previous_day1,precipitation_previous_day1")


def ask_api(url, settings):
    # ask the website for data, try up to 3 times if it fails
    for attempt in range(3):
        try:
            answer = requests.get(url, params=settings, timeout=60)
            data = answer.json()
            if "hourly" in data:
                return data
            print("API said:", data.get("reason", "no data"))
        except Exception as error:
            print("Problem:", error)
        time.sleep(3)
    return None


def get_weather_forecasts(start, today):
    # download the day-ahead weather forecasts in pieces of 90 days
    pieces = []
    piece_start = start

    while piece_start <= today:
        piece_end = piece_start + pd.Timedelta(days=89)
        if piece_end > today:
            piece_end = today

        settings = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": forecast_columns,
            "start_date": piece_start.strftime("%Y-%m-%d"),
            "end_date": piece_end.strftime("%Y-%m-%d"),
            "timezone": "Asia/Karachi",
        }
        data = ask_api(previous_runs_url, settings)

        if data is None:
            print("Could not download the weather forecasts for", settings["start_date"], "to", settings["end_date"])
            print("Continuing without them (the rest of the project still works).")
            break

        pieces.append(pd.DataFrame(data["hourly"]))
        piece_start = piece_end + pd.Timedelta(days=1)

    if len(pieces) == 0:
        return None

    forecast_table = pd.concat(pieces)
    forecast_table["time"] = pd.to_datetime(forecast_table["time"])
    forecast_table = forecast_table.drop_duplicates(subset="time")
    return forecast_table


def get_history(history_days):
    today = pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None).normalize()
    start = today - pd.Timedelta(days=history_days)

    # ---------- air quality, in pieces of 90 days ----------
    air_pieces = []
    piece_start = start

    while piece_start <= today:
        piece_end = piece_start + pd.Timedelta(days=89)
        if piece_end > today:
            piece_end = today

        settings = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": air_columns,
            "start_date": piece_start.strftime("%Y-%m-%d"),
            "end_date": piece_end.strftime("%Y-%m-%d"),
            "timezone": "Asia/Karachi",
        }
        data = ask_api(air_url, settings)

        if data is None:
            print("Could not download air quality for", settings["start_date"], "to", settings["end_date"])
            print("Using what was downloaded so far.")
            break

        air_pieces.append(pd.DataFrame(data["hourly"]))
        piece_start = piece_end + pd.Timedelta(days=1)

    if len(air_pieces) == 0:
        print("No air quality data downloaded. Check your internet connection.")
        return None

    air_table = pd.concat(air_pieces)
    air_table["time"] = pd.to_datetime(air_table["time"])
    air_table = air_table.drop_duplicates(subset="time")

    first_air_time = air_table["time"].min()

    # ---------- weather ----------
    weather_pieces = []

    # older days come from the archive (it is about 5 days behind)
    archive_end = today - pd.Timedelta(days=6)
    if first_air_time <= archive_end:
        settings = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": weather_columns,
            "start_date": first_air_time.strftime("%Y-%m-%d"),
            "end_date": archive_end.strftime("%Y-%m-%d"),
            "timezone": "Asia/Karachi",
        }
        data = ask_api(weather_archive_url, settings)
        if data is not None:
            weather_pieces.append(pd.DataFrame(data["hourly"]))

    # the last few days come from the normal forecast service
    settings = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": weather_columns,
        "past_days": 10,
        "forecast_days": 1,
        "timezone": "Asia/Karachi",
    }
    data = ask_api(weather_forecast_url, settings)
    if data is not None:
        weather_pieces.append(pd.DataFrame(data["hourly"]))

    if len(weather_pieces) == 0:
        print("No weather data downloaded. Check your internet connection.")
        return None

    weather_table = pd.concat(weather_pieces)
    weather_table["time"] = pd.to_datetime(weather_table["time"])
    weather_table = weather_table.drop_duplicates(subset="time")

    # ---------- join and remove future hours ----------
    table = pd.merge(air_table, weather_table, on="time")

    # add the day-ahead weather forecasts (kept even if some hours are missing)
    forecast_table = get_weather_forecasts(first_air_time, today)
    if forecast_table is not None:
        table = pd.merge(table, forecast_table, on="time", how="left")

    now = pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None)
    table = table[table["time"] <= now]
    table = table.sort_values("time")
    table = table.reset_index(drop=True)

    return table
