# A small interactive dashboard for the Karachi air quality project
# Run it with:  streamlit run app.py
# Needs these files in the same folder: clean_data.csv, spike_flags.csv,
# model.pkl and model_comparison.csv (the last two come from the notebook)

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Karachi Air Quality", layout="wide")

# the SARIMA model looks at the last 14 days of history when it makes a forecast
history_window = 24 * 14

# US EPA PM2.5 health categories (2024 update): (upper limit, name, color, text color, tip)
# They are made for 24-hour averages, so hourly values are only a guide. Confirm the standard you cite.
categories = [
    (9.0, "Good", "#00e400", "black", "Air is clean. Normal outdoor activity is fine."),
    (35.4, "Moderate", "#ffff00", "black", "Acceptable. Unusually sensitive people may want to limit long outdoor effort."),
    (55.4, "Unhealthy for sensitive groups", "#ff7e00", "black", "People with asthma or heart problems, children and older adults should reduce long or heavy outdoor activity."),
    (125.4, "Unhealthy", "#ff0000", "white", "Everyone should reduce long or heavy outdoor activity. Sensitive groups should avoid it. Consider a well-fitted N95 mask outdoors."),
    (225.4, "Very unhealthy", "#8f3f97", "white", "Avoid outdoor activity. Stay indoors with windows closed if you can."),
    (100000, "Hazardous", "#7e0023", "white", "Stay indoors and avoid all outdoor activity."),
]


def get_category(value):
    for upper, name, color, text_color, tip in categories:
        if value <= upper:
            return name, color, text_color, tip
    return "Hazardous", "#7e0023", "white", "Stay indoors."


# the +24 hour predictions from the notebook: what the models predicted 24 hours before each hour
@st.cache_data
def load_backtest(clean_time):
    try:
        backtest = pd.read_csv("backtest_24h.csv")
        backtest["time"] = pd.to_datetime(backtest["time"])
        return backtest
    except FileNotFoundError:
        return None


# typical error from the notebook results (optional files)
@st.cache_data
def load_error_info(clean_time):
    info = {}
    try:
        error_table = pd.read_csv("model_comparison.csv")
    except FileNotFoundError:
        try:
            error_table = pd.read_csv("sarima_results.csv")
        except FileNotFoundError:
            error_table = None
    info["errors"] = error_table

    try:
        info["coverage"] = pd.read_csv("band_coverage.csv")
    except FileNotFoundError:
        info["coverage"] = None
    return info


# ---------- load the data ----------
# the two file times are passed in so the data reloads whenever the files change
@st.cache_data
def load_data(clean_time, flags_time):
    clean = pd.read_csv("clean_data.csv")
    clean["time"] = pd.to_datetime(clean["time"])

    flags = pd.read_csv("spike_flags.csv")
    flags["time"] = pd.to_datetime(flags["time"])
    flags = flags[["time", "if_spike", "rule_spike"]]

    joined = pd.merge(clean, flags, on="time")
    joined = joined.sort_values("time")
    joined = joined.reset_index(drop=True)
    return joined


# ---------- fit SARIMA one time (same settings as the notebook) ----------
@st.cache_resource
def fit_model(train_values):
    model = SARIMAX(train_values, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
    results = model.fit(disp=False)
    return results


# ---------- load the saved Random Forest (+24 hours) from Phase 6 ----------
@st.cache_resource
def load_saved_model():
    try:
        with open("model.pkl", "rb") as file:
            saved = pickle.load(file)
        return saved
    except FileNotFoundError:
        return None


clean_time = os.path.getmtime("clean_data.csv")
flags_time = os.path.getmtime("spike_flags.csv")
table = load_data(clean_time, flags_time)
saved_model = load_saved_model()
pm = table["pm2_5"]
test_start = int(len(pm) * 0.8)

# fit on the last 90 days before the test period (same as the notebook)
results = fit_model(pm[:test_start][-24 * 90:])


# ---------- title ----------
st.title("Karachi Air Quality")
st.caption("Data: Open-Meteo (CAMS model, not ground sensors). Forecast model: SARIMA.")

latest_hour = table["time"].max().strftime("%d %b %Y, %H:%M")
files_updated = pd.Timestamp(clean_time, unit="s").strftime("%d %b %Y, %H:%M")
st.info("Latest hour in the data: " + latest_hour + "  |  clean_data.csv last updated: " + files_updated + "  |  Run refresh.py to update.")


# ---------- sidebar controls ----------
st.sidebar.header("Controls")
horizon = st.sidebar.selectbox("Forecast hours ahead", [6, 24, 48, 72], index=1)
show_spikes = st.sidebar.checkbox("Show spikes", value=True)
show_band = st.sidebar.checkbox("Show 95% band", value=True)
show_past = st.sidebar.checkbox("Show past predictions (made 24 h earlier)", value=True)
days_before = st.sidebar.slider("Hours of past to show", 24, 240, 96)


# ---------- time cursor ----------
first_time = table["time"].min().to_pydatetime()
last_time = table["time"].max().to_pydatetime()

st.write("Move the cursor to look at a different moment. Far right = latest data.")
cursor_time = st.slider(
    "Time cursor",
    min_value=first_time,
    max_value=last_time,
    value=last_time,
    format="DD MMM, HH:mm",
)

# find the row closest to the chosen time
differences = (table["time"] - cursor_time).abs()
position = differences.idxmin()

# the model needs some past hours, so the cursor cannot be at the very start
if position < 48:
    position = 48

row = table.loc[position]


# ---------- forecast from the cursor ----------
history = pm[max(0, position + 1 - history_window) : position + 1]
new_results = results.apply(history, refit=False)
forecast = new_results.get_forecast(steps=horizon)
forecast_values = forecast.predicted_mean
band = forecast.conf_int()

forecast_at_end = forecast_values.iloc[horizon - 1]

# Random Forest forecast for +24 hours (only if model.pkl exists)
rf_forecast = None
if saved_model is not None:
    ml_input = table.loc[[position]].copy()
    ml_input["target_hour"] = ml_input["hour"]        # +24 hours = same hour of day
    ml_input = ml_input[saved_model["features"]]
    rf_forecast = saved_model["model"].predict(ml_input)[0]


# ---------- numbers at the top ----------
category_name, category_color, category_text, category_tip = get_category(row["pm2_5"])

if row["if_spike"] == 1:
    spike_text = "Yes (Isolation Forest)"
else:
    spike_text = "No"

col1, col2, col3, col4 = st.columns(4)
col1.metric("PM2.5 at cursor", str(round(row["pm2_5"], 1)) + " ug/m3")
col2.metric("Air quality category", category_name)
col3.metric("Forecast in " + str(horizon) + " h", str(round(forecast_at_end, 1)) + " ug/m3")
col4.metric("Spike flagged?", spike_text)

col5, col6, col7, col8 = st.columns(4)
col5.metric("Temperature", str(round(row["temperature_2m"], 1)) + " C")
col6.metric("Humidity", str(round(row["relative_humidity_2m"])) + " %")
col7.metric("Wind", str(round(row["wind_speed_10m"], 1)) + " km/h")

if rf_forecast is not None:
    col8.metric("Random Forest in 24 h", str(round(rf_forecast, 1)) + " ug/m3")
else:
    col8.metric("Random Forest in 24 h", "run Phase 6")


# ---------- how much to trust the forecast ----------
error_info = load_error_info(clean_time)
trust_text = "Run the notebook to see the typical forecast error."

if error_info["errors"] is not None:
    error_row = error_info["errors"][error_info["errors"]["horizon_hours"] == horizon]
    if len(error_row) == 1:
        typical_error = error_row["SARIMA_RMSE"].values[0]
        trust_text = "At +" + str(horizon) + " h the SARIMA forecast is typically off by about " + str(round(typical_error, 1)) + " ug/m3 (RMSE on the test period)."

if error_info["coverage"] is not None:
    coverage_row = error_info["coverage"][error_info["coverage"]["horizon_hours"] == horizon]
    if len(coverage_row) == 1:
        inside = coverage_row["percent_inside_95_band"].values[0]
        trust_text = trust_text + " The 95% band contained the real value " + str(round(inside)) + "% of the time."

st.info(trust_text)


# ---------- chart ----------
start = max(0, position - days_before)
past_table = table[start : position + 1]

# the real values after the cursor (only if they exist)
end = min(len(table), position + 1 + horizon)
future_table = table[position + 1 : end]

# time labels for the forecast hours
future_times = []
for step in range(1, horizon + 1):
    future_times.append(row["time"] + pd.Timedelta(hours=step))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(past_table["time"], past_table["pm2_5"], color="blue", label="Real PM2.5 (past)")

if len(future_table) > 0:
    ax.plot(future_table["time"], future_table["pm2_5"], color="gray", label="Real PM2.5 (after cursor)")

ax.plot(future_times, forecast_values.values, color="green", linestyle="--", label="SARIMA forecast")

if show_band:
    ax.fill_between(future_times, band.iloc[:, 0].values, band.iloc[:, 1].values, color="green", alpha=0.2, label="95% band")

if show_spikes:
    spike_points = past_table[past_table["if_spike"] == 1]
    ax.scatter(spike_points["time"], spike_points["pm2_5"], color="red", s=40, label="Spike")

# what the models predicted in the past, versus what really happened (the blue line)
past_note = ""
if show_past:
    backtest = load_backtest(clean_time)

    if backtest is None:
        past_note = "Past predictions need backtest_24h.csv. Run the notebook (Part 3) to create it."
    else:
        in_window = backtest[(backtest["time"] >= past_table["time"].min()) & (backtest["time"] <= row["time"])]

        if len(in_window) == 0:
            past_note = "No past predictions for this window. They exist only for the test period (the last 20% of the data)."
        else:
            ax.plot(in_window["time"], in_window["sarima_24h"], color="green", marker="o", markersize=3, linewidth=1, alpha=0.8, label="SARIMA (made 24 h earlier)")
            ax.plot(in_window["time"], in_window["random_forest_24h"], color="purple", marker="o", markersize=3, linewidth=1, alpha=0.8, label="Random Forest (made 24 h earlier)")

            sarima_miss = (in_window["real"] - in_window["sarima_24h"]).abs().mean()
            forest_miss = (in_window["real"] - in_window["random_forest_24h"]).abs().mean()
            past_note = ("In the past part of this window (" + str(len(in_window)) + " points), the +24 h predictions were off by about "
                         + str(round(sarima_miss, 1)) + " ug/m3 (SARIMA) and " + str(round(forest_miss, 1)) + " ug/m3 (Random Forest) on average.")

if rf_forecast is not None:
    ax.scatter([row["time"] + pd.Timedelta(hours=24)], [rf_forecast], color="purple", s=60, marker="D", label="Random Forest (+24 h)")

# colored health bands behind the lines
y_top = ax.get_ylim()[1]
band_low = 0
for upper, name, color, text_color, tip in categories:
    if band_low < y_top:
        ax.axhspan(band_low, min(upper, y_top), color=color, alpha=0.12)
    band_low = upper
ax.set_ylim(0, y_top)

ax.axvline(row["time"], color="black", linestyle="--")
ax.set_xlabel("Time")
ax.set_ylabel("PM2.5 (ug/m3)")
ax.legend(loc="upper left")
fig.autofmt_xdate()

st.pyplot(fig)

if past_note != "":
    st.caption(past_note)


# ---------- health banner and alert ----------
banner = "<div style='background-color:" + category_color + "; color:" + category_text + "; padding:12px; border-radius:8px;'>"
banner = banner + "<b>" + category_name + "</b>: " + category_tip + "</div>"
st.markdown(banner, unsafe_allow_html=True)
st.caption("General guidance based on US EPA categories, not medical advice. Categories are defined for 24-hour averages.")

if row["if_spike"] == 1:
    st.error("Unusual spike at this time (Isolation Forest).")


# ---------- model comparison ----------
st.subheader("Model comparison (RMSE by forecast horizon)")
try:
    compare = pd.read_csv("model_comparison.csv")

    rmse_columns = []
    for column in compare.columns:
        if column.endswith("_RMSE"):
            rmse_columns.append(column)

    st.line_chart(compare.set_index("horizon_hours")[rmse_columns])
    st.dataframe(compare)
except FileNotFoundError:
    st.info("Run the notebook (Part 3) first to create model_comparison.csv.")
