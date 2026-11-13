# A small interactive dashboard for the Karachi air quality project
# Run it with:  streamlit run app.py
# Needs these files in the same folder: clean_data.csv, spike_flags.csv,
# model.pkl and model_comparison.csv (the last two come from analysis_phase6.ipynb)

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Karachi Air Quality", layout="wide")

# example limit for "unhealthy" (confirm the standard you cite in the report)
limit = 35


# ---------- load the data ----------
@st.cache_data
def load_data():
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


table = load_data()
saved_model = load_saved_model()
pm = table["pm2_5"]
test_start = int(len(pm) * 0.8)

results = fit_model(pm[:test_start])


# ---------- title ----------
st.title("Karachi Air Quality")
st.caption("Data: Open-Meteo (CAMS model, not ground sensors). Forecast model: SARIMA.")


# ---------- sidebar controls ----------
st.sidebar.header("Controls")
horizon = st.sidebar.selectbox("Forecast hours ahead", [6, 24, 48, 72], index=1)
show_spikes = st.sidebar.checkbox("Show spikes", value=True)
show_band = st.sidebar.checkbox("Show 95% band", value=True)
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
history = pm[: position + 1]
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
if row["pm2_5"] > limit:
    status = "Above limit"
else:
    status = "Within limit"

if row["if_spike"] == 1:
    spike_text = "Yes (Isolation Forest)"
else:
    spike_text = "No"

col1, col2, col3, col4 = st.columns(4)
col1.metric("PM2.5 at cursor", str(round(row["pm2_5"], 1)) + " ug/m3")
col2.metric("Status (limit " + str(limit) + ")", status)
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

if rf_forecast is not None:
    ax.scatter([row["time"] + pd.Timedelta(hours=24)], [rf_forecast], color="purple", s=60, marker="D", label="Random Forest (+24 h)")

ax.axhline(limit, color="red", linestyle=":", label="Limit (example)")
ax.axvline(row["time"], color="black", linestyle="--")
ax.set_xlabel("Time")
ax.set_ylabel("PM2.5 (ug/m3)")
ax.legend(loc="upper left")
fig.autofmt_xdate()

st.pyplot(fig)


# ---------- alert ----------
if row["if_spike"] == 1:
    st.error("Unusual spike at this time (Isolation Forest).")
elif row["pm2_5"] > limit:
    st.warning("PM2.5 is above the example limit at this time.")
else:
    st.success("PM2.5 looks normal at this time.")


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
    st.info("Run analysis_phase6.ipynb first to create model_comparison.csv.")
