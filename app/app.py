# Karachi air quality dashboard (plain-language version)
# Start it from the project folder with:   python -m streamlit run app/app.py
#
# It reads files made by the notebook and the scripts:
#   data/processed/clean_data.csv and spike_flags.csv   (python src/refresh.py makes them)
#   results/...  and models/model.pkl                   (the notebook makes them)

import os
import pickle
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

# the project folder is one level above this file
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Karachi Air Quality", layout="wide")

# the SARIMA model looks at the last 14 days of history when it makes a forecast
history_window = 24 * 14

# US EPA PM2.5 health categories (2024 update): (upper limit, name, color, text color, tip)
# They are made for 24-hour averages, so hourly values are only a guide.
categories = [
    (9.0, "Good", "#00e400", "black", "The air is clean. Normal outdoor activity is fine."),
    (35.4, "Moderate", "#ffff00", "black", "The air is acceptable. Unusually sensitive people may want to limit long outdoor effort."),
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


def show_badge(text, color, text_color):
    html = "<div style='background-color:" + color + "; color:" + text_color + "; padding:12px 16px; border-radius:8px; font-size:1.05rem;'>"
    html = html + text + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def add_health_bands(ax):
    y_top = ax.get_ylim()[1]
    band_low = 0
    for upper, name, color, text_color, tip in categories:
        if band_low < y_top:
            ax.axhspan(band_low, min(upper, y_top), color=color, alpha=0.13)
        band_low = upper
    ax.set_ylim(0, y_top)


def trend_words(now_value, later_value):
    difference = later_value - now_value
    if difference > 3:
        return "getting worse"
    if difference < -3:
        return "getting better"
    return "about the same"


def hours_label(hours):
    return "Next " + str(hours) + " hours"


def days_label(days):
    return str(days) + " days"


def trust_words(hours):
    if hours <= 6:
        return "High. Forecasts up to about 6 hours ahead are the most reliable."
    if hours <= 24:
        return "Medium. Useful for the general direction, not for exact numbers."
    return "Low. Treat it as a rough guide only."


# ---------------------------------------------------------------- loading
@st.cache_data
def load_data(clean_time, flags_time):
    clean = pd.read_csv(os.path.join(ROOT, "data", "processed", "clean_data.csv"))
    clean["time"] = pd.to_datetime(clean["time"])

    flags = pd.read_csv(os.path.join(ROOT, "data", "processed", "spike_flags.csv"))
    flags["time"] = pd.to_datetime(flags["time"])
    flags = flags[["time", "if_spike", "rule_spike"]]

    joined = pd.merge(clean, flags, on="time")
    joined = joined.sort_values("time")
    joined = joined.reset_index(drop=True)
    return joined


@st.cache_resource
def fit_model(train_values):
    model = SARIMAX(train_values, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
    return model.fit(disp=False)


@st.cache_resource
def load_saved_model():
    try:
        with open(os.path.join(ROOT, "models", "model.pkl"), "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None


@st.cache_data
def load_backtest(clean_time):
    try:
        backtest = pd.read_csv(os.path.join(ROOT, "results", "backtest_24h.csv"))
        backtest["time"] = pd.to_datetime(backtest["time"])
        return backtest
    except FileNotFoundError:
        return None


@st.cache_data
def load_results(clean_time):
    info = {}
    for key, name in [("compare", "model_comparison.csv"), ("sarima", "sarima_results.csv"), ("coverage", "band_coverage.csv")]:
        try:
            info[key] = pd.read_csv(os.path.join(ROOT, "results", name))
        except FileNotFoundError:
            info[key] = None
    return info


# the forecast for the next 72 hours from one moment (kept in memory so moving between tabs is fast)
@st.cache_data
def make_forecast(_results, _pm, position, stamp):
    history = _pm[max(0, position + 1 - history_window) : position + 1]
    new_results = _results.apply(history, refit=False)
    forecast = new_results.get_forecast(steps=72)
    values = forecast.predicted_mean.values
    band = forecast.conf_int()
    return values, band.iloc[:, 0].values, band.iloc[:, 1].values


# ---------------------------------------------------------------- data checks
data_folder = os.path.join(ROOT, "data", "processed")
clean_path = os.path.join(data_folder, "clean_data.csv")
flags_path = os.path.join(data_folder, "spike_flags.csv")

if not (os.path.exists(clean_path) and os.path.exists(flags_path)):
    st.title("Karachi Air Quality")
    st.error("The data files are missing, so there is nothing to show yet.")
    st.write("Run these two commands from the project folder, then reload this page:")
    st.code("python src/backfill.py\npython src/refresh.py")
    st.stop()

clean_time = os.path.getmtime(clean_path)
flags_time = os.path.getmtime(flags_path)
table = load_data(clean_time, flags_time)
saved_model = load_saved_model()
all_results = load_results(clean_time)
pm = table["pm2_5"]
test_start = int(len(pm) * 0.8)

with st.spinner("Loading the forecasting model. The first time takes about a minute..."):
    sarima_results = fit_model(pm[:test_start][-24 * 90:])

first_time = table["time"][48].to_pydatetime()
last_time = table["time"].max().to_pydatetime()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Data status")
st.sidebar.write("Latest hour in the data:")
st.sidebar.write("**" + table["time"].max().strftime("%d %b %Y, %H:%M") + "**")
st.sidebar.write("Hours of data: " + str(len(table)))
st.sidebar.caption("Files updated: " + pd.Timestamp(clean_time, unit="s").strftime("%d %b %Y, %H:%M"))
st.sidebar.caption("To load newer data, run `python src/refresh.py` and reload this page.")

st.sidebar.header("Display options")
days_before = st.sidebar.slider("Hours of the past to show", 24, 240, 96)
show_bands = st.sidebar.checkbox("Color the charts by health category", value=True)
show_spikes = st.sidebar.checkbox("Mark spikes on the charts", value=True)
show_band = st.sidebar.checkbox("Show the uncertainty band", value=True)

# ---------------------------------------------------------------- title and help
st.title("Karachi air quality: what is it now, and what is coming?")
st.write("This page shows the amount of fine dust (PM2.5) in Karachi's air, what our models expect next, and how good their earlier predictions were.")

with st.expander("How to use this page (read this first)"):
    st.markdown(
        "1. **Pick a moment** with the slider or the buttons below. The far right is the latest data.\n"
        "2. Open the tabs, left to right:\n"
        "   - **Right now**: how the air is at that moment, in plain words.\n"
        "   - **Forecast**: what the models expect over the next hours.\n"
        "   - **Past predictions**: how close earlier predictions were to what really happened.\n"
        "   - **Spikes**: unusual jumps in pollution.\n"
        "   - **Model scores**: which model is the most accurate.\n"
        "   - **About**: where the data comes from and what the words mean.\n"
        "3. Move the slider **left** to replay the past. That is the best way to check the forecast against reality."
    )

# ---------------------------------------------------------------- pick a moment
st.markdown("#### Step 1: pick a moment")

if "cursor_time" not in st.session_state:
    st.session_state["cursor_time"] = last_time

# keep the saved moment inside the data (the data may have been refreshed)
saved_time = st.session_state["cursor_time"]
if saved_time > last_time:
    st.session_state["cursor_time"] = last_time
if saved_time < first_time:
    st.session_state["cursor_time"] = first_time


def move_time(hours):
    new_time = st.session_state["cursor_time"] + datetime.timedelta(hours=hours)
    if new_time > last_time:
        new_time = last_time
    if new_time < first_time:
        new_time = first_time
    st.session_state["cursor_time"] = new_time


def go_latest():
    st.session_state["cursor_time"] = last_time


button_columns = st.columns(5)
button_columns[0].button("Latest", on_click=go_latest, key="b_latest")
button_columns[1].button("1 week back", on_click=move_time, args=(-168,), key="b_week")
button_columns[2].button("1 day back", on_click=move_time, args=(-24,), key="b_day_back")
button_columns[3].button("1 hour back", on_click=move_time, args=(-1,), key="b_hour_back")
button_columns[4].button("1 hour forward", on_click=move_time, args=(1,), key="b_hour_forward")

cursor_time = st.slider("Moment to look at", min_value=first_time, max_value=last_time, key="cursor_time", format="DD MMM, HH:mm")

# find the row closest to the chosen moment
differences = (table["time"] - cursor_time).abs()
position = differences.idxmin()
row = table.loc[position]
when_text = row["time"].strftime("%d %b %Y, %H:%M")

if position == len(table) - 1:
    st.caption("You are looking at the **latest** data. Nothing after this has happened yet, so there is no real value to compare the forecast with.")
else:
    st.caption("You are looking at the **past**. The page shows what the models would have predicted at this moment, so you can compare it with what really happened.")

# the forecast and the Random Forest guess for this moment
forecast_values, forecast_low, forecast_high = make_forecast(sarima_results, pm, position, clean_time)

rf_forecast = None
if saved_model is not None:
    ml_input = table.loc[[position]].copy()
    ml_input["target_hour"] = ml_input["hour"]
    ml_input = ml_input[saved_model["features"]]
    rf_forecast = saved_model["model"].predict(ml_input)[0]

now_value = row["pm2_5"]
now_name, now_color, now_text, now_tip = get_category(now_value)

# ---------------------------------------------------------------- tabs
tab_now, tab_forecast, tab_past, tab_spikes, tab_scores, tab_about = st.tabs(
    ["Right now", "Forecast", "Past predictions", "Spikes", "Model scores", "About"]
)

# ================================================================ tab 1: right now
with tab_now:
    st.subheader("How is the air at " + when_text + "?")

    show_badge("PM2.5 is <b>" + str(round(now_value, 1)) + " ug/m3</b>. Air quality: <b>" + now_name + "</b>.<br>" + now_tip, now_color, now_text)
    st.caption("General guidance based on US EPA categories, not medical advice. The categories are made for 24-hour averages, so a single hour is only a guide.")

    later_value = forecast_values[23]
    later_name, later_color, later_text, later_tip = get_category(later_value)
    st.markdown(
        "**Looking ahead:** in 24 hours the forecast is about **" + str(round(later_value, 1)) + " ug/m3** (likely between "
        + str(round(forecast_low[23], 1)) + " and " + str(round(forecast_high[23], 1)) + "), which is **" + later_name
        + "**. The air is **" + trend_words(now_value, later_value) + "**."
    )

    columns = st.columns(4)
    columns[0].metric("PM2.5 now", str(round(now_value, 1)) + " ug/m3")
    columns[1].metric("Category", now_name)
    columns[2].metric("In 24 hours", str(round(later_value, 1)) + " ug/m3", delta=str(round(later_value - now_value, 1)) + " vs now", delta_color="inverse")
    if row["if_spike"] == 1:
        columns[3].metric("Spike?", "Yes")
    else:
        columns[3].metric("Spike?", "No")

    if row["if_spike"] == 1:
        st.warning("This hour was flagged as a **spike**: PM2.5 is unusually high for this time of day (found by Isolation Forest). See the Spikes tab.")

    st.markdown("**The weather at that moment**")
    weather_columns = st.columns(3)
    weather_columns[0].metric("Temperature", str(round(row["temperature_2m"], 1)) + " C")
    weather_columns[1].metric("Humidity", str(round(row["relative_humidity_2m"])) + " %")
    weather_columns[2].metric("Wind", str(round(row["wind_speed_10m"], 1)) + " km/h")
    st.caption("In our data, stronger wind and higher temperature go with lower PM2.5. Weak wind lets dust build up. This is a pattern, not proof of cause.")

    with st.expander("What do these numbers mean?"):
        st.markdown(
            "- **PM2.5** is the amount of very small dust particles in the air, in micrograms per cubic metre (ug/m3). Lower is better.\n"
            "- **Category** is a colour-coded label for how healthy the air is.\n"
            "- **In 24 hours** is the SARIMA model's forecast. The change next to it compares it with now (green = better).\n"
            "- **Spike** means the reading is unusual for that time of day."
        )

# ================================================================ tab 2: forecast
with tab_forecast:
    st.subheader("What do the models expect next?")

    horizon = st.radio("How far ahead?", [6, 24, 48, 72], index=1, horizontal=True, key="horizon",
                       format_func=hours_label)

    end_value = forecast_values[horizon - 1]
    end_low = forecast_low[horizon - 1]
    end_high = forecast_high[horizon - 1]
    end_name, end_color, end_text, end_tip = get_category(end_value)
    end_time = (row["time"] + pd.Timedelta(hours=horizon)).strftime("%d %b, %H:%M")

    show_badge("At <b>" + end_time + "</b> (" + str(horizon) + " hours after the moment you picked), PM2.5 is expected to be about <b>"
               + str(round(end_value, 1)) + " ug/m3</b>, likely between " + str(round(end_low, 1)) + " and " + str(round(end_high, 1))
               + ". That is <b>" + end_name + "</b>.", end_color, end_text)

    trust_text = "**How much to trust it:** " + trust_words(horizon)
    if all_results["compare"] is not None:
        error_row = all_results["compare"][all_results["compare"]["horizon_hours"] == horizon]
        if len(error_row) == 1:
            trust_text = trust_text + " On the test period, this forecast was typically off by about " + str(round(error_row["SARIMA_RMSE"].values[0], 1)) + " ug/m3."
    if all_results["coverage"] is not None:
        coverage_row = all_results["coverage"][all_results["coverage"]["horizon_hours"] == horizon]
        if len(coverage_row) == 1:
            trust_text = trust_text + " The shaded band contained the real value " + str(round(coverage_row["percent_inside_95_band"].values[0])) + "% of the time."
    st.markdown(trust_text)

    # the chart
    recent_table = table[max(0, position - 48) : position + 1]
    future_table = table[position + 1 : min(len(table), position + 1 + horizon)]
    future_times = []
    for step in range(1, horizon + 1):
        future_times.append(row["time"] + pd.Timedelta(hours=step))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(recent_table["time"], recent_table["pm2_5"], color="black", linewidth=2, label="What happened (the 48 hours before)")
    if len(future_table) > 0:
        ax.plot(future_table["time"], future_table["pm2_5"], color="gray", linewidth=2, label="What really happened afterwards")
    ax.plot(future_times, forecast_values[:horizon], color="green", linestyle="--", linewidth=2, label="SARIMA forecast")
    if show_band:
        ax.fill_between(future_times, forecast_low[:horizon], forecast_high[:horizon], color="green", alpha=0.2, label="Likely range (95%)")
    if rf_forecast is not None and horizon >= 24:
        ax.scatter([row["time"] + pd.Timedelta(hours=24)], [rf_forecast], color="purple", s=80, marker="D", zorder=4,
                   edgecolors="black", label="Random Forest guess for +24 h")
    ax.axvline(row["time"], color="black", linestyle=":")
    if show_bands:
        add_health_bands(ax)
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.autofmt_xdate()
    st.pyplot(fig)
    st.caption("The dotted vertical line is the moment you picked. Everything to its right is a forecast.")

    # reality check when the future is known
    if position + horizon < len(table):
        real_end = table.loc[position + horizon, "pm2_5"]
        st.markdown("**Reality check:** after " + str(horizon) + " hours the real PM2.5 was **" + str(round(real_end, 1))
                    + " ug/m3**. The forecast was off by **" + str(round(abs(end_value - real_end), 1)) + " ug/m3**.")
    if rf_forecast is not None and horizon >= 24:
        st.markdown("**Second opinion:** the Random Forest model guesses **" + str(round(rf_forecast, 1)) + " ug/m3** for +24 hours.")

    # table and download
    table_rows = []
    for step in [1, 3, 6, 12, 24, 36, 48, 72]:
        if step <= horizon:
            name, color, text_color, tip = get_category(forecast_values[step - 1])
            table_rows.append({
                "When": (row["time"] + pd.Timedelta(hours=step)).strftime("%d %b, %H:%M"),
                "Hours ahead": step,
                "Forecast (ug/m3)": round(forecast_values[step - 1], 1),
                "Likely range": str(round(forecast_low[step - 1], 1)) + " to " + str(round(forecast_high[step - 1], 1)),
                "Category": name,
            })
    with st.expander("Show the forecast as a table"):
        st.dataframe(pd.DataFrame(table_rows))

    download_rows = pd.DataFrame({
        "time": future_times,
        "forecast_pm2_5": forecast_values[:horizon],
        "low_95": forecast_low[:horizon],
        "high_95": forecast_high[:horizon],
    })
    st.download_button("Download this forecast (CSV)", download_rows.to_csv(index=False).encode("utf-8"), file_name="forecast.csv", mime="text/csv")

    with st.expander("How to read this chart"):
        st.markdown(
            "- The **black line** is what really happened before your moment.\n"
            "- The **green dashed line** is the SARIMA forecast. The **green band** is the range where the real value should fall about 95% of the time.\n"
            "- The **gray line** shows what really happened afterwards, when that is already known.\n"
            "- A wide band means the model is less sure. It gets wider the further ahead you look.\n"
            "- The coloured background shows the health category for each level."
        )

# ================================================================ tab 3: past predictions
with tab_past:
    st.subheader("Were the earlier predictions any good?")
    st.write("Every dot is a prediction that was made **24 hours earlier** for that hour. The closer a dot is to the black line, the better the prediction.")

    model_choice = st.radio("Which model?", ["SARIMA", "Random Forest"], horizontal=True, key="model_choice")
    if model_choice == "SARIMA":
        column = "sarima_24h"
        model_color = "#00C853"
    else:
        column = "random_forest_24h"
        model_color = "purple"

    start = max(0, position - days_before)
    past_table = table[start : position + 1]
    backtest = load_backtest(clean_time)

    past_points = None
    message = ""
    if backtest is None:
        message = "Past predictions need results/backtest_24h.csv. Run the notebook to create it."
    else:
        past_points = backtest[(backtest["time"] >= past_table["time"].min()) & (backtest["time"] <= row["time"])]
        if len(past_points) == 0:
            past_points = None
            message = "There are no past predictions for this window. They exist only for the test period (the last 20% of the data), so move the moment to the right."

    if past_points is None:
        st.info(message)
    else:
        miss = past_points[column] - past_points["real"]
        average_miss = miss.abs().mean()
        within_10 = (miss.abs() <= 10).mean() * 100

        summary = st.columns(3)
        summary[0].metric("Predictions in this window", str(len(past_points)))
        summary[1].metric("Average miss", str(round(average_miss, 1)) + " ug/m3")
        summary[2].metric("Within 10 ug/m3", str(round(within_10)) + "%")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(past_table["time"], past_table["pm2_5"], color="black", linewidth=2, label="What really happened")
        if show_spikes:
            spike_points = past_table[past_table["if_spike"] == 1]
            if len(spike_points) > 0:
                ax1.scatter(spike_points["time"], spike_points["pm2_5"], color="#FF5E5B", s=50, zorder=3, edgecolors="black", label="Spike flagged")
        for index, point in past_points.iterrows():
            ax1.plot([point["time"], point["time"]], [point["real"], point[column]], color="#888888", linewidth=1)
        ax1.scatter(past_points["time"], past_points[column], color=model_color, s=40, zorder=4, edgecolors="black",
                    label=model_choice + " prediction (made 24 h earlier)")
        if show_bands:
            add_health_bands(ax1)
        ax1.set_ylabel("PM2.5 (ug/m3)")
        ax1.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

        error_colors = []
        for value in miss:
            if value >= 0:
                error_colors.append("#FF9F43")
            else:
                error_colors.append("#4FC3F7")
        ax2.bar(past_points["time"], miss, width=0.08, color=error_colors, edgecolor="black", linewidth=0.8)
        ax2.axhline(0, color="black", linewidth=1.5)
        ax2.set_ylabel("Miss (ug/m3)\n up = too high")
        fig.autofmt_xdate()
        st.pyplot(fig)

        st.caption("Orange bars: the prediction was too high. Blue bars: too low. Predictions exist every 3 hours.")
        gap_hours = (row["time"] - past_points["time"].max()).total_seconds() / 3600
        if gap_hours > 24:
            st.caption("The dots stop about " + str(round(gap_hours / 24)) + " days before your moment, because the notebook only tests forecasts that have 72 hours of real data after them.")

# ================================================================ tab 4: spikes
with tab_spikes:
    st.subheader("Unusual jumps in pollution")
    st.write("A **spike** is an hour that is unusually high for the time of day. Two methods look for them: **Isolation Forest** (a model that spots odd readings) and a **simple rule** (PM2.5 above the 95th percentile of the training period).")

    look_back = st.select_slider("How far back to look", options=[3, 7, 14, 30], value=7, key="look_back",
                                 format_func=days_label)
    window = table[(table["time"] >= row["time"] - pd.Timedelta(days=look_back)) & (table["time"] <= row["time"])]

    forest_spikes = window[window["if_spike"] == 1]
    rule_spikes = window[window["rule_spike"] == 1]

    counts = st.columns(3)
    counts[0].metric("Hours in this window", str(len(window)))
    counts[1].metric("Isolation Forest spikes", str(len(forest_spikes)))
    counts[2].metric("Simple rule spikes", str(len(rule_spikes)))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(window["time"], window["pm2_5"], color="black", linewidth=1.5, label="PM2.5")
    if len(rule_spikes) > 0:
        ax.scatter(rule_spikes["time"], rule_spikes["pm2_5"], color="orange", marker="x", s=40, label="Simple rule spike")
    if len(forest_spikes) > 0:
        ax.scatter(forest_spikes["time"], forest_spikes["pm2_5"], color="#FF5E5B", s=60, edgecolors="black", zorder=3, label="Isolation Forest spike")
    if show_bands:
        add_health_bands(ax)
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.autofmt_xdate()
    st.pyplot(fig)

    flagged = window[(window["if_spike"] == 1) | (window["rule_spike"] == 1)]
    if len(flagged) == 0:
        st.success("No spikes in this window. The air did not jump unusually.")
    else:
        spike_table = pd.DataFrame({
            "Time": flagged["time"].dt.strftime("%d %b, %H:%M"),
            "PM2.5 (ug/m3)": flagged["pm2_5"].round(1),
            "Isolation Forest": flagged["if_spike"].map({1: "flagged", 0: ""}),
            "Simple rule": flagged["rule_spike"].map({1: "flagged", 0: ""}),
        })
        with st.expander("Show the flagged hours (" + str(len(flagged)) + ")"):
            st.dataframe(spike_table)

    st.caption("There are no true spike labels, so the two methods are compared with each other and cannot be scored for accuracy.")

# ================================================================ tab 5: model scores
with tab_scores:
    st.subheader("Which model is the most accurate?")
    st.write("These scores come from the notebook. Each model was tested on later data it had never seen. **Lower is better.** The number is the typical error in ug/m3.")

    compare = all_results["compare"]
    if compare is None:
        st.info("Run the notebook first to create results/model_comparison.csv.")
    else:
        rmse_columns = []
        for name in compare.columns:
            if name.endswith("_RMSE"):
                rmse_columns.append(name)

        # best model for every horizon, in plain words
        best_rows = []
        for index, line in compare.iterrows():
            best_name = ""
            best_value = 1000000
            for name in rmse_columns:
                if line[name] < best_value:
                    best_value = line[name]
                    best_name = name.replace("_RMSE", "")
            best_rows.append({"Hours ahead": int(line["horizon_hours"]), "Most accurate model": best_name, "Typical error (ug/m3)": round(best_value, 2)})
        st.markdown("**Best model for each horizon**")
        st.dataframe(pd.DataFrame(best_rows))

        st.markdown("**Typical error by horizon, all models**")
        st.line_chart(compare.set_index("horizon_hours")[rmse_columns])
        st.caption("Errors grow the further ahead you look. A model that is not below the Naive line is not adding anything over 'same hour yesterday'.")

        with st.expander("Show the full table"):
            st.dataframe(compare)

    coverage = all_results["coverage"]
    if coverage is not None:
        st.markdown("**How honest is the green band?**")
        st.write("A 95% band should contain the real value about 95% of the time.")
        st.bar_chart(coverage.set_index("horizon_hours")["percent_inside_95_band"])

# ================================================================ tab 6: about
with tab_about:
    st.subheader("What is going on behind this page?")
    st.markdown(
        "**1. Data.** Every hour of air quality and weather for Karachi comes from the free Open-Meteo service. The air quality values are **model output (CAMS), not sensor readings**, so they approximate real conditions.\n\n"
        "**2. Cleaning.** The scripts remove duplicates, fill small gaps and add helpful columns, such as the PM2.5 value 1, 3 and 24 hours earlier.\n\n"
        "**3. Models.** SARIMA repeats the daily pattern and gives a likely range. Random Forest, Linear Regression and Gradient Boosting are trained on past data. Isolation Forest finds spikes.\n\n"
        "**4. Testing.** Models are tested on the last 20% of the data, which they never saw, never on shuffled data.\n\n"
        "**5. This page** replays any moment so you can compare each forecast with what really happened."
    )

    st.markdown("**Words used on this page**")
    glossary = pd.DataFrame([
        {"Word": "PM2.5", "Meaning": "Tiny dust particles in the air. Lower is better."},
        {"Word": "ug/m3", "Meaning": "Micrograms of dust per cubic metre of air."},
        {"Word": "Forecast", "Meaning": "What a model expects to happen next."},
        {"Word": "95% band", "Meaning": "The range where the real value should fall about 95% of the time."},
        {"Word": "SARIMA", "Meaning": "A time-series model that repeats the daily pattern."},
        {"Word": "Random Forest", "Meaning": "Many decision trees that vote on an answer."},
        {"Word": "Isolation Forest", "Meaning": "A model that finds unusual readings."},
        {"Word": "RMSE", "Meaning": "The typical size of a model's error. Lower is better."},
        {"Word": "Naive baseline", "Meaning": "The simple guess: the same hour as yesterday."},
    ])
    st.dataframe(glossary)

    st.markdown("**Things to keep in mind**")
    st.markdown(
        "- Forecasts are most reliable up to about 6 hours ahead. After 24 hours they are a rough guide.\n"
        "- The data is model output, not sensor readings.\n"
        "- Health categories are for 24-hour averages. This page gives general guidance, not medical advice."
    )
