# Real-Time Air Quality Forecasting and Spike Detection for Karachi

Course project for Introduction to Data Science.

**Team:** Naresh Kumar (25K-2501) and Syed Zaid Khalid (25K-2522)

## What this project does

This project collects hourly air quality and weather data for Karachi and uses it to:

1. Forecast PM2.5 for the next 24 to 72 hours.
2. Flag unusual pollution spikes.

New readings are collected live with a small script, so the dataset keeps growing until the final evaluation.

## Data source

- **API:** Open-Meteo Air Quality API and Open-Meteo Weather API (free, no key needed).
- **Location:** Karachi (latitude 24.86, longitude 67.01).
- **Air quality columns:** `pm2_5`, `pm10`, `nitrogen_dioxide`, `ozone`
- **Weather columns:** `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`

**Limitation:** Open-Meteo's air quality values come from an atmospheric model (CAMS), not from ground sensors, so they approximate real readings.

## Files

| File | What it does |
|---|---|
| `backfill.py` | Downloads about 92 days of past hourly data and saves `karachi_aq_history.csv` (run once) |
| `collector.py` | Saves one new reading every hour to `live_data.csv` (keep it running) |
| `analysis.ipynb` | Phase 2 (cleaning and features) and Phase 3 (EDA with 5 charts). Saves `clean_data.csv` |
| `app.py` | Interactive dashboard (time cursor, forecast horizon, spike toggle). Run with `streamlit run app.py` |
| `analysis_phase4_5.ipynb` | Phase 4 (SARIMA forecasting) and Phase 5 (spike detection). Saves `sarima_results.csv` and `spike_flags.csv` |

Files created when you run the code:

| File | Made by |
|---|---|
| `karachi_aq_history.csv` | `backfill.py` |
| `live_data.csv` | `collector.py` |
| `clean_data.csv` | `analysis.ipynb` |
| `sarima_results.csv` | `analysis_phase4_5.ipynb` |
| `spike_flags.csv` | `analysis_phase4_5.ipynb` |

## Setup

You need Python 3 and these libraries:

```
pip install requests pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter streamlit
```

## How to run (in this order)

1. Put all files in one folder.
2. Run the backfill once:
   ```
   python backfill.py
   ```
3. Start the live collector and leave it running (laptop on). Stop it with Ctrl + C:
   ```
   python collector.py
   ```
4. Open `analysis.ipynb` and run all cells from top to bottom (Phases 2 and 3).
5. Open `analysis_phase4_5.ipynb` and run all cells from top to bottom (Phases 4 and 5). The SARIMA step can take about a minute.
6. Start the dashboard:
   ```
   streamlit run app.py
   ```

## Methods

| Method | Used for | Status |
|---|---|---|
| SARIMA | Main forecasting model, with a 24-hour seasonal cycle and 95% confidence bands | Done (Phase 4) |
| Naive baseline | Minimum score to beat (same hour on the last known day) | Done (Phase 4) |
| Isolation Forest | Main spike detection | Done (Phase 5) |
| Simple rule (95th percentile) | Comparison for spike detection | Done (Phase 5) |
| DBSCAN | Optional second opinion for spike detection | Done (Phase 5) |
| Linear Regression, Random Forest, Gradient Boosting | Baselines, added after each is taught in class | Planned (Phase 6) |

**Evaluation:** RMSE and MAE at +1, +6, +24, +48, and +72 hours. The data is split by time (earlier 80% for training, latest 20% for testing), never shuffled. Lag and rolling features use only past values so the models never see the future.

## Project phases

| Phase | Work | Status |
|---|---|---|
| 1 | Setup and data collection | Done |
| 2 | Cleaning and features | Done |
| 3 | Exploration (EDA) | Done |
| 4 | SARIMA forecasting | Done |
| 5 | Spike detection | Done |
| 6 | Baselines and model comparison | Planned |
| 7 | Streamlit dashboard (`app.py`) | First version done |
| 8 | Final report (PDF) and presentation slides | Planned |

## Notes

- Spikes have no true labels, so they are compared with a simple rule and between methods, not scored for accuracy.
- Each forecast horizon is tested from about 16 starting points, so the error numbers are approximate.
- The 35 µg/m³ line used on some charts is an example limit. Confirm the standard you cite in the report.
- The final evaluation should use a frozen copy of the data, so the results stay the same. The live feed is used only for the demo.
