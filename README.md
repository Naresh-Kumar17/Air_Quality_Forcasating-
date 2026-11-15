# Real-Time Air Quality Forecasting and Spike Detection for Karachi

Course project for Introduction to Data Science.

**Team:** Naresh Kumar (25K-2501) and Syed Zaid Khalid (25K-2522)

## What this project does

This project collects hourly air quality and weather data for Karachi and uses it to:

1. Forecast PM2.5 for the next 1 to 72 hours.
2. Flag unusual pollution spikes.
3. Check how far the data and the forecasts can be trusted.

## Data source

- **API:** Open-Meteo Air Quality API and Open-Meteo Weather API (free, no key needed).
- **Location:** Karachi (latitude 24.86, longitude 67.01).
- **History:** 365 days by default (set `history_days` in `backfill.py` and `refresh.py`). The air quality service has data from 2022, so longer is possible.
- **Air quality columns:** `pm2_5`, `pm10`, `nitrogen_dioxide`, `ozone`
- **Weather columns:** `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`

**Limitation:** Open-Meteo's air quality values come from an atmospheric model (CAMS), not from ground sensors. Part 4 of the notebook compares them with a real sensor if you provide one.

## Files

| File | What it does |
|---|---|
| `karachi_air_quality_full.ipynb` | The whole project in one notebook (Parts 1 to 4, see below) |
| `backfill.py` | Downloads the history and saves `karachi_aq_history.csv` (run once) |
| `refresh.py` | Downloads fresh data and rebuilds `clean_data.csv` and `spike_flags.csv` for the dashboard |
| `data_download.py` | Shared download code used by `backfill.py` and `refresh.py` |
| `collector.py` | Saves a reading every hour to `live_data.csv` and `forecast_log.csv` (keep it running) |
| `collect_once.py` | Takes one reading (used by `collector.py` and the GitHub Action) |
| `.github/workflows/collect.yml` | Hourly collection on GitHub, so your laptop can be off |
| `app.py` | Interactive dashboard (run with `python -m streamlit run app.py`) |
| `run_all.py` | Rebuilds everything with one command |
| `requirements.txt` | Libraries to install |

The older notebooks (`analysis.ipynb`, `analysis_phase4_5.ipynb`, `analysis_phase6.ipynb`) are the earlier separate versions. Use the full notebook.

## Setup

```
pip install -r requirements.txt
```

## How to run

**Option 1: one command**
```
python run_all.py
python -m streamlit run app.py
```

**Option 2: step by step**
1. Run `python backfill.py` once.
2. Open `karachi_air_quality_full.ipynb` and run all cells from top to bottom (it takes a few minutes).
3. Start the dashboard: `python -m streamlit run app.py`
4. Keep `python collector.py` running (or use the GitHub Action) to collect live data for Part 4, Check B.

To see newer data in the dashboard at any time, run `python refresh.py` and restart the dashboard.

## The notebook

| Part | What it does |
|---|---|
| 1 | Data quality report, cleaning, features, and 5 EDA charts |
| 2 | SARIMA forecasting, residual check, confidence-band check, and spike detection (Isolation Forest, DBSCAN) |
| 3 | Baselines (Linear Regression, Random Forest, Gradient Boosting), comparison with SARIMA, a predicted-vs-actual chart (+24 hours), a significance test, feature importance, and a sensitivity check |
| 4 | Real-world checks: compare with a ground sensor (Check A) and with Open-Meteo's own 24-hour forecast (Check B) |

Set `use_gradient_boosting = False` in Part 3 until Gradient Boosting is taught in class.

## Methods and evaluation

- **Models:** SARIMA (main), naive baseline, Isolation Forest and DBSCAN (spikes), then Linear Regression, Random Forest, and Gradient Boosting as baselines added after they are taught in class.
- **Split:** by time (earlier 80% to train, latest 20% to test), never shuffled. Lag and rolling features use only past values.
- **Starting points:** forecasts are repeated every 3 hours across the test period (`origin_step`), so each horizon has hundreds of errors.
- **Metrics:** RMSE and MAE at +1, +6, +24, +48, and +72 hours.
- **Trust checks:** SARIMA residual plots and Ljung-Box test, 95% band coverage, Diebold-Mariano test between models, a sensitivity check (without weather, shorter training window), and the real-world checks in Part 4.

## Dashboard

- Time cursor (far right = latest data, move left to replay the past) and a choice of forecast horizon.
- Forecast with a 95% band, spike alerts, and a Random Forest forecast for +24 hours.
- Past predictions: what SARIMA and Random Forest predicted 24 hours earlier, drawn over the real line, with the average miss for the window (needs `backtest_24h.csv` from the notebook).
- Health category (US EPA PM2.5 categories, 2024) with colored bands, a banner, and a general health tip.
- The typical forecast error and band coverage for the chosen horizon.
- A model comparison chart and table.

## Notes

- Health categories are made for 24-hour averages, so hourly values are only a guide. This is general guidance, not medical advice.
- Spikes have no true labels, so they are compared with a simple rule and between methods, not scored for accuracy.
- Freeze a copy of your data (for example `clean_data_final.csv`) before the final run, so the numbers in the report stay the same.
- Add exact library versions with `pip freeze > requirements.txt` before you submit, so others can reproduce your results.
