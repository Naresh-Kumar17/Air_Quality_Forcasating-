# Real-Time Air Quality Forecasting and Spike Detection for Karachi

Course project for Introduction to Data Science.

**Team:** Naresh Kumar (25K-2501) and Syed Zaid Khalid (25K-2522)

Forecasts PM2.5 in Karachi for 1 to 72 hours ahead, flags unusual pollution spikes, and shows everything in an interactive dashboard. The full write-up is in [`docs/Karachi_Air_Quality_Report.docx`](docs/Karachi_Air_Quality_Report.docx).

## Project structure

```
Air_Quality_Forecasting/
├── README.md
├── requirements.txt
├── src/                      scripts that download and prepare data
│   ├── paths.py                  where every file lives
│   ├── data_download.py          shared Open-Meteo download code
│   ├── backfill.py               download the history (run once)
│   ├── refresh.py                download fresh data and rebuild the dashboard files
│   ├── collector.py              save a reading every hour (keep it running)
│   ├── collect_once.py           save one reading (used by collector.py and the GitHub Action)
│   ├── weather_forecast_test.py  quick test of tomorrow's weather forecast as an input
│   └── run_all.py                rebuild everything with one command
├── notebooks/
│   ├── karachi_air_quality_full.ipynb   the whole analysis (Parts 1 to 5)
│   └── executed/                        a finished run with all outputs
├── app/
│   └── app.py                the Streamlit dashboard
├── data/
│   ├── raw/                  downloaded data (history, live readings, forecast log)
│   └── processed/            cleaned data and spike flags
├── results/                  score tables made by the notebook
├── models/                   saved models (not uploaded to GitHub)
├── docs/                     the project report
├── archive/                  older notebooks, kept for reference
└── .github/workflows/        hourly data collection on GitHub
```

## Data

- **Source:** Open-Meteo Air Quality API and Weather API (free, no key). The air quality values are model data (CAMS), not ground sensors.
- **Place and period:** Karachi (24.86, 67.01), hourly, 1500 days by default (set `history_days` in `src/backfill.py` and `src/refresh.py`).
- **Columns:** `pm2_5`, `pm10`, `nitrogen_dioxide`, `ozone`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, plus day-ahead weather forecasts.

## Quick start

Run everything from the project folder.

```
pip install -r requirements.txt
python src/backfill.py
python src/run_all.py
python -m streamlit run app/app.py
```

`run_all.py` downloads the data and runs the whole notebook (several minutes). If you already ran `backfill.py`, you can run the notebook yourself instead:
`python -m jupyter notebook`, open `notebooks/karachi_air_quality_full.ipynb`, and choose Run > Run All Cells.

| I want to... | Command |
|---|---|
| Update the dashboard with newer data | `python src/refresh.py`, then restart the dashboard |
| Collect live data every hour | `python src/collector.py` |
| Test tomorrow's weather forecast as an input | `python src/weather_forecast_test.py` |

## Notebook parts

| Part | What it does |
|---|---|
| 1 | Data quality report, cleaning, features, and exploration (EDA) |
| 2 | SARIMA forecasting, residual and band checks, spike detection |
| 3 | Baselines (Linear Regression, Random Forest, Gradient Boosting), comparison, significance test, sensitivity check, classification |
| 4 | Real-world checks: ground sensor (optional) and Open-Meteo's own forecast |
| 5 | Ways to improve accuracy: categories, better regression, averaging, learning curve, weather forecast |

Set `use_gradient_boosting = False` in Part 3 until Gradient Boosting is taught in class.

## Methods and evaluation

- **Models:** SARIMA (main), naive baseline, Isolation Forest and DBSCAN (spikes), then Linear Regression, Random Forest, and Gradient Boosting as comparison models.
- **Split:** by time (earlier 80% to train, latest 20% to test), never shuffled. Lag and rolling features use only past values.
- **Metrics:** RMSE, MAE, and R2 at +1, +6, +24, +48, and +72 hours, a Diebold-Mariano significance test, and a check of the 95% confidence band.

## Dashboard

The dashboard explains itself in plain language. Pick a moment with the slider or the quick buttons (Latest, 1 week back, 1 day back, 1 hour back, 1 hour forward), then read the tabs from left to right:

| Tab | What it shows |
|---|---|
| Right now | How the air is at that moment in a sentence, its health category and advice, the weather, and where PM2.5 is heading |
| Forecast | Choose 6, 24, 48 or 72 hours ahead. Chart with a likely range, a reality check against what really happened, a table, and a CSV download |
| Past predictions | Dots for earlier predictions against the real line, a chart of the misses, and the average miss for the window |
| Spikes | Unusual jumps found by Isolation Forest and a simple rule, with a table of flagged hours |
| Model scores | The most accurate model for each horizon, the error chart, and how honest the confidence band is |
| About | Where the data comes from, how the pipeline works, and a glossary |

Moving the moment to the left replays the past, so you can check each forecast against what really happened.

## Notes

- Health categories are made for 24-hour averages, so hourly values are only a guide. This is general guidance, not medical advice.
- Freeze a copy of your data before the final run so the report numbers stay the same.
- Add exact library versions with `pip freeze > requirements.txt` before you submit.
