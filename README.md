# Electricity Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37%2B-red)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)

Hourly electricity demand forecasting with XGBoost, served through an interactive Streamlit application.

**Live demo:** https://electricity-demand-forecasting-gnaxcfqfpan6fvvrcwnafc.streamlit.app/

## Table of contents

- [Overview](#overview)
- [Key features](#key-features)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model and results](#model-and-results)
- [Application walkthrough](#application-walkthrough)
- [Getting started](#getting-started)
- [Deployment](#deployment)
- [Limitations and future work](#limitations-and-future-work)

## Overview

This project forecasts hourly electricity demand from calendar attributes, weather conditions, and recent demand history. It includes:

1. An exploratory and training notebook (`ML+Project+-+Electricity+Demand+Forecasting.ipynb`)
2. A serialized XGBoost regressor (`electicity_xgb_prediction_model.pkl`)
3. A production Streamlit app (`app.py`) for single predictions, batch scoring, performance analysis, and data exploration

## Key features

- Single-hour demand prediction from date, time, temperature, and humidity
- Automatic derivation of calendar features (hour, day of week, month, year, day of year, ISO week, quarter, weekend flag)
- History-aware lag defaults: historical values when the timestamp exists in the data, otherwise editable dataset medians
- Batch scoring: upload a CSV with the 14 model features, download predictions
- Performance dashboard: MAE, RMSE, R², actual-vs-predicted plot, error table, and XGBoost feature importance
- Data explorer: raw preview, descriptive statistics, and daily, hourly, and monthly demand charts
- Cached model and dataset loading for fast Streamlit reruns

## Project structure

```text
.
├── app.py                                          # Streamlit application
├── requirements.txt                                # Deployment dependencies
├── electicity_xgb_prediction_model.pkl             # Trained XGBRegressor
├── Electricity+Demand+Dataset.csv                  # Raw hourly data
└── ML+Project+-+Electricity+Demand+Forecasting.ipynb  # Training and EDA
```

## Dataset

Source file: `Electricity+Demand+Dataset.csv`

| Column | Description |
| --- | --- |
| Timestamp | Calendar date (date resolution; hour is stored separately) |
| hour | Hour of day, 0–23 |
| dayofweek | Monday = 0 through Sunday = 6 |
| month | 1–12 |
| year | 2020–2024 |
| dayofyear | 1–366 |
| Temperature | Ambient temperature |
| Humidity | Relative humidity (%) |
| Demand | Target variable: hourly electricity demand |

Coverage is approximately 43,848 hourly records from 2020-01-01 to 2024-12-31, with scattered missing values handled during preprocessing.

## Methodology

The app reproduces the notebook pipeline exactly:

1. Parse `Timestamp` into a sorted `DatetimeIndex`; drop fully-empty rows
2. Forward-fill calendar columns; backward-fill temperature and humidity; time-interpolate `Demand`
3. Engineer calendar features: `quarter`, ISO `weekofyear`, and `is_weekend`
4. Engineer history features:
   - `Demand_lag_24hr`: demand at the same hour on the previous day
   - `demand_lag_168hr`: demand at the same hour one week earlier
   - `demand_rolling_mean_24hr` and `demand_rolling_std_24hr`: 24-hour rolling statistics
5. Drop incomplete warm-up rows, yielding approximately 43,676 usable rows
6. Chronological split: train through 2023-12-31, test from 2024-01-01

Final model input is 14 features in the order stored in `model.feature_names_in_`:

```text
hour, dayofweek, month, year, dayofyear, weekofyear, quarter,
is_weekend, Temperature, Humidity,
Demand_lag_24hr, demand_lag_168hr,
demand_rolling_mean_24hr, demand_rolling_std_24hr
```

## Model and results

- Algorithm: `XGBRegressor(n_estimators=1000, learning_rate=0.01, early_stopping_rounds=50, random_state=42)`
- Evaluation on the post-2024-01-01 holdout, reproduced locally:

| Metric | Value |
| --- | --- |
| MAE | 123.38 |
| RMSE | 174.56 |
| R² | 0.9847 |

## Application walkthrough

- **Predict:** select date and time, set temperature and humidity, review or edit the four lag and rolling inputs, run the prediction, and optionally score a batch CSV.
- **Model performance:** inspect holdout metrics, the actual-versus-predicted chart, a sampled comparison table, and feature importance.
- **Data explorer:** review raw rows, summary statistics, and aggregated demand patterns.

## Getting started

Prerequisites: Python 3.11+ is recommended.

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run locally from the project directory:

```powershell
streamlit run app.py
```

Streamlit prints two addresses for the same app:

```text
Local URL:   http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Use the Local URL on the machine running the app; use the Network URL from another device on the same local network.

For batch scoring, use the Predict tab to download `feature_template.csv`, populate all 14 feature columns, upload the file, and export `demand_predictions.csv`.

## Deployment

The repository is deployment-ready for Streamlit Community Cloud:

- Entrypoint: `app.py`
- Dependencies: `requirements.txt` containing `streamlit`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, and `joblib`
- After pushing changes to the `main` branch, reboot the Cloud app if it does not redeploy automatically

A missing `requirements.txt` previously caused `ModuleNotFoundError: No module named 'joblib'` on Cloud because only Streamlit was installed. That file is now included.

## Limitations and future work

- Future-hour forecasts require assumed lag and rolling values; the app currently uses medians unless historical context or user-provided history is available.
- The serialized model may emit an XGBoost version warning when loaded by a newer XGBoost release; predictions remain valid.
- Possible extensions: recursive multi-hour forecasting, weather-forecast integration, holiday features, experiment tracking, and automated retraining.
