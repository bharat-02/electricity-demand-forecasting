# Electricity Demand Forecasting

**Live demo:** https://electricity-demand-forecasting-gnaxcfqfpan6fvvrcwnafc.streamlit.app/

XGBoost regression model + Streamlit app for hourly electricity demand forecasting (2020–2024).

## Overview

Predicts hourly electricity demand from calendar features, weather, and recent demand history. The Streamlit app supports single predictions, batch CSV scoring, model performance evaluation, and data exploration.

## Project files

- `app.py` — Streamlit application
- `electicity_xgb_prediction_model.pkl` — trained `XGBRegressor` (joblib)
- `Electricity+Demand+Dataset.csv` — raw hourly data
- `ML+Project+-+Electricity+Demand+Forecasting.ipynb` — training / EDA notebook

## Dataset

Raw columns in `Electricity+Demand+Dataset.csv`:

`Timestamp, hour, dayofweek, month, year, dayofyear, Temperature, Humidity, Demand`

- ~43,848 hourly rows, 2020-01-01 to 2024-12-31
- `Timestamp` is date-only; `hour` is a separate column
- Contains scattered missing values (handled in preprocessing)

## Model input features (14, in order)

```text
hour, dayofweek, month, year, dayofyear, weekofyear, quarter,
is_weekend, Temperature, Humidity,
Demand_lag_24hr, demand_lag_168hr,
demand_rolling_mean_24hr, demand_rolling_std_24hr
```

Target: `Demand`

## Preprocessing (same as notebook)

1. Parse `Timestamp`, set as sorted `DatetimeIndex`, drop fully-empty rows
2. `ffill` time columns, `bfill` temperature/humidity, time-interpolate `Demand`
3. Add `quarter`, `weekofyear` (ISO), `is_weekend` (Sat/Sun = 1)
4. Add lags: `shift(24)`, `shift(168)`; rolling 24h mean/std
5. `dropna` → ~43,676 rows
6. Train: up to `2023-12-31`; Test: from `2024-01-01`

## Model

- `XGBRegressor(n_estimators=1000, learning_rate=0.01, early_stopping_rounds=50, random_state=42)`
- Test performance (reproduced locally):
  - MAE: 123.38
  - RMSE: 174.56
  - R²: 0.9847

## App tabs

- Predict: date/time + temperature/humidity sliders, lag/rolling inputs (pre-filled from history when the timestamp exists, otherwise dataset medians), single prediction, batch CSV scoring with downloadable template
- Model performance: test metrics, actual vs predicted plot, sample table, feature importance
- Data explorer: raw head, summary stats, daily/hourly/monthly demand charts

## Installation

Requires Python with:

```text
streamlit
pandas
numpy
matplotlib
scikit-learn
xgboost
joblib
```

Example:

```powershell
pip install streamlit pandas numpy matplotlib scikit-learn xgboost joblib
```

Note: in this workspace the working interpreter is the Anaconda Python at `C:\Users\lenovo\anaconda3\python.exe`. The `C:\Program Files\Python310\python.exe` install is broken.

## Usage

From the project folder:

```powershell
& 'C:\Users\lenovo\anaconda3\python.exe' -m streamlit run app.py
```


Batch prediction: in Predict tab, download `feature_template.csv`, fill the 14 feature columns, upload it, then download `demand_predictions.csv`.

## Notes

- Lag/rolling features need recent demand history. For future timestamps the app uses medians as defaults — override them if you know recent demand.
- Loading the `.pkl` may emit an XGBoost version warning about serialized models. It still loads and predicts correctly; re-saving with `Booster.save_model` removes the warning.
