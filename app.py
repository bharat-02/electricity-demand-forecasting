"""Electricity Demand Forecasting — Streamlit app (XGBoost).

Matches the training notebook `ML Project - Electricity Demand Forecasting.ipynb`:
features (in order):
  hour, dayofweek, month, year, dayofyear, weekofyear, quarter,
  is_weekend, Temperature, Humidity,
  Demand_lag_24hr, demand_lag_168hr,
  demand_rolling_mean_24hr, demand_rolling_std_24hr
target: Demand
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# Config / paths
# ---------------------------------------------------------
st.set_page_config(
    page_title="Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "electicity_xgb_prediction_model.pkl"
DATA_PATH = BASE_DIR / "Electricity+Demand+Dataset.csv"

FALLBACK_FEATURES = [
    "hour",
    "dayofweek",
    "month",
    "year",
    "dayofyear",
    "weekofyear",
    "quarter",
    "is_weekend",
    "Temperature",
    "Humidity",
    "Demand_lag_24hr",
    "demand_lag_168hr",
    "demand_rolling_mean_24hr",
    "demand_rolling_std_24hr",
]
TARGET = "Demand"


# ---------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading XGBoost model…")
def load_model():
    model = joblib.load(MODEL_PATH)
    features = list(getattr(model, "feature_names_in_", FALLBACK_FEATURES))
    return model, features


@st.cache_data(show_spinner="Loading dataset…")
def load_raw():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data(show_spinner="Preprocessing dataset…")
def get_processed(_raw_json_hash: str = ""):
    """Reproduce the notebook preprocessing exactly."""
    # _raw_json_hash is only a cache-buster placeholder; actual data comes from load_raw().
    df_raw = load_raw()
    df = df_raw.copy()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp").sort_index()

    # Drop fully-empty rows (4 rows on 2021-02-20 in the notebook)
    df = df.dropna(how="all")

    # Same imputations as notebook
    time_cols = ["hour", "dayofweek", "month", "year", "dayofyear"]
    df[time_cols] = df[time_cols].ffill()
    df[["Temperature", "Humidity"]] = df[["Temperature", "Humidity"]].bfill()
    df["Demand"] = df["Demand"].interpolate(method="time")
    df = df.dropna(subset=["Demand", "Temperature", "Humidity"])
    df[time_cols] = df[time_cols].ffill().astype(int)

    # Calendar features (same insertion order as notebook)
    if "quarter" not in df.columns:
        df.insert(5, "quarter", df.index.quarter)
    if "weekofyear" not in df.columns:
        df.insert(5, "weekofyear", df.index.isocalendar().week.astype(int))
    if "is_weekend" not in df.columns:
        df.insert(7, "is_weekend", df.index.dayofweek.isin([5, 6]).astype(int))

    # Lag / rolling features
    df["Demand_lag_24hr"] = df["Demand"].shift(24)
    df["demand_lag_168hr"] = df["Demand"].shift(168)
    df["demand_rolling_mean_24hr"] = df["Demand"].rolling(window=24).mean()
    df["demand_rolling_std_24hr"] = df["Demand"].rolling(window=24).std()

    df = df.dropna().sort_index()
    return df


def derive_calendar_features(ts: pd.Timestamp) -> dict:
    dow = int(ts.dayofweek)
    return {
        "hour": int(ts.hour),
        "dayofweek": dow,
        "month": int(ts.month),
        "year": int(ts.year),
        "dayofyear": int(ts.dayofyear),
        "weekofyear": int(ts.isocalendar().week),
        "quarter": int(ts.quarter),
        "is_weekend": int(dow >= 5),
    }


# ---------------------------------------------------------
# Load model + data
# ---------------------------------------------------------
try:
    model, FEATURES = load_model()
except Exception as e:
    st.error(f"❌ Could not load model `{MODEL_PATH.name}`: {e}")
    st.stop()

try:
    df_raw = load_raw()
except Exception as e:
    st.error(f"❌ Could not load dataset `{DATA_PATH.name}`: {e}")
    st.stop()

try:
    df = get_processed()
except Exception as e:
    st.error(f"❌ Preprocessing failed: {e}")
    st.stop()

X_full = df[FEATURES]
y_full = df[TARGET]
medians = X_full.median(numeric_only=True)

# ---------------------------------------------------------
# Header / sidebar
# ---------------------------------------------------------
st.title("⚡ Electricity Demand Forecasting")
st.markdown(
    "XGBoost regression model trained on hourly demand, weather and "
    "lag/rolling features (2020–2024). Pick a tab below."
)

with st.sidebar:
    st.title("⚙️ Model info")
    st.success("✅ Model loaded")
    st.write("**Algorithm:** XGBoost Regressor")
    st.write("**Target:** Demand")
    st.write(f"**Rows (processed):** {len(df):,}")
    st.write(f"**Date range:** {df.index.min().date()} → {df.index.max().date()}")
    with st.expander("Features used (14)", expanded=False):
        for f in FEATURES:
            st.write(f"• `{f}`")
    st.caption(f"Model file: `{MODEL_PATH.name}`")

tab_pred, tab_perf, tab_data = st.tabs(
    ["🔮 Predict", "📈 Model performance", "🗂️ Data explorer"]
)

# ---------------------------------------------------------
# TAB 1 — Single + batch prediction
# ---------------------------------------------------------
with tab_pred:
    st.header("🔮 Predict electricity demand")
    st.write(
        "Time features are derived automatically from the date/time you pick. "
        "Lag features need recent demand history — defaults are dataset medians, "
        "or historical values when your timestamp exists in the data."
    )

    c1, c2 = st.columns(2)
    with c1:
        sel_date = st.date_input("📅 Date", value=pd.Timestamp("2024-06-15").date(),
                                 min_value=df.index.min().date(),
                                 max_value=pd.Timestamp("2030-12-31").date())
    with c2:
        sel_time = st.time_input("⏰ Time (hourly)", value=pd.Timestamp("2024-06-15 12:00").time())

    ts = pd.Timestamp(f"{sel_date} {sel_time}").floor("h")
    cal = derive_calendar_features(ts)

    c3, c4 = st.columns(2)
    with c3:
        temperature = st.slider("🌡️ Temperature", min_value=3.0, max_value=50.0,
                                value=25.0, step=0.1)
    with c4:
        humidity = st.slider("💧 Humidity (%)", min_value=20.0, max_value=95.0,
                             value=60.0, step=0.1)

    # Historical lookup: index is date-only with duplicates, so match date + hour col
    hist_row = None
    try:
        day_mask = df.index.date == ts.date()
        hour_match = df[day_mask]
        hour_match = hour_match[hour_match["hour"] == cal["hour"]]
        if len(hour_match):
            hist_row = hour_match.iloc[0]
    except Exception:
        hist_row = None

    def _default(col):
        if hist_row is not None and col in hist_row.index and pd.notna(hist_row[col]):
            return float(hist_row[col])
        return float(medians[col])

    with st.expander("🕰️ Historical context (lag / rolling features)", expanded=True):
        if hist_row is not None:
            st.info(f"Historical record found for {ts:%d %b %Y %H:%M} — values pre-filled from history.")
        else:
            st.caption("No exact history for this timestamp — defaults are dataset medians. Adjust if you know recent demand.")
        l1, l2 = st.columns(2)
        with l1:
            lag_24 = st.number_input("Demand same hour yesterday (lag 24h)",
                                     min_value=0.0, value=_default("Demand_lag_24hr"), step=10.0)
            roll_mean = st.number_input("Rolling mean demand (last 24h)",
                                        min_value=0.0, value=_default("demand_rolling_mean_24hr"), step=10.0)
        with l2:
            lag_168 = st.number_input("Demand same hour last week (lag 168h)",
                                      min_value=0.0, value=_default("demand_lag_168hr"), step=10.0)
            roll_std = st.number_input("Rolling std demand (last 24h)",
                                       min_value=0.0, value=_default("demand_rolling_std_24hr"), step=1.0)

    row = {
        **cal,
        "Temperature": float(temperature),
        "Humidity": float(humidity),
        "Demand_lag_24hr": float(lag_24),
        "demand_lag_168hr": float(lag_168),
        "demand_rolling_mean_24hr": float(roll_mean),
        "demand_rolling_std_24hr": float(roll_std),
    }
    input_df = pd.DataFrame([row])[FEATURES]  # enforce model order

    st.subheader("📋 Model input")
    st.dataframe(input_df, use_container_width=True, hide_index=True)

    if st.button("🚀 Predict demand", type="primary", use_container_width=True):
        try:
            pred = float(model.predict(input_df)[0])
            st.success("Prediction completed.")
            st.metric("⚡ Predicted demand", f"{pred:,.2f}")
            st.info(
                f"For **{ts:%d %B %Y at %H:%M}** — 🌡️ {temperature:.1f}, "
                f"💧 {humidity:.1f}%, ⚡ **{pred:,.2f}**"
            )
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

    st.divider()
    st.subheader("📦 Batch prediction (CSV)")
    st.caption("Upload a CSV containing the 14 feature columns (same names as above) to score many rows at once.")
    template = pd.DataFrame(columns=FEATURES)
    st.download_button("⬇️ Download feature template", template.to_csv(index=False),
                       file_name="feature_template.csv", mime="text/csv")
    up = st.file_uploader("Upload features CSV", type=["csv"])
    if up is not None:
        try:
            bdf = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in bdf.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                bdf["predicted_demand"] = model.predict(bdf[FEATURES])
                st.dataframe(bdf.head(100), use_container_width=True)
                st.download_button("⬇️ Download predictions", bdf.to_csv(index=False),
                                   file_name="demand_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not score file: {e}")

# ---------------------------------------------------------
# TAB 2 — Performance
# ---------------------------------------------------------
with tab_perf:
    st.header("📈 Actual vs predicted")
    split = "2024-01-01"
    X_test = X_full.loc[split:]
    y_test = y_full.loc[split:]

    if len(X_test) == 0:
        st.warning("No test rows at/after 2024-01-01.")
    else:
        with st.spinner("Scoring test set…"):
            y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{mae:,.2f}")
        m2.metric("RMSE", f"{rmse:,.2f}")
        m3.metric("R²", f"{r2:.4f}")
        m4.metric("Test rows", f"{len(X_test):,}")

        n = st.slider("Points to plot (most recent)", 200, min(len(X_test), 5000),
                      value=min(len(X_test), 1000), step=100)
        yt, yp = y_test.iloc[-n:], y_pred[-n:]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(yt.index, yt.values, label="Actual", linewidth=1)
        ax.plot(yt.index, yp, label="Predicted", linewidth=1, linestyle="--")
        ax.set_title("XGBoost — Actual vs Predicted Demand (test period)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Demand")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("📋 Actual vs predicted (sample)")
        comp = pd.DataFrame({"timestamp": yt.index, "actual": yt.values,
                             "predicted": yp})
        comp["error"] = comp["actual"] - comp["predicted"]
        st.dataframe(comp.head(100), use_container_width=True, hide_index=True)

        if hasattr(model, "feature_importances_"):
            st.subheader("📌 Feature importance")
            imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.barh(imp.index, imp.values)
            ax2.set_title("XGBoost feature importance")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

# ---------------------------------------------------------
# TAB 3 — Data explorer
# ---------------------------------------------------------
with tab_data:
    st.header("🗂️ Dataset overview")
    st.write(f"Raw rows: **{len(df_raw):,}** · Processed rows: **{len(df):,}**")
    st.dataframe(df_raw.head(50), use_container_width=True)

    st.subheader("Summary statistics")
    st.dataframe(df[["Temperature", "Humidity", "Demand"]].describe(), use_container_width=True)

    st.subheader("Daily mean demand")
    daily = df["Demand"].resample("D").mean()
    st.line_chart(daily)

    st.subheader("Mean demand by hour")
    st.bar_chart(df.groupby("hour")["Demand"].mean())

    st.subheader("Mean demand by month")
    st.bar_chart(df.groupby("month")["Demand"].mean())

st.divider()
st.caption("⚡ Electricity Demand Forecasting · XGBoost · Streamlit")
