from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
STATIONS_PATH = ARTIFACTS_DIR / "stations.csv"
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from air_pollution.data import build_inference_frame


st.set_page_config(page_title="AIMS Air Pollution Forecaster", page_icon="AQ", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_assets() -> tuple[dict[str, object], pd.DataFrame]:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    stations = pd.read_csv(STATIONS_PATH, dtype={"Station code": str})
    return metrics, stations


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(18, 118, 145, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(104, 169, 92, 0.18), transparent 28%),
                linear-gradient(180deg, #f4f8f8 0%, #e9f0f2 100%);
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #0b5d7a 0%, #133b5c 60%, #0f8b8d 100%);
            color: white;
            box-shadow: 0 22px 60px rgba(10, 48, 74, 0.22);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin: 0.55rem 0 0 0;
            max-width: 52rem;
            opacity: 0.92;
        }
        .panel {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(12, 58, 83, 0.08);
            backdrop-filter: blur(8px);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 34px rgba(21, 45, 64, 0.08);
        }
        .small-label {
            text-transform: uppercase;
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            color: #4d6c7c;
            margin-bottom: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def air_quality_band(pm25_value: float) -> str:
    if pm25_value <= 15:
        return "Good"
    if pm25_value <= 35:
        return "Moderate"
    if pm25_value <= 55:
        return "Unhealthy for sensitive groups"
    return "Unhealthy"


if not MODEL_PATH.exists() or not METRICS_PATH.exists() or not STATIONS_PATH.exists():
    st.error("Model artifacts are missing. Run `python scripts/train_model.py` first.")
    st.stop()


model = load_model()
metrics, stations = load_assets()
inject_styles()

station_lookup = stations.set_index("Station code")
station_code = st.sidebar.selectbox("Monitoring station", stations["Station code"].tolist())
station_details = station_lookup.loc[station_code]

st.markdown(
    f"""
    <section class="hero">
        <div class="small-label">AIMS Submission Dashboard</div>
        <h1>Air Pollution Forecasting Studio</h1>
        <p>
            Forecast next-hour PM2.5 with a time-series aware model, inspect validation quality,
            and demonstrate deployment from the same interface.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

selected_label = metrics.get("selected_model", {}).get("label", "Trained model")

metric_columns = st.columns(3)
metric_columns[0].metric("Model MAE", metrics["model"]["mae"])
metric_columns[1].metric("Model RMSE", metrics["model"]["rmse"])
metric_columns[2].metric("Model R2", metrics["model"]["r2"])
st.caption(f"Champion model: {selected_label}")

with st.expander("Station metadata", expanded=True):
    st.write(f"Address: {station_details['Address']}")
    st.write(f"Latitude: {station_details['Latitude']}")
    st.write(f"Longitude: {station_details['Longitude']}")

forecast_tab, validation_tab, visuals_tab, deploy_tab = st.tabs(
    ["Forecast", "Validation", "Visuals", "Deployment"]
)

with forecast_tab:
    left_column, right_column = st.columns(2)
    with left_column:
        input_date = st.date_input("Measurement date")
        input_hour = st.slider("Hour of day", 0, 23, 12)
        so2 = st.number_input("SO2", min_value=0.0, value=0.004, step=0.001, format="%.3f")
        no2 = st.number_input("NO2", min_value=0.0, value=0.030, step=0.001, format="%.3f")
        o3 = st.number_input("O3", min_value=0.0, value=0.020, step=0.001, format="%.3f")
        pm25_lag_1 = st.number_input("PM2.5 one hour ago", min_value=0.0, value=20.0, step=1.0)
        pm25_lag_2 = st.number_input("PM2.5 two hours ago", min_value=0.0, value=21.0, step=1.0)
        pm25_lag_3 = st.number_input("PM2.5 three hours ago", min_value=0.0, value=19.0, step=1.0)
        pm25_lag_24 = st.number_input("PM2.5 24 hours ago", min_value=0.0, value=17.0, step=1.0)

    with right_column:
        co = st.number_input("CO", min_value=0.0, value=0.5, step=0.1, format="%.1f")
        pm10 = st.number_input("Current PM10", min_value=0.0, value=35.0, step=1.0)
        pm25 = st.number_input("Current PM2.5", min_value=0.0, value=18.0, step=1.0)
        pm25_roll_mean_24 = st.number_input("Average PM2.5 over last 24 hours", min_value=0.0, value=18.6, step=0.1)
        pm25_roll_std_24 = st.number_input("PM2.5 volatility over last 24 hours", min_value=0.0, value=4.8, step=0.1)
        pm10_lag_1 = st.number_input("PM10 one hour ago", min_value=0.0, value=38.0, step=1.0)
        pm10_lag_24 = st.number_input("PM10 24 hours ago", min_value=0.0, value=33.0, step=1.0)
        pm10_roll_mean_24 = st.number_input("Average PM10 over last 24 hours", min_value=0.0, value=34.7, step=0.1)

    timestamp = pd.Timestamp(input_date) + pd.to_timedelta(input_hour, unit="h")
    input_frame = build_inference_frame(
        {
            "Measurement date": timestamp,
            "Station code": station_code,
            "Latitude": float(station_details["Latitude"]),
            "Longitude": float(station_details["Longitude"]),
            "SO2": so2,
            "NO2": no2,
            "O3": o3,
            "CO": co,
            "PM10": pm10,
            "PM2.5": pm25,
            "pm25_lag_1": pm25_lag_1,
            "pm25_lag_2": pm25_lag_2,
            "pm25_lag_3": pm25_lag_3,
            "pm25_lag_24": pm25_lag_24,
            "pm25_roll_mean_24": pm25_roll_mean_24,
            "pm25_roll_std_24": pm25_roll_std_24,
            "pm10_lag_1": pm10_lag_1,
            "pm10_lag_24": pm10_lag_24,
            "pm10_roll_mean_24": pm10_roll_mean_24,
        }
    )

    prediction = float(model.predict(input_frame)[0])
    prediction = max(prediction, 0.0)
    band = air_quality_band(prediction)

    result_columns = st.columns(2)
    result_columns[0].metric("Predicted next-hour PM2.5", f"{prediction:.2f}")
    result_columns[1].metric("Air quality band", band)

    st.markdown(
        "<div class='panel'><strong>Why this forecast is stronger</strong><br/>"
        "The deployed model uses current measurements plus recent hourly and daily history from the same station, which makes it much more realistic for operational forecasting.</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Model input sent to the predictor")
    st.dataframe(input_frame, use_container_width=True)

with validation_tab:
    comparison_rows = [
        {"Model": key, "MAE": values["mae"], "RMSE": values["rmse"], "R2": values["r2"]}
        for key, values in metrics.get("models", {}).items()
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    st.markdown("<div class='panel'>Model-family comparison on the 2019 holdout period.</div>", unsafe_allow_html=True)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    backtest_rows = metrics.get("backtesting", {}).get("quarters", [])
    if backtest_rows:
        backtest_df = pd.DataFrame(backtest_rows)
        chart_df = backtest_df.set_index("period")[["baseline_rmse", "model_rmse"]]
        st.line_chart(chart_df)
        st.dataframe(backtest_df, use_container_width=True, hide_index=True)
        summary = metrics.get("backtesting", {}).get("summary", {})
        summary_cols = st.columns(3)
        summary_cols[0].metric("Mean backtest MAE", summary.get("mean_mae", "n/a"))
        summary_cols[1].metric("Mean backtest RMSE", summary.get("mean_rmse", "n/a"))
        summary_cols[2].metric("Mean backtest R2", summary.get("mean_r2", "n/a"))

with visuals_tab:
    visuals_left, visuals_right = st.columns(2)
    with visuals_left:
        st.image(str(ROOT / "visuals" / "predicted_vs_actual.png"), caption="Predicted vs actual next-hour PM2.5")
        st.image(str(ROOT / "visuals" / "model_comparison.png"), caption="Holdout RMSE by model family")
    with visuals_right:
        st.image(str(ROOT / "visuals" / "feature_importance.png"), caption="Top feature importances")
        st.image(str(ROOT / "visuals" / "quarterly_backtest_rmse.png"), caption="Quarterly walk-forward RMSE")

with deploy_tab:
    st.markdown(
        "<div class='panel'><strong>Deployment checklist</strong><br/>"
        "1. Push this project to GitHub.<br/>"
        "2. Open Streamlit Community Cloud and choose <code>app.py</code> as the entrypoint.<br/>"
        "3. Make sure the trained artifacts are committed if you want instant demo deployment.<br/>"
        "4. Alternatively retrain on the deployment machine with <code>python scripts/train_model.py</code>.</div>",
        unsafe_allow_html=True,
    )
    st.code("streamlit run app.py", language="bash")
    st.info(
        "This app is deployment-ready for local use or Streamlit Community Cloud after committing the project to GitHub."
    )