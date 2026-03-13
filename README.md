# Air Pollution Forecasting for AIMS Submission

This project turns the provided air-quality dataset into a complete machine learning submission: data preparation, time-aware model training, evaluation artifacts, and a deployment-ready web app.

## Project Objective

The goal is to forecast **next-hour PM2.5 concentration** at Seoul monitoring stations using:

- current pollutant measurements: SO2, NO2, O3, CO, PM10, PM2.5
- station metadata: station code, latitude, longitude
- calendar signals: hour, month, day of week, weekend indicator
- time-series context: recent PM2.5 and PM10 lags, rolling means, and short-term trends

This is a stronger framing for academic submission than predicting PM2.5 at the same time step, because it answers an operational forecasting question and avoids trivial target leakage.

## Publication-Grade Evaluation Additions

- Two model families are trained and compared on the same holdout period:
   - HistGradientBoostingRegressor
   - Ridge Regression
- Quarterly walk-forward backtesting is run over 2019 to measure temporal stability.
- Comparison and backtest outputs are exported as both CSV artifacts and plots.

## Dataset Snapshot

- File: `Measurement_summary.csv`
- Rows: 647,511
- Stations: 25
- Time span: 2017-01-01 to 2019-12-31
- Frequency: hourly

## Project Structure

```text
air-pollution/
|-- Measurement_summary.csv
|-- app.py
|-- requirements.txt
|-- scripts/
|   `-- train_model.py
|-- src/
|   `-- air_pollution/
|       |-- __init__.py
|       `-- data.py
|-- artifacts/                # created after training
|-- reports/                  # created after training
`-- visuals/                  # created after training
```

## Modeling Approach

1. Parse the measurement timestamp.
2. Replace sensor sentinel values of `-1` with missing values.
3. Create a target called `pm25_next_hour` by shifting PM2.5 one hour ahead within each station.
4. Add stronger time-series features per station:
   - PM2.5 lags at 1h, 2h, 3h, and 24h
   - PM2.5 rolling mean and volatility over the recent 24 hours
   - PM10 lag and rolling mean signals
   - cyclical hour and month encoding
5. Split by time:
   - training: 2017-01-01 to 2018-12-31
   - testing: 2019-01-01 to 2019-12-31
6. Train a tuned `HistGradientBoostingRegressor` inside a preprocessing pipeline.
7. Compare the model against a persistence baseline where the next hour is assumed to equal the current PM2.5 reading.

## Why This Is Submission-Ready

- It uses a realistic time-based split.
- It includes a baseline, not just one model.
- It saves reusable artifacts for deployment.
- It produces plots and a written report for interpretation.
- It includes a simple dashboard for demonstration.

## Installation

Use the configured Python environment or create a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run:

```bash
python scripts/train_model.py
```

Outputs created after training:

- `artifacts/model.joblib`
- `artifacts/metrics.json`
- `artifacts/model_comparison.csv`
- `artifacts/quarterly_backtest.csv`
- `artifacts/stations.csv`
- `artifacts/sample_input.json`
- `reports/model_report.md`
- `visuals/predicted_vs_actual.png`
- `visuals/feature_importance.png`
- `visuals/model_comparison.png`
- `visuals/quarterly_backtest_rmse.png`

## Deployment

### Local deployment with Streamlit

```bash
streamlit run app.py
```

### Cloud deployment with Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open Streamlit Community Cloud.
3. Select the repository and set the main file to `app.py`.
4. Confirm `requirements.txt` is detected.
5. If you commit `artifacts/` and `visuals/`, the dashboard will come up immediately with the trained model and evaluation plots.
6. Deploy.

## Local Demo Flow

1. Run `python scripts/train_model.py`
2. Run `streamlit run app.py`
3. Open the Validation tab to show model-family comparison and quarterly backtesting
4. Open the Visuals tab to show plots suitable for report screenshots

## Recommended Submission Contents for AIMS

Include these in your final submission package:

- the full project code
- `reports/model_report.md`
- the trained model metrics from `artifacts/metrics.json`
- the two visuals in `visuals/`
- a short presentation or PDF with problem statement, methodology, results, and limitations

## Suggested Final Report Sections

1. Introduction and motivation
2. Dataset description
3. Data cleaning and feature engineering
4. Modeling approach
5. Evaluation strategy
6. Results and interpretation
7. Limitations
8. Deployment demonstration
9. Conclusion and future work