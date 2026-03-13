from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from air_pollution.data import FEATURE_COLUMNS, TARGET_COLUMN, prepare_training_frame, split_train_test


ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
VISUALS_DIR = ROOT / "visuals"
DATASET_PATH = ROOT / "Measurement_summary.csv"


def ensure_directories() -> None:
    for directory in (ARTIFACTS_DIR, REPORTS_DIR, VISUALS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_tree_pipeline() -> Pipeline:
    numeric_features = [column for column in FEATURE_COLUMNS if column != "Station code"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Station code"]),
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
        ]
    )
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=12,
        max_iter=500,
        min_samples_leaf=20,
        l2_regularization=0.05,
        random_state=42,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_linear_pipeline() -> Pipeline:
    numeric_features = [column for column in FEATURE_COLUMNS if column != "Station code"]
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Station code"]),
            ("numeric", numeric_pipeline, numeric_features),
        ]
    )
    model = Ridge(alpha=2.0)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_model_builders() -> dict[str, tuple[str, callable]]:
    return {
        "hist_gradient_boosting": ("HistGradientBoostingRegressor", build_tree_pipeline),
        "ridge_regression": ("Ridge Regression", build_linear_pipeline),
    }


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(mean_squared_error(y_true, y_pred) ** 0.5, 4),
        "r2": round(r2_score(y_true, y_pred), 4),
    }


def build_prediction_frame(
    source_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    baseline_pred: pd.Series,
    period_label: str | None = None,
) -> pd.DataFrame:
    frame = source_df[["Measurement date", "Station code"]].copy()
    frame["actual"] = y_true.to_numpy()
    frame["prediction"] = y_pred.to_numpy()
    frame["baseline_prediction"] = baseline_pred.to_numpy()
    frame["residual"] = frame["actual"] - frame["prediction"]
    frame["month_number"] = frame["Measurement date"].dt.month
    frame["month_label"] = frame["Measurement date"].dt.strftime("%b")
    if period_label is not None:
        frame["period"] = period_label
    return frame


def evaluate_candidate_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[str, str, Pipeline, dict[str, dict[str, float]], dict[str, pd.Series]]:
    comparison_metrics: dict[str, dict[str, float]] = {}
    prediction_store: dict[str, pd.Series] = {}
    fitted_models: dict[str, Pipeline] = {}
    labels: dict[str, str] = {}

    for model_key, (label, builder) in get_model_builders().items():
        pipeline = builder()
        pipeline.fit(x_train, y_train)
        predictions = pd.Series(pipeline.predict(x_test), index=y_test.index)
        comparison_metrics[model_key] = regression_metrics(y_test, predictions)
        prediction_store[model_key] = predictions
        fitted_models[model_key] = pipeline
        labels[model_key] = label

    best_model_key = min(comparison_metrics, key=lambda key: comparison_metrics[key]["rmse"])
    return best_model_key, labels[best_model_key], fitted_models[best_model_key], comparison_metrics, prediction_store


def save_prediction_plot(y_true: pd.Series, y_pred: pd.Series) -> None:
    sample = pd.DataFrame({"Actual PM2.5": y_true, "Predicted PM2.5": y_pred}).sample(
        n=min(3000, len(y_true)), random_state=42
    )
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=sample, x="Actual PM2.5", y="Predicted PM2.5", s=18, alpha=0.55)
    axis_max = max(sample["Actual PM2.5"].max(), sample["Predicted PM2.5"].max())
    plt.plot([0, axis_max], [0, axis_max], linestyle="--", color="black", linewidth=1)
    plt.title("Predicted vs Actual Next-Hour PM2.5")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "predicted_vs_actual.png", dpi=180)
    plt.close()


def save_model_comparison_plot(model_metrics: dict[str, dict[str, float]]) -> None:
    comparison_frame = pd.DataFrame(
        [
            {"model": key, "rmse": values["rmse"], "mae": values["mae"], "r2": values["r2"]}
            for key, values in model_metrics.items()
        ]
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison_frame, x="model", y="rmse", hue="model", palette="crest", legend=False)
    plt.title("Holdout RMSE by Model Family")
    plt.xlabel("")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "model_comparison.png", dpi=180)
    plt.close()


def get_quarter_periods(year: int) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    return [
        (f"{year}-Q1", pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-04-01")),
        (f"{year}-Q2", pd.Timestamp(f"{year}-04-01"), pd.Timestamp(f"{year}-07-01")),
        (f"{year}-Q3", pd.Timestamp(f"{year}-07-01"), pd.Timestamp(f"{year}-10-01")),
        (f"{year}-Q4", pd.Timestamp(f"{year}-10-01"), pd.Timestamp(f"{year + 1}-01-01")),
    ]


def save_importance_plot(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> list[dict[str, float]]:
    sample_x = x_test.sample(n=min(8000, len(x_test)), random_state=42)
    sample_y = y_test.loc[sample_x.index]
    importance = permutation_importance(
        model,
        sample_x,
        sample_y,
        n_repeats=5,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    feature_rows = [
        {"feature": feature, "importance": float(score)}
        for feature, score in sorted(
            zip(FEATURE_COLUMNS, importance.importances_mean, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    top_rows = feature_rows[:10]
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=pd.DataFrame(top_rows),
        x="importance",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
    )
    plt.title("Top Permutation Importances")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "feature_importance.png", dpi=180)
    plt.close()
    return top_rows


def collect_period_predictions(
    prepared_df: pd.DataFrame,
    best_model_key: str,
    periods: list[tuple[str, pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    builder = get_model_builders()[best_model_key][1]
    prediction_frames: list[pd.DataFrame] = []

    for label, start, end in periods:
        train_fold = prepared_df[prepared_df["Measurement date"] < start].copy()
        test_fold = prepared_df[
            (prepared_df["Measurement date"] >= start) & (prepared_df["Measurement date"] < end)
        ].copy()
        if train_fold.empty or test_fold.empty:
            continue

        x_train = train_fold[FEATURE_COLUMNS]
        y_train = train_fold[TARGET_COLUMN]
        x_test = test_fold[FEATURE_COLUMNS]
        y_test = test_fold[TARGET_COLUMN]

        fold_model = builder()
        fold_model.fit(x_train, y_train)
        fold_predictions = pd.Series(fold_model.predict(x_test), index=y_test.index)
        prediction_frames.append(
            build_prediction_frame(test_fold, y_test, fold_predictions, x_test["PM2.5"], label)
        )

    return pd.concat(prediction_frames, ignore_index=True)


def summarize_quarterly_backtest(prediction_frame: pd.DataFrame) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    backtest_rows: list[dict[str, float | str]] = []
    for period, period_df in prediction_frame.groupby("period", sort=False):
        fold_metrics = regression_metrics(period_df["actual"], period_df["prediction"])
        baseline_metrics = regression_metrics(period_df["actual"], period_df["baseline_prediction"])
        backtest_rows.append(
            {
                "period": str(period),
                "rows": int(len(period_df)),
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_mae": baseline_metrics["mae"],
                "model_rmse": fold_metrics["rmse"],
                "model_mae": fold_metrics["mae"],
                "model_r2": fold_metrics["r2"],
            }
        )

    summary = {
        "mean_rmse": round(sum(row["model_rmse"] for row in backtest_rows) / len(backtest_rows), 4),
        "mean_mae": round(sum(row["model_mae"] for row in backtest_rows) / len(backtest_rows), 4),
        "mean_r2": round(sum(row["model_r2"] for row in backtest_rows) / len(backtest_rows), 4),
    }
    return backtest_rows, summary


def save_backtest_plot(backtest_rows: list[dict[str, float | str]]) -> None:
    backtest_frame = pd.DataFrame(backtest_rows)
    melted = backtest_frame.melt(
        id_vars=["period"],
        value_vars=["baseline_rmse", "model_rmse"],
        var_name="series",
        value_name="rmse",
    )
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=melted, x="period", y="rmse", hue="series", marker="o", linewidth=2.4)
    plt.title("Quarterly Walk-Forward Backtest RMSE")
    plt.xlabel("")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "quarterly_backtest_rmse.png", dpi=180)
    plt.close()


def build_monthly_uncertainty_profile(calibration_predictions: pd.DataFrame) -> pd.DataFrame:
    profile = (
        calibration_predictions.groupby(["month_number", "month_label"], sort=True)["residual"]
        .agg(
            residual_q10=lambda values: np.quantile(values, 0.10),
            residual_q90=lambda values: np.quantile(values, 0.90),
            residual_median=lambda values: np.quantile(values, 0.50),
            residual_std="std",
            sample_size="size",
        )
        .reset_index()
        .sort_values("month_number")
        .reset_index(drop=True)
    )
    profile[["residual_q10", "residual_q90", "residual_median", "residual_std"]] = profile[[
        "residual_q10",
        "residual_q90",
        "residual_median",
        "residual_std",
    ]].round(4)
    profile["interval_width"] = (profile["residual_q90"] - profile["residual_q10"]).round(4)
    return profile


def apply_uncertainty_profile(
    prediction_frame: pd.DataFrame, uncertainty_profile: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    profiled = prediction_frame.merge(
        uncertainty_profile[["month_number", "residual_q10", "residual_q90", "interval_width"]],
        on="month_number",
        how="left",
    )
    profiled["lower_bound"] = profiled["prediction"] + profiled["residual_q10"]
    profiled["upper_bound"] = profiled["prediction"] + profiled["residual_q90"]
    profiled["interval_width_value"] = profiled["upper_bound"] - profiled["lower_bound"]
    profiled["within_interval"] = (
        (profiled["actual"] >= profiled["lower_bound"]) & (profiled["actual"] <= profiled["upper_bound"])
    ).astype(int)
    summary = {
        "interval_level": 0.8,
        "coverage": round(float(profiled["within_interval"].mean()), 4),
        "average_width": round(float(profiled["interval_width_value"].mean()), 4),
    }
    return profiled, summary


def build_monthly_residual_analysis(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    analysis = (
        prediction_frame.groupby(["month_number", "month_label"], sort=True)
        .agg(
            rows=("residual", "size"),
            bias=("residual", "mean"),
            mae=("residual", lambda series: series.abs().mean()),
            rmse=("residual", lambda series: float(np.sqrt(np.mean(np.square(series))))),
            residual_std=("residual", "std"),
            interval_coverage=("within_interval", "mean"),
            average_interval_width=("interval_width_value", "mean"),
        )
        .reset_index()
        .sort_values("month_number")
        .reset_index(drop=True)
    )
    numeric_columns = [
        "bias",
        "mae",
        "rmse",
        "residual_std",
        "interval_coverage",
        "average_interval_width",
    ]
    analysis[numeric_columns] = analysis[numeric_columns].round(4)
    return analysis


def save_monthly_residual_plot(monthly_analysis: pd.DataFrame) -> None:
    fig, axis_left = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=monthly_analysis, x="month_label", y="bias", color="#2a9d8f", ax=axis_left)
    axis_left.axhline(0, color="#264653", linewidth=1)
    axis_left.set_ylabel("Mean residual (Actual - Predicted)")
    axis_left.set_xlabel("")
    axis_left.set_title("Monthly Residual Bias and MAE")

    axis_right = axis_left.twinx()
    sns.lineplot(
        data=monthly_analysis,
        x="month_label",
        y="mae",
        marker="o",
        linewidth=2.2,
        color="#e76f51",
        ax=axis_right,
    )
    axis_right.set_ylabel("Monthly MAE")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "monthly_residual_analysis.png", dpi=180)
    plt.close(fig)


def save_uncertainty_coverage_plot(monthly_analysis: pd.DataFrame) -> None:
    fig, axis_left = plt.subplots(figsize=(10, 5.5))
    sns.lineplot(
        data=monthly_analysis,
        x="month_label",
        y="interval_coverage",
        marker="o",
        linewidth=2.4,
        color="#0b5d7a",
        ax=axis_left,
    )
    axis_left.axhline(0.8, color="#6c757d", linestyle="--", linewidth=1)
    axis_left.set_ylim(0, 1)
    axis_left.set_ylabel("Empirical coverage")
    axis_left.set_xlabel("")
    axis_left.set_title("Monthly Uncertainty Coverage and Interval Width")

    axis_right = axis_left.twinx()
    sns.barplot(data=monthly_analysis, x="month_label", y="average_interval_width", color="#84a59d", ax=axis_right)
    axis_right.set_ylabel("Average interval width")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "monthly_uncertainty_coverage.png", dpi=180)
    plt.close(fig)


def save_uncertainty_band_example_plot(prediction_frame: pd.DataFrame) -> None:
    station_code = prediction_frame["Station code"].mode().iat[0]
    sample = (
        prediction_frame[prediction_frame["Station code"] == station_code]
        .sort_values("Measurement date")
        .head(168)
        .copy()
    )
    plt.figure(figsize=(11, 5.5))
    plt.plot(sample["Measurement date"], sample["actual"], label="Actual", color="#133b5c", linewidth=1.8)
    plt.plot(sample["Measurement date"], sample["prediction"], label="Prediction", color="#f4a261", linewidth=1.8)
    plt.fill_between(
        sample["Measurement date"],
        sample["lower_bound"],
        sample["upper_bound"],
        color="#2a9d8f",
        alpha=0.18,
        label="80% uncertainty band",
    )
    plt.title(f"Prediction Interval Example for Station {station_code}")
    plt.xlabel("")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "uncertainty_band_example.png", dpi=180)
    plt.close()


def build_report(
    dataset_summary: dict[str, object],
    baseline_metrics: dict[str, float],
    best_model_key: str,
    best_model_label: str,
    model_metrics: dict[str, dict[str, float]],
    backtest_rows: list[dict[str, float | str]],
    backtest_summary: dict[str, float],
    uncertainty_summary: dict[str, float],
    monthly_analysis: pd.DataFrame,
    top_features: list[dict[str, float]],
) -> str:
    champion_metrics = model_metrics[best_model_key]
    rmse_gain = round(((baseline_metrics["rmse"] - champion_metrics["rmse"]) / baseline_metrics["rmse"]) * 100, 2)
    top_feature_lines = "\n".join(
        f"- {row['feature']}: {row['importance']:.4f}" for row in top_features
    )
    comparison_lines = "\n".join(
        f"- {key}: MAE {values['mae']}, RMSE {values['rmse']}, R2 {values['r2']}"
        for key, values in model_metrics.items()
    )
    backtest_lines = "\n".join(
        f"- {row['period']}: RMSE {row['model_rmse']}, baseline RMSE {row['baseline_rmse']}, R2 {row['model_r2']}"
        for row in backtest_rows
    )
    monthly_lines = "\n".join(
        f"- {row['month_label']}: bias {row['bias']}, MAE {row['mae']}, RMSE {row['rmse']}, coverage {row['interval_coverage']}, width {row['average_interval_width']}"
        for row in monthly_analysis.to_dict(orient="records")
    )
    return f"""# Air Pollution Forecasting Report

## Project Goal
Forecast next-hour PM2.5 concentration using current pollutant readings, station metadata, calendar signals, and recent station history.

## Dataset Summary
- Rows used for modeling: {dataset_summary['rows_used']}
- Stations: {dataset_summary['stations']}
- Time span: {dataset_summary['min_date']} to {dataset_summary['max_date']}
- Training period: 2017-01-01 to 2018-12-31
- Test period: 2019-01-01 to 2019-12-31

## Baseline
- Persistence baseline MAE: {baseline_metrics['mae']}
- Persistence baseline RMSE: {baseline_metrics['rmse']}
- Persistence baseline R2: {baseline_metrics['r2']}

## Model Family Comparison
{comparison_lines}

## Selected Final Model
- Champion model: {best_model_label}
- MAE: {champion_metrics['mae']}
- RMSE: {champion_metrics['rmse']}
- R2: {champion_metrics['r2']}
- RMSE improvement vs baseline: {rmse_gain}%

## Time-Series Features Added
- Recent PM2.5 lags: 1 hour, 2 hours, 3 hours, and 24 hours
- PM2.5 rolling signals: 3-hour mean, 24-hour mean, and 24-hour volatility
- PM2.5 trend signals: current minus 1-hour lag, current minus 24-hour lag
- PM10 memory signals: 1-hour lag, 24-hour lag, and 24-hour rolling mean
- Cyclical calendar encoding for hour of day and month of year

## Quarterly Walk-Forward Backtesting
{backtest_lines}

Average across quarters:
- Mean MAE: {backtest_summary['mean_mae']}
- Mean RMSE: {backtest_summary['mean_rmse']}
- Mean R2: {backtest_summary['mean_r2']}

## Monthly Residual Analysis
{monthly_lines}

## Uncertainty Bands
- Interval level: {int(uncertainty_summary['interval_level'] * 100)}%
- Empirical holdout coverage: {uncertainty_summary['coverage']}
- Average interval width: {uncertainty_summary['average_width']}
- Calibration source: 2018 quarter-by-quarter walk-forward residuals grouped by calendar month

## Most Influential Features
{top_feature_lines}

## Interpretation
The evaluation is now stronger in three ways. First, a second model family is trained on the same feature set to test whether the gains come from the model design or only from the features. Second, quarterly walk-forward backtesting measures how stable the selected model is through time instead of relying on one aggregate holdout score. Third, monthly residual diagnostics and empirical uncertainty bands reveal when the model is biased, how error magnitude shifts through the year, and how reliable the prediction interval is.

## Artifacts
- Model: artifacts/model.joblib
- Metrics: artifacts/metrics.json
- Model comparison: artifacts/model_comparison.csv
- Quarterly backtest: artifacts/quarterly_backtest.csv
- Monthly residual analysis: artifacts/monthly_residual_analysis.csv
- Monthly uncertainty profile: artifacts/monthly_uncertainty_profile.csv
- Station metadata: artifacts/stations.csv
- Prediction plot: visuals/predicted_vs_actual.png
- Feature importance plot: visuals/feature_importance.png
- Model comparison plot: visuals/model_comparison.png
- Quarterly backtest plot: visuals/quarterly_backtest_rmse.png
- Monthly residual plot: visuals/monthly_residual_analysis.png
- Monthly uncertainty plot: visuals/monthly_uncertainty_coverage.png
- Uncertainty band example: visuals/uncertainty_band_example.png
"""


def main() -> None:
    ensure_directories()
    prepared = prepare_training_frame(DATASET_PATH)
    train_df, test_df = split_train_test(prepared.frame)

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    baseline_predictions = x_test["PM2.5"]
    baseline_metrics = regression_metrics(y_test, baseline_predictions)

    best_model_key, best_model_label, model, comparison_metrics, prediction_store = evaluate_candidate_models(
        x_train, y_train, x_test, y_test
    )
    predictions = prediction_store[best_model_key]
    holdout_prediction_frame = build_prediction_frame(test_df, y_test, predictions, baseline_predictions)

    dataset_summary = {
        "rows_used": int(len(prepared.frame)),
        "stations": int(prepared.station_catalog["Station code"].nunique()),
        "min_date": str(prepared.frame["Measurement date"].min()),
        "max_date": str(prepared.frame["Measurement date"].max()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }

    backtest_prediction_frame = collect_period_predictions(prepared.frame, best_model_key, get_quarter_periods(2019))
    backtest_rows, backtest_summary = summarize_quarterly_backtest(backtest_prediction_frame)
    calibration_prediction_frame = collect_period_predictions(prepared.frame, best_model_key, get_quarter_periods(2018))
    uncertainty_profile = build_monthly_uncertainty_profile(calibration_prediction_frame)
    holdout_prediction_frame, uncertainty_summary = apply_uncertainty_profile(
        holdout_prediction_frame, uncertainty_profile
    )
    monthly_analysis = build_monthly_residual_analysis(holdout_prediction_frame)

    top_features = save_importance_plot(model, x_test, y_test)
    save_prediction_plot(y_test, predictions)
    save_model_comparison_plot(comparison_metrics)
    save_backtest_plot(backtest_rows)
    save_monthly_residual_plot(monthly_analysis)
    save_uncertainty_coverage_plot(monthly_analysis)
    save_uncertainty_band_example_plot(holdout_prediction_frame)

    metrics_payload = {
        "dataset": dataset_summary,
        "baseline": baseline_metrics,
        "selected_model": {"key": best_model_key, "label": best_model_label},
        "model": comparison_metrics[best_model_key],
        "models": comparison_metrics,
        "backtesting": {
            "granularity": "quarter",
            "quarters": backtest_rows,
            "summary": backtest_summary,
        },
        "uncertainty": {
            **uncertainty_summary,
            "month_profiles": uncertainty_profile.to_dict(orient="records"),
            "monthly_analysis": monthly_analysis.to_dict(orient="records"),
        },
    }

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    prepared.station_catalog.to_csv(ARTIFACTS_DIR / "stations.csv", index=False)
    with (ARTIFACTS_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    pd.DataFrame(
        [
            {"model": key, **values}
            for key, values in comparison_metrics.items()
        ]
    ).to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(backtest_rows).to_csv(ARTIFACTS_DIR / "quarterly_backtest.csv", index=False)
    monthly_analysis.to_csv(ARTIFACTS_DIR / "monthly_residual_analysis.csv", index=False)
    uncertainty_profile.to_csv(ARTIFACTS_DIR / "monthly_uncertainty_profile.csv", index=False)
    with (ARTIFACTS_DIR / "sample_input.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "Station code": prepared.station_catalog.iloc[0]["Station code"],
                "Measurement date": "2019-07-17 14:00:00",
                "Latitude": float(prepared.station_catalog.iloc[0]["Latitude"]),
                "Longitude": float(prepared.station_catalog.iloc[0]["Longitude"]),
                "SO2": 0.004,
                "NO2": 0.03,
                "O3": 0.02,
                "CO": 0.5,
                "PM10": 35.0,
                "PM2.5": 18.0,
                "pm25_lag_1": 20.0,
                "pm25_lag_2": 21.0,
                "pm25_lag_3": 19.0,
                "pm25_lag_24": 17.0,
                "pm25_roll_mean_24": 18.6,
                "pm25_roll_std_24": 4.8,
                "pm10_lag_1": 38.0,
                "pm10_lag_24": 33.0,
                "pm10_roll_mean_24": 34.7,
            },
            handle,
            indent=2,
        )

    report_text = build_report(
        dataset_summary,
        baseline_metrics,
        best_model_key,
        best_model_label,
        comparison_metrics,
        backtest_rows,
        backtest_summary,
        uncertainty_summary,
        monthly_analysis,
        top_features,
    )
    with (REPORTS_DIR / "model_report.md").open("w", encoding="utf-8") as handle:
        handle.write(report_text)

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()