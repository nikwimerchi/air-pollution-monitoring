from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

RAW_NUMERIC_COLUMNS = [
    "Latitude",
    "Longitude",
    "SO2",
    "NO2",
    "O3",
    "CO",
    "PM10",
    "PM2.5",
]

FEATURE_COLUMNS = [
    "Station code",
    "Latitude",
    "Longitude",
    "SO2",
    "NO2",
    "O3",
    "CO",
    "PM10",
    "PM2.5",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "pm25_lag_1",
    "pm25_lag_2",
    "pm25_lag_3",
    "pm25_lag_24",
    "pm25_roll_mean_3",
    "pm25_roll_mean_24",
    "pm25_roll_std_24",
    "pm25_trend_1",
    "pm25_trend_24",
    "pm10_lag_1",
    "pm10_lag_24",
    "pm10_roll_mean_24",
]

TARGET_COLUMN = "pm25_next_hour"


@dataclass(frozen=True)
class PreparedDataset:
    frame: pd.DataFrame
    station_catalog: pd.DataFrame


def load_raw_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Measurement date"])
    df[RAW_NUMERIC_COLUMNS] = df[RAW_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df[RAW_NUMERIC_COLUMNS] = df[RAW_NUMERIC_COLUMNS].replace(-1, np.nan)
    df[RAW_NUMERIC_COLUMNS] = df[RAW_NUMERIC_COLUMNS].astype(float)
    df["Station code"] = df["Station code"].astype(str)
    df = df.sort_values(["Station code", "Measurement date"]).reset_index(drop=True)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["Measurement date"].dt.hour
    df["day_of_week"] = df["Measurement date"].dt.dayofweek
    df["month"] = df["Measurement date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_station_history_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("Station code", sort=False)

    df["pm25_lag_1"] = grouped["PM2.5"].shift(1)
    df["pm25_lag_2"] = grouped["PM2.5"].shift(2)
    df["pm25_lag_3"] = grouped["PM2.5"].shift(3)
    df["pm25_lag_24"] = grouped["PM2.5"].shift(24)
    df["pm25_roll_mean_3"] = grouped["PM2.5"].transform(
        lambda series: series.shift(1).rolling(window=3, min_periods=3).mean()
    )
    df["pm25_roll_mean_24"] = grouped["PM2.5"].transform(
        lambda series: series.shift(1).rolling(window=24, min_periods=24).mean()
    )
    df["pm25_roll_std_24"] = grouped["PM2.5"].transform(
        lambda series: series.shift(1).rolling(window=24, min_periods=24).std()
    )
    df["pm25_trend_1"] = df["PM2.5"] - df["pm25_lag_1"]
    df["pm25_trend_24"] = df["PM2.5"] - df["pm25_lag_24"]

    df["pm10_lag_1"] = grouped["PM10"].shift(1)
    df["pm10_lag_24"] = grouped["PM10"].shift(24)
    df["pm10_roll_mean_24"] = grouped["PM10"].transform(
        lambda series: series.shift(1).rolling(window=24, min_periods=24).mean()
    )
    return df


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_station_history_features(df)
    return df


def prepare_training_frame(csv_path: str | Path) -> PreparedDataset:
    df = load_raw_dataset(csv_path)
    df = enrich_features(df)
    df[TARGET_COLUMN] = df.groupby("Station code", sort=False)["PM2.5"].shift(-1)

    station_catalog = (
        df[["Station code", "Address", "Latitude", "Longitude"]]
        .drop_duplicates(subset=["Station code"])
        .sort_values("Station code")
        .reset_index(drop=True)
    )

    clean_df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    return PreparedDataset(frame=clean_df, station_catalog=station_catalog)


def build_inference_frame(raw_input: dict[str, float | int | str | pd.Timestamp]) -> pd.DataFrame:
    timestamp = pd.Timestamp(raw_input["Measurement date"])
    current_pm25 = float(raw_input["PM2.5"])
    current_pm10 = float(raw_input["PM10"])
    pm25_lag_1 = float(raw_input["pm25_lag_1"])
    pm25_lag_2 = float(raw_input["pm25_lag_2"])
    pm25_lag_3 = float(raw_input["pm25_lag_3"])
    pm25_lag_24 = float(raw_input["pm25_lag_24"])
    pm10_lag_1 = float(raw_input["pm10_lag_1"])
    pm10_lag_24 = float(raw_input["pm10_lag_24"])

    frame = pd.DataFrame(
        [
            {
                "Measurement date": timestamp,
                "Station code": str(raw_input["Station code"]),
                "Latitude": float(raw_input["Latitude"]),
                "Longitude": float(raw_input["Longitude"]),
                "SO2": float(raw_input["SO2"]),
                "NO2": float(raw_input["NO2"]),
                "O3": float(raw_input["O3"]),
                "CO": float(raw_input["CO"]),
                "PM10": current_pm10,
                "PM2.5": current_pm25,
                "hour": timestamp.hour,
                "day_of_week": timestamp.dayofweek,
                "month": timestamp.month,
                "is_weekend": int(timestamp.dayofweek >= 5),
                "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
                "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),
                "month_sin": np.sin(2 * np.pi * timestamp.month / 12),
                "month_cos": np.cos(2 * np.pi * timestamp.month / 12),
                "pm25_lag_1": pm25_lag_1,
                "pm25_lag_2": pm25_lag_2,
                "pm25_lag_3": pm25_lag_3,
                "pm25_lag_24": pm25_lag_24,
                "pm25_roll_mean_3": (current_pm25 + pm25_lag_1 + pm25_lag_2) / 3,
                "pm25_roll_mean_24": float(raw_input["pm25_roll_mean_24"]),
                "pm25_roll_std_24": float(raw_input["pm25_roll_std_24"]),
                "pm25_trend_1": current_pm25 - pm25_lag_1,
                "pm25_trend_24": current_pm25 - pm25_lag_24,
                "pm10_lag_1": pm10_lag_1,
                "pm10_lag_24": pm10_lag_24,
                "pm10_roll_mean_24": float(raw_input["pm10_roll_mean_24"]),
            }
        ]
    )
    return frame[FEATURE_COLUMNS]


def split_train_test(prepared_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = prepared_df[prepared_df["Measurement date"] < "2019-01-01"].copy()
    test_df = prepared_df[prepared_df["Measurement date"] >= "2019-01-01"].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Time-based split failed. Expected both training and test data.")
    return train_df, test_df