"""Aggregate synthetic demand and build time-series forecast features."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

DAILY_DEMAND_FILENAME = "daily_demand.csv"
FORECAST_FEATURES_FILENAME = "forecast_features.csv"
LAG_PERIODS = (1, 7, 14)
ROLL_WINDOWS = (7, 14)


def preprocess_demand(
    df: pd.DataFrame,
    processed_dir: str | Path | None = None,
    by_blood_group: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate daily demand, engineer features, and save processed CSVs.

    Parameters
    ----------
    df
        Cleaned demand frame from :func:`forecasting.data_loader.load_demand_data`.
    processed_dir
        Directory for processed outputs. Defaults to ``data/processed``.
    by_blood_group
        If True, keep one series per blood group. If False, sum all groups
        into a single daily total.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(daily_demand, forecast_features)``.
    """
    daily = aggregate_daily_demand(df, by_blood_group=by_blood_group)
    features = create_forecast_features(daily)
    save_processed_data(daily, features, processed_dir)
    return daily, features


def aggregate_daily_demand(df: pd.DataFrame, by_blood_group: bool = True) -> pd.DataFrame:
    """Collapse possibly duplicate rows into one observation per day (and group)."""
    if df.empty:
        raise ValueError("Cannot aggregate an empty demand DataFrame.")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()

    group_keys: list[str] = ["date"]
    if by_blood_group:
        group_keys.append("blood_group")

    aggregated = (
        work.groupby(group_keys, as_index=False)
        .agg(
            platelet_demand_units=("platelet_demand_units", "sum"),
            seasonal_factor=("seasonal_factor", "mean"),
            demand_condition=("demand_condition", _mode_or_unknown),
        )
    )
    aggregated["day_of_week"] = aggregated["date"].dt.day_name()
    if not by_blood_group:
        aggregated["blood_group"] = "ALL"

    aggregated = fill_calendar_gaps(aggregated)
    aggregated["platelet_demand_units"] = aggregated["platelet_demand_units"].round(4)
    return aggregated.sort_values(["blood_group", "date"]).reset_index(drop=True)


def fill_calendar_gaps(daily: pd.DataFrame) -> pd.DataFrame:
    """Reindex each blood group onto a continuous daily calendar."""
    frames: list[pd.DataFrame] = []
    for blood_group, group in daily.groupby("blood_group", sort=False):
        group = group.sort_values("date")
        full_idx = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        aligned = group.set_index("date").reindex(full_idx)
        aligned.index.name = "date"
        aligned["blood_group"] = blood_group
        aligned["platelet_demand_units"] = aligned["platelet_demand_units"].interpolate(
            limit_direction="both"
        )
        aligned["seasonal_factor"] = aligned["seasonal_factor"].fillna(
            calendar_seasonal_factor(pd.Series(aligned.index, index=aligned.index))
        )
        aligned["day_of_week"] = aligned.index.day_name()
        aligned["demand_condition"] = aligned["demand_condition"].fillna("unknown")
        frames.append(aligned.reset_index())
    return pd.concat(frames, ignore_index=True)


def calendar_seasonal_factor(dates: pd.Series) -> pd.Series:
    """Deterministic seasonal multiplier from calendar day-of-year (demo only)."""
    dayofyear = pd.to_datetime(dates).dt.dayofyear
    return 1.0 + 0.08 * np.sin(2.0 * np.pi * dayofyear / 365.25)


def create_forecast_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, lag, and rolling features used by XGBoost."""
    if daily.empty:
        raise ValueError("Cannot create features from an empty daily demand table.")

    frames: list[pd.DataFrame] = []
    for blood_group, group in daily.groupby("blood_group", sort=False):
        g = group.sort_values("date").copy()
        g["dayofweek"] = g["date"].dt.dayofweek
        g["month"] = g["date"].dt.month
        g["weekofyear"] = g["date"].dt.isocalendar().week.astype(int)
        g["dayofyear"] = g["date"].dt.dayofyear
        g["is_weekend"] = (g["dayofweek"] >= 5).astype(int)
        g["quarter"] = g["date"].dt.quarter
        g["blood_group_code"] = _blood_group_code(blood_group)
        if "seasonal_factor" not in g.columns or g["seasonal_factor"].isna().any():
            g["seasonal_factor"] = calendar_seasonal_factor(g["date"])

        for lag in LAG_PERIODS:
            g[f"lag_{lag}"] = g["platelet_demand_units"].shift(lag)
        for window in ROLL_WINDOWS:
            g[f"roll_mean_{window}"] = (
                g["platelet_demand_units"].shift(1).rolling(window).mean()
            )
            g[f"roll_std_{window}"] = (
                g["platelet_demand_units"].shift(1).rolling(window).std()
            )
        frames.append(g)

    features = pd.concat(frames, ignore_index=True)
    lag_cols = [f"lag_{lag}" for lag in LAG_PERIODS] + [
        f"roll_mean_{w}" for w in ROLL_WINDOWS
    ]
    before = len(features)
    features = features.dropna(subset=lag_cols)
    logger.info("Dropped %s warmup rows without lag history", before - len(features))
    return features.reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    test_days: int,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dated table into train/test using the last ``test_days`` calendar dates.

    Uses only the global date axis so later days never appear in training.
    """
    if test_days < 1:
        raise ValueError("test_days must be >= 1.")
    if date_col not in df.columns:
        raise ValueError(f"DataFrame must include a {date_col} column.")
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    unique_dates = pd.DatetimeIndex(sorted(work[date_col].dt.normalize().unique()))
    if len(unique_dates) <= test_days:
        raise ValueError("Not enough distinct dates for the requested chronological split.")
    test_start = unique_dates[-test_days]
    train = work.loc[work[date_col] < test_start].sort_values(date_col)
    test = work.loc[work[date_col] >= test_start].sort_values(date_col)
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def feature_column_names() -> list[str]:
    """Return the XGBoost input column names."""
    return [
        "blood_group_code",
        "dayofweek",
        "month",
        "weekofyear",
        "dayofyear",
        "is_weekend",
        "quarter",
        "seasonal_factor",
        *[f"lag_{lag}" for lag in LAG_PERIODS],
        *[f"roll_mean_{w}" for w in ROLL_WINDOWS],
        *[f"roll_std_{w}" for w in ROLL_WINDOWS],
    ]


def save_processed_data(
    daily: pd.DataFrame,
    features: pd.DataFrame,
    processed_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write daily demand and feature tables to ``data/processed``."""
    out_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_path = out_dir / DAILY_DEMAND_FILENAME
    features_path = out_dir / FORECAST_FEATURES_FILENAME
    daily.to_csv(daily_path, index=False)
    features.to_csv(features_path, index=False)
    logger.info("Wrote processed demand to %s and %s", daily_path, features_path)
    return daily_path, features_path


def _mode_or_unknown(series: pd.Series) -> str:
    modes = series.dropna().mode()
    if modes.empty:
        return "unknown"
    return str(modes.iloc[0])


def _blood_group_code(blood_group: str) -> int:
    from config import BLOOD_GROUPS

    if blood_group in BLOOD_GROUPS:
        return BLOOD_GROUPS.index(blood_group)
    # Stable encoding for unexpected labels (PYTHONHASHSEED must not affect this).
    return sum(ord(ch) for ch in str(blood_group)) % 1000
