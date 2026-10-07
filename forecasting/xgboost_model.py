"""XGBoost demand forecasting on synthetic time-series features."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from forecasting.preprocessing import (
    LAG_PERIODS,
    calendar_seasonal_factor,
    chronological_split,
    feature_column_names,
)

logger = logging.getLogger(__name__)


@dataclass
class XGBoostForecastModel:
    """Trained XGBoost regressor plus the encoding used at train time."""

    estimator: XGBRegressor
    feature_columns: list[str]
    blood_group_codes: dict[str, int]


def fit_xgboost(
    train_features: pd.DataFrame,
    target_col: str = "platelet_demand_units",
    random_state: int = 42,
    n_estimators: int = 120,
) -> XGBoostForecastModel:
    """Train an XGBoost regressor on engineered forecast features."""
    feature_cols = feature_column_names()
    missing = [col for col in feature_cols + [target_col] if col not in train_features.columns]
    if missing:
        raise ValueError(f"Training features missing columns: {missing}")

    work = train_features.dropna(subset=feature_cols + [target_col]).copy()
    if work.empty:
        raise ValueError("No complete training rows available for XGBoost.")

    codes = (
        work.drop_duplicates("blood_group")
        .set_index("blood_group")["blood_group_code"]
        .astype(int)
        .to_dict()
    )

    estimator = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=1,
        tree_method="hist",
    )
    estimator.fit(work[feature_cols], work[target_col])
    logger.info("Fitted XGBoost on %s rows, %s features", len(work), len(feature_cols))
    return XGBoostForecastModel(
        estimator=estimator,
        feature_columns=feature_cols,
        blood_group_codes=codes,
    )


def forecast_xgboost(
    model: XGBoostForecastModel,
    history: pd.DataFrame,
    horizon_dates: pd.DatetimeIndex | list[pd.Timestamp],
    demand_col: str = "platelet_demand_units",
) -> pd.DataFrame:
    """Recursive multi-step forecast that does not peek at test actuals."""
    dates = pd.DatetimeIndex(horizon_dates)
    hist = history[["date", "blood_group", demand_col]].copy()
    hist["date"] = pd.to_datetime(hist["date"])
    rows: list[dict] = []

    for blood_group, group in hist.groupby("blood_group", sort=False):
        series = group.sort_values("date").set_index("date")[demand_col].astype(float)
        code = int(model.blood_group_codes.get(str(blood_group), 0))
        for date in dates:
            features = _feature_row(date, code, series)
            pred = float(model.estimator.predict(features)[0])
            pred = max(0.0, pred)
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "blood_group": blood_group,
                    "model": "XGBoost",
                    "prediction": pred,
                }
            )
            series.loc[pd.Timestamp(date)] = pred
            series = series.sort_index()

    return pd.DataFrame(rows)


def forecast_xgboost_with_actuals(
    model: XGBoostForecastModel,
    history: pd.DataFrame,
    test_daily: pd.DataFrame,
    horizon_dates: pd.DatetimeIndex | list[pd.Timestamp],
    demand_col: str = "platelet_demand_units",
) -> pd.DataFrame:
    """Recursive XGBoost forecast merged with hold-out actuals (no test leakage)."""
    preds = forecast_xgboost(
        model,
        history=history,
        horizon_dates=horizon_dates,
        demand_col=demand_col,
    )
    actual = test_daily[["date", "blood_group", demand_col]].rename(
        columns={demand_col: "actual"}
    )
    return (
        preds.merge(actual, on=["date", "blood_group"], how="left")
        .sort_values(["date", "blood_group"])
        .reset_index(drop=True)
    )


def split_features_chronologically(
    features: pd.DataFrame,
    test_days: int,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split of engineered features."""
    return chronological_split(features, test_days=test_days, date_col=date_col)


def _feature_row(date: pd.Timestamp, blood_group_code: int, series: pd.Series) -> pd.DataFrame:
    """Build one XGBoost row from calendar fields and lagged history."""
    date = pd.Timestamp(date)
    seasonal = float(calendar_seasonal_factor(pd.Series([date])).iloc[0])
    lag_values = {}
    for lag in LAG_PERIODS:
        lag_date = date - pd.Timedelta(days=lag)
        lag_values[f"lag_{lag}"] = float(series.loc[lag_date]) if lag_date in series.index else float(series.iloc[-1])

    def _roll(window: int) -> tuple[float, float]:
        past = series[series.index < date].tail(window)
        if past.empty:
            return float(series.iloc[-1]), 0.0
        std = float(past.std(ddof=0)) if len(past) > 1 else 0.0
        return float(past.mean()), std

    roll7_mean, roll7_std = _roll(7)
    roll14_mean, roll14_std = _roll(14)
    row = {
        "blood_group_code": blood_group_code,
        "dayofweek": int(date.dayofweek),
        "month": int(date.month),
        "weekofyear": int(date.isocalendar().week),
        "dayofyear": int(date.dayofyear),
        "is_weekend": int(date.dayofweek >= 5),
        "quarter": int(date.quarter),
        "seasonal_factor": seasonal,
        **lag_values,
        "roll_mean_7": roll7_mean,
        "roll_mean_14": roll14_mean,
        "roll_std_7": roll7_std,
        "roll_std_14": roll14_std,
    }
    return pd.DataFrame([row], columns=feature_column_names())
