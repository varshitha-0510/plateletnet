"""Simple moving-average (SMA) demand forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SMAModel:
    """Fitted SMA configuration: last training window and mean forecast."""

    window: int
    last_window_values: tuple[float, ...]
    forecast_level: float
    blood_group: str


def fit_sma(
    series: pd.Series | np.ndarray,
    window: int = 7,
    blood_group: str = "ALL",
) -> SMAModel:
    """Fit an SMA model as the mean of the last ``window`` observations."""
    values = _as_float_array(series)
    if window < 1:
        raise ValueError("SMA window must be >= 1.")
    if values.size == 0:
        raise ValueError("Cannot fit SMA on an empty series.")

    used = values[-window:]
    level = float(np.mean(used))
    return SMAModel(
        window=window,
        last_window_values=tuple(float(v) for v in used),
        forecast_level=level,
        blood_group=blood_group,
    )


def forecast_sma(model: SMAModel, horizon: int) -> np.ndarray:
    """Repeat the SMA level for ``horizon`` days (classic constant SMA forecast)."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be >= 1.")
    return np.full(horizon, max(0.0, model.forecast_level), dtype=float)


def forecast_sma_from_history(
    history: pd.Series | np.ndarray,
    horizon: int,
    window: int = 7,
    blood_group: str = "ALL",
) -> np.ndarray:
    """Fit SMA on ``history`` and forecast ``horizon`` steps."""
    model = fit_sma(history, window=window, blood_group=blood_group)
    return forecast_sma(model, horizon)


def sma_forecast_frame(
    history: pd.Series | np.ndarray,
    horizon: int,
    window: int = 7,
    blood_group: str = "ALL",
    dates: pd.DatetimeIndex | list[pd.Timestamp] | None = None,
    actual: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Return SMA predictions in a DataFrame the evaluation pipeline can reuse.

    Parameters
    ----------
    history
        Training observations only (no test leakage).
    horizon
        Number of days to forecast.
    window
        Moving-average window size.
    dates
        Optional forecast dates. If omitted, an integer step index is used.
    actual
        Optional hold-out actuals, aligned with ``horizon``.
    """
    preds = forecast_sma_from_history(
        history, horizon=horizon, window=window, blood_group=blood_group
    )
    if dates is None:
        date_index = pd.RangeIndex(start=1, stop=horizon + 1, name="step")
        frame = pd.DataFrame(
            {
                "step": date_index,
                "blood_group": blood_group,
                "model": "SMA",
                "prediction": preds,
            }
        )
    else:
        date_index = pd.DatetimeIndex(dates)
        frame = pd.DataFrame(
            {
                "date": date_index,
                "blood_group": blood_group,
                "model": "SMA",
                "prediction": preds,
            }
        )
    if actual is not None:
        actual_values = np.asarray(actual, dtype=float)
        if actual_values.shape[0] != horizon:
            raise ValueError("actual length must match the forecast horizon.")
        frame["actual"] = actual_values
    return frame


def forecast_sma_by_group(
    daily: pd.DataFrame,
    train_end: pd.Timestamp,
    horizon_dates: pd.DatetimeIndex | list[pd.Timestamp],
    window: int = 7,
    demand_col: str = "platelet_demand_units",
) -> pd.DataFrame:
    """Produce SMA forecasts for every blood group over the same dates."""
    dates = pd.DatetimeIndex(horizon_dates)
    rows: list[dict] = []
    for blood_group, group in daily.groupby("blood_group", sort=False):
        history = group.loc[group["date"] <= train_end, demand_col]
        preds = forecast_sma_from_history(
            history, horizon=len(dates), window=window, blood_group=str(blood_group)
        )
        for date, pred in zip(dates, preds):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "blood_group": blood_group,
                    "model": "SMA",
                    "prediction": float(pred),
                }
            )
    return pd.DataFrame(rows)


def sma_forecast_with_actuals(
    daily: pd.DataFrame,
    train_end: pd.Timestamp,
    horizon_dates: pd.DatetimeIndex | list[pd.Timestamp],
    window: int = 7,
    demand_col: str = "platelet_demand_units",
) -> pd.DataFrame:
    """SMA forecasts merged with hold-out actual demand."""
    preds = forecast_sma_by_group(
        daily,
        train_end=train_end,
        horizon_dates=horizon_dates,
        window=window,
        demand_col=demand_col,
    )
    actual = daily.loc[daily["date"].isin(pd.DatetimeIndex(horizon_dates)), [
        "date",
        "blood_group",
        demand_col,
    ]].rename(columns={demand_col: "actual"})
    return (
        preds.merge(actual, on=["date", "blood_group"], how="left")
        .sort_values(["date", "blood_group"])
        .reset_index(drop=True)
    )


def _as_float_array(series: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    return values
