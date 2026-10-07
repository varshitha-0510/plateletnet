"""SARIMA demand forecasting with fallbacks for short synthetic series."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

# Weekly seasonality first; simpler ARIMA fallbacks if seasonal fit fails.
_CANDIDATE_ORDERS: tuple[tuple[tuple[int, int, int], tuple[int, int, int, int]], ...] = (
    ((1, 1, 1), (1, 1, 1, 7)),
    ((1, 1, 1), (0, 1, 1, 7)),
    ((1, 1, 1), (0, 0, 0, 0)),
    ((1, 0, 1), (0, 0, 0, 0)),
    ((1, 1, 0), (0, 0, 0, 0)),
)


@dataclass
class SARIMAGroupModel:
    """Fitted SARIMAX result or a mean fallback for one blood group."""

    blood_group: str
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    result: object | None
    fallback_level: float


@dataclass
class SARIMAModel:
    """Collection of per-blood-group SARIMA fits."""

    group_models: dict[str, SARIMAGroupModel]


def fit_sarima(
    series: pd.Series | np.ndarray,
    blood_group: str = "ALL",
    order: tuple[int, int, int] | None = None,
    seasonal_order: tuple[int, int, int, int] | None = None,
    min_observations: int = 16,
) -> SARIMAGroupModel:
    """Fit SARIMA on a single daily series, falling back if estimation fails.

    Parameters
    ----------
    order
        Optional ARIMA ``(p, d, q)``. If omitted, candidate orders are tried.
    seasonal_order
        Optional seasonal ``(P, D, Q, s)``. Used with ``order`` when both are set.
    min_observations
        Below this length the seasonal model is skipped and a mean fallback is used.
    """
    values = pd.Series(np.asarray(series, dtype=float)).dropna()
    if values.empty:
        raise ValueError(f"Cannot fit SARIMA on an empty series ({blood_group}).")

    fallback = float(np.clip(values.iloc[-7:].mean() if len(values) else 0.0, 0.0, None))
    if len(values) < min_observations:
        logger.warning(
            "Series for %s has only %s points; using mean fallback",
            blood_group,
            len(values),
        )
        return SARIMAGroupModel(
            blood_group=blood_group,
            order=(0, 0, 0),
            seasonal_order=(0, 0, 0, 0),
            result=None,
            fallback_level=fallback,
        )

    indexed = values.reset_index(drop=True)
    last_error: Exception | None = None
    if order is not None:
        seasonal = seasonal_order if seasonal_order is not None else (0, 0, 0, 0)
        candidates = ((order, seasonal),)
    else:
        candidates = _CANDIDATE_ORDERS
    for order, seasonal_order in candidates:
        if seasonal_order[3] and len(indexed) < 3 * seasonal_order[3]:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    indexed,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False, maxiter=75)
            if result is None or not np.isfinite(getattr(result, "aic", np.nan)):
                continue
            logger.info(
                "SARIMA %s fit %s x %s (AIC=%.2f)",
                blood_group,
                order,
                seasonal_order,
                float(result.aic),
            )
            return SARIMAGroupModel(
                blood_group=blood_group,
                order=order,
                seasonal_order=seasonal_order,
                result=result,
                fallback_level=fallback,
            )
        except (ValueError, np.linalg.LinAlgError, Exception) as exc:  # noqa: BLE001
            last_error = exc
            continue

    logger.warning(
        "SARIMA fit failed for %s (%s); using mean fallback",
        blood_group,
        last_error,
    )
    return SARIMAGroupModel(
        blood_group=blood_group,
        order=(0, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        result=None,
        fallback_level=fallback,
    )


def forecast_sarima(model: SARIMAGroupModel, horizon: int) -> np.ndarray:
    """Forecast ``horizon`` days from a fitted group model."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be >= 1.")
    if model.result is None:
        return np.full(horizon, max(0.0, model.fallback_level), dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = np.asarray(model.result.forecast(steps=horizon), dtype=float)
        pred = np.where(np.isfinite(pred), pred, model.fallback_level)
        return np.clip(pred, 0.0, None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SARIMA forecast failed for %s: %s", model.blood_group, exc)
        return np.full(horizon, max(0.0, model.fallback_level), dtype=float)


def fit_sarima_by_group(
    daily: pd.DataFrame,
    train_end: pd.Timestamp,
    demand_col: str = "platelet_demand_units",
    order: tuple[int, int, int] | None = None,
    seasonal_order: tuple[int, int, int, int] | None = None,
) -> SARIMAModel:
    """Fit one SARIMA model per blood group on training history."""
    group_models: dict[str, SARIMAGroupModel] = {}
    for blood_group, group in daily.groupby("blood_group", sort=False):
        history = group.loc[group["date"] <= train_end].sort_values("date")
        group_models[str(blood_group)] = fit_sarima(
            history[demand_col],
            blood_group=str(blood_group),
            order=order,
            seasonal_order=seasonal_order,
        )
    return SARIMAModel(group_models=group_models)


def forecast_sarima_by_group(
    model: SARIMAModel,
    horizon_dates: pd.DatetimeIndex | list[pd.Timestamp],
) -> pd.DataFrame:
    """Forecast each fitted blood-group SARIMA over ``horizon_dates``."""
    dates = pd.DatetimeIndex(horizon_dates)
    rows: list[dict] = []
    for blood_group, group_model in model.group_models.items():
        preds = forecast_sarima(group_model, horizon=len(dates))
        for date, pred in zip(dates, preds):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "blood_group": blood_group,
                    "model": "SARIMA",
                    "prediction": float(pred),
                }
            )
    return pd.DataFrame(rows)


def sarima_forecast_frame(
    daily: pd.DataFrame,
    train_end: pd.Timestamp,
    horizon_dates: pd.DatetimeIndex | list[pd.Timestamp],
    demand_col: str = "platelet_demand_units",
    order: tuple[int, int, int] | None = None,
    seasonal_order: tuple[int, int, int, int] | None = None,
) -> tuple[pd.DataFrame, SARIMAModel]:
    """Fit SARIMA on training history and return predictions with actuals."""
    model = fit_sarima_by_group(
        daily,
        train_end=train_end,
        demand_col=demand_col,
        order=order,
        seasonal_order=seasonal_order,
    )
    preds = forecast_sarima_by_group(model, horizon_dates)
    actual = daily.loc[daily["date"].isin(pd.DatetimeIndex(horizon_dates)), [
        "date",
        "blood_group",
        demand_col,
    ]].rename(columns={demand_col: "actual"})
    frame = preds.merge(actual, on=["date", "blood_group"], how="left")
    return frame.sort_values(["date", "blood_group"]).reset_index(drop=True), model
