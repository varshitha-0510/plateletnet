"""Train SMA, XGBoost, and SARIMA on synthetic demand and persist results."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import FORECAST_RESULTS_DIR, PROCESSED_DATA_DIR
from forecasting.data_loader import DEFAULT_DEMAND_CSV, load_demand_data
from forecasting.evaluate import comparison_table, select_best_model
from forecasting.model_registry import save_forecast_artifacts
from forecasting.preprocessing import chronological_split, preprocess_demand
from forecasting.sarima import sarima_forecast_frame
from forecasting.sma import SMAModel, fit_sma, sma_forecast_with_actuals
from forecasting.xgboost_model import fit_xgboost, forecast_xgboost_with_actuals

logger = logging.getLogger(__name__)

DEFAULT_TEST_DAYS = 28
DEFAULT_SMA_WINDOW = 7


def train_models(
    csv_path: str | Path | None = None,
    processed_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    test_days: int = DEFAULT_TEST_DAYS,
    sma_window: int = DEFAULT_SMA_WINDOW,
    xgb_estimators: int = 120,
) -> dict:
    """Run the full forecasting pipeline and write artifacts.

    Returns a dictionary with metrics, the selected model name, forecasts,
    and output paths.
    """
    if test_days < 1:
        raise ValueError("test_days must be >= 1.")

    processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_DATA_DIR
    results_dir = Path(results_dir) if results_dir is not None else FORECAST_RESULTS_DIR
    path = Path(csv_path) if csv_path is not None else DEFAULT_DEMAND_CSV

    logger.info("Loading synthetic demand from %s", path)
    raw = load_demand_data(path)
    daily, features = preprocess_demand(raw, processed_dir=processed_dir)

    train_daily, test_daily = chronological_split(daily, test_days=test_days)
    train_end = pd.Timestamp(train_daily["date"].max())
    test_start = pd.Timestamp(test_daily["date"].min())
    last_date = pd.Timestamp(test_daily["date"].max())
    horizon_dates = pd.DatetimeIndex(sorted(test_daily["date"].unique()))
    train_features = features.loc[features["date"] <= train_end].copy()

    logger.info(
        "Train through %s (%s rows); test %s to %s",
        train_end.date(),
        len(train_daily),
        test_start.date(),
        last_date.date(),
    )

    sma_frame = sma_forecast_with_actuals(
        daily, train_end=train_end, horizon_dates=horizon_dates, window=sma_window
    )
    xgb_model = fit_xgboost(train_features, n_estimators=xgb_estimators)
    xgb_frame = forecast_xgboost_with_actuals(
        xgb_model, train_daily, test_daily, horizon_dates
    )
    sarima_frame, sarima_model = sarima_forecast_frame(
        daily, train_end=train_end, horizon_dates=horizon_dates
    )

    forecasts = _wide_forecast_table(test_daily, sma_frame, xgb_frame, sarima_frame)
    metrics = comparison_table(sma_frame, xgb_frame, sarima_frame)
    selected = select_best_model(metrics)
    selection_reason = (
        f"{selected} was selected because it achieved the lowest MAE "
        f"({float(metrics.loc[metrics['model'] == selected, 'MAE'].iloc[0]):.4f}) "
        "on the synthetic hold-out set among SMA, XGBoost, and SARIMA."
    )

    sma_models = _sma_metadata(train_daily, sma_window)
    sarima_metadata = [
        {
            "blood_group": gm.blood_group,
            "order": gm.order,
            "seasonal_order": gm.seasonal_order,
            "used_fallback": gm.result is None,
            "fallback_level": gm.fallback_level,
        }
        for gm in sarima_model.group_models.values()
    ]
    selected_metadata = {
        "selected_model": selected,
        "selection_rule": "lowest_MAE",
        "selection_reason": selection_reason,
        "test_days": test_days,
        "test_start": str(test_start.date()),
        "test_end": str(last_date.date()),
        "sma_window": sma_window,
        "source_csv": str(path),
        "data_note": "synthetic/demo demand only — not a clinical system",
        "metrics": metrics.to_dict(orient="records"),
    }
    paths = save_forecast_artifacts(
        metrics=metrics,
        forecasts=forecasts,
        sma_models=sma_models,
        xgboost_model=xgb_model,
        sarima_metadata=sarima_metadata,
        selected_metadata=selected_metadata,
        results_dir=results_dir,
        sma_forecast=sma_frame,
        xgboost_forecast=xgb_frame,
        sarima_forecast=sarima_frame,
    )
    logger.info("Selected forecasting model: %s", selected)
    return {
        "metrics": metrics,
        "forecasts": forecasts,
        "selected_model": selected,
        "selection_reason": selection_reason,
        "paths": paths,
        "daily": daily,
        "features": features,
    }


def _wide_forecast_table(
    test_daily: pd.DataFrame,
    sma_frame: pd.DataFrame,
    xgb_frame: pd.DataFrame,
    sarima_frame: pd.DataFrame,
) -> pd.DataFrame:
    actual = test_daily.rename(columns={"platelet_demand_units": "actual"})[
        ["date", "blood_group", "actual"]
    ]
    wide = actual.merge(
        sma_frame.rename(columns={"prediction": "SMA"})[["date", "blood_group", "SMA"]],
        on=["date", "blood_group"],
        how="left",
    )
    wide = wide.merge(
        xgb_frame.rename(columns={"prediction": "XGBoost"})[["date", "blood_group", "XGBoost"]],
        on=["date", "blood_group"],
        how="left",
    )
    wide = wide.merge(
        sarima_frame.rename(columns={"prediction": "SARIMA"})[["date", "blood_group", "SARIMA"]],
        on=["date", "blood_group"],
        how="left",
    )
    if wide[["SMA", "XGBoost", "SARIMA"]].isna().any().any():
        raise RuntimeError("Forecast merge produced missing predictions.")
    return wide.sort_values(["date", "blood_group"]).reset_index(drop=True)


def _sma_metadata(train_daily: pd.DataFrame, window: int) -> list[dict]:
    records: list[dict] = []
    for blood_group, group in train_daily.groupby("blood_group", sort=False):
        model: SMAModel = fit_sma(
            group.sort_values("date")["platelet_demand_units"],
            window=window,
            blood_group=str(blood_group),
        )
        records.append(
            {
                "blood_group": model.blood_group,
                "window": model.window,
                "forecast_level": model.forecast_level,
                "last_window_values": list(model.last_window_values),
            }
        )
    return records
