"""Store and load the selected forecasting model and comparison metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config import FORECAST_RESULTS_DIR

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "selected_model.json"
METRICS_FILENAME = "metrics.csv"
COMPARISON_FILENAME = "model_comparison.csv"
FORECAST_FILENAME = "test_forecasts.csv"
SMA_FORECAST_FILENAME = "sma_forecast.csv"
XGBOOST_FORECAST_FILENAME = "xgboost_forecast.csv"
SARIMA_FORECAST_FILENAME = "sarima_forecast.csv"
XGBOOST_MODEL_FILENAME = "xgboost_model.json"
SMA_FILENAME = "sma_models.json"
SARIMA_FILENAME = "sarima_metadata.json"


def save_registry(
    metadata: dict[str, Any],
    results_dir: str | Path | None = None,
) -> Path:
    """Write selected-model metadata as JSON."""
    out_dir = Path(results_dir) if results_dir is not None else FORECAST_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REGISTRY_FILENAME
    payload = dict(metadata)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote model registry to %s", path)
    return path


def load_registry(results_dir: str | Path | None = None) -> dict[str, Any]:
    """Load selected-model metadata."""
    out_dir = Path(results_dir) if results_dir is not None else FORECAST_RESULTS_DIR
    path = out_dir / REGISTRY_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_model(results_dir: str | Path | None = None) -> dict[str, Any]:
    """Return registry contents for the recommended forecasting model."""
    return load_registry(results_dir)


def save_forecast_artifacts(
    metrics: pd.DataFrame,
    forecasts: pd.DataFrame,
    sma_models: list[dict[str, Any]],
    xgboost_model,
    sarima_metadata: list[dict[str, Any]],
    selected_metadata: dict[str, Any],
    results_dir: str | Path | None = None,
    sma_forecast: pd.DataFrame | None = None,
    xgboost_forecast: pd.DataFrame | None = None,
    sarima_forecast: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Persist metrics, per-model forecasts, and lightweight model artifacts."""
    out_dir = Path(results_dir) if results_dir is not None else FORECAST_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / METRICS_FILENAME
    comparison_path = out_dir / COMPARISON_FILENAME
    forecast_path = out_dir / FORECAST_FILENAME
    sma_forecast_path = out_dir / SMA_FORECAST_FILENAME
    xgb_forecast_path = out_dir / XGBOOST_FORECAST_FILENAME
    sarima_forecast_path = out_dir / SARIMA_FORECAST_FILENAME
    sma_path = out_dir / SMA_FILENAME
    sarima_path = out_dir / SARIMA_FILENAME
    xgb_path = out_dir / XGBOOST_MODEL_FILENAME

    metrics.to_csv(metrics_path, index=False)
    metrics.to_csv(comparison_path, index=False)
    forecasts.to_csv(forecast_path, index=False)

    sma_out = sma_forecast if sma_forecast is not None else _long_forecast(forecasts, "SMA")
    xgb_out = (
        xgboost_forecast
        if xgboost_forecast is not None
        else _long_forecast(forecasts, "XGBoost")
    )
    sarima_out = (
        sarima_forecast
        if sarima_forecast is not None
        else _long_forecast(forecasts, "SARIMA")
    )
    sma_out.to_csv(sma_forecast_path, index=False)
    xgb_out.to_csv(xgb_forecast_path, index=False)
    sarima_out.to_csv(sarima_forecast_path, index=False)

    sma_path.write_text(json.dumps(sma_models, indent=2, default=str), encoding="utf-8")
    sarima_path.write_text(
        json.dumps(sarima_metadata, indent=2, default=str), encoding="utf-8"
    )

    if xgboost_model is not None:
        xgboost_model.estimator.save_model(xgb_path)
        selected_metadata = dict(selected_metadata)
        selected_metadata["xgboost_model_path"] = str(xgb_path)
        selected_metadata["feature_columns"] = xgboost_model.feature_columns
        selected_metadata["blood_group_codes"] = xgboost_model.blood_group_codes

    registry_path = save_registry(selected_metadata, out_dir)
    return {
        "metrics": metrics_path,
        "model_comparison": comparison_path,
        "forecasts": forecast_path,
        "sma_forecast": sma_forecast_path,
        "xgboost_forecast": xgb_forecast_path,
        "sarima_forecast": sarima_forecast_path,
        "sma": sma_path,
        "sarima": sarima_path,
        "xgboost": xgb_path,
        "registry": registry_path,
    }


def _long_forecast(wide: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Convert a wide comparison table into one model forecast file."""
    frame = wide[["date", "blood_group", "actual"]].copy()
    frame["model"] = model_name
    frame["prediction"] = wide[model_name]
    return frame


def load_xgboost_estimator(results_dir: str | Path | None = None):
    """Load a previously saved XGBoost JSON model, if present."""
    from xgboost import XGBRegressor

    out_dir = Path(results_dir) if results_dir is not None else FORECAST_RESULTS_DIR
    path = out_dir / XGBOOST_MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {path}")
    estimator = XGBRegressor()
    estimator.load_model(path)
    return estimator
