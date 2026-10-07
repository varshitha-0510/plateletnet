"""Forecast accuracy metrics and model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_DISPLAY_ORDER = ("SMA", "XGBoost", "SARIMA")


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean absolute error."""
    actual, pred = _align(y_true, y_pred)
    if actual.size == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - pred)))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root mean squared error."""
    actual, pred = _align(y_true, y_pred)
    if actual.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean absolute percentage error (zeros in actuals are skipped)."""
    actual, pred = _align(y_true, y_pred)
    mask = actual != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100.0)


def evaluate_models(
    actual: np.ndarray | pd.Series,
    predictions: dict[str, np.ndarray | pd.Series],
) -> pd.DataFrame:
    """Compare models on MAE, RMSE, and MAPE; lowest MAE is ranked first."""
    if not predictions:
        raise ValueError("predictions must contain at least one model.")

    rows: list[dict] = []
    for name, pred in predictions.items():
        rows.append(
            {
                "model": name,
                "MAE": mae(actual, pred),
                "RMSE": rmse(actual, pred),
                "MAPE": mape(actual, pred),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["tie_rank"] = metrics["model"].map(
        {name: i for i, name in enumerate(MODEL_DISPLAY_ORDER)}
    ).fillna(len(MODEL_DISPLAY_ORDER))
    metrics = metrics.sort_values(["MAE", "tie_rank"], kind="mergesort").drop(
        columns=["tie_rank"]
    )
    metrics["rank"] = np.arange(1, len(metrics) + 1)
    return metrics.reset_index(drop=True)


def select_best_model(metrics: pd.DataFrame) -> str:
    """Return the model name with the lowest MAE."""
    if metrics.empty or "model" not in metrics.columns or "MAE" not in metrics.columns:
        raise ValueError("metrics must include model and MAE columns.")
    ordered = metrics.sort_values("MAE", kind="mergesort")
    return str(ordered.iloc[0]["model"])


def comparison_table(
    sma: pd.DataFrame,
    xgboost: pd.DataFrame,
    sarima: pd.DataFrame,
    actual_col: str = "actual",
    pred_col: str = "prediction",
) -> pd.DataFrame:
    """Build an MAE/RMSE/MAPE comparison table for SMA, XGBoost, and SARIMA."""
    frames = {"SMA": sma, "XGBoost": xgboost, "SARIMA": sarima}
    predictions: dict[str, pd.Series] = {}
    actual = None
    for name, frame in frames.items():
        if actual_col not in frame.columns or pred_col not in frame.columns:
            raise ValueError(f"{name} forecast must include {actual_col} and {pred_col}.")
        sort_cols = [col for col in ("date", "blood_group") if col in frame.columns]
        aligned = frame.sort_values(sort_cols, kind="mergesort").reset_index(drop=True) if sort_cols else frame.reset_index(drop=True)
        predictions[name] = aligned[pred_col]
        if actual is None:
            actual = aligned[actual_col]
        elif len(actual) != len(aligned[actual_col]):
            raise ValueError(f"{name} forecast length does not match the other models.")
    return evaluate_models(actual, predictions)


def evaluate_forecast_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Score SMA / XGBoost / SARIMA columns against ``actual`` in a wide table."""
    required = {"actual", "SMA", "XGBoost", "SARIMA"}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"Forecast table missing columns: {sorted(missing)}")
    return evaluate_models(
        forecasts["actual"],
        {
            "SMA": forecasts["SMA"],
            "XGBoost": forecasts["XGBoost"],
            "SARIMA": forecasts["SARIMA"],
        },
    )


def _align(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if actual.shape != pred.shape:
        raise ValueError(
            f"actual and predicted length mismatch: {actual.shape} vs {pred.shape}"
        )
    mask = np.isfinite(actual) & np.isfinite(pred)
    return actual[mask], pred[mask]
