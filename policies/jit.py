"""Just-In-Time recommendation engine (synthetic/demo research only).

Combines Phase 1 forecast demand with Phase 2 usable inventory.

This module does not make clinical recommendations, does not extend
expiry dates, and does not implement Simulated Micro-Expiry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import BLOOD_GROUPS, FORECAST_RESULTS_DIR
from forecasting.model_registry import (
    FORECAST_FILENAME,
    SARIMA_FORECAST_FILENAME,
    SMA_FORECAST_FILENAME,
    XGBOOST_FORECAST_FILENAME,
    load_registry,
)
from inventory.inventory_manager import InventoryManager
from inventory.platelet_unit import DateLike, parse_date

STATUS_SHORTAGE = "shortage"
STATUS_SUFFICIENT = "sufficient"

DEFAULT_HORIZON_DAYS = 7

MODEL_FORECAST_FILES = {
    "XGBoost": XGBOOST_FORECAST_FILENAME,
    "SMA": SMA_FORECAST_FILENAME,
    "SARIMA": SARIMA_FORECAST_FILENAME,
}

LONG_FORECAST_COLUMNS = ("date", "blood_group", "prediction")


@dataclass(frozen=True)
class JITRecommendation:
    """One blood-group JIT recommendation for a future dashboard or demo."""

    blood_group: str
    forecast_horizon: int
    expected_demand: float
    available_inventory: int
    projected_shortage: float
    recommended_quantity: float
    status: str
    selected_model: str | None = None

    def to_dict(self) -> dict:
        return {
            "blood_group": self.blood_group,
            "forecast_horizon": self.forecast_horizon,
            "expected_demand": self.expected_demand,
            "available_inventory": self.available_inventory,
            "projected_shortage": self.projected_shortage,
            "recommended_quantity": self.recommended_quantity,
            "status": self.status,
            "selected_model": self.selected_model,
        }


def load_phase1_forecast(
    results_dir: str | Path | None = None,
    model_name: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load the selected Phase 1 forecast (long format: date, blood_group, prediction).

    Uses ``selected_model.json`` plus the matching forecast CSV. Falls back to
    the wide ``test_forecasts.csv`` comparison table if needed.
    """
    out_dir = Path(results_dir) if results_dir is not None else FORECAST_RESULTS_DIR
    registry = load_registry(out_dir)
    selected = model_name or str(registry["selected_model"])

    long_name = MODEL_FORECAST_FILES.get(selected)
    if long_name is not None:
        long_path = out_dir / long_name
        if long_path.exists():
            frame = pd.read_csv(long_path)
            return _normalize_forecast_frame(frame, selected_model=selected), selected

    wide_path = out_dir / FORECAST_FILENAME
    if not wide_path.exists():
        raise FileNotFoundError(
            f"Phase 1 forecast not found for model {selected!r} in {out_dir}"
        )
    wide = pd.read_csv(wide_path)
    if selected not in wide.columns:
        raise ValueError(
            f"Selected model {selected!r} is not a column in {wide_path}"
        )
    frame = wide[["date", "blood_group"]].copy()
    frame["model"] = selected
    frame["prediction"] = wide[selected]
    return _normalize_forecast_frame(frame, selected_model=selected), selected


def expected_demand_by_group(
    forecast: pd.DataFrame,
    horizon_days: int,
    start_date: DateLike | None = None,
) -> dict[str, float]:
    """Sum Phase 1 ``prediction`` values for the next ``horizon_days`` forecast dates."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1.")
    frame = _normalize_forecast_frame(forecast)
    if start_date is not None:
        start = pd.Timestamp(parse_date(start_date))
        frame = frame.loc[frame["date"] >= start].copy()

    horizon_dates = sorted(frame["date"].unique())[:horizon_days]
    window = frame.loc[frame["date"].isin(horizon_dates)]
    totals: dict[str, float] = {}
    for blood_group, group_rows in window.groupby("blood_group", sort=False):
        demand = float(group_rows["prediction"].clip(lower=0).sum())
        totals[str(blood_group)] = demand
    return totals


def recommend_jit(
    inventory: InventoryManager,
    forecast: pd.DataFrame | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    *,
    results_dir: str | Path | None = None,
    as_of: DateLike | None = None,
    start_date: DateLike | None = None,
    selected_model: str | None = None,
) -> list[JITRecommendation]:
    """Recommend units per blood group from forecast demand minus usable stock.

    ``recommended_quantity = max(expected_demand - available_inventory, 0)``.
    Expired and depleted batches are excluded via Phase 2 ``InventoryManager``.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1.")
    if not isinstance(inventory, InventoryManager):
        raise TypeError("inventory must be an InventoryManager.")

    model_name = selected_model
    if forecast is None:
        forecast, loaded_model = load_phase1_forecast(
            results_dir=results_dir, model_name=selected_model
        )
        model_name = loaded_model
    else:
        forecast = _normalize_forecast_frame(forecast, selected_model=selected_model)
        if model_name is None and "model" in forecast.columns:
            models = [str(value) for value in forecast["model"].dropna().unique()]
            if len(models) == 1:
                model_name = models[0]

    demand_by_group = expected_demand_by_group(
        forecast, horizon_days=horizon_days, start_date=start_date
    )
    inventory_groups = {
        batch.blood_group for batch in inventory.get_inventory(as_of=as_of)
    }
    forecast_groups = {str(value) for value in forecast["blood_group"].unique()}
    groups = _ordered_blood_groups(forecast_groups | inventory_groups)

    recommendations: list[JITRecommendation] = []
    for blood_group in groups:
        expected = float(demand_by_group.get(blood_group, 0.0))
        if expected < 0:
            expected = 0.0
        available = int(
            inventory.available_units(blood_group=blood_group, as_of=as_of)
        )
        shortage = max(expected - available, 0.0)
        status = STATUS_SHORTAGE if available < expected else STATUS_SUFFICIENT
        recommendations.append(
            JITRecommendation(
                blood_group=blood_group,
                forecast_horizon=horizon_days,
                expected_demand=expected,
                available_inventory=available,
                projected_shortage=shortage,
                recommended_quantity=shortage,
                status=status,
                selected_model=model_name,
            )
        )
    return recommendations


def apply_jit_policy(*args, **kwargs):
    """Alias for :func:`recommend_jit` (research/demo policy hook)."""
    return recommend_jit(*args, **kwargs)


def _ordered_blood_groups(present: set[str]) -> list[str]:
    ordered = [group for group in BLOOD_GROUPS if group in present]
    extras = sorted(group for group in present if group not in BLOOD_GROUPS)
    return ordered + extras


def _normalize_forecast_frame(
    forecast: pd.DataFrame,
    selected_model: str | None = None,
) -> pd.DataFrame:
    if not isinstance(forecast, pd.DataFrame):
        raise TypeError("forecast must be a pandas DataFrame.")
    frame = forecast.copy()
    if "prediction" not in frame.columns and selected_model and selected_model in frame.columns:
        frame["prediction"] = frame[selected_model]
    missing = [column for column in LONG_FORECAST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Forecast frame is missing required columns "
            f"{missing}. Expected Phase 1 columns {list(LONG_FORECAST_COLUMNS)}."
        )
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["blood_group"] = frame["blood_group"].astype(str)
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    if frame["prediction"].isna().any():
        raise ValueError("Forecast prediction column contains non-numeric values.")
    return frame
