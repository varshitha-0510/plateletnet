"""Forecasting service for the local research dashboard.

Reads existing Phase 1 artifacts. Does not retrain models.
"""

from __future__ import annotations

from pathlib import Path

from config import FORECAST_RESULTS_DIR
from forecasting.model_registry import (
    SARIMA_FORECAST_FILENAME,
    SMA_FORECAST_FILENAME,
    XGBOOST_FORECAST_FILENAME,
    load_registry,
)
from policies.jit import (
    DEFAULT_HORIZON_DAYS,
    MODEL_FORECAST_FILES,
    expected_demand_by_group,
    load_phase1_forecast,
)

DASHBOARD_MODEL_ORDER = ("SMA", "XGBoost", "SARIMA")
DASHBOARD_MODEL_FILES = {
    "SMA": SMA_FORECAST_FILENAME,
    "XGBoost": XGBOOST_FORECAST_FILENAME,
    "SARIMA": SARIMA_FORECAST_FILENAME,
}


def get_forecast(
    results_dir: str | Path | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    start_date: str | None = None,
) -> dict:
    """Return the selected Phase 1 forecast summary for the dashboard."""
    out_dir = Path(results_dir) if results_dir is not None else FORECAST_RESULTS_DIR
    forecast, selected = load_phase1_forecast(results_dir=out_dir)
    registry = load_registry(out_dir)
    demand = expected_demand_by_group(
        forecast, horizon_days=horizon_days, start_date=start_date
    )
    long_name = MODEL_FORECAST_FILES.get(selected)
    forecast_file = str(out_dir / long_name) if long_name else str(out_dir)
    window = forecast.copy()
    if start_date is not None:
        window = window.loc[window["date"] >= pd_timestamp(start_date)]
    horizon_dates = sorted(window["date"].unique())[:horizon_days]
    series = window.loc[window["date"].isin(horizon_dates)].copy()
    by_group = [
        {"blood_group": group, "expected_demand": float(demand.get(group, 0.0))}
        for group in _group_order(demand)
    ]
    return {
        "label": "Synthetic / Research Forecast",
        "selected_model": selected,
        "selection_reason": registry.get("selection_reason"),
        "selection_rule": registry.get("selection_rule"),
        "horizon_days": horizon_days,
        "start_date": start_date,
        "forecast_file": forecast_file,
        "test_start": registry.get("test_start"),
        "test_end": registry.get("test_end"),
        "metrics": registry.get("metrics"),
        "by_blood_group": by_group,
        "series": [
            {
                "date": str(pd_date(row.date)),
                "blood_group": str(row.blood_group),
                "prediction": float(row.prediction),
            }
            for row in series.itertuples(index=False)
        ],
        "comparison": _comparison_from_registry(registry),
        "models": _phase1_model_payloads(out_dir, registry),
        "data_note": "synthetic/demo demand only — not a clinical system",
    }


def get_forecast_models(results_dir: str | Path | None = None) -> dict:
    """Return SMA, XGBoost, and SARIMA hold-out series from Phase 1 files."""
    out_dir = Path(results_dir) if results_dir is not None else FORECAST_RESULTS_DIR
    registry = load_registry(out_dir)
    return {
        "label": "Synthetic / Research Forecast",
        "selected_model": registry.get("selected_model"),
        "selection_reason": registry.get("selection_reason"),
        "selection_rule": registry.get("selection_rule"),
        "test_start": registry.get("test_start"),
        "test_end": registry.get("test_end"),
        "comparison": _comparison_from_registry(registry),
        "models": _phase1_model_payloads(out_dir, registry),
        "data_note": "synthetic/demo demand only — not a clinical system",
    }


def pd_timestamp(value: str):
    import pandas as pd

    return pd.Timestamp(value)


def pd_date(value) -> str:
    import pandas as pd

    return pd.Timestamp(value).date().isoformat()


def _comparison_from_registry(registry: dict) -> list[dict]:
    """Pass through Phase 1 MAE/RMSE/MAPE ranks. Do not recompute metrics."""
    rows = []
    for item in registry.get("metrics") or []:
        rows.append(
            {
                "model": item.get("model"),
                "MAE": item.get("MAE"),
                "RMSE": item.get("RMSE"),
                "MAPE": item.get("MAPE"),
                "rank": item.get("rank"),
                "selected": item.get("model") == registry.get("selected_model"),
            }
        )
    order = {name: index for index, name in enumerate(DASHBOARD_MODEL_ORDER)}
    rows.sort(key=lambda row: order.get(row["model"], 99))
    return rows


def _phase1_model_payloads(out_dir: Path, registry: dict) -> list[dict]:
    selected = registry.get("selected_model")
    payloads = []
    for name in DASHBOARD_MODEL_ORDER:
        filename = DASHBOARD_MODEL_FILES[name]
        path = out_dir / filename
        series, dates, groups = _read_phase1_forecast_file(path)
        payloads.append(
            {
                "name": name,
                "file": str(path),
                "selected": name == selected,
                "horizon_start": dates[0] if dates else None,
                "horizon_end": dates[-1] if dates else None,
                "horizon_days": len(dates),
                "blood_groups": groups,
                "series": series,
            }
        )
    return payloads


def _read_phase1_forecast_file(path: Path) -> tuple[list[dict], list[str], list[str]]:
    import pandas as pd

    if not path.exists():
        return [], [], []
    frame = pd.read_csv(path)
    has_actual = "actual" in frame.columns
    series = []
    for row in frame.itertuples(index=False):
        actual = None
        if has_actual:
            value = getattr(row, "actual")
            if value is not None and not pd.isna(value):
                actual = float(value)
        series.append(
            {
                "date": str(pd_date(row.date)),
                "blood_group": str(row.blood_group),
                "prediction": float(row.prediction),
                "actual": actual,
            }
        )
    dates = sorted({row["date"] for row in series})
    groups = _group_order({row["blood_group"]: 0.0 for row in series})
    return series, dates, groups


def _group_order(demand: dict[str, float]) -> list[str]:
    from config import BLOOD_GROUPS

    ordered = [group for group in BLOOD_GROUPS if group in demand]
    extras = sorted(group for group in demand if group not in BLOOD_GROUPS)
    return ordered + extras
