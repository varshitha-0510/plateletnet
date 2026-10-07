"""Assemble the research dashboard payload from Phases 1–3.

Python remains the source of truth. The frontend only displays this payload.
"""

from __future__ import annotations

from pathlib import Path

from config import APP_NAME, MICRO_EXPIRY_LABEL, SIMULATED_MICRO_EXPIRY_ENABLED
from backend.services.forecast_service import get_forecast
from backend.services.inventory_service import get_inventory
from backend.services.jit_service import recommend_jit
from inventory.demo_loader import load_demo_inventory_config


def get_dashboard(
    csv_path: str | Path | None = None,
    config_path: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> dict:
    """Return one JSON document for the local PlateletNet dashboard."""
    config = load_demo_inventory_config(config_path)
    as_of = str(config["as_of"])
    horizon_days = int(config.get("horizon_days", 7))
    inventory = get_inventory(csv_path=csv_path, config_path=config_path, as_of=as_of)
    forecast = get_forecast(
        results_dir=results_dir, horizon_days=horizon_days, start_date=as_of
    )
    jit = recommend_jit(
        csv_path=csv_path,
        config_path=config_path,
        results_dir=results_dir,
        horizon_days=horizon_days,
        as_of=as_of,
    )
    return {
        "meta": {
            "app_name": APP_NAME,
            "title": "PlateletNet Research Prototype",
            "research_prototype": True,
            "synthetic_data": True,
            "clinical_system": False,
            "micro_expiry_implemented": False,
            "micro_expiry_enabled": bool(SIMULATED_MICRO_EXPIRY_ENABLED),
            "micro_expiry_label": MICRO_EXPIRY_LABEL,
            "disclaimer": (
                "This is not a clinical medical system. Synthetic/demo data only. "
                "Do not use real patient information. Forecasts and JIT values are "
                "demo analytics, not clinical advice. Expiry dates are not extended."
            ),
        },
        "forecast": forecast,
        "inventory": inventory,
        "jit": jit,
    }
