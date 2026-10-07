"""JIT recommendation service for the local research dashboard.

Delegates to the Phase 3 engine in ``policies.jit``.
"""

from __future__ import annotations

from pathlib import Path

from inventory.demo_loader import load_demo_inventory, load_demo_inventory_config
from inventory.inventory_manager import InventoryManager
from policies.jit import DEFAULT_HORIZON_DAYS, recommend_jit as recommend_jit_engine


def recommend_jit(
    inventory: InventoryManager | None = None,
    csv_path: str | Path | None = None,
    config_path: str | Path | None = None,
    results_dir: str | Path | None = None,
    horizon_days: int | None = None,
    as_of: str | None = None,
) -> dict:
    """Run Phase 3 JIT on Phase 2 inventory and Phase 1 forecast artifacts."""
    config = load_demo_inventory_config(config_path)
    manager = inventory if inventory is not None else load_demo_inventory(
        csv_path=csv_path, config_path=config_path
    )
    reference = as_of or str(config["as_of"])
    days = int(horizon_days if horizon_days is not None else config.get("horizon_days", DEFAULT_HORIZON_DAYS))
    rows = recommend_jit_engine(
        manager,
        horizon_days=days,
        results_dir=results_dir,
        as_of=reference,
        start_date=reference,
    )
    payload = [row.to_dict() for row in rows]
    return {
        "data_note": "synthetic/demo JIT recommendations — not clinical advice",
        "as_of": reference[:10],
        "horizon_days": days,
        "recommendations": payload,
        "shortage_groups": [row["blood_group"] for row in payload if row["status"] == "shortage"],
        "sufficient_groups": [
            row["blood_group"] for row in payload if row["status"] == "sufficient"
        ],
    }
