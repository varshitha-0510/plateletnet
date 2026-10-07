"""Inventory query service for the local research dashboard.

Uses Phase 2 InventoryManager, FIFO selection, and expiry alerts.
Does not change expiry dates and does not implement micro-expiry.
"""

from __future__ import annotations

from pathlib import Path

from inventory.demo_loader import (
    DEFAULT_DEMO_INVENTORY_CSV,
    load_demo_inventory,
    load_demo_inventory_config,
)
from inventory.fifo import select_fifo
from inventory.platelet_unit import DateLike


def get_inventory(
    csv_path: str | Path | None = None,
    config_path: str | Path | None = None,
    as_of: DateLike | None = None,
) -> dict:
    """Return synthetic demo inventory state from Phase 2 classes."""
    config = load_demo_inventory_config(config_path)
    manager = load_demo_inventory(csv_path=csv_path, config_path=config_path)
    reference = as_of if as_of is not None else str(config["as_of"])
    batches = [batch.to_dict() for batch in manager.get_inventory(as_of=reference)]
    fifo_batches = [
        batch.to_dict()
        for batch in select_fifo(manager.get_inventory(as_of=reference), as_of=reference)
    ]
    stock = manager.stock_by_blood_group(as_of=reference)
    low_stock = manager.low_stock_report(as_of=reference)
    by_group = [
        {
            "blood_group": group,
            "available_units": int(stock[group]),
            "threshold": int(low_stock[group]["threshold"]),
            "low_stock": bool(low_stock[group]["low_stock"]),
        }
        for group in stock
    ]
    return {
        "data_note": config.get(
            "data_note", "synthetic/demo inventory only — not a clinical system"
        ),
        "source_csv": str(csv_path or DEFAULT_DEMO_INVENTORY_CSV),
        "as_of": str(reference)[:10],
        "low_stock_threshold": manager.low_stock_threshold,
        "near_expiry_days": manager.near_expiry_days,
        "total_available": int(sum(stock.values())),
        "by_blood_group": by_group,
        "low_stock_groups": [
            row["blood_group"] for row in by_group if row["low_stock"]
        ],
        "expiry_alerts": manager.expiry_alerts(as_of=reference),
        "batches": batches,
        "fifo_order": fifo_batches,
    }
