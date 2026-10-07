"""Load the labeled synthetic demo inventory into Phase 2 InventoryManager.

Demo/research data only. This is not a clinical inventory feed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import SYNTHETIC_DATA_DIR
from inventory.inventory_manager import InventoryManager
from inventory.platelet_unit import PlateletUnit

DEFAULT_DEMO_INVENTORY_CSV = SYNTHETIC_DATA_DIR / "demo_inventory.csv"
DEFAULT_DEMO_INVENTORY_CONFIG = SYNTHETIC_DATA_DIR / "demo_inventory_config.json"
REQUIRED_COLUMNS = (
    "batch_id",
    "blood_group",
    "collection_date",
    "expiry_date",
    "quantity",
)


def load_demo_inventory_config(path: str | Path | None = None) -> dict:
    """Load dashboard demo inventory settings (as_of, thresholds, horizon)."""
    config_path = Path(path) if path is not None else DEFAULT_DEMO_INVENTORY_CONFIG
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload


def load_demo_batches(csv_path: str | Path | None = None) -> list[PlateletUnit]:
    """Read synthetic demo batches from CSV into Phase 2 ``PlateletUnit`` objects."""
    path = Path(csv_path) if csv_path is not None else DEFAULT_DEMO_INVENTORY_CSV
    if not path.exists():
        raise FileNotFoundError(f"Demo inventory CSV not found: {path}")
    frame = pd.read_csv(path, comment="#")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Demo inventory CSV is missing columns {missing}.")
    batches: list[PlateletUnit] = []
    for row in frame.itertuples(index=False):
        batches.append(
            PlateletUnit(
                batch_id=str(row.batch_id),
                blood_group=str(row.blood_group),
                collection_date=row.collection_date,
                expiry_date=row.expiry_date,
                quantity=int(row.quantity),
            )
        )
    return batches


def load_demo_inventory(
    csv_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> InventoryManager:
    """Build an ``InventoryManager`` from the synthetic demo inventory CSV."""
    config = load_demo_inventory_config(config_path)
    manager = InventoryManager(
        low_stock_threshold=int(config.get("low_stock_threshold", 10)),
        near_expiry_days=int(config.get("near_expiry_days", 2)),
        reference_date=str(config["as_of"]),
    )
    for batch in load_demo_batches(csv_path):
        manager.add_batch(batch)
    return manager
