"""Synthetic platelet inventory (demo/research only)."""

from inventory.demo_loader import load_demo_inventory, load_demo_inventory_config
from inventory.expiry_manager import ExpiryAlert, get_expiry_alerts
from inventory.fifo import consume_fifo, select_fifo
from inventory.inventory_manager import InventoryManager
from inventory.platelet_unit import PlateletBatch, PlateletUnit

__all__ = [
    "ExpiryAlert",
    "InventoryManager",
    "PlateletBatch",
    "PlateletUnit",
    "consume_fifo",
    "get_expiry_alerts",
    "load_demo_inventory",
    "load_demo_inventory_config",
    "select_fifo",
]
