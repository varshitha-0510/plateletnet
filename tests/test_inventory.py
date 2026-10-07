"""Tests for Phase 2 synthetic inventory, FIFO, and expiry tracking."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inventory.expiry_manager import get_expiry_alerts, is_expired, is_near_expiry
from inventory.fifo import select_fifo
from inventory.inventory_manager import InventoryManager
from inventory.platelet_unit import PlateletUnit


AS_OF = date(2026, 1, 10)


def _batch(
    batch_id: str,
    blood_group: str = "A+",
    collection: str = "2026-01-01",
    expiry: str = "2026-01-15",
    quantity: int = 10,
) -> PlateletUnit:
    return PlateletUnit(
        batch_id=batch_id,
        blood_group=blood_group,
        collection_date=collection,
        expiry_date=expiry,
        quantity=quantity,
    )


def _manager(*batches: PlateletUnit, threshold: int = 10) -> InventoryManager:
    manager = InventoryManager(
        low_stock_threshold=threshold,
        near_expiry_days=2,
        reference_date=AS_OF,
    )
    for batch in batches:
        manager.add_batch(batch)
    return manager


def test_add_inventory() -> None:
    manager = _manager(_batch("B1", quantity=10), _batch("B2", quantity=5, collection="2026-01-05"))
    inventory = manager.get_inventory()
    assert len(inventory) == 2
    assert {b.batch_id for b in inventory} == {"B1", "B2"}
    assert manager.available_units("A+") == 15


def test_duplicate_batch_id_rejected() -> None:
    manager = _manager(_batch("B1"))
    with pytest.raises(ValueError, match="Duplicate batch_id"):
        manager.add_batch(_batch("B1", quantity=3))


def test_consume_inventory_reduces_quantity() -> None:
    manager = _manager(_batch("B1", quantity=10))
    events = manager.consume("A+", 4)
    assert events == [
        {
            "batch_id": "B1",
            "blood_group": "A+",
            "quantity_consumed": 4,
            "remaining_quantity": 6,
            "status": "available",
        }
    ]
    assert manager.available_units("A+") == 6


def test_fifo_older_collection_date_first() -> None:
    older = _batch("A", collection="2026-01-01", quantity=10)
    newer = _batch("B", collection="2026-01-05", quantity=20)
    manager = _manager(newer, older)
    ordered = [b.batch_id for b in manager.available_batches("A+")]
    assert ordered == ["A", "B"]
    events = manager.consume("A+", 12)
    assert [e["batch_id"] for e in events] == ["A", "B"]
    assert events[0]["quantity_consumed"] == 10
    assert events[0]["remaining_quantity"] == 0
    assert events[0]["status"] == "depleted"
    assert events[1]["quantity_consumed"] == 2
    assert events[1]["remaining_quantity"] == 18
    assert manager.get_inventory_by_blood_group("A+")
    remaining = {b.batch_id: b.quantity for b in manager.get_inventory()}
    assert remaining["A"] == 0
    assert remaining["B"] == 18


def test_blood_group_filtering_is_independent() -> None:
    manager = _manager(
        _batch("A1", blood_group="A+", quantity=8),
        _batch("O1", blood_group="O+", quantity=12),
    )
    assert manager.available_units("A+") == 8
    assert manager.available_units("O+") == 12
    manager.consume("A+", 3)
    assert manager.available_units("A+") == 5
    assert manager.available_units("O+") == 12
    groups = {b.blood_group for b in manager.get_inventory_by_blood_group("O+")}
    assert groups == {"O+"}


def test_expired_batch_excluded_from_available_and_fifo() -> None:
    valid = _batch("OK", collection="2026-01-06", expiry="2026-01-12", quantity=7)
    expired = _batch("OLD", collection="2025-12-20", expiry="2026-01-09", quantity=9)
    manager = _manager(valid, expired)
    assert manager.available_units("A+") == 7
    assert [b.batch_id for b in select_fifo(manager.get_inventory(), "A+", AS_OF)] == ["OK"]
    events = manager.consume("A+", 7)
    assert events[0]["batch_id"] == "OK"
    leftover = {b.batch_id: (b.quantity, b.status) for b in manager.get_inventory()}
    assert leftover["OLD"] == (9, "expired")
    assert leftover["OK"] == (0, "depleted")


def test_near_expiry_detection_uses_reference_date() -> None:
    soon = _batch("SOON", expiry="2026-01-12", quantity=4)
    later = _batch("LATER", collection="2026-01-02", expiry="2026-01-20", quantity=4)
    expired = _batch("GONE", collection="2025-12-01", expiry="2026-01-08", quantity=4)
    manager = _manager(soon, later, expired, threshold=10)
    alerts = manager.expiry_alerts()
    by_id = {a["batch_id"]: a for a in alerts}
    assert by_id["GONE"]["alert_type"] == "expired"
    assert by_id["SOON"]["alert_type"] == "near_expiry"
    assert by_id["SOON"]["days_until_expiry"] == 2
    assert "LATER" not in by_id
    assert is_near_expiry(soon, AS_OF, warning_days=2)
    assert not is_near_expiry(later, AS_OF, warning_days=2)
    assert is_expired(expired, AS_OF)


def test_low_stock_detection_ignores_expired_and_depleted() -> None:
    manager = InventoryManager(
        low_stock_threshold=10,
        near_expiry_days=2,
        reference_date=AS_OF,
    )
    manager.add_batch(_batch("LIVE", quantity=8))
    manager.add_batch(_batch("EMPTY", collection="2026-01-02", quantity=0))
    manager.add_batch(
        _batch("EXP", collection="2025-12-01", expiry="2026-01-01", quantity=50)
    )
    report = manager.low_stock_report()
    assert report["A+"]["available_units"] == 8
    assert report["A+"]["threshold"] == 10
    assert report["A+"]["low_stock"] is True
    assert report["O+"]["available_units"] == 0
    assert report["O+"]["low_stock"] is True


def test_empty_and_depleted_batches_are_not_consumed() -> None:
    empty = _batch("E0", quantity=0)
    live = _batch("E1", collection="2026-01-04", quantity=5)
    manager = _manager(empty, live)
    events = manager.consume("A+", 5)
    assert [e["batch_id"] for e in events] == ["E1"]
    assert manager._batches["E0"].quantity == 0
    assert manager._batches["E0"].status == "depleted"


def test_consume_exact_available_quantity() -> None:
    manager = _manager(_batch("X1", quantity=6), _batch("X2", collection="2026-01-03", quantity=4))
    events = manager.consume("A+", 10)
    assert sum(e["quantity_consumed"] for e in events) == 10
    assert manager.available_units("A+") == 0
    assert all(b.status == "depleted" for b in manager.get_inventory())


def test_consume_more_than_available_raises_and_does_not_change_stock() -> None:
    manager = _manager(_batch("Y1", quantity=5))
    with pytest.raises(ValueError, match="only 5 available"):
        manager.consume("A+", 6)
    assert manager.available_units("A+") == 5
    assert manager._batches["Y1"].quantity == 5


def test_zero_and_negative_quantity_validation() -> None:
    with pytest.raises(ValueError, match="negative"):
        _batch("BAD", quantity=-1)
    manager = _manager(_batch("Z1", quantity=3))
    with pytest.raises(ValueError, match="positive"):
        manager.consume("A+", 0)
    with pytest.raises(ValueError, match="positive"):
        manager.consume("A+", -2)
    assert manager.available_units("A+") == 3


def test_expiry_alerts_are_deterministic_with_fixed_reference_date() -> None:
    batches = [
        _batch("N1", expiry="2026-01-11", quantity=2),
        _batch("N2", collection="2026-01-02", expiry="2026-01-09", quantity=2),
    ]
    alerts = get_expiry_alerts(batches, as_of=AS_OF, warning_days=2)
    assert [a.alert_type for a in alerts] == ["expired", "near_expiry"]
    assert [a.batch_id for a in alerts] == ["N2", "N1"]
    assert alerts[0].expiry_date == date(2026, 1, 9)
    # Expiry dates must not be rewritten by alert generation.
    assert batches[0].expiry_date == date(2026, 1, 11)
    assert batches[1].expiry_date == date(2026, 1, 9)
