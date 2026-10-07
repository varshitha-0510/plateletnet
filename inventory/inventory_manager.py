"""Blood-group-wise inventory manager for synthetic platelet batches.

Expired batches never count as available stock. Consumption uses FIFO
and never goes negative. This is demo/research software, not a clinical
inventory system.
"""

from __future__ import annotations

from datetime import date

from config import BLOOD_GROUPS
from inventory.expiry_manager import get_expiry_alerts
from inventory.fifo import consume_fifo, select_fifo
from inventory.platelet_unit import DateLike, PlateletUnit, parse_date


class InventoryManager:
    """In-memory manager for synthetic platelet batches."""

    def __init__(
        self,
        low_stock_threshold: int = 10,
        near_expiry_days: int = 2,
        reference_date: DateLike | None = None,
    ) -> None:
        if low_stock_threshold < 0:
            raise ValueError("low_stock_threshold must be >= 0.")
        if near_expiry_days < 0:
            raise ValueError("near_expiry_days must be >= 0.")
        self.low_stock_threshold = low_stock_threshold
        self.near_expiry_days = near_expiry_days
        self.reference_date = (
            parse_date(reference_date) if reference_date is not None else None
        )
        self._batches: dict[str, PlateletUnit] = {}

    def _as_of(self, as_of: DateLike | None) -> date:
        if as_of is not None:
            return parse_date(as_of)
        if self.reference_date is not None:
            return self.reference_date
        return date.today()

    def add_batch(self, batch: PlateletUnit) -> PlateletUnit:
        """Add a platelet batch. Duplicate ``batch_id`` values are rejected."""
        if not isinstance(batch, PlateletUnit):
            raise TypeError("batch must be a PlateletUnit.")
        if batch.batch_id in self._batches:
            raise ValueError(f"Duplicate batch_id: {batch.batch_id}")
        batch.refresh_status(self._as_of(None))
        self._batches[batch.batch_id] = batch
        return batch

    def get_inventory(self, as_of: DateLike | None = None) -> list[PlateletUnit]:
        """Return all batches (including expired and depleted)."""
        as_of_date = self._as_of(as_of)
        batches = list(self._batches.values())
        for batch in batches:
            batch.refresh_status(as_of_date)
        return batches

    def get_inventory_by_blood_group(
        self,
        blood_group: str,
        as_of: DateLike | None = None,
    ) -> list[PlateletUnit]:
        """Return all batches for one blood group."""
        return [
            batch
            for batch in self.get_inventory(as_of=as_of)
            if batch.blood_group == blood_group
        ]

    def available_batches(
        self,
        blood_group: str | None = None,
        as_of: DateLike | None = None,
    ) -> list[PlateletUnit]:
        """Return usable batches in FIFO order."""
        return select_fifo(
            self.get_inventory(as_of=as_of),
            blood_group=blood_group,
            as_of=self._as_of(as_of),
        )

    def available_units(
        self,
        blood_group: str | None = None,
        as_of: DateLike | None = None,
    ) -> int:
        """Total usable units (expired and depleted excluded)."""
        return sum(batch.quantity for batch in self.available_batches(blood_group, as_of))

    def stock_by_blood_group(self, as_of: DateLike | None = None) -> dict[str, int]:
        """Available units for each configured blood group (plus any extras)."""
        as_of_date = self._as_of(as_of)
        groups = list(BLOOD_GROUPS)
        for batch in self._batches.values():
            if batch.blood_group not in groups:
                groups.append(batch.blood_group)
        return {
            group: self.available_units(blood_group=group, as_of=as_of_date)
            for group in groups
        }

    def consume(
        self,
        blood_group: str,
        quantity: int,
        as_of: DateLike | None = None,
    ) -> list[dict]:
        """Consume units of ``blood_group`` using FIFO. Never goes negative."""
        as_of_date = self._as_of(as_of)
        return consume_fifo(
            list(self._batches.values()),
            blood_group=blood_group,
            quantity=quantity,
            as_of=as_of_date,
        )

    def low_stock_report(self, as_of: DateLike | None = None) -> dict[str, dict]:
        """Per-group available stock vs the configured low-stock threshold."""
        as_of_date = self._as_of(as_of)
        stock = self.stock_by_blood_group(as_of=as_of_date)
        report: dict[str, dict] = {}
        for group, available in stock.items():
            report[group] = {
                "available_units": available,
                "threshold": self.low_stock_threshold,
                "low_stock": available < self.low_stock_threshold,
            }
        return report

    def expiry_alerts(self, as_of: DateLike | None = None) -> list[dict]:
        """Expired and near-expiry alerts. Does not change expiry dates."""
        as_of_date = self._as_of(as_of)
        alerts = get_expiry_alerts(
            self.get_inventory(as_of=as_of_date),
            as_of=as_of_date,
            warning_days=self.near_expiry_days,
        )
        return [alert.to_dict() for alert in alerts]
