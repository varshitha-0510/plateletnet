"""Synthetic platelet batch/unit model.

Demo/research inventory only. This is not a clinical product record and
does not store patient information.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Union

DateLike = Union[date, datetime, str]

STATUS_AVAILABLE = "available"
STATUS_DEPLETED = "depleted"
STATUS_EXPIRED = "expired"
VALID_STATUSES = (STATUS_AVAILABLE, STATUS_DEPLETED, STATUS_EXPIRED)


def parse_date(value: DateLike) -> date:
    """Convert a date, datetime, or ISO date string to ``datetime.date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise ValueError(f"Invalid date value: {value!r}") from exc


@dataclass
class PlateletUnit:
    """One synthetic platelet batch tracked in demo inventory.

    ``quantity`` is the remaining units in this batch. Status is a simple
    inventory state (available / depleted / expired), not a clinical label.
    """

    batch_id: str
    blood_group: str
    collection_date: date
    expiry_date: date
    quantity: int
    status: str = STATUS_AVAILABLE

    def __post_init__(self) -> None:
        self.batch_id = str(self.batch_id).strip()
        if not self.batch_id:
            raise ValueError("batch_id is required.")

        self.blood_group = str(self.blood_group).strip()
        if not self.blood_group:
            raise ValueError("blood_group is required.")

        self.collection_date = parse_date(self.collection_date)
        self.expiry_date = parse_date(self.expiry_date)
        if self.expiry_date < self.collection_date:
            raise ValueError(
                f"expiry_date {self.expiry_date} is before collection_date "
                f"{self.collection_date}."
            )

        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise ValueError("quantity must be an integer number of units.")
        if self.quantity < 0:
            raise ValueError("quantity cannot be negative.")

        self.status = str(self.status).strip().lower()
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Unknown status {self.status!r}. Expected one of {VALID_STATUSES}."
            )
        if self.quantity == 0:
            self.status = STATUS_DEPLETED

    def refresh_status(self, as_of: DateLike) -> str:
        """Update status from remaining quantity and expiry (does not change dates)."""
        as_of_date = parse_date(as_of)
        if self.quantity <= 0:
            self.status = STATUS_DEPLETED
        elif self.expiry_date < as_of_date:
            self.status = STATUS_EXPIRED
        else:
            self.status = STATUS_AVAILABLE
        return self.status

    def is_available(self, as_of: DateLike) -> bool:
        """True when the batch still has units and has not expired as of ``as_of``."""
        self.refresh_status(as_of)
        return self.status == STATUS_AVAILABLE and self.quantity > 0

    def to_dict(self) -> dict:
        """Plain dict for tests, later dashboard, and demo output."""
        return {
            "batch_id": self.batch_id,
            "blood_group": self.blood_group,
            "collection_date": self.collection_date.isoformat(),
            "expiry_date": self.expiry_date.isoformat(),
            "quantity": self.quantity,
            "status": self.status,
        }


# Readable alias: each PlateletUnit row is one inventory batch.
PlateletBatch = PlateletUnit
