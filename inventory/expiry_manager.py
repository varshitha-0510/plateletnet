"""Expiry tracking for synthetic platelet batches.

Detects expired and near-expiry batches relative to a reference date.
This module does not change expiry dates and does not implement
Simulated Micro-Expiry (that is a later research-only phase).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from inventory.platelet_unit import DateLike, PlateletUnit, parse_date

ALERT_EXPIRED = "expired"
ALERT_NEAR_EXPIRY = "near_expiry"


@dataclass
class ExpiryAlert:
    """One expiry warning for a synthetic batch."""

    batch_id: str
    blood_group: str
    expiry_date: date
    quantity: int
    alert_type: str
    days_until_expiry: int

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "blood_group": self.blood_group,
            "expiry_date": self.expiry_date.isoformat(),
            "quantity": self.quantity,
            "alert_type": self.alert_type,
            "days_until_expiry": self.days_until_expiry,
        }


def is_expired(batch: PlateletUnit, as_of: DateLike) -> bool:
    """True when ``expiry_date`` is strictly before the reference date."""
    as_of_date = parse_date(as_of)
    return batch.expiry_date < as_of_date


def is_near_expiry(
    batch: PlateletUnit,
    as_of: DateLike,
    warning_days: int,
) -> bool:
    """True when the batch is still usable and expires within ``warning_days``."""
    if warning_days < 0:
        raise ValueError("warning_days must be >= 0.")
    as_of_date = parse_date(as_of)
    if is_expired(batch, as_of_date):
        return False
    if batch.quantity <= 0:
        return False
    days_left = (batch.expiry_date - as_of_date).days
    return 0 <= days_left <= warning_days


def expired_batches(
    batches: list[PlateletUnit],
    as_of: DateLike,
) -> list[PlateletUnit]:
    """Return batches whose expiry_date is before ``as_of`` (quantity may remain)."""
    as_of_date = parse_date(as_of)
    found: list[PlateletUnit] = []
    for batch in batches:
        batch.refresh_status(as_of_date)
        if is_expired(batch, as_of_date):
            found.append(batch)
    return found


def near_expiry_batches(
    batches: list[PlateletUnit],
    as_of: DateLike,
    warning_days: int,
) -> list[PlateletUnit]:
    """Return still-usable batches that expire within the warning window."""
    as_of_date = parse_date(as_of)
    return [
        batch
        for batch in batches
        if is_near_expiry(batch, as_of_date, warning_days=warning_days)
    ]


def get_expiry_alerts(
    batches: list[PlateletUnit],
    as_of: DateLike,
    warning_days: int,
) -> list[ExpiryAlert]:
    """Build expired and near-expiry alerts (does not modify expiry dates)."""
    as_of_date = parse_date(as_of)
    alerts: list[ExpiryAlert] = []
    for batch in batches:
        batch.refresh_status(as_of_date)
        if batch.quantity <= 0:
            continue
        days_left = (batch.expiry_date - as_of_date).days
        if is_expired(batch, as_of_date):
            alerts.append(
                ExpiryAlert(
                    batch_id=batch.batch_id,
                    blood_group=batch.blood_group,
                    expiry_date=batch.expiry_date,
                    quantity=batch.quantity,
                    alert_type=ALERT_EXPIRED,
                    days_until_expiry=days_left,
                )
            )
        elif is_near_expiry(batch, as_of_date, warning_days=warning_days):
            alerts.append(
                ExpiryAlert(
                    batch_id=batch.batch_id,
                    blood_group=batch.blood_group,
                    expiry_date=batch.expiry_date,
                    quantity=batch.quantity,
                    alert_type=ALERT_NEAR_EXPIRY,
                    days_until_expiry=days_left,
                )
            )
    alerts.sort(key=lambda a: (a.expiry_date, a.batch_id))
    return alerts


def update_simulated_expiry(*args, **kwargs):
    """Reserved for a later Simulated Micro-Expiry research phase.

    Phase 2 does not change or extend expiry dates.
    """
    raise NotImplementedError(
        "Simulated Micro-Expiry is not part of Phase 2. "
        "Expiry dates are not modified."
    )
