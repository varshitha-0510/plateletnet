"""First-in, first-out issue order for synthetic platelet batches.

FIFO uses collection_date (oldest first) within a blood group.
Expired and empty batches are never selected for consumption.
"""

from __future__ import annotations

from datetime import date

from inventory.platelet_unit import DateLike, PlateletUnit, parse_date


def select_fifo(
    batches: list[PlateletUnit],
    blood_group: str | None = None,
    as_of: DateLike | None = None,
) -> list[PlateletUnit]:
    """Return available batches in FIFO order (oldest collection_date first).

    Parameters
    ----------
    batches
        Current inventory batches.
    blood_group
        If set, only that blood group is included.
    as_of
        Reference date for expiry. Defaults to today.
    """
    as_of_date = parse_date(as_of) if as_of is not None else date.today()
    eligible: list[PlateletUnit] = []
    for batch in batches:
        if blood_group is not None and batch.blood_group != blood_group:
            continue
        if batch.is_available(as_of_date):
            eligible.append(batch)
    eligible.sort(key=lambda b: (b.collection_date, b.batch_id))
    return eligible


def consume_fifo(
    batches: list[PlateletUnit],
    blood_group: str,
    quantity: int,
    as_of: DateLike | None = None,
) -> list[dict]:
    """Consume ``quantity`` units from ``batches`` using FIFO.

    Mutates matching batches in place. Raises ``ValueError`` if there is
    not enough available stock of that blood group.
    """
    if quantity <= 0:
        raise ValueError("quantity to consume must be a positive integer.")
    as_of_date = parse_date(as_of) if as_of is not None else date.today()
    ordered = select_fifo(batches, blood_group=blood_group, as_of=as_of_date)
    available = sum(batch.quantity for batch in ordered)
    if quantity > available:
        raise ValueError(
            f"Cannot consume {quantity} units of {blood_group}: "
            f"only {available} available (expired/empty batches excluded)."
        )

    remaining = quantity
    events: list[dict] = []
    for batch in ordered:
        if remaining == 0:
            break
        taken = min(batch.quantity, remaining)
        batch.quantity -= taken
        remaining -= taken
        batch.refresh_status(as_of_date)
        events.append(
            {
                "batch_id": batch.batch_id,
                "blood_group": batch.blood_group,
                "quantity_consumed": taken,
                "remaining_quantity": batch.quantity,
                "status": batch.status,
            }
        )
    return events
