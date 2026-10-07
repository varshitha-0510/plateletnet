"""Load and validate synthetic platelet demand CSVs.

This module reads demo/research demand data only. It does not handle
real patient or clinical records.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import BLOOD_GROUPS, RAW_DATA_DIR

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "date",
    "blood_group",
    "platelet_demand_units",
    "demand_condition",
    "day_of_week",
    "seasonal_factor",
)

DEFAULT_DEMAND_CSV = RAW_DATA_DIR / "platelet_demand.csv"
VALID_DAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


class DemandDataError(ValueError):
    """Raised when the demand CSV cannot be loaded or validated."""


def load_demand_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load platelet demand from CSV and return a cleaned DataFrame.

    Parameters
    ----------
    csv_path
        Path to the demand CSV. Defaults to ``data/raw/platelet_demand.csv``.

    Returns
    -------
    pandas.DataFrame
        Cleaned demand table with a datetime ``date`` column.

    Raises
    ------
    DemandDataError
        If the file is missing, unreadable, or lacks required columns.
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_DEMAND_CSV
    if not path.exists():
        raise DemandDataError(
            f"Demand CSV not found: {path}. Expected synthetic demo data at "
            "data/raw/platelet_demand.csv."
        )

    try:
        df = pd.read_csv(path)
    except OSError as exc:
        raise DemandDataError(f"Could not read demand CSV {path}: {exc}") from exc
    except pd.errors.ParserError as exc:
        raise DemandDataError(f"Could not parse demand CSV {path}: {exc}") from exc

    if df.empty:
        raise DemandDataError(f"Demand CSV is empty: {path}")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DemandDataError(
            f"Demand CSV is missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    cleaned = _clean_demand_frame(df)
    if cleaned.empty:
        raise DemandDataError(
            f"No valid demand rows remained after cleaning {path}."
        )

    logger.info("Loaded %s demand rows from %s", len(cleaned), path)
    return cleaned.reset_index(drop=True)


def _clean_demand_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, repair recoverable issues, and drop invalid rows."""
    out = df.loc[:, list(REQUIRED_COLUMNS)].copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce", format="mixed")
    invalid_dates = int(out["date"].isna().sum())
    if invalid_dates:
        logger.warning("Dropping %s rows with invalid dates", invalid_dates)
        out = out.loc[out["date"].notna()]

    out["blood_group"] = (
        out["blood_group"].astype("string").str.strip().replace({"": pd.NA})
    )
    unknown_groups = sorted(
        set(out["blood_group"].dropna().unique()) - set(BLOOD_GROUPS)
    )
    if unknown_groups:
        logger.warning("Demand file contains unexpected blood groups: %s", unknown_groups)

    missing_groups = int(out["blood_group"].isna().sum())
    if missing_groups:
        logger.warning("Dropping %s rows with missing blood_group", missing_groups)
        out = out.loc[out["blood_group"].notna()]

    out["platelet_demand_units"] = pd.to_numeric(
        out["platelet_demand_units"], errors="coerce"
    )
    negative = out["platelet_demand_units"] < 0
    if negative.any():
        logger.warning(
            "Replacing %s negative demand values with NA for interpolation",
            int(negative.sum()),
        )
        out.loc[negative, "platelet_demand_units"] = pd.NA

    out["seasonal_factor"] = pd.to_numeric(out["seasonal_factor"], errors="coerce")
    out["seasonal_factor"] = out["seasonal_factor"].fillna(1.0)
    out.loc[out["seasonal_factor"] <= 0, "seasonal_factor"] = 1.0

    out["demand_condition"] = (
        out["demand_condition"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"": pd.NA})
        .fillna("unknown")
    )

    out["day_of_week"] = out["day_of_week"].astype("string").str.strip()
    derived = out["date"].dt.day_name()
    invalid_dow = ~out["day_of_week"].isin(VALID_DAY_NAMES)
    if invalid_dow.any():
        logger.warning(
            "Replacing %s missing/invalid day_of_week values from the date",
            int(invalid_dow.sum()),
        )
        out.loc[invalid_dow, "day_of_week"] = derived.loc[invalid_dow]

    out = out.sort_values(["blood_group", "date"])
    out["platelet_demand_units"] = out.groupby("blood_group")["platelet_demand_units"].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    still_missing = int(out["platelet_demand_units"].isna().sum())
    if still_missing:
        logger.warning("Dropping %s rows with unrecoverable demand values", still_missing)
        out = out.loc[out["platelet_demand_units"].notna()]

    out["platelet_demand_units"] = out["platelet_demand_units"].clip(lower=0)
    return out
