"""Tests for Phase 3 JIT recommendation engine (synthetic/demo only)."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inventory.inventory_manager import InventoryManager
from inventory.platelet_unit import PlateletUnit
from policies.jit import (
    JITRecommendation,
    apply_jit_policy,
    expected_demand_by_group,
    load_phase1_forecast,
    recommend_jit,
)

AS_OF = date(2026, 1, 10)


def _forecast(
    rows: list[tuple[str, str, float]],
    model: str = "XGBoost",
) -> pd.DataFrame:
    """Build a Phase 1 long-format forecast table (date, blood_group, prediction)."""
    records = [
        {
            "date": day,
            "blood_group": group,
            "model": model,
            "prediction": prediction,
            "actual": prediction,
        }
        for day, group, prediction in rows
    ]
    return pd.DataFrame(records)


def _batch(
    batch_id: str,
    blood_group: str,
    quantity: int,
    collection: str = "2026-01-01",
    expiry: str = "2026-01-20",
) -> PlateletUnit:
    return PlateletUnit(
        batch_id=batch_id,
        blood_group=blood_group,
        collection_date=collection,
        expiry_date=expiry,
        quantity=quantity,
    )


def _manager(*batches: PlateletUnit) -> InventoryManager:
    manager = InventoryManager(
        low_stock_threshold=10,
        near_expiry_days=2,
        reference_date=AS_OF,
    )
    for batch in batches:
        manager.add_batch(batch)
    return manager


def test_sufficient_inventory_produces_zero_recommendation() -> None:
    forecast = _forecast(
        [
            ("2026-01-10", "B+", 4.0),
            ("2026-01-11", "B+", 6.0),
        ]
    )
    manager = _manager(_batch("B1", "B+", quantity=20))
    result = recommend_jit(manager, forecast, horizon_days=2, as_of=AS_OF)[0]
    assert result.blood_group == "B+"
    assert result.expected_demand == 10.0
    assert result.available_inventory == 20
    assert result.projected_shortage == 0.0
    assert result.recommended_quantity == 0.0
    assert result.status == "sufficient"


def test_insufficient_inventory_produces_correct_shortage() -> None:
    forecast = _forecast(
        [
            ("2026-01-10", "A+", 10.0),
            ("2026-01-11", "A+", 15.0),
        ]
    )
    manager = _manager(_batch("A1", "A+", quantity=15))
    result = recommend_jit(manager, forecast, horizon_days=2, as_of=AS_OF)[0]
    assert result.expected_demand == 25.0
    assert result.available_inventory == 15
    assert result.projected_shortage == 10.0
    assert result.recommended_quantity == 10.0
    assert result.status == "shortage"


def test_recommendation_is_never_negative() -> None:
    forecast = _forecast([("2026-01-10", "O+", 8.0)])
    manager = _manager(_batch("O1", "O+", quantity=50))
    result = recommend_jit(manager, forecast, horizon_days=1, as_of=AS_OF)[0]
    assert result.recommended_quantity >= 0
    assert result.projected_shortage >= 0
    assert result.recommended_quantity == 0.0


def test_blood_groups_are_evaluated_independently() -> None:
    forecast = _forecast(
        [
            ("2026-01-10", "A+", 12.0),
            ("2026-01-10", "O+", 12.0),
        ]
    )
    manager = _manager(
        _batch("A1", "A+", quantity=12),
        _batch("O1", "O+", quantity=3),
    )
    by_group = {row.blood_group: row for row in recommend_jit(manager, forecast, horizon_days=1)}
    assert by_group["A+"].status == "sufficient"
    assert by_group["A+"].recommended_quantity == 0.0
    assert by_group["O+"].status == "shortage"
    assert by_group["O+"].recommended_quantity == 9.0
    assert by_group["A+"].available_inventory == 12
    assert by_group["O+"].available_inventory == 3


def test_expired_inventory_is_excluded() -> None:
    forecast = _forecast([("2026-01-10", "A+", 10.0)])
    manager = _manager(
        _batch("LIVE", "A+", quantity=4, collection="2026-01-05", expiry="2026-01-15"),
        _batch("EXP", "A+", quantity=40, collection="2025-12-01", expiry="2026-01-09"),
    )
    result = recommend_jit(manager, forecast, horizon_days=1, as_of=AS_OF)[0]
    assert result.available_inventory == 4
    assert result.recommended_quantity == 6.0
    assert result.status == "shortage"


def test_depleted_inventory_is_excluded() -> None:
    forecast = _forecast([("2026-01-10", "A+", 5.0)])
    manager = _manager(
        _batch("LIVE", "A+", quantity=5, collection="2026-01-04"),
        _batch("EMPTY", "A+", quantity=0, collection="2026-01-02"),
    )
    result = recommend_jit(manager, forecast, horizon_days=1, as_of=AS_OF)[0]
    assert result.available_inventory == 5
    assert result.recommended_quantity == 0.0
    assert result.status == "sufficient"


def test_forecast_horizon_is_respected() -> None:
    forecast = _forecast(
        [
            ("2026-01-10", "A+", 10.0),
            ("2026-01-11", "A+", 20.0),
            ("2026-01-12", "A+", 40.0),
        ]
    )
    one = expected_demand_by_group(forecast, horizon_days=1)
    two = expected_demand_by_group(forecast, horizon_days=2)
    three = expected_demand_by_group(forecast, horizon_days=3)
    assert one["A+"] == 10.0
    assert two["A+"] == 30.0
    assert three["A+"] == 70.0

    manager = _manager(_batch("A1", "A+", quantity=0))
    rec_one = recommend_jit(manager, forecast, horizon_days=1, as_of=AS_OF)[0]
    rec_two = recommend_jit(manager, forecast, horizon_days=2, as_of=AS_OF)[0]
    assert rec_one.forecast_horizon == 1
    assert rec_two.forecast_horizon == 2
    assert rec_one.expected_demand == 10.0
    assert rec_two.expected_demand == 30.0
    assert rec_one.recommended_quantity != rec_two.recommended_quantity


def test_expected_demand_is_calculated_from_forecast_predictions() -> None:
    forecast = _forecast(
        [
            ("2026-01-10", "AB+", 1.5),
            ("2026-01-11", "AB+", 2.25),
            ("2026-01-12", "AB+", 99.0),
        ]
    )
    demand = expected_demand_by_group(forecast, horizon_days=2)
    assert demand["AB+"] == pytest.approx(3.75)
    assert 99.0 not in demand.values()


def test_multi_blood_group_recommendation_produces_separate_results() -> None:
    forecast = _forecast(
        [
            ("2026-01-10", "A+", 25.0),
            ("2026-01-10", "B+", 10.0),
            ("2026-01-11", "A+", 0.0),
            ("2026-01-11", "B+", 0.0),
        ]
    )
    manager = _manager(
        _batch("A1", "A+", quantity=15),
        _batch("B1", "B+", quantity=20),
    )
    results = recommend_jit(manager, forecast, horizon_days=2, as_of=AS_OF)
    by_group = {row.blood_group: row for row in results}
    assert set(by_group) == {"A+", "B+"}
    assert by_group["A+"].status == "shortage"
    assert by_group["B+"].status == "sufficient"
    assert all(isinstance(row, JITRecommendation) for row in results)
    assert [row.blood_group for row in results] == ["A+", "B+"]


def test_jit_engine_loads_existing_phase1_forecast_output(tmp_path: Path) -> None:
    """Use the real Phase 1 CSV schema and selected_model.json layout."""
    forecast = pd.DataFrame(
        {
            "date": ["2024-12-04", "2024-12-04", "2024-12-05", "2024-12-05"],
            "blood_group": ["A+", "B+", "A+", "B+"],
            "model": ["XGBoost", "XGBoost", "XGBoost", "XGBoost"],
            "prediction": [17.42213249206543, 13.19157600402832, 18.362457275390625, 12.932600975036621],
            "actual": [18.0, 14.0, 23.0, 13.0],
        }
    )
    forecast.to_csv(tmp_path / "xgboost_forecast.csv", index=False)
    (tmp_path / "selected_model.json").write_text(
        json.dumps(
            {
                "selected_model": "XGBoost",
                "selection_rule": "lowest_MAE",
                "data_note": "synthetic/demo demand only — not a clinical system",
            }
        ),
        encoding="utf-8",
    )

    loaded, model_name = load_phase1_forecast(tmp_path)
    assert model_name == "XGBoost"
    assert list(loaded.columns)[:3] == ["date", "blood_group", "model"]
    assert "prediction" in loaded.columns

    demand = expected_demand_by_group(loaded, horizon_days=2)
    assert demand["A+"] == pytest.approx(17.42213249206543 + 18.362457275390625)
    assert demand["B+"] == pytest.approx(13.19157600402832 + 12.932600975036621)

    manager = _manager(
        _batch("A1", "A+", quantity=10),
        _batch("B1", "B+", quantity=50),
    )
    results = recommend_jit(
        manager,
        horizon_days=2,
        results_dir=tmp_path,
        as_of=AS_OF,
    )
    by_group = {row.blood_group: row for row in results}
    assert by_group["A+"].selected_model == "XGBoost"
    assert by_group["A+"].status == "shortage"
    assert by_group["B+"].status == "sufficient"
    assert by_group["A+"].recommended_quantity == pytest.approx(demand["A+"] - 10)
    assert by_group["B+"].recommended_quantity == 0.0


def test_apply_jit_policy_matches_recommend_jit() -> None:
    forecast = _forecast([("2026-01-10", "O-", 3.0)])
    manager = _manager(_batch("N1", "O-", quantity=1))
    direct = recommend_jit(manager, forecast, horizon_days=1, as_of=AS_OF)
    aliased = apply_jit_policy(manager, forecast, horizon_days=1, as_of=AS_OF)
    assert [row.to_dict() for row in direct] == [row.to_dict() for row in aliased]
