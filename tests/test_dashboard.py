"""Tests for the local research dashboard integration layer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import create_app
from backend.services.dashboard_service import get_dashboard
from backend.services.forecast_service import get_forecast
from backend.services.inventory_service import get_inventory
from backend.services.jit_service import recommend_jit
from inventory.demo_loader import (
    DEFAULT_DEMO_INVENTORY_CSV,
    load_demo_inventory,
    load_demo_inventory_config,
)
from policies.jit import expected_demand_by_group, load_phase1_forecast, recommend_jit as jit_engine

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def test_demo_inventory_loads_phase2_manager() -> None:
    config = load_demo_inventory_config()
    manager = load_demo_inventory()
    as_of = config["as_of"]
    assert DEFAULT_DEMO_INVENTORY_CSV.exists()
    assert manager.available_units("O+", as_of=as_of) == 180
    assert manager.available_units("O-", as_of=as_of) == 8
    assert manager.available_units("AB-", as_of=as_of) == 2
    expired = {row["batch_id"] for row in manager.expiry_alerts(as_of=as_of) if row["alert_type"] == "expired"}
    assert "OPLUS-EXP" in expired
    assert "AMINUS-EXP" in expired
    fifo_ids = [batch.batch_id for batch in manager.available_batches("O+", as_of=as_of)]
    assert fifo_ids == ["OPLUS-A", "OPLUS-B"]


def test_forecast_service_loads_phase1_artifacts() -> None:
    config = load_demo_inventory_config()
    payload = get_forecast(horizon_days=int(config["horizon_days"]), start_date=config["as_of"])
    assert payload["selected_model"]
    assert payload["label"] == "Synthetic / Research Forecast"
    assert payload["horizon_days"] == 7
    assert payload["by_blood_group"]
    groups = {row["blood_group"] for row in payload["by_blood_group"]}
    assert {"O+", "A+", "AB-"}.issubset(groups)


def test_forecast_service_exposes_all_three_phase1_models() -> None:
    from forecasting.model_registry import load_registry

    registry = load_registry()
    payload = get_forecast()
    names = [model["name"] for model in payload["models"]]
    assert names == ["SMA", "XGBoost", "SARIMA"]
    by_name = {model["name"]: model for model in payload["models"]}
    assert by_name[payload["selected_model"]]["selected"] is True
    assert by_name["SMA"]["series"]
    assert by_name["XGBoost"]["series"]
    assert by_name["SARIMA"]["series"]
    assert "actual" in by_name["SARIMA"]["series"][0]
    assert "prediction" in by_name["SARIMA"]["series"][0]
    comparison = {row["model"]: row for row in payload["comparison"]}
    for row in registry["metrics"]:
        assert comparison[row["model"]]["MAE"] == pytest.approx(row["MAE"])
        assert comparison[row["model"]]["RMSE"] == pytest.approx(row["RMSE"])
        assert comparison[row["model"]]["MAPE"] == pytest.approx(row["MAPE"])
        assert comparison[row["model"]]["selected"] is (row["model"] == registry["selected_model"])


def test_inventory_service_uses_phase2_classes() -> None:
    payload = get_inventory()
    assert payload["total_available"] > 0
    assert len(payload["by_blood_group"]) >= 8
    assert payload["expiry_alerts"]
    assert payload["fifo_order"]
    assert "OPLUS-EXP" not in {row["batch_id"] for row in payload["fifo_order"]}


def test_jit_service_matches_phase3_engine() -> None:
    config = load_demo_inventory_config()
    manager = load_demo_inventory()
    service = recommend_jit()
    engine = jit_engine(
        manager,
        horizon_days=int(config["horizon_days"]),
        as_of=config["as_of"],
        start_date=config["as_of"],
    )
    by_service = {row["blood_group"]: row for row in service["recommendations"]}
    by_engine = {row.blood_group: row.to_dict() for row in engine}
    assert set(by_service) == set(by_engine)
    for group, row in by_engine.items():
        assert by_service[group]["available_inventory"] == row["available_inventory"]
        assert by_service[group]["status"] == row["status"]
        assert by_service[group]["recommended_quantity"] == pytest.approx(row["recommended_quantity"])


def test_dashboard_payload_structure() -> None:
    payload = get_dashboard()
    assert payload["meta"]["clinical_system"] is False
    assert payload["meta"]["synthetic_data"] is True
    assert payload["meta"]["micro_expiry_implemented"] is False
    assert "forecast" in payload and "inventory" in payload and "jit" in payload
    assert payload["forecast"]["selected_model"]
    assert payload["inventory"]["by_blood_group"]
    assert payload["jit"]["recommendations"]
    example_hardcoded = {"blood_group": "A+", "expected_demand": 25, "available_inventory": 15}
    assert example_hardcoded not in payload["jit"]["recommendations"]


def test_dashboard_http_endpoints_return_live_structure() -> None:
    client = TestClient(create_app())
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["synthetic_data"] is True

    response = client.get("/api/dashboard")
    assert response.status_code == 200
    body = response.json()
    assert body["forecast"]["by_blood_group"]
    assert body["inventory"]["total_available"] == get_inventory()["total_available"]
    assert body["jit"]["recommendations"][0]["blood_group"]

    assert client.get("/api/forecast").status_code == 200
    models = client.get("/api/forecast/models")
    assert models.status_code == 200
    assert {row["name"] for row in models.json()["models"]} == {"SMA", "XGBoost", "SARIMA"}
    assert client.get("/api/inventory").status_code == 200
    assert client.get("/api/jit").status_code == 200


def test_frontend_does_not_hardcode_example_jit_numbers() -> None:
    html = (ROOT / "frontend" / "PlateletNet.html").read_text(encoding="utf-8")
    assert "/api/dashboard" in html
    assert "A+ | 25 | 15 | 10 | 10 | Shortage" not in html
    assert "fetch(\"/api/dashboard\")" in html
    assert 'data-view="dashboard"' in html
    assert 'data-view="inventory"' in html
    assert 'data-view="forecast"' in html
    assert 'data-view="jit"' in html
    assert 'data-view="alerts"' in html
    assert "SMA Forecast" in html
    assert "XGBoost Forecast" in html
    assert "SARIMA Forecast" in html
    assert "Research Prototype — Synthetic/Demo Data" in html
    assert "Not a clinical decision-support system." in html
    assert "showView" in html


def test_dashboard_forecast_demand_matches_phase1_file() -> None:
    config = load_demo_inventory_config()
    forecast, model = load_phase1_forecast()
    demand = expected_demand_by_group(
        forecast, horizon_days=int(config["horizon_days"]), start_date=config["as_of"]
    )
    payload = get_forecast(horizon_days=int(config["horizon_days"]), start_date=config["as_of"])
    by_group = {row["blood_group"]: row["expected_demand"] for row in payload["by_blood_group"]}
    assert payload["selected_model"] == model
    for group, value in demand.items():
        assert by_group[group] == pytest.approx(value)
