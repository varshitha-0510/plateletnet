"""Dashboard API for the local academic demonstration."""

from fastapi import APIRouter

from backend.services.dashboard_service import get_dashboard
from backend.services.forecast_service import get_forecast, get_forecast_models
from backend.services.inventory_service import get_inventory
from backend.services.jit_service import recommend_jit

router = APIRouter()


@router.get("/dashboard")
def dashboard():
    """Return Phase 1 forecast + Phase 2 inventory + Phase 3 JIT in one payload."""
    return get_dashboard()


@router.get("/forecast")
def forecast():
    """Return the selected Phase 1 forecast summary."""
    return get_forecast()


@router.get("/forecast/models")
def forecast_models():
    """Return SMA, XGBoost, and SARIMA series from Phase 1 result files."""
    return get_forecast_models()


@router.get("/inventory")
def inventory():
    """Return Phase 2 synthetic demo inventory, FIFO order, and expiry alerts."""
    return get_inventory()


@router.get("/jit")
def jit():
    """Return Phase 3 JIT recommendations."""
    return recommend_jit()
