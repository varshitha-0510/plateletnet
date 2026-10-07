"""Tests for Phase 1 synthetic demand forecasting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecasting.data_loader import DEFAULT_DEMAND_CSV, DemandDataError, load_demand_data
from forecasting.evaluate import evaluate_models, mae, mape, rmse, select_best_model
from forecasting.model_registry import load_registry
from forecasting.preprocessing import aggregate_daily_demand, chronological_split, preprocess_demand
from forecasting.sarima import fit_sarima, forecast_sarima, forecast_sarima_by_group, fit_sarima_by_group
from forecasting.sma import forecast_sma_from_history, fit_sma, sma_forecast_frame
from forecasting.train import train_models
from forecasting.xgboost_model import fit_xgboost, forecast_xgboost


def _tiny_demand_frame(start: str = "2024-01-01", days: int = 60) -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="D")
    rows: list[dict] = []
    for blood_group, base in (("O+", 12.0), ("A+", 8.0)):
        for date in dates:
            seasonal = 1.0 + 0.05 * np.sin(2 * np.pi * date.dayofyear / 365.25)
            weekday = 1.15 if date.dayofweek < 5 else 0.8
            demand = max(1.0, base * seasonal * weekday + (date.dayofyear % 5) * 0.2)
            condition = "high" if demand > base * 1.1 else "normal"
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "blood_group": blood_group,
                    "platelet_demand_units": demand,
                    "demand_condition": condition,
                    "day_of_week": date.day_name(),
                    "seasonal_factor": seasonal,
                }
            )
    return pd.DataFrame(rows)


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.write_text(df.to_csv(index=False), encoding="utf-8")
    return path


def test_load_demand_validates_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("date,blood_group\n2024-01-01,O+\n", encoding="utf-8")
    with pytest.raises(DemandDataError, match="missing required columns"):
        load_demand_data(csv_path)


def test_load_demand_cleans_invalid_values(tmp_path: Path) -> None:
    df = _tiny_demand_frame(days=10).astype(
        {
            "date": "object",
            "platelet_demand_units": "object",
            "seasonal_factor": "object",
            "blood_group": "object",
            "day_of_week": "object",
        }
    )
    df.loc[0, "date"] = "not-a-date"
    df.loc[1, "platelet_demand_units"] = -5
    df.loc[2, "day_of_week"] = ""
    df.loc[3, "seasonal_factor"] = ""
    df.loc[4, "blood_group"] = ""
    csv_path = _write_csv(tmp_path / "messy.csv", df)
    cleaned = load_demand_data(csv_path)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    assert (cleaned["platelet_demand_units"] >= 0).all()
    assert cleaned["blood_group"].ne("").all()
    assert cleaned["day_of_week"].isin(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ).all()
    assert (cleaned["seasonal_factor"] > 0).all()


def test_missing_file_raises() -> None:
    with pytest.raises(DemandDataError, match="not found"):
        load_demand_data(Path("data/raw/does_not_exist.csv"))


def test_aggregate_and_features(tmp_path: Path) -> None:
    raw = _tiny_demand_frame(days=30)
    daily, features = preprocess_demand(raw, processed_dir=tmp_path)
    assert (tmp_path / "daily_demand.csv").exists()
    assert (tmp_path / "forecast_features.csv").exists()
    assert set(["date", "blood_group", "platelet_demand_units"]).issubset(daily.columns)
    assert features["lag_1"].notna().all()
    assert features["lag_7"].notna().all()
    assert set(daily["blood_group"]) == {"O+", "A+"}


def test_preprocessing_sorts_and_splits_chronologically() -> None:
    raw = _tiny_demand_frame(days=20)
    daily = aggregate_daily_demand(raw)
    assert daily.groupby("blood_group")["date"].apply(lambda s: s.is_monotonic_increasing).all()
    train, test = chronological_split(daily, test_days=5)
    assert train["date"].max() < test["date"].min()
    assert test["date"].nunique() == 5


def test_aggregate_without_blood_group_split() -> None:
    raw = _tiny_demand_frame(days=5)
    daily = aggregate_daily_demand(raw, by_blood_group=False)
    assert list(daily["blood_group"].unique()) == ["ALL"]
    assert len(daily) == 5


def test_sma_forecast_is_nonnegative_constant() -> None:
    history = np.array([10.0, 12.0, 8.0, 11.0, 9.0, 13.0, 7.0], dtype=float)
    model = fit_sma(history, window=7, blood_group="O+")
    preds = forecast_sma_from_history(history, horizon=5, window=7)
    assert preds.shape == (5,)
    assert np.allclose(preds, model.forecast_level)
    assert (preds >= 0).all()


def test_sma_forecast_returns_dataframe() -> None:
    history = np.array([10.0, 12.0, 8.0, 11.0, 9.0, 13.0, 7.0], dtype=float)
    actual = np.array([11.0, 10.0, 12.0, 9.0, 8.0])
    dates = pd.date_range("2024-02-01", periods=5, freq="D")
    frame = sma_forecast_frame(
        history,
        horizon=5,
        window=3,
        blood_group="O+",
        dates=dates,
        actual=actual,
    )
    assert list(frame.columns) == ["date", "blood_group", "model", "prediction", "actual"]
    assert len(frame) == 5
    expected = float(np.mean(history[-3:]))
    assert np.allclose(frame["prediction"], expected)
    assert np.allclose(frame["actual"], actual)


def test_metrics_known_values() -> None:
    actual = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 18.0, 33.0])
    assert mae(actual, pred) == pytest.approx(7.0 / 3.0)
    assert rmse(actual, pred) == pytest.approx(float(np.sqrt((4 + 4 + 9) / 3)))
    assert mape(actual, pred) == pytest.approx((0.2 + 0.1 + 0.1) / 3 * 100)


def test_select_model_lowest_mae() -> None:
    metrics = evaluate_models(
        np.array([1.0, 2.0, 3.0]),
        {
            "SMA": np.array([2.0, 3.0, 4.0]),
            "XGBoost": np.array([1.0, 2.0, 3.5]),
            "SARIMA": np.array([5.0, 5.0, 5.0]),
        },
    )
    assert select_best_model(metrics) == "XGBoost"
    assert list(metrics.sort_values("MAE")["model"])[0] == "XGBoost"


def test_load_project_synthetic_csv() -> None:
    assert DEFAULT_DEMAND_CSV.exists()
    loaded = load_demand_data(DEFAULT_DEMAND_CSV)
    required = {
        "date",
        "blood_group",
        "platelet_demand_units",
        "demand_condition",
        "day_of_week",
        "seasonal_factor",
    }
    assert required.issubset(loaded.columns)
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
    assert loaded["blood_group"].nunique() == 8
    assert (loaded["platelet_demand_units"] >= 0).all()


def test_sarima_forecasts_nonnegative_series() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(60)
    history = 10.0 + 2.0 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.3, size=60)
    model = fit_sarima(history, blood_group="O+")
    preds = forecast_sarima(model, horizon=7)
    assert preds.shape == (7,)
    assert np.all(np.isfinite(preds))
    assert (preds >= 0).all()


def test_sarima_short_series_uses_fallback() -> None:
    model = fit_sarima(np.array([4.0, 5.0, 6.0], dtype=float), blood_group="B+")
    preds = forecast_sarima(model, horizon=3)
    assert model.result is None
    assert np.allclose(preds, model.fallback_level)


def test_sarima_by_group_covers_horizon() -> None:
    raw = _tiny_demand_frame(days=40)
    daily = aggregate_daily_demand(raw)
    train_end = daily["date"].max() - pd.Timedelta(days=5)
    horizon = pd.DatetimeIndex(sorted(daily["date"].unique())[-5:])
    model = fit_sarima_by_group(daily, train_end=train_end)
    preds = forecast_sarima_by_group(model, horizon)
    assert set(preds["blood_group"]) == {"O+", "A+"}
    assert len(preds) == 10
    assert (preds["prediction"] >= 0).all()


def test_xgboost_learns_on_tiny_panel() -> None:
    from forecasting.preprocessing import create_forecast_features

    raw = _tiny_demand_frame(days=40)
    daily = aggregate_daily_demand(raw)
    features = create_forecast_features(daily)
    train = features.iloc[:-5]
    model = fit_xgboost(train, n_estimators=20)
    horizon = pd.DatetimeIndex(sorted(daily["date"].unique())[-5:])
    history = daily.loc[daily["date"] < horizon.min()]
    preds = forecast_xgboost(model, history, horizon)
    assert len(preds) == 5 * daily["blood_group"].nunique()
    assert (preds["prediction"] >= 0).all()


def test_train_pipeline_writes_artifacts(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "platelet_demand.csv", _tiny_demand_frame(days=50))
    processed = tmp_path / "processed"
    results = tmp_path / "forecasts"
    result = train_models(
        csv_path=csv_path,
        processed_dir=processed,
        results_dir=results,
        test_days=7,
        sma_window=5,
        xgb_estimators=25,
    )
    assert result["selected_model"] in {"SMA", "XGBoost", "SARIMA"}
    assert (processed / "daily_demand.csv").exists()
    assert (processed / "forecast_features.csv").exists()
    assert (results / "metrics.csv").exists()
    assert (results / "model_comparison.csv").exists()
    assert (results / "sma_forecast.csv").exists()
    assert (results / "xgboost_forecast.csv").exists()
    assert (results / "sarima_forecast.csv").exists()
    assert (results / "test_forecasts.csv").exists()
    assert (results / "selected_model.json").exists()
    registry = load_registry(results)
    assert registry["selected_model"] == result["selected_model"]
    assert {"MAE", "RMSE", "MAPE"}.issubset(result["metrics"].columns)
    forecasts = result["forecasts"]
    assert not forecasts[["SMA", "XGBoost", "SARIMA"]].isna().any().any()
