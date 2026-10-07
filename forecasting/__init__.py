"""Demand forecasting for synthetic platelet inventory experiments."""

from forecasting.data_loader import DemandDataError, load_demand_data
from forecasting.evaluate import evaluate_models, mae, mape, rmse, select_best_model
from forecasting.model_registry import get_model, load_registry, save_registry
from forecasting.preprocessing import chronological_split, preprocess_demand
from forecasting.sarima import fit_sarima
from forecasting.sma import fit_sma
from forecasting.train import train_models
from forecasting.xgboost_model import fit_xgboost

__all__ = [
    "DemandDataError",
    "evaluate_models",
    "fit_sarima",
    "fit_sma",
    "fit_xgboost",
    "get_model",
    "load_demand_data",
    "load_registry",
    "mae",
    "mape",
    "chronological_split",
    "preprocess_demand",
    "rmse",
    "save_registry",
    "select_best_model",
    "train_models",
]
