"""Application configuration for PlateletNet.

Placeholder settings for later stages. No database connections or
model training are performed here.

Safety: Simulated Micro-Expiry is research/demo only and must never be
presented as a real medical expiry extension.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
RESULTS_DIR = BASE_DIR / "results"
FORECAST_RESULTS_DIR = RESULTS_DIR / "forecasts"
SIMULATION_RESULTS_DIR = RESULTS_DIR / "simulations"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

APP_NAME = os.getenv("PLATELETNET_APP_NAME", "PlateletNet")
ENV = os.getenv("PLATELETNET_ENV", "development")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./plateletnet.db")
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Demo/research flags — not clinical controls.
USE_SYNTHETIC_DATA_ONLY = os.getenv("USE_SYNTHETIC_DATA_ONLY", "true").lower() == "true"
SIMULATED_MICRO_EXPIRY_ENABLED = (
    os.getenv("SIMULATED_MICRO_EXPIRY_ENABLED", "false").lower() == "true"
)
MICRO_EXPIRY_LABEL = "Simulated Micro-Expiry — Research/Demo Only"

BLOOD_GROUPS = ("O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-")
