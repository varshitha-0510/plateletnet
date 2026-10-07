"""Run the complete PlateletNet forecasting pipeline on synthetic demand."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecasting.train import train_models


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = train_models()
    metrics = result["metrics"]
    print("PlateletNet forecasting (synthetic/demo data only)")
    print(metrics.to_string(index=False))
    print(f"Selected model: {result['selected_model']}")
    print(result["selection_reason"])
    print("Artifacts:")
    for name, path in result["paths"].items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
