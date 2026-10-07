# PlateletNet

Student/research software prototype for **AI-driven platelet inventory management**.

PlateletNet demonstrates a demand-driven software approach with **Just-In-Time (JIT) processing recommendations** and **simulated dynamic micro-expiry** in a blood-bank inventory setting. It is a research and teaching prototype, not a clinical or operational medical system.

## Project objective

Build a working demonstration application that shows:

1. AI-based platelet demand forecasting
2. Blood-group-wise platelet inventory management
3. Just-In-Time (JIT) processing recommendations
4. Simulated dynamic micro-expiry management
5. Shortage-risk detection
6. Wastage monitoring
7. Comparison of Traditional vs JIT-only vs JIT + Micro-Expiry policies
8. Simulation and analytics
9. Backend APIs
10. An interactive frontend dashboard

All demand, inventory, and outcome data used by this project is **synthetic/demo data**.

## Research motivation

Platelets have a short usable window compared with many other blood products, so mismatch between supply and demand can produce both **wastage** (units discarded at expiry) and **shortage** (unmet requests). Research on demand-driven inventory, delayed or just-in-time processing, and hypothetical approved shelf-life extensions explores whether software policy can reduce those mismatches.

This prototype is inspired by research on:

> Demand-Driven Software Approach with Dynamic Micro-Expiry and Just-In-Time Processing to Reduce Platelet Wastage in Blood Banks.

The software is meant to **illustrate and compare policies in simulation**, not to operate a real blood bank or to change medical product dating.

## Safety disclaimer

**This is not a clinical medical system.**

- Use **synthetic/demo data only**. Do not store or process real patient information.
- The application **cannot** and **must not** be used to extend the medical expiry date of platelets.
- **Micro-Expiry** is implemented only as a **simulated / research** feature representing a **hypothetical approved extension scenario**.
- Always label this capability as: **Simulated Micro-Expiry — Research/Demo Only**.
- Forecasts, JIT recommendations, shortage-risk scores, and wastage metrics are **demo analytics**. They are not clinical advice and are not validated for real-world transfusion or inventory decisions.

## Technology stack

| Area | Technologies |
| --- | --- |
| Backend | Python, FastAPI, SQLite |
| AI / data | Pandas, NumPy, scikit-learn, XGBoost, Statsmodels, SciPy |
| Visualization | Plotly (and Matplotlib for experiment figures) |
| Frontend | HTML, CSS, JavaScript |
| Testing | Pytest |
| Configuration | pydantic, python-dotenv |

See `requirements.txt` for the pinned package list used in development.

## System architecture

```
Frontend (PlateletNet.html)
        |
        v
FastAPI backend (backend/main.py)
        |
        +-- /forecast     forecasting services
        +-- /inventory    inventory services
        +-- /requests     request / shortage services
        +-- /simulation   policy simulation services
        |
        v
Domain packages
        +-- forecasting/   demand models (planned)
        +-- inventory/     units, FIFO, expiry tracking (planned)
        +-- policies/      traditional, JIT, simulated micro-expiry (planned)
        +-- simulation/    scenario runner (planned)
        +-- evaluation/    metrics, comparison, plots (planned)
        |
        v
SQLite (demo store) + data/synthetic + results/
```

This first development stage creates **folders and placeholders only**. Algorithms, APIs, database code, and dashboard behavior are not implemented yet.

## Inventory policies (planned)

### Traditional strategy

A baseline policy: process or hold platelet units according to conventional, relatively static inventory rules (for example, keep a target stock and issue oldest units first). This policy is the comparison baseline for wastage and shortage in simulation.

### JIT-only strategy

A **Just-In-Time** policy: use forecasted demand to recommend **when** to process or release units so inventory is closer to expected need. The goal in simulation is fewer units sitting unused until expiry, while still covering predicted requests. This is a software recommendation in a demo environment, not an operational manufacturing instruction.

### JIT + simulated Micro-Expiry strategy

The same JIT logic plus a **hypothetical, simulation-only** extra usable window called **Simulated Micro-Expiry — Research/Demo Only**. In the simulator, selected units may receive a **fictional additional shelf-life increment** under an assumed “approved extension” scenario. This does **not** represent a real regulatory or medical expiry change. Results from this policy are research comparison numbers only.

## AI forecasting (planned)

Demand forecasting will operate on **synthetic** time series by blood group. Planned model families:

- Simple moving average (SMA) as a transparent baseline
- SARIMA for seasonal/time-series structure
- XGBoost for feature-based demand prediction

Training, evaluation, and a model registry are planned under `forecasting/`. No forecasting code is implemented in this scaffolding stage.

## Inventory management (planned)

Inventory will track **synthetic platelet units** by blood group, with FIFO issue order, remaining simulated shelf life, shortage-risk flags, and wastage when a unit reaches simulated expiry. Database persistence is planned for a later stage.

## Simulation (planned)

A discrete simulation will generate synthetic demand, apply each of the three policies, and write metrics (wastage, shortage, fill rate, and related statistics) to `results/`. Experiment entry points live in `experiments/`.

## Repository layout

```
plateletnet/
├── README.md
├── requirements.txt
├── config.py
├── .env.example
├── .gitignore
├── data/           raw, processed, synthetic datasets
├── forecasting/    demand-forecasting placeholders
├── inventory/      unit and stock-management placeholders
├── policies/       traditional / JIT / micro-expiry placeholders
├── simulation/     scenario and runner placeholders
├── evaluation/     metrics, comparison, and plotting placeholders
├── backend/        FastAPI app and API placeholders
├── frontend/       dashboard HTML
├── experiments/    experiment scripts (placeholders)
├── tests/          pytest placeholders
└── results/        forecasts, simulations, figures, reports
```

## Planned development stages

1. **Scaffolding (this stage)** — folder structure, placeholders, configuration, documentation. No algorithms, APIs, or UI logic.
2. **Synthetic data and configuration** — demo demand and inventory datasets; environment settings.
3. **Forecasting** — SMA, SARIMA, XGBoost training and evaluation on synthetic series.
4. **Inventory core** — unit model, FIFO, simulated expiry, shortage and wastage counters.
5. **Policies** — traditional, JIT-only, and simulated micro-expiry comparison logic.
6. **Simulation and evaluation** — scenario runner, metrics, statistical tests, plots.
7. **Backend APIs** — FastAPI routes and SQLite demo persistence.
8. **Frontend dashboard** — interactive views for forecasts, inventory, JIT recommendations, and policy comparison.
9. **Tests and experiments** — pytest coverage and reproducible experiment scripts.

## Setup (later stages)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Running the API, dashboard, and experiments will be documented when those features are implemented.

## License and use

Intended for student and research demonstration. Not for clinical decision-making, blood-product dating, or production blood-bank operations.
