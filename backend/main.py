"""FastAPI entry point for the local PlateletNet research dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import API_HOST, API_PORT, APP_NAME, BASE_DIR
from backend.api.dashboard import router as dashboard_router

FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_FILE = FRONTEND_DIR / "PlateletNet.html"


def create_app(*args, **kwargs) -> FastAPI:
    """Build the local demo FastAPI app (research/prototype only)."""
    app = FastAPI(
        title=APP_NAME,
        description=(
            "Local academic dashboard for synthetic platelet demand, inventory, "
            "and JIT recommendations. Not a clinical system."
        ),
        version="0.3.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(dashboard_router, prefix="/api")

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "app": APP_NAME,
            "clinical_system": False,
            "synthetic_data": True,
        }

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_FILE)

    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
