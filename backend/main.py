"""FastAPI application entry point for the SyncVerity web demo (Phase 8).

Run with: ``uvicorn backend.main:app --reload`` (or ``python -m backend.main``).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers.analyze import router as analyze_router
from src.config import load_config

app = FastAPI(
    title="SyncVerity",
    description="Cross-modal consistency deepfake detection API.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_results_dir = Path(load_config()["paths"]["results_dir"]).resolve()
_results_dir.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(_results_dir)), name="results")

app.include_router(analyze_router, prefix="/api")


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)