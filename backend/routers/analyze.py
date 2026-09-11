"""FastAPI router for the analysis endpoint (Phase 8)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@router.post("/analyze")
async def analyze_clip(file: UploadFile = File(...)) -> dict:
    """Analyse an uploaded video and return verdict + explanations.

    Args:
        file: The uploaded video file.

    Returns:
        Analysis dictionary (verdict, confidence, scores, explanation paths).

    Raises:
        HTTPException: If the file type is unsupported or analysis fails.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix or 'unknown'}")

    with tempfile.TemporaryDirectory(prefix="syncverity_upload_") as tmp:
        video_path = Path(tmp) / f"upload{suffix}"
        video_path.write_bytes(await file.read())
        try:
            from src.inference import analyze

            return analyze(video_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - structured errors not raw traces
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc