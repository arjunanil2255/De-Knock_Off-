"""FastAPI router for the analysis endpoint (Phase 8)."""
from __future__ import annotations

import tempfile
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.config import load_config

router = APIRouter()
logger = logging.getLogger("backend.analyze")

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
        raise HTTPException(
            status_code=415,
            detail={"code": "unsupported_file_type", "message": f"Unsupported file type: {suffix or 'unknown'}"},
        )

    api_config = load_config().get("api", {})
    max_bytes = int(api_config.get("max_upload_mb", 200)) * 1024 * 1024
    chunk_size = int(api_config.get("upload_chunk_bytes", 1024 * 1024))

    with tempfile.TemporaryDirectory(prefix="syncverity_upload_") as tmp:
        video_path = Path(tmp) / f"upload{suffix}"
        size = 0
        try:
            with video_path.open("wb") as destination:
                while chunk := await file.read(chunk_size):
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail={"code": "upload_too_large", "message": "Video exceeds the upload size limit."},
                        )
                    destination.write(chunk)
        finally:
            await file.close()
        try:
            from src.inference import analyze

            return analyze(video_path, clip_id=f"upload-{uuid4().hex}")
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "model_unavailable", "message": "The analysis model is not available."},
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_media", "message": str(exc)},
            ) from exc
        except Exception as exc:  # noqa: BLE001 - structured errors not raw traces
            logger.exception("Analysis failed")
            raise HTTPException(
                status_code=500,
                detail={"code": "analysis_failed", "message": "Analysis could not be completed."},
            ) from exc
