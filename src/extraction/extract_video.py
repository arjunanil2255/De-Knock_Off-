"""ArcFace face embedding extraction from video clips (Phase 1).

Detects and aligns faces per sampled frame, then produces a 512-d ArcFace
embedding per frame. Results are cached as ``{clip_id}_video.pt`` under
``data/processed``. Frames in which no face is detected are skipped and
logged rather than aborting the clip.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.config import load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("extract_video")
_DETECTORS: dict[tuple[int, str], Any] = {}

# Canonical 112x112 ArcFace landmark targets for similarity alignment.
_ARCFACE_REF = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def _align_face_crop(frame: np.ndarray, landmarks: np.ndarray, crop_size: int = 112) -> np.ndarray:
    """Warp a frame so a detected face occupies the canonical ArcFace layout.

    Args:
        frame: BGR frame.
        landmarks: ``(5, 2)`` face landmarks (right eye, left eye, nose,
            mouth corners).
        crop_size: Output crop edge length in pixels.

    Returns:
        Aligned ``(crop_size, crop_size, 3)`` BGR crop.
    """
    transform, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), _ARCFACE_REF, method=cv2.RANSAC
    )
    if transform is None:
        x1, y1 = landmarks[:, 0].min(), landmarks[:, 1].min()
        x2, y2 = landmarks[:, 0].max(), landmarks[:, 1].max()
        return _align_face_crop_bbox(frame, x1, y1, x2, y2, crop_size)
    return cv2.warpAffine(frame, transform, (crop_size, crop_size), borderValue=(0, 0, 0))


def _align_face_crop_bbox(
    frame: np.ndarray, x1: float, y1: float, x2: float, y2: float, crop_size: int
) -> np.ndarray:
    """Fallback square crop from a bounding box when alignment fails."""
    height, width = frame.shape[:2]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * 0.6
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    crop = frame[int(y1) : int(y2), int(x1) : int(x2)]
    if crop.size == 0:
        crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)


def _get_detector(ctx_id: int, model_name: str = "buffalo_l") -> Any:
    """Build a lazy-loaded InsightFace FaceAnalysis detector.

    Imported inside the function so the heavy ``insightface`` dependency is
    only pulled in when extraction is actually run.

    Args:
        ctx_id: Device index for the detector (``0`` GPU, ``-1`` CPU).

    Returns:
        A prepared ``FaceAnalysis`` instance.
    """
    from insightface.app import FaceAnalysis

    cache_key = (ctx_id, model_name)
    if cache_key in _DETECTORS:
        return _DETECTORS[cache_key]
    app = FaceAnalysis(
        name=model_name,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    _DETECTORS[cache_key] = app
    return app


def sample_frames(video_path: Path, frame_rate: int) -> list[np.ndarray]:
    """Decode a video and return frames sampled at ``frame_rate`` fps (BGR).

    Args:
        video_path: Path to the input video file.
        frame_rate: Desired sampling rate in frames per second.

    Returns:
        List of BGR frames evenly sampled across the clip.

    Raises:
        RuntimeError: If the video cannot be opened.
    """
    frames, _ = _sample_frames_with_rate(video_path, frame_rate)
    return frames


def _sample_frames_with_rate(video_path: Path, frame_rate: int) -> tuple[list[np.ndarray], float]:
    """Sample frames and return the effective sampling rate used."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(source_fps / max(float(frame_rate), 1e-6))))

    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames, source_fps / interval


def extract_embeddings(
    detector: Any, frames: list[np.ndarray], threshold: float, crop_size: int = 112
) -> tuple[np.ndarray, list[int], list[np.ndarray]]:
    """Run ArcFace over sampled frames, keeping the largest face per frame.

    Args:
        detector: Prepared ``FaceAnalysis`` instance.
        frames: List of BGR frames.
        threshold: Minimum detection score to accept a face.
        crop_size: Aligned face crop edge length.

    Returns:
        A ``(embeddings, kept_frame_indices, crops)`` tuple where
        ``embeddings`` is a ``(T, 512)`` float array, ``kept_frame_indices``
        records which original frames each embedding came from, and ``crops``
        holds the aligned ``(T, crop_size, crop_size, 3)`` BGR face crops.
    """
    embeddings: list[np.ndarray] = []
    kept: list[int] = []
    crops: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        faces = detector.get(frame)
        if not faces:
            logger.warning("No face detected in frame %d; skipping.", idx)
            continue
        face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        if face.det_score is not None and float(face.det_score) < threshold:
            logger.info("Face below confidence threshold in frame %d; skipping.", idx)
            continue
        embedding = (
            face.normed_embedding if face.normed_embedding is not None else face.embedding
        )
        if embedding is None:
            logger.warning("No embedding produced for frame %d; skipping.", idx)
            continue
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        kept.append(idx)
        if face.kps is not None:
            crops.append(_align_face_crop(frame, np.asarray(face.kps, dtype=np.float32), crop_size))
        else:
            x1, y1, x2, y2 = face.bbox.astype(int)
            crops.append(_align_face_crop_bbox(frame, x1, y1, x2, y2, crop_size))
    if not embeddings:
        return (
            np.zeros((0, 512), dtype=np.float32),
            kept,
            np.zeros((0, crop_size, crop_size, 3), dtype=np.uint8),
        )
    return np.stack(embeddings), kept, np.stack(crops)


def process_video(
    video_path: Path,
    clip_id: str,
    output_dir: Path,
    ctx_id: int = 0,
    force: bool = False,
) -> Path:
    """Extract and cache ArcFace embeddings for a single clip.

    Args:
        video_path: Path to the source video.
        clip_id: Unique identifier used for the cache key.
        output_dir: Directory where ``{clip_id}_video.pt`` is written.
        ctx_id: Device index for the detector.
        force: Recompute even if the cache file already exists.

    Returns:
        Path of the written cache file.

    Raises:
        RuntimeError: If the video cannot be processed at all (e.g. no decoder).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{clip_id}_video.pt"
    if out_file.exists() and not force:
        logger.info("Cache hit, skipping %s", video_path)
        return out_file

    config = load_config()
    detector = _get_detector(ctx_id, str(config["video"]["face_detector"]))
    crop_size = int(config["video"]["crop_size"])
    frames, effective_frame_rate = _sample_frames_with_rate(
        video_path, int(config["video"]["frame_rate"])
    )
    if not frames:
        logger.warning("No frames decoded for %s; writing empty cache.", video_path)
        torch.save(
            {
                "embeddings": torch.zeros(0, int(config["video"]["embedding_dim"])),
                "crops": torch.zeros(0, crop_size, crop_size, 3, dtype=torch.uint8),
                "sampling_fps": effective_frame_rate,
                "clip_id": clip_id,
            },
            out_file,
        )
        return out_file

    embeddings, kept, crops = extract_embeddings(
        detector,
        frames,
        float(config["video"]["face_detector_threshold"]),
        crop_size=crop_size,
    )
    payload = {
        "embeddings": torch.from_numpy(embeddings),
        "frame_indices": torch.tensor(kept, dtype=torch.long),
        "crops": torch.from_numpy(crops),
        "sampling_fps": effective_frame_rate,
        "clip_id": clip_id,
    }
    torch.save(payload, out_file)
    logger.info("Cached %d embeddings for %s -> %s", len(kept), clip_id, out_file)
    return out_file


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: extract embeddings for one or more video files.

    Args:
        argv: Optional command line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Extract ArcFace embeddings from videos.")
    parser.add_argument("videos", nargs="+", help="Video file paths.")
    parser.add_argument("--clip-id", help="Override clip id (defaults to file stem).")
    parser.add_argument(
        "--ctx-id", type=int, default=0, help="Detector device index (0.. for GPU, -1 CPU)."
    )
    parser.add_argument("--force", action="store_true", help="Recompute cached clips.")
    args = parser.parse_args(argv)

    config = load_config()
    output_dir = Path(config["paths"]["processed_data_dir"])
    for path in args.videos:
        video_path = Path(path)
        clip_id = args.clip_id or video_path.stem
        try:
            process_video(video_path, clip_id, output_dir, args.ctx_id, args.force)
        except Exception as exc:  # noqa: BLE001 - batch jobs must keep going
            logger.error("Failed to process %s: %s", video_path, exc)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
