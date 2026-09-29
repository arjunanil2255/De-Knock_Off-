"""End-to-end inference pipeline shared by backend and CLI (Phase 8).

Runs a trained model over one clip: extract embeddings, align to a shared
time grid, fuse, classify, and produce all explainability artifacts
(Grad-CAM frames, spectrogram highlight, attention matrices). Returns a
plain-dict result plus saved artifact file paths.
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import load_config
from src.explain.attention_viz import attention_matrix_image, top_attention_spans
from src.explain.gradcam import grad_cam_sequence, overlay_heatmap
from src.explain.spectrogram_highlight import spectrogram_highlight
from src.extraction.extract_audio import process_audio
from src.extraction.extract_video import process_video
from src.models.align import align_pair, build_audio_times, build_video_times, grid_validity
from src.models.model import SyncVerityModel

logger = logging.getLogger("inference")

MODE_FLAGS: dict[str, dict[str, bool]] = {
    "fusion": {"enable_fusion": True, "enable_video_artifact": False, "enable_audio_artifact": False},
    "artifact": {"enable_fusion": False, "enable_video_artifact": True, "enable_audio_artifact": True},
    "combined": {"enable_fusion": True, "enable_video_artifact": True, "enable_audio_artifact": True},
}
_MODEL_CACHE: dict[tuple[str, str, int], SyncVerityModel] = {}


def _safe_clip_id(value: str) -> str:
    """Return a path-safe result directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return cleaned or "clip"


def load_model(config: dict[str, Any], device: torch.device) -> SyncVerityModel:
    """Load the configured checkpoint into eval mode.

    Args:
        config: Project configuration.
        device: Torch device.

    Returns:
        The trained model in eval mode.

    Raises:
        FileNotFoundError: If ``paths.model_checkpoint`` does not exist.
    """
    checkpoint_path = Path(config["paths"]["model_checkpoint"]).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_path} not found. Train a `combined` model first "
            "and point `paths.model_checkpoint` at it."
        )
    cache_key = (str(checkpoint_path), str(device), checkpoint_path.stat().st_mtime_ns)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    mode = checkpoint.get("mode", "combined")
    if mode not in MODE_FLAGS:
        raise ValueError(f"Unsupported checkpoint mode: {mode}")
    model = SyncVerityModel(config, **MODE_FLAGS[mode])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    _MODEL_CACHE.clear()
    _MODEL_CACHE[cache_key] = model
    return model


def _build_batch(
    config: dict[str, Any],
    video_emb: torch.Tensor,
    frame_indices: torch.Tensor,
    audio_emb: torch.Tensor,
    waveform: torch.Tensor,
    sample_rate: int,
    crops: torch.Tensor,
    sampling_fps: float | None = None,
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    """Align a single clip and pad it into a batch of one."""
    max_grid = int(config["alignment"]["max_grid_steps"])
    grid_rate = float(config["alignment"]["time_grid_rate"])
    max_crops = int(config["video"].get("max_crops", 100))
    max_samples = int(config["audio"].get("max_samples", 192000))

    frame_rate = float(sampling_fps or config["video"]["frame_rate"])
    video_times = build_video_times(frame_indices, frame_rate)
    audio_fps = sample_rate / 320.0 if sample_rate else 50.0
    audio_times = build_audio_times(audio_emb.shape[0], fps=audio_fps)

    video_grid, audio_grid, grid_times = align_pair(
        video_emb, video_times, audio_emb, audio_times, grid_rate
    )
    grid_len = grid_times.shape[0]
    grid_len = min(grid_len, max_grid)

    video_padded = torch.zeros(max_grid, video_grid.shape[-1])
    audio_padded = torch.zeros(max_grid, audio_grid.shape[-1])
    video_padded[:grid_len] = video_grid[:grid_len]
    audio_padded[:grid_len] = audio_grid[:grid_len]

    shared_duration = max(
        float(video_times[-1]) if video_times.numel() else 0.0,
        float(audio_times[-1]) if audio_times.numel() else 0.0,
    )
    video_valid = grid_validity(video_times, video_grid.shape[0], shared_duration)[:grid_len]
    audio_valid = grid_validity(audio_times, audio_grid.shape[0], shared_duration)[:grid_len]
    video_mask = torch.ones(max_grid, dtype=torch.bool)
    audio_mask = torch.ones(max_grid, dtype=torch.bool)
    video_mask[:grid_len] = ~video_valid
    audio_mask[:grid_len] = ~audio_valid

    crops_float = crops.float().div_(255.0)
    n_crops = min(crops_float.shape[0], max_crops)
    crops_padded = torch.zeros(max_crops, *crops_float.shape[1:])
    crops_padded[:n_crops] = crops_float[:n_crops]
    crop_mask = torch.ones(max_crops, dtype=torch.bool)
    crop_mask[:n_crops] = False

    samples = min(waveform.numel(), max_samples)
    waveform_padded = torch.zeros(max_samples, dtype=torch.float32)
    waveform_padded[:samples] = waveform[:samples]
    waveform_mask = torch.ones(max_samples, dtype=torch.bool)
    waveform_mask[:samples] = False

    batch = {
        "video_emb": video_padded.unsqueeze(0),
        "audio_emb": audio_padded.unsqueeze(0),
        "video_pad_mask": video_mask.unsqueeze(0),
        "audio_pad_mask": audio_mask.unsqueeze(0),
        "crops": crops_padded.unsqueeze(0),
        "crop_mask": crop_mask.unsqueeze(0),
        "waveform": waveform_padded.unsqueeze(0),
        "waveform_mask": waveform_mask.unsqueeze(0),
    }
    return batch, grid_times[:grid_len].numpy()


def _save_artifacts(
    clip_id: str, config: dict[str, Any], images: dict[str, np.ndarray]
) -> dict[str, str]:
    """Persist explainability images under ``results/<clip_id>/``.

    Returns paths relative to the results dir so the backend can expose them
    as static URLs.
    """
    import cv2

    out_dir = Path(config["paths"]["results_dir"])
    clip_dir = out_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for name, image in images.items():
        path = clip_dir / f"{name}.png"
        if image.ndim == 3:
            cv2.imwrite(str(path), image[..., ::-1])
        else:
            cv2.imwrite(str(path), image)
        saved[name] = path.relative_to(out_dir).as_posix()
    return saved


def analyze(
    video_path: str | Path,
    config: dict[str, Any] | None = None,
    device: torch.device | None = None,
    clip_id: str | None = None,
    ctx_id: int = 0,
) -> dict[str, Any]:
    """Analyse a single clip and return verdict + explanations.

    Args:
        video_path: Path to the source video file.
        config: Optional loaded config (loads itself when ``None``).
        device: Torch device (uses auto when ``None``).
        clip_id: Optional clip id (defaults to the file stem).
        ctx_id: Detector device index for face extraction.

    Returns:
        Dict with ``clip_id``, ``verdict``, ``confidence``, ``scores``,
        ``explanation_paths`` and attention ``spans``.
    """
    config = config or load_config()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_id = _safe_clip_id(clip_id or Path(video_path).stem)
    video_path = Path(video_path)

    model = load_model(config, device)
    with tempfile.TemporaryDirectory(prefix="syncverity_") as tmp:
        tmp_dir = Path(tmp)
        video_cache = process_video(
            video_path, clip_id, tmp_dir, ctx_id=ctx_id, force=True
        )
        audio_cache = process_audio(
            video_path, clip_id, tmp_dir, device=str(device), force=True
        )

        video_payload = torch.load(video_cache, map_location="cpu")
        audio_payload = torch.load(audio_cache, map_location="cpu")

        video_emb = video_payload["embeddings"].float()
        frame_indices = video_payload.get(
            "frame_indices", torch.arange(video_emb.shape[0], dtype=torch.long)
        )
        sampling_fps = float(video_payload.get("sampling_fps", config["video"]["frame_rate"]))
        crops = video_payload.get(
            "crops", torch.zeros(0, 112, 112, 3, dtype=torch.uint8)
        )
        audio_emb = audio_payload["embeddings"].squeeze(0).float()
        waveform = audio_payload.get("waveform", torch.zeros(0, dtype=torch.float32))
        sample_rate = int(audio_payload.get("sample_rate", 16000))

        if video_emb.shape[0] == 0:
            raise ValueError("No usable face was detected in the uploaded clip.")
        if audio_emb.shape[0] == 0 or waveform.numel() == 0:
            raise ValueError("No usable non-silent audio was detected in the uploaded clip.")

        batch, grid_times = _build_batch(
            config, video_emb, frame_indices, audio_emb, waveform, sample_rate, crops,
            sampling_fps=sampling_fps,
        )
        device_batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            output = model(device_batch)

        probs = torch.softmax(output["logits"], dim=-1)[0]
        predicted_class = int(probs.argmax())
        verdict = "fake" if predicted_class == 1 else "real"
        confidence = float(probs.max())

        images: dict[str, np.ndarray] = {}

        if getattr(model, "enable_fusion", False):
            attn_va = output["attn_va"][0].cpu().numpy()
            attn_av = output["attn_av"][0].cpu().numpy()
            images["attention_video_to_audio"] = attention_matrix_image(
                attn_va[: grid_times.shape[0], : grid_times.shape[0]], grid_times
            )
            images["attention_audio_to_video"] = attention_matrix_image(
                attn_av[: grid_times.shape[0], : grid_times.shape[0]], grid_times
            )

        if getattr(model, "enable_video_artifact", False) and crops.shape[0] > 0:
            heatmaps, _ = grad_cam_sequence(
                model.video_artifact,
                crops.float().div_(255.0).permute(0, 3, 1, 2).to(device),
                target_class=predicted_class,
            )
            for idx in [0, min(1, heatmaps.shape[0] - 1), min(2, heatmaps.shape[0] - 1)]:
                overlaid = overlay_heatmap(crops[idx].numpy(), heatmaps[idx])
                images[f"gradcam_frame_{idx}"] = overlaid

        if getattr(model, "enable_audio_artifact", False):
            if waveform.numel() > 0:
                spec = spectrogram_highlight(
                    model.audio_artifact,
                    waveform.to(device),
                    sample_rate=sample_rate,
                    target_class=predicted_class,
                )
                images["audio_spectrogram_highlight"] = spec["overlay"]

        result = {
            "clip_id": clip_id,
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "scores": {
                # This is a diagnostic embedding magnitude, not a calibrated
                # probability or stand-alone mismatch verdict.
                "fusion_embedding_norm": round(float(torch.linalg.vector_norm(output["consistency"][0])), 4),
                "video_artifact": round(float(output["video_score"][0]), 4),
                "audio_artifact": round(float(output["audio_score"][0]), 4),
            },
        }

        if images:
            result["explanation_paths"] = _save_artifacts(clip_id, config, images)

        if "attn_va" in output:
            spans_map: dict[str, list[dict[str, float]]] = {}
            for direction, matrix in (
                ("video_to_audio", output["attn_va"][0].cpu().numpy()),
                ("audio_to_video", output["attn_av"][0].cpu().numpy()),
            ):
                n = grid_times.shape[0]
                spans_map[direction] = top_attention_spans(
                    matrix[:n, :n], grid_times[:n], k=int(config["explainability"]["top_attention_spans"])
                )
            result["attention_spans"] = spans_map

        return result


def main_cli(video_path: str, device: str = "auto") -> int:
    """CLI wrapper for a single-file analysis run.

    Args:
        video_path: Path to the clip to analyse.
        device: Torch device string.

    Returns:
        Process exit code.
    """
    config = load_config()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    result = analyze(video_path, config=config, device=dev)
    print(result)
    return 0


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Analyse one clip.")
    parser.add_argument("video", help="Path to the video clip.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    sys.exit(main_cli(args.video, args.device))
