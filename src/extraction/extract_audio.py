"""wav2vec2 voice embedding extraction from audio tracks (Phase 1).

Loads the audio stream of a clip, runs ``faceboook/wav2vec2-base-960h``
feature extraction, and caches the 768-d frame-level embeddings plus the
resampled waveform (needed later for spectrogram explainability).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from src.config import load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("extract_audio")


def _find_ffmpeg() -> str:
    """Locate an ffmpeg executable, with a fallback for Windows winget installs."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    for candidate in (
        Path.home()
        / "AppData"
        / "Local"
        / "Microsoft"
        / "WinGet"
        / "Packages"
    ).rglob("ffmpeg.exe"):
        return str(candidate)
    return "ffmpeg"


def load_audio(video_path: Path, sample_rate: int) -> np.ndarray:
    """Decode the audio track of a video at ``sample_rate`` (mono float32).

    Args:
        video_path: Path to the source video (or audio file).
        sample_rate: Target sample rate in Hz.

    Returns:
        Mono waveform as a float32 array in ``[-1, 1]``.

    Raises:
        RuntimeError: If the audio cannot be decoded by any backend.
    """
    import librosa

    try:
        waveform, _ = librosa.load(str(video_path), sr=sample_rate, mono=True)
        return np.asarray(waveform, dtype=np.float32)
    except Exception as first_error:  # noqa: BLE001
        try:
            import io
            import subprocess

            ffmpeg_cmd = [
                _find_ffmpeg(), "-i", str(video_path), "-vn", "-acodec", "pcm_f32le",
                "-ar", str(sample_rate), "-ac", "1", "-f", "f32le", "-",
            ]
            proc = subprocess.run(  # noqa: S603
                ffmpeg_cmd,
                capture_output=True,
                check=True,
            )
            waveform = np.frombuffer(proc.stdout, dtype=np.float32).copy()
            if waveform.size == 0:
                raise ValueError("ffmpeg produced an empty stream")
            return waveform
        except Exception as second_error:  # noqa: BLE001
            raise RuntimeError(
                f"Could not decode audio from {video_path} (librosa: {first_error}; "
                f"ffmpeg: {second_error})."
            ) from second_error


def extract_audio_track(
    model: torch.nn.Module,
    processor: "Any",
    waveform: np.ndarray,
    sample_rate: int,
) -> torch.Tensor:
    """Extract frame-level wav2vec2 hidden states from a waveform.

    Args:
        model: The loaded ``Wav2Vec2Model``.
        processor: The paired ``Wav2Vec2FeatureExtractor``.
        waveform: Mono float32 waveform.
        sample_rate: Sample rate of ``waveform``.

    Returns:
        ``(1, T, 768)`` tensor of hidden states.
    """
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs.input_values)
    return outputs.last_hidden_state


def process_audio(
    video_path: Path,
    clip_id: str,
    output_dir: Path,
    device: str = "cpu",
    force: bool = False,
) -> Path:
    """Extract and cache wav2vec2 embeddings for a single clip.

    Args:
        video_path: Path to the source video/audio file.
        clip_id: Unique identifier used for the cache key.
        output_dir: Directory where cache files are written.
        device: Torch device string (``cuda`` / ``cpu``).
        force: Recompute even if the cache file already exists.

    Returns:
        Path of the written embedding cache file.

    Raises:
        RuntimeError: If audio is silent / undecodable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{clip_id}_audio.pt"
    if out_file.exists() and not force:
        logger.info("Cache hit, skipping %s", video_path)
        return out_file

    config = load_config()
    sample_rate = int(config["audio"]["sample_rate"])

    waveform = load_audio(video_path, sample_rate)
    if waveform.size == 0 or np.max(np.abs(waveform)) == 0.0:
        logger.warning("Silent or empty audio for %s; writing empty cache.", video_path)
        torch.save(
            {
                "embeddings": torch.zeros(1, 0, int(config["audio"]["embedding_dim"])),
                "waveform": torch.zeros(0, dtype=torch.float32),
                "clip_id": clip_id,
            },
            out_file,
        )
        return out_file

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    model_name = config["audio"]["model_name"]
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).eval().to(device)

    embeddings = extract_audio_track(model, processor, waveform, sample_rate).to("cpu")
    payload = {
        "embeddings": embeddings,
        "waveform": torch.from_numpy(waveform),
        "sample_rate": sample_rate,
        "clip_id": clip_id,
    }
    torch.save(payload, out_file)
    logger.info("Cached %d audio frames for %s -> %s", embeddings.shape[1], clip_id, out_file)
    return out_file


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: extract wav2vec2 embeddings for one or more clips.

    Args:
        argv: Optional command line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Extract wav2vec2 embeddings from audio.")
    parser.add_argument("clips", nargs="+", help="Video or audio file paths.")
    parser.add_argument("--clip-id", help="Override clip id (defaults to file stem).")
    parser.add_argument(
        "--device", default="cpu", help="Torch device (``cuda``/``cpu``)."
    )
    parser.add_argument("--force", action="store_true", help="Recompute cached clips.")
    args = parser.parse_args(argv)

    config = load_config()
    output_dir = Path(config["paths"]["processed_data_dir"])
    for path in args.clips:
        clip_path = Path(path)
        clip_id = args.clip_id or clip_path.stem
        try:
            process_audio(clip_path, clip_id, output_dir, args.device, args.force)
        except Exception as exc:  # noqa: BLE001 - batch jobs must keep going
            logger.error("Failed to process %s: %s", clip_path, exc)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())