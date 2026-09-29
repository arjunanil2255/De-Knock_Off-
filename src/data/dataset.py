"""PyTorch Dataset over cached embeddings + crops + waveforms (Phase 3).

Loads ``data/processed/{clip_id}_{video,audio}.pt`` keyed by a manifest CSV,
aligns embedding sequences onto the shared time grid, and pads everything
(crops, waveform) to fixed lengths so a default collation function can batch
clips of different durations. All masks are returned so padding never leaks
into attention, pooling, or the final verdict.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import load_config
from src.models.align import align_pair, build_audio_times, build_video_times, grid_validity


class SyncVerityDataset(Dataset):
    """Loads cached per-clip data and returns fixed-size grid-padded batches.

    Args:
        processed_dir: Directory containing the cached ``.pt`` embeddings.
        manifest_path: Path to the ``manifest.csv`` table.
        split: Split name to select (``train``/``val``/``generalization``).
        max_grid_steps: Fixed padded grid length per sample.
        grid_rate: Alignment grid steps per second (from config by default).
        max_clips: Optional class-balanced limit for a development run.
        seed: Seed used to select the development subset.

    Attributes:
        clip_ids: List of clip ids after filtering clips that failed
            extraction or have no usable audio/video.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        manifest_path: str | Path,
        split: str,
        max_grid_steps: int | None = None,
        grid_rate: float | None = None,
        max_clips: int | None = None,
        seed: int = 42,
    ) -> None:
        config = load_config()
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.grid_rate = float(grid_rate if grid_rate is not None else config["alignment"]["time_grid_rate"])
        self.max_grid_steps = int(
            max_grid_steps
            if max_grid_steps is not None
            else config["alignment"].get("max_grid_steps", 128)
        )
        self.frame_rate = float(config["video"]["frame_rate"])
        self.crop_size = int(config["video"]["crop_size"])
        self.max_crops = int(config["video"].get("max_crops", 100))
        self.max_samples = int(config["audio"].get("max_samples", 192000))

        manifest = pd.read_csv(manifest_path)
        self.rows = manifest[manifest["split"] == split].to_dict("records")
        self.rows_by_clip_id = {str(row["clip_id"]): row for row in self.rows}
        self.clip_ids: list[str] = []
        for row in self.rows:
            clip_id = str(row["clip_id"])
            video_path = self.processed_dir / f"{clip_id}_video.pt"
            audio_path = self.processed_dir / f"{clip_id}_audio.pt"
            if not video_path.exists() or not audio_path.exists():
                continue
            # A clip without either modality cannot be used by a cross-modal
            # detector. Filtering it here prevents all-key-masked attention
            # (which otherwise produces NaNs) and keeps training statistics
            # honest.
            try:
                video = torch.load(video_path, map_location="cpu")
                audio = torch.load(audio_path, map_location="cpu")
                if video["embeddings"].shape[0] == 0 or audio["embeddings"].shape[-2] == 0:
                    continue
            except (KeyError, RuntimeError, OSError):
                continue
            self.clip_ids.append(clip_id)
        if max_clips is not None:
            self._limit_clips(max_clips, seed)

    def _limit_clips(self, max_clips: int, seed: int) -> None:
        """Keep a deterministic, approximately class-balanced dev subset."""
        if max_clips < 1:
            raise ValueError("max_clips must be at least 1 when specified.")
        if len(self.clip_ids) <= max_clips:
            return
        by_label: dict[int, list[str]] = {}
        for clip_id in self.clip_ids:
            by_label.setdefault(int(self.rows_by_clip_id[clip_id]["label"]), []).append(clip_id)
        rng = random.Random(seed)
        for clip_ids in by_label.values():
            rng.shuffle(clip_ids)

        selected: list[str] = []
        labels = sorted(by_label)
        while len(selected) < max_clips and any(by_label.values()):
            for label in labels:
                if by_label[label] and len(selected) < max_clips:
                    selected.append(by_label[label].pop())
        self.clip_ids = selected

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip_id = self.clip_ids[index]
        row = self.rows_by_clip_id[clip_id]

        video_raw = torch.load(self.processed_dir / f"{clip_id}_video.pt", map_location="cpu")
        audio_raw = torch.load(self.processed_dir / f"{clip_id}_audio.pt", map_location="cpu")

        video_emb = video_raw["embeddings"].float()
        frame_indices = video_raw.get(
            "frame_indices", torch.arange(video_emb.shape[0], dtype=torch.long)
        )
        crops_u8 = video_raw.get("crops", torch.zeros(0, self.crop_size, self.crop_size, 3, dtype=torch.uint8))

        audio_emb = audio_raw["embeddings"].squeeze(0).float()
        waveform = audio_raw.get("waveform", torch.zeros(0, dtype=torch.float32))
        sample_rate = int(audio_raw.get("sample_rate", 16000))

        sampling_fps = float(video_raw.get("sampling_fps", self.frame_rate))
        video_times = build_video_times(frame_indices, sampling_fps)
        audio_fps = sample_rate / 320.0 if sample_rate else 50.0
        audio_times = build_audio_times(audio_emb.shape[0], fps=audio_fps)

        video_grid, audio_grid, _ = align_pair(
            video_emb, video_times, audio_emb, audio_times, self.grid_rate
        )
        grid_len = video_grid.shape[0]
        pad = self.max_grid_steps - grid_len
        if pad < 0:
            video_grid = video_grid[: self.max_grid_steps]
            audio_grid = audio_grid[: self.max_grid_steps]
            pad = 0

        video_padded = torch.zeros(self.max_grid_steps, video_grid.shape[-1])
        video_padded[: video_grid.shape[0]] = video_grid
        audio_padded = torch.zeros(self.max_grid_steps, audio_grid.shape[-1])
        audio_padded[: audio_grid.shape[0]] = audio_grid

        valid_steps = min(grid_len, self.max_grid_steps)
        shared_duration = max(
            float(video_times[-1]) if video_times.numel() else 0.0,
            float(audio_times[-1]) if audio_times.numel() else 0.0,
        )
        video_valid = grid_validity(video_times, grid_len, shared_duration)[:valid_steps]
        audio_valid = grid_validity(audio_times, grid_len, shared_duration)[:valid_steps]
        video_pad_mask = torch.ones(self.max_grid_steps, dtype=torch.bool)
        audio_pad_mask = torch.ones(self.max_grid_steps, dtype=torch.bool)
        video_pad_mask[:valid_steps] = ~video_valid
        audio_pad_mask[:valid_steps] = ~audio_valid

        # Crops (uint8 -> float32 in [0, 1]) padded to self.max_crops frames
        crops_float = crops_u8.float().div_(255.0)
        n_crops = crops_float.shape[0]
        if n_crops > self.max_crops:
            crops_float = crops_float[: self.max_crops]
            n_crops = self.max_crops
        crops_padded = torch.zeros(self.max_crops, *crops_float.shape[1:])
        crops_padded[:n_crops] = crops_float
        crop_mask = torch.ones(self.max_crops, dtype=torch.bool)
        crop_mask[:n_crops] = False

        # Waveform padded to self.max_samples
        samples = waveform.numel()
        if samples > self.max_samples:
            waveform = waveform[: self.max_samples]
            samples = self.max_samples
        waveform_padded = torch.zeros(self.max_samples, dtype=torch.float32)
        waveform_padded[:samples] = waveform
        waveform_mask = torch.ones(self.max_samples, dtype=torch.bool)
        waveform_mask[:samples] = False

        return {
            "video_emb": video_padded,
            "audio_emb": audio_padded,
            "video_pad_mask": video_pad_mask,
            "audio_pad_mask": audio_pad_mask,
            "crops": crops_padded,
            "crop_mask": crop_mask,
            "waveform": waveform_padded,
            "waveform_mask": waveform_mask,
            "frame_indices": frame_indices,
            "label": int(row["label"]),
            "clip_id": clip_id,
        }
