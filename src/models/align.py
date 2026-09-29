"""Time-alignment and pooling utilities (Phase 2).

Video embeddings are sampled at a low frame rate while wav2vec2 outputs hide
den states at ~50 Hz. Every cross-modal path must first pool both sequences
onto a shared time grid so each grid step ``t`` refers to the same physical
moment in both modalities. Tested in isolation before being wired into the
full pipeline (see ``phase.md``).
"""
from __future__ import annotations

import torch


def sequence_to_grid(
    embeddings: torch.Tensor,
    times: torch.Tensor,
    grid_steps: int,
    duration: float | None = None,
) -> torch.Tensor:
    """Average-pool a sequence of embeddings onto a fixed time grid.

    Each embedding is assigned to a bin over the supplied clip ``duration``.
    When ``duration`` is omitted, the final timestamp is used for backwards
    compatibility.  Cross-modal callers must pass their *shared* duration;
    scaling video and audio independently would erase a real timing offset.

    Args:
        embeddings: ``(S, D)`` sequence of embeddings.
        times: ``(S,)`` per-embedding times in seconds (monotonic).
        grid_steps: Number of output grid steps ``G``.
        duration: Physical duration covered by the grid in seconds.

    Returns:
        ``(G, D)`` pooled sequence; empty bins are zero vectors.

    Raises:
        ValueError: If ``embeddings`` and ``times`` length mismatch, or if
            ``grid_steps < 1``.
    """
    seq_len, dim = embeddings.shape
    if times.shape[0] != seq_len:
        raise ValueError(
            f"embeddings ({seq_len}) and times ({times.shape[0]}) length mismatch"
        )
    if grid_steps < 1:
        raise ValueError(f"grid_steps must be >= 1, got {grid_steps}")
    if seq_len == 0:
        return torch.zeros(grid_steps, dim, dtype=embeddings.dtype, device=embeddings.device)

    t_max = float(duration) if duration is not None else float(times[-1].item())
    t_max = t_max if t_max > 0.0 else 1.0
    bins = torch.floor(times * (grid_steps - 1) / t_max).long().clamp(min=0, max=grid_steps - 1)

    sums = torch.zeros(grid_steps, dim, dtype=embeddings.dtype, device=embeddings.device)
    counts = torch.zeros(grid_steps, dtype=torch.long, device=embeddings.device)
    sums.index_add_(0, bins, embeddings)
    counts.scatter_add_(0, bins, torch.ones_like(bins))

    non_empty = counts > 0
    sums[non_empty] /= counts[non_empty].unsqueeze(1).to(dtype=embeddings.dtype)
    return sums


def grid_validity(times: torch.Tensor, grid_steps: int, duration: float) -> torch.Tensor:
    """Return ``True`` for shared-grid bins containing an observed sample.

    Empty alignment bins must remain masked: a zero vector can mean either
    padding or a valid embedding whose value happens to be near zero.
    """
    valid = torch.zeros(grid_steps, dtype=torch.bool, device=times.device)
    if times.numel() == 0:
        return valid
    safe_duration = duration if duration > 0.0 else 1.0
    bins = torch.floor(times * (grid_steps - 1) / safe_duration).long()
    bins = bins.clamp(min=0, max=grid_steps - 1)
    valid[bins] = True
    return valid


def build_audio_times(seq_len: int, fps: float = 50.0) -> torch.Tensor:
    """Approximate per-embedding timestamps for wav2vec2 hidden states.

    wav2vec2 base hidden states advance at roughly ``sample_rate / 320``
    (about 50 Hz for 16 kHz audio).

    Args:
        seq_len: Number of hidden-state frames.
        fps: Hidden-state frames per second.

    Returns:
        ``(seq_len,)`` tensor of ascending times in seconds.
    """
    return torch.arange(seq_len, dtype=torch.float32) / fps


def build_video_times(frame_indices: torch.Tensor, frame_rate: float) -> torch.Tensor:
    """Convert kept-frame indices back into absolute times in seconds.

    Args:
        frame_indices: Original (un-sampled) frame indices of kept faces.
        frame_rate: Extraction frame rate in fps.

    Returns:
        ``(len(frame_indices),)`` tensor of ascending times in seconds.
    """
    return frame_indices.float() / frame_rate


def align_pair(
    video_emb: torch.Tensor,
    video_times: torch.Tensor,
    audio_emb: torch.Tensor,
    audio_times: torch.Tensor,
    grid_rate: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool a video/audio embedding pair onto a shared time grid.

    The grid covers ``[0, t_max]`` where ``t_max`` is the later of the two
    modality end times; both sequences are padded to the same step count.

    Args:
        video_emb: ``(V, d_v)`` video embeddings.
        video_times: ``(V,)`` video times in seconds.
        audio_emb: ``(A, d_a)`` audio embeddings.
        audio_times: ``(A,)`` audio times in seconds.
        grid_rate: Grid steps per second.

    Returns:
        ``(video_grid, audio_grid, grid_times)`` where each grid is
        ``(G, d)`` and ``grid_times`` is ``(G,)``.
    """
    t_max = max(
        float(video_times[-1]) if video_times.numel() else 0.0,
        float(audio_times[-1]) if audio_times.numel() else 0.0,
    )
    if t_max <= 0.0:
        return (
            torch.zeros(1, video_emb.shape[-1], dtype=video_emb.dtype),
            torch.zeros(1, audio_emb.shape[-1], dtype=audio_emb.dtype),
            torch.zeros(1, dtype=video_times.dtype),
        )

    # Include both endpoints so a 3-second clip at 10 Hz has timestamps
    # 0.0, 0.1, ..., 3.0 rather than a slightly compressed grid.
    grid_steps = max(1, int(torch.ceil(torch.tensor(t_max * grid_rate)).item()) + 1)
    video_grid = sequence_to_grid(video_emb, video_times, grid_steps, duration=t_max)
    audio_grid = sequence_to_grid(audio_emb, audio_times, grid_steps, duration=t_max)
    grid_times = torch.linspace(0.0, t_max, grid_steps)
    return video_grid, audio_grid, grid_times
