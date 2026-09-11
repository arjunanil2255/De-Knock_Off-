"""Attention-weight visualization for cross-modal misalignment (Phase 6).

The cross-attention matrices show which video grid step each audio step
attends to (and vice versa). Off-diagonal structure is the quantitative
"does the face match the voice over time" signal, so these weights are kept
from the first implementation and rendered here for every verdict.
"""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")


def attention_matrix_image(
    attention: np.ndarray,
    grid_times: np.ndarray,
    title: str = "Cross-attention",
    colormap: str = "viridis",
) -> np.ndarray:
    """Render an attention matrix as an RGB image.

    Args:
        attention: ``(G, G)`` attention weights (rows = queries).
        grid_times: ``(G,)`` grid times in seconds (row/column labels).
        title: Plot title.
        colormap: Matplotlib colormap name.

    Returns:
        ``(H, W, 3)`` uint8 RGB image of the figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=110)
    im = ax.imshow(attention, cmap=colormap, aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_count = min(6, len(grid_times))
    tick_positions = np.linspace(0, len(grid_times) - 1, tick_count).astype(int)
    tick_labels = [f"{grid_times[pos]:.1f}s" for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("modality time")
    ax.set_ylabel("query time")
    ax.set_title(title)

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    image = buf[..., :3].copy()
    plt.close(fig)
    return image


def top_attention_spans(
    attention: np.ndarray,
    grid_times: np.ndarray,
    k: int = 5,
) -> list[dict[str, float]]:
    """Return the strongest (query, key) attention pairs as spans.

    Args:
        attention: ``(G, G)`` attention weights.
        grid_times: ``(G,)`` grid times in seconds.
        k: Number of spans to return.

    Returns:
        List of dicts with ``query_time``, ``key_time`` and ``weight``.
    """
    grid_steps = len(grid_times)
    flat = attention.reshape(-1)
    order = np.argsort(flat)[::-1][:k]
    spans: list[dict[str, float]] = []
    for linear_index in order:
        query_idx, key_idx = divmod(int(linear_index), grid_steps)
        spans.append({
            "query_time": float(grid_times[query_idx]),
            "key_time": float(grid_times[key_idx]),
            "weight": float(flat[linear_index]),
        })
    return spans