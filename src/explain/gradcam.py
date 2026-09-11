"""Grad-CAM heatmaps for the video artifact branch (Phase 6).

Produces per-frame spatial heatmaps highlighting image regions that drive the
artifact CNN's verdict, and overlays them on the aligned face crops.
"""
from __future__ import annotations

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")


def _last_conv_layer(network: nn.Module) -> nn.Module:
    """Return the last Conv2d module found by a backward walk."""
    candidates: list[nn.Module] = []
    def _walk(module: nn.Module) -> None:
        for child in module.children():
            if isinstance(child, nn.Conv2d):
                candidates.append(child)
            _walk(child)
    _walk(network)
    if not candidates:
        raise ValueError("No Conv2d layer found in the given network.")
    return candidates[-1]


def grad_cam_sequence(
    model: nn.Module,
    crops: torch.Tensor,
    target_class: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-frame Grad-CAM heatmaps for a face-crop sequence.

    Args:
        model: A ``VideoArtifactClassifier`` in eval mode.
        crops: ``(T, 3, H, W)`` aligned face crops normalised to ``[0, 1]``.
        target_class: Class whose logit is explained (default 1 = fake).

    Returns:
        ``(heatmaps, logits)``: heatmaps as a ``(T, out_h, out_w)`` float
        array in ``[0, 1]`` and the per-frame ``(T, 2)`` logits.
    """
    model.eval()

    target = _last_conv_layer(model.backbone)
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def _hook_forward(_module, _input, output) -> None:
        activations["feat"] = output.detach()

    def _hook_backward(_module, _grad_input, grad_output) -> None:
        gradients["feat"] = grad_output[0].detach()

    forward_handle = target.register_forward_hook(_hook_forward)
    backward_handle = target.register_full_backward_hook(_hook_backward)

    seq = crops.unsqueeze(0)  # (1, T, 3, H, W)
    with torch.enable_grad():
        logits = model.forward(seq)  # mean-pooled over frames
        target_score = logits[0, target_class]
        target_score.backward()

    forward_handle.remove()
    backward_handle.remove()

    activations_map = activations["feat"]          # (T, C, h, w)
    gradients_map = gradients["feat"]              # (T, C, h, w)
    alpha = gradients_map.mean(dim=(2, 3), keepdim=True)  # (T, C, 1, 1)
    heatmaps = F.relu(alpha * activations_map).sum(dim=1)  # (T, h, w)

    heatmaps_np = heatmaps.cpu().numpy()
    frame_logits = logits.detach().cpu().numpy()

    normalized = np.zeros_like(heatmaps_np)
    for idx in range(heatmaps_np.shape[0]):
        frame = heatmaps_np[idx]
        max_value = frame.max()
        if max_value > 0:
            normalized[idx] = frame / max_value
    return normalized, frame_logits


def overlay_heatmap(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "magma",
) -> np.ndarray:
    """Blend a heatmap over a BGR/RGB frame and return an RGB image.

    Args:
        frame: ``(H, W, 3)`` image (any dtype; uint8 or float in [0, 1]).
        heatmap: ``(h, w)`` float map in ``[0, 1]``.
        alpha: Overlay strength in ``[0, 1]``.
        colormap: Matplotlib colormap name.

    Returns:
        ``(H, W, 3)`` uint8 RGB image with the heatmap overlaid.
    """
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)

    height, width = frame.shape[:2]
    heatmap_float = heatmap.astype(np.float32)
    if heatmap_float.shape[:2] != (height, width):
        heatmap_float = cv2.resize(heatmap_float, (width, height), interpolation=cv2.INTER_LINEAR)

    cmap = matplotlib.colormaps.get_cmap(colormap)
    heatmap_rgba = cmap(np.clip(heatmap_float, 0.0, 1.0))          # (H, W, 4)
    heatmap_rgb = (heatmap_rgba[..., :3] * 255).astype(np.uint8)

    if frame.ndim == 3 and frame.shape[2] == 3:
        if isinstance(frame, np.ndarray) and frame.shape[2] == 3 and np.allclose(frame[..., 0], frame[..., 2]):
            base = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            base = frame
    blended = cv2.addWeighted(base, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return blended