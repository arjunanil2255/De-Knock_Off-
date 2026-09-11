"""Spectrogram highlight for the audio artifact branch (Phase 6).

Computes a log-mel spectrogram of the clip and overlays the regions the audio
artifact CNN focuses on (Grad-CAM on the final spectrogram conv layer), so the
explanation shows *which* spectral content drove the audio verdict.
"""
from __future__ import annotations

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


def spectrogram_highlight(
    model: nn.Module,
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    target_class: int = 1,
    colormap: str = "inferno",
) -> dict[str, np.ndarray]:
    """Render a log-mel spectrogram with a Grad-CAM highlight overlay.

    Args:
        model: An ``AudioArtifactClassifier`` in eval mode.
        waveform: ``(L,)`` mono float waveform in ``[-1, 1]``.
        sample_rate: Audio sample rate of ``waveform``.
        target_class: Class whose logit is explained (default 1 = fake).
        colormap: Matplotlib colormap for the overlay.

    Returns:
        Dictionary with ``spectrogram`` (grayscale ``(H, W)`` log-mel),
        ``heatmap`` ``(H, W)`` in ``[0, 1]``, and ``overlay`` an ``(H, W, 3)``
        uint8 RGB image combining both.
    """
    model.eval()

    waveform_batch = waveform.unsqueeze(0).to(next(model.parameters()).device)  # (1, L)
    with torch.no_grad():
        spec_db = _log_mel(model, waveform_batch)                   # (1, 1, n_mels, T)
    spec_np = spec_db.squeeze(0).squeeze(0).cpu().numpy()

    target = _last_conv_layer(model.conv_stack)
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def _hook_forward(_module, _input, output) -> None:
        activations["feat"] = output.detach()

    def _hook_backward(_module, _grad_input, grad_output) -> None:
        gradients["feat"] = grad_output[0].detach()

    forward_handle = target.register_forward_hook(_hook_forward)
    backward_handle = target.register_full_backward_hook(_hook_backward)

    target_batch = waveform_batch.unsqueeze(1) if waveform_batch.ndim == 2 else waveform_batch
    with torch.enable_grad():
        logits = model.forward(waveform_batch)
        model.zero_grad()
        logits[0, target_class].backward()

    forward_handle.remove()
    backward_handle.remove()

    activations_map = activations["feat"]                               # (1, C, h, w)
    gradients_map = gradients["feat"]                                   # (1, C, h, w)
    alpha = gradients_map.mean(dim=(2, 3), keepdim=True)
    heatmap = F.relu(alpha * activations_map).sum(dim=1).squeeze(0)     # (h, w)
    heatmap_np = heatmap.cpu().numpy()
    if heatmap_np.max() > 0:
        heatmap_np = heatmap_np / heatmap_np.max()

    overlay = _render_overlay(spec_np, heatmap_np, colormap)
    return {
        "spectrogram": spec_np,
        "heatmap": heatmap_np,
        "overlay": overlay,
    }


def _log_mel(model: nn.Module, waveform_batch: torch.Tensor) -> torch.Tensor:
    """Compute the same log-mel representation the model classifies."""
    mel = model.mel(waveform_batch)
    db = model.to_db(mel)
    db = (db - db.mean()) / (db.std() + 1e-6)
    return db


def _render_overlay(spectrogram: np.ndarray, heatmap: np.ndarray, colormap: str) -> np.ndarray:
    """Compose the grayscale spectrogram with a colored overlay."""
    height, width = spectrogram.shape
    if heatmap.shape != (height, width):
        import cv2

        heatmap = cv2.resize(heatmap.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)

    normalized = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-6)
    gray = (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)

    cmap = matplotlib.colormaps.get_cmap(colormap)
    overlay_color = (cmap(np.clip(heatmap, 0.0, 1.0))[..., :3] * 255).astype(np.uint8)

    blended = np.stack([gray, gray, gray], axis=-1)
    blend_mask = heatmap > 0.15
    alpha = (0.75 * blend_mask).astype(np.float32)[..., np.newaxis]
    blended = (blended.astype(np.float32) * (1.0 - alpha) + overlay_color.astype(np.float32) * alpha).astype(np.uint8)
    return blended