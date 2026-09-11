"""Cross-attention fusion module (Phase 2 — core novelty).

Video sequences attend to audio and vice versa, producing aligned
representations that capture whether face motion and speech are genuinely
consistent.  **Attention weights are always retained** on every forward pass
so the explainability layer can visualise temporal misalignment without
retrofitting.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between video and audio grid sequences.

    Each modality is projected to a shared ``d_model`` dimension, then
    stacked cross-attention layers let video query audio and audio query
    video.  The module returns both attended representations **and** raw
    attention weight matrices (``(B, H, G, G)`` each) so the explainability
    layer can inspect which video frames are being paired with which audio
    frames.

    Args:
        video_dim: Raw video embedding dimension (e.g. 512).
        audio_dim: Raw audio embedding dimension (e.g. 768).
        d_model: Shared hidden dimension inside the transformer.
        n_heads: Number of multi-head attention heads.
        num_layers: Number of stacked cross-attention layers.
        dropout: Dropout probability in MHA and feed-forward sub-layers.
        ff_dim: Inner dimension of the feed-forward network per layer.
            Defaults to ``4 * d_model`` when ``None``.
    """

    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        ff_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.proj_video = nn.Linear(video_dim, d_model)
        self.proj_audio = nn.Linear(audio_dim, d_model)

        self.video_layers = nn.ModuleList()
        self.audio_layers = nn.ModuleList()
        self.video_ff_layers = nn.ModuleList()
        self.audio_ff_layers = nn.ModuleList()
        self.video_norm1 = nn.ModuleList()
        self.video_norm2 = nn.ModuleList()
        self.audio_norm1 = nn.ModuleList()
        self.audio_norm2 = nn.ModuleList()

        _ff_dim = ff_dim if ff_dim is not None else d_model * 4

        for _ in range(num_layers):
            self.video_layers.append(
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            )
            self.audio_layers.append(
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            )
            self.video_ff_layers.append(nn.Sequential(
                nn.Linear(d_model, _ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(_ff_dim, d_model),
                nn.Dropout(dropout),
            ))
            self.audio_ff_layers.append(nn.Sequential(
                nn.Linear(d_model, _ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(_ff_dim, d_model),
                nn.Dropout(dropout),
            ))
            self.video_norm1.append(nn.LayerNorm(d_model))
            self.video_norm2.append(nn.LayerNorm(d_model))
            self.audio_norm1.append(nn.LayerNorm(d_model))
            self.audio_norm2.append(nn.LayerNorm(d_model))

        self.consistency_proj = nn.Linear(d_model * 2, d_model)

    def forward(
        self,
        video_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        video_pad_mask: torch.Tensor | None = None,
        audio_pad_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run the full cross-attention fusion.

        Args:
            video_emb: ``(B, G, video_dim)`` video grid.
            audio_emb: ``(B, G, audio_dim)`` audio grid.
            video_pad_mask: Optional ``(B, G)`` boolean mask, ``True`` in
                padded grid positions; applied to audio→video attention keys.
            audio_pad_mask: Optional ``(B, G)`` boolean mask, ``True`` in
                padded grid positions; applied to video→audio attention keys.

        Returns:
            Dictionary with keys:
            - ``video_attended``: ``(B, G, d_model)``
            - ``audio_attended``: ``(B, G, d_model)``
            - ``consistency``: ``(B, d_model)`` concatenated + projected token
              representing the fused cross-modal representation.
            - ``attn_va``: ``(B, H, G, G)`` video-to-audio attention weights.
            - ``attn_av``: ``(B, H, G, G)`` audio-to-video attention weights.
        """
        x_v = self.proj_video(video_emb)
        x_a = self.proj_audio(audio_emb)

        attn_va_list: list[torch.Tensor] = []
        attn_av_list: list[torch.Tensor] = []

        for i in range(len(self.video_layers)):
            # Video attends to audio
            residual = x_v
            out_v, w_va = self.video_layers[i](
                query=x_v,
                key=x_a,
                value=x_a,
                key_padding_mask=audio_pad_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x_v = self.video_norm1[i](residual + out_v)
            x_v = self.video_norm2[i](x_v + self.video_ff_layers[i](x_v))
            attn_va_list.append(w_va)

            # Audio attends to video
            residual = x_a
            out_a, w_av = self.audio_layers[i](
                query=x_a,
                key=x_v,
                value=x_v,
                key_padding_mask=video_pad_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x_a = self.audio_norm1[i](residual + out_a)
            x_a = self.audio_norm2[i](x_a + self.audio_ff_layers[i](x_a))
            attn_av_list.append(w_av)

        attn_va = torch.stack(attn_va_list, dim=1)   # (B, L, H, G, G)
        attn_av = torch.stack(attn_av_list, dim=1)   # (B, L, H, G, G)

        # Collapse to last layer's mean head for storage / viz
        attn_va_last = attn_va[:, -1].mean(dim=1)     # (B, G, G)
        attn_av_last = attn_av[:, -1].mean(dim=1)     # (B, G, G)

        # Consistency token: mask-aware mean-pool each attended sequence
        if video_pad_mask is not None:
            valid_v = (~video_pad_mask).unsqueeze(-1)
            v_pooled = (x_v * valid_v).sum(dim=1) / valid_v.sum(dim=1).clamp_min(1.0)
        else:
            v_pooled = x_v.mean(dim=1)

        if audio_pad_mask is not None:
            valid_a = (~audio_pad_mask).unsqueeze(-1)
            a_pooled = (x_a * valid_a).sum(dim=1) / valid_a.sum(dim=1).clamp_min(1.0)
        else:
            a_pooled = x_a.mean(dim=1)

        consistency = self.consistency_proj(torch.cat([v_pooled, a_pooled], dim=-1))  # (B, d_model)

        return {
            "video_attended": x_v,
            "audio_attended": x_a,
            "consistency": consistency,
            "attn_va": attn_va_last,
            "attn_av": attn_av_last,
            "attn_va_full": attn_va,
            "attn_av_full": attn_av,
        }