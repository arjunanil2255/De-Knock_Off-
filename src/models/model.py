"""Composed SyncVerity model tying together all branches (Phase 3/4).

Selects which branches participate via flags so one module backs the
ablation table: ``fusion``-only, ``artifact``-only, and ``combined``.
Attention weights from the cross-attention module are always kept.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.artifact_audio import AudioArtifactClassifier
from src.models.artifact_video import VideoArtifactClassifier
from src.models.classifier import FusionClassifier
from src.models.cross_attention import CrossAttentionFusion


class SyncVerityModel(nn.Module):
    """Full model: cross-attention fusion + artifact branches + MLP head.

    Args:
        config: Loaded project configuration dict.
        enable_fusion: Use the cross-attention consistency branch.
        enable_video_artifact: Use the video artifact CNN branch.
        enable_audio_artifact: Use the audio artifact CNN branch.

    Attributes:
        consistency_dim: Dimension of the consistency vector (exposed for the
            classifier and ablations).
    """

    def __init__(
        self,
        config: dict[str, Any],
        enable_fusion: bool = True,
        enable_video_artifact: bool = False,
        enable_audio_artifact: bool = False,
    ) -> None:
        super().__init__()
        self.enable_fusion = enable_fusion
        self.enable_video_artifact = enable_video_artifact
        self.enable_audio_artifact = enable_audio_artifact
        self.artifact_auxiliary_loss_weight = float(
            config.get("training", {}).get("artifact_auxiliary_loss_weight", 0.5)
        )

        fusion_cfg = config["fusion"]
        self.consistency_dim = int(fusion_cfg["d_model"])

        if enable_fusion:
            self.fusion = CrossAttentionFusion(
                video_dim=int(config["video"]["embedding_dim"]),
                audio_dim=int(config["audio"]["embedding_dim"]),
                d_model=int(fusion_cfg["d_model"]),
                n_heads=int(fusion_cfg["n_heads"]),
                num_layers=int(fusion_cfg["num_layers"]),
                dropout=float(fusion_cfg["dropout"]),
            )
        if enable_video_artifact:
            self.video_artifact = VideoArtifactClassifier()
        if enable_audio_artifact:
            self.audio_artifact = AudioArtifactClassifier(
                sample_rate=int(config["audio"]["sample_rate"])
            )

        classifier_cfg = config.get("classifier", {})
        self.classifier = FusionClassifier(
            consistency_dim=self.consistency_dim,
            n_artifact_scores=2,
            hidden_sizes=classifier_cfg.get("hidden_sizes"),
            dropout=float(classifier_cfg.get("dropout", 0.3)),
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Compute logits plus explainability artifacts for a batch.

        Args:
            batch: Batch produced by :class:`SyncVerityDataset`.

        Returns:
            Dictionary with ``logits`` ``(B, 2)``, ``consistency`` ``(B, d)``,
            ``video_score``/``audio_score`` ``(B,)``, and (when fusion is
            enabled) ``attn_va``/``attn_av`` ``(B, G, G)`` attention maps.
        """
        device = batch["video_emb"].device
        batch_size = batch["video_emb"].shape[0]

        fused_output: dict[str, Any] = {}
        if self.enable_fusion:
            fused_output = self.fusion(
                batch["video_emb"],
                batch["audio_emb"],
                video_pad_mask=batch.get("video_pad_mask"),
                audio_pad_mask=batch.get("audio_pad_mask"),
            )
            consistency = fused_output["consistency"]
        else:
            consistency = torch.zeros(
                batch_size, self.consistency_dim, device=device, dtype=batch["video_emb"].dtype
            )

        if self.enable_video_artifact:
            crops = batch["crops"].float().permute(0, 1, 4, 2, 3)  # (B, T, 3, H, W)
            video_artifact_logits = self.video_artifact(
                crops, crop_pad_mask=batch.get("crop_mask")
            )
            video_score = torch.softmax(video_artifact_logits, dim=-1)[:, 1]
        else:
            video_artifact_logits = None
            video_score = torch.zeros(batch_size, device=device)

        if self.enable_audio_artifact:
            audio_artifact_logits = self.audio_artifact(
                batch["waveform"], waveform_pad_mask=batch.get("waveform_mask")
            )
            audio_score = torch.softmax(audio_artifact_logits, dim=-1)[:, 1]
        else:
            audio_artifact_logits = None
            audio_score = torch.zeros(batch_size, device=device)

        logits = self.classifier(consistency, video_score, audio_score)

        output: dict[str, Any] = {
            "logits": logits,
            "consistency": consistency,
            "video_score": video_score,
            "audio_score": audio_score,
        }
        if video_artifact_logits is not None:
            output["video_artifact_logits"] = video_artifact_logits
        if audio_artifact_logits is not None:
            output["audio_artifact_logits"] = audio_artifact_logits
        if self.enable_fusion:
            output.update(
                {
                    "attn_va": fused_output["attn_va"],
                    "attn_av": fused_output["attn_av"],
                    "attn_va_full": fused_output["attn_va_full"],
                    "attn_av_full": fused_output["attn_av_full"],
                    "video_attended": fused_output["video_attended"],
                    "audio_attended": fused_output["audio_attended"],
                }
            )
        return output
