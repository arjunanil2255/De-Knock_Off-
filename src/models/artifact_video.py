"""Video artifact baseline: CNN over aligned face crops (Phase 4).

A pretrained EfficientNet scores aligned face crops for blending/temporal
artifacts. Outputs a per-clip artifact score (higher = more fake-like) which
feeds the fusion classifier alongside the cross-attention consistency signal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoArtifactClassifier(nn.Module):
    """EfficientNet-based per-clip artifact scorer.

    Args:
        backbone: timm model name to use as feature extractor.
        pretrained: Load ImageNet-pretrained backbone weights.
        num_classes: Number of output classes (real / fake artifacts).
        dropout: Dropout applied to pooled features before the head.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=3
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """Score a batch of aligned face-crop sequences.

        Args:
            crops: ``(B, T, 3, H, W)`` float tensors in ``[0, 1]``.

        Returns:
            ``(B, num_classes)`` logits (mean-pooled over frames).
        """
        batch_size, seq_len, channels, height, width = crops.shape
        flat = crops.reshape(batch_size * seq_len, channels, height, width)
        features = self.backbone(flat)
        logits = self.head(self.drop(features))
        logits = logits.reshape(batch_size, seq_len, -1).mean(dim=1)
        return logits

    def artifact_score(self, crops: torch.Tensor) -> torch.Tensor:
        """Return the fake-class probability as the artifact score.

        Args:
            crops: ``(B, T, 3, H, W)`` aligned face crops.

        Returns:
            ``(B,)`` artifact probabilities.
        """
        logits = self.forward(crops)
        return F.softmax(logits, dim=-1)[:, 1]