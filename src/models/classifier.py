"""Fusion classifier head: MLP on consistency + artifact scores.

Takes the cross-attention consistency vector together with the two
per-modality artifact scores and produces the final real/fake logits.  For
ablations the caller zeroes whichever inputs are intentionally disabled
(fusion-only or artifact-only), so the same head can back every row of the
ablation table (see ``rules.md``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionClassifier(nn.Module):
    """MLP mapping fused signals to a binary real/fake verdict.

    Args:
        consistency_dim: Dimension of the cross-attention consistency vector.
        n_artifact_scores: Number of per-modality artifact scores (default 2:
            video + audio).
        hidden_sizes: Hidden layer sizes for the MLP.
        dropout: Dropout probability between hidden layers.
        n_classes: Number of output classes (default 2).

    """

    def __init__(
        self,
        consistency_dim: int,
        n_artifact_scores: int = 2,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.3,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [512, 128]
        input_dim = consistency_dim + n_artifact_scores

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, n_classes))
        self.head = nn.Sequential(*layers)

        self.input_dim = input_dim
        self.n_classes = n_classes

    def forward(
        self,
        consistency: torch.Tensor,
        video_artifact_score: torch.Tensor,
        audio_artifact_score: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits for a batch.

        Args:
            consistency: ``(B, consistency_dim)`` fused signal.
            video_artifact_score: ``(B,)`` or ``(B, 1)`` video artifact score.
            audio_artifact_score: ``(B,)`` or ``(B, 1)`` audio artifact score.

        Returns:
            ``(B, n_classes)`` logits.
        """
        video_score = video_artifact_score.reshape(-1, 1)
        audio_score = audio_artifact_score.reshape(-1, 1)
        x = torch.cat([consistency, video_score, audio_score], dim=-1)
        return self.head(x)

    def predict(self, consistency: torch.Tensor, video_score: torch.Tensor, audio_score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class prediction and softmax confidence.

        Args:
            consistency: ``(B, consistency_dim)`` fused signal.
            video_score: ``(B,)`` or ``(B, 1)`` video artifact score.
            audio_score: ``(B,)`` or ``(B, 1)`` audio artifact score.

        Returns:
            Tuple of predicted class indices ``(B,)`` and confidences ``(B,)``.
        """
        logits = self.forward(consistency, video_score, audio_score)
        probs = F.softmax(logits, dim=-1)
        return probs.argmax(dim=-1), probs.max(dim=-1).values