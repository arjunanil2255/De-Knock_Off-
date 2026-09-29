"""Audio artifact baseline: spectral spoof classifier (Phase 4).

A small CNN over log-mel spectrograms flags spectral artifacts left by
voice conversion / cloning. Outputs a per-clip artifact score (higher =
more fake-like) that feeds the fusion classifier with the video artifact
score and the cross-attention consistency signal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


class AudioArtifactClassifier(nn.Module):
    """CNN spoof classifier over log-mel spectrograms.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_fft: FFT window size.
        hop_length: STFT hop length.
        n_mels: Number of mel filterbanks.
        channels: List of convolutional channel counts.
        kernel_size: Conv kernel size.
        num_classes: Number of output classes (real / fake artifacts).
        dropout: Dropout probability in the classifier head.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 64,
        channels: list[int] | None = None,
        kernel_size: int = 5,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128]
        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.to_db = AmplitudeToDB()

        layers: list[nn.Module] = []
        in_ch = 1
        for ch in channels:
            layers.append(
                nn.Conv2d(in_ch, ch, kernel_size, padding=kernel_size // 2)
            )
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            in_ch = ch
        self.conv_stack = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def _forward_one(self, waveform: torch.Tensor) -> torch.Tensor:
        """Classify one unpadded waveform while preserving its true length."""
        if waveform.numel() < self.mel.n_fft:
            waveform = F.pad(waveform, (0, self.mel.n_fft - waveform.numel()))
        spec = self.to_db(self.mel(waveform.unsqueeze(0).unsqueeze(0)))
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        features = self.conv_stack(spec)
        pooled = self.global_pool(features).flatten(1)
        return self.classifier(pooled)

    def forward(
        self, waveform: torch.Tensor, waveform_pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Score a batch of waveforms.

        Args:
            waveform: ``(B, L)`` mono float waveforms in ``[-1, 1]``.
            waveform_pad_mask: Optional ``(B, L)`` mask with ``True`` for
                padding. Each clip is trimmed before its spectrogram is made.

        Returns:
            ``(B, num_classes)`` logits.
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        logits: list[torch.Tensor] = []
        for index in range(waveform.shape[0]):
            clip = waveform[index]
            if waveform_pad_mask is not None:
                clip = clip[~waveform_pad_mask[index]]
            logits.append(self._forward_one(clip))
        return torch.cat(logits, dim=0)

    def artifact_score(
        self, waveform: torch.Tensor, waveform_pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the fake-class probability as the artifact score.

        Args:
            waveform: ``(B, L)`` mono float waveforms.
            waveform_pad_mask: Optional ``(B, L)`` padding mask.

        Returns:
            ``(B,)`` artifact probabilities.
        """
        logits = self.forward(waveform, waveform_pad_mask=waveform_pad_mask)
        return F.softmax(logits, dim=-1)[:, 1]
