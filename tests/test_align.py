from __future__ import annotations

import unittest

import torch

from src.models.align import align_pair, grid_validity


class AlignmentTests(unittest.TestCase):
    def test_shorter_modality_keeps_its_physical_timestamp(self) -> None:
        video, audio, grid_times = align_pair(
            torch.tensor([[1.0], [2.0]]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([[10.0], [20.0]]),
            torch.tensor([0.0, 2.0]),
            grid_rate=2.0,
        )
        self.assertEqual(grid_times.tolist(), [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(video[:, 0].tolist(), [1.0, 0.0, 2.0, 0.0, 0.0])
        self.assertEqual(audio[:, 0].tolist(), [10.0, 0.0, 0.0, 0.0, 20.0])

    def test_empty_grid_bins_are_identified_for_attention_masks(self) -> None:
        valid = grid_validity(torch.tensor([0.0, 1.0]), grid_steps=5, duration=2.0)
        self.assertEqual(valid.tolist(), [True, False, True, False, False])


if __name__ == "__main__":
    unittest.main()
