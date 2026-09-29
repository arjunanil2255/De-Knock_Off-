from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd

from src.data.build_manifest import build_manifest
from src.data.build_avdeepfake_manifest import build_avdeepfake_manifest
from src.data.dataset import SyncVerityDataset


class ManifestTests(unittest.TestCase):
    def test_ids_are_unique_and_source_groups_do_not_cross_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            for relative in (
                "real/alice_01.mp4",
                "fake/alice_02.mp4",
                "real/bob_01.mp4",
                "fake/bob_02.mp4",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            output = Path(directory) / "manifest.csv"
            build_manifest(root, output, val_fraction=0.5, seed=7)
            manifest = pd.read_csv(output)
            self.assertTrue(manifest["clip_id"].is_unique)
            self.assertTrue((manifest.groupby("source_group")["split"].nunique() == 1).all())

    def test_generalization_rows_are_never_marked_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            path = root / "fake" / "alice_01.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            output = Path(directory) / "manifest.csv"
            build_manifest(root, output, split="generalization", val_fraction=0.5)
            manifest = pd.read_csv(output)
            self.assertEqual(set(manifest["split"]), {"generalization"})

    def test_avdeepfake_metadata_uses_modify_type_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "avdeepfake"
            (root / "val").mkdir(parents=True)
            (root / "val" / "real.mp4").touch()
            (root / "val" / "fake.mp4").touch()
            (root / "val_metadata.json").write_text(
                json.dumps([
                    {"file": "val/real.mp4", "modify_type": "real", "original": "source-a"},
                    {"file": "val/fake.mp4", "modify_type": "audio_modified", "original": "source-a"},
                ]),
                encoding="utf-8",
            )
            output = Path(directory) / "manifest.csv"
            build_avdeepfake_manifest(root, output)
            manifest = pd.read_csv(output)
            self.assertEqual(manifest["label"].tolist(), [0, 1])
            self.assertEqual(set(manifest["split"]), {"generalization"})

    def test_dev_subset_is_deterministic_and_represents_both_classes(self) -> None:
        dataset = SyncVerityDataset.__new__(SyncVerityDataset)
        dataset.clip_ids = ["real-1", "real-2", "fake-1", "fake-2"]
        dataset.rows_by_clip_id = {
            "real-1": {"label": 0}, "real-2": {"label": 0},
            "fake-1": {"label": 1}, "fake-2": {"label": 1},
        }
        dataset._limit_clips(2, seed=42)
        labels = [dataset.rows_by_clip_id[clip_id]["label"] for clip_id in dataset.clip_ids]
        self.assertEqual(len(labels), 2)
        self.assertEqual(set(labels), {0, 1})


if __name__ == "__main__":
    unittest.main()
