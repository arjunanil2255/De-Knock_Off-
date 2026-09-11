"""Build the training manifest from raw dataset folders (Phase 0/3).

Scans a dataset root for video files, infers their split and label from the
directory layout, and writes ``data/processed/manifest.csv`` with columns
``{clip_id, video_path, label, split}``. Generalization-test data is never
touched here (see ``rules.md``).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from src.config import load_config

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def infer_label(relative: str) -> int:
    """Return 1 (fake) / 0 (real) from a relative path.

    A path containing "fake" is labelled fake; "real" labels it real;
    otherwise a ``label`` column expected by the caller is required.

    Args:
        relative: Relative path of the video below the dataset root.

    Returns:
        Label integer (1 = fake, 0 = real).

    Raises:
        ValueError: If the path is ambiguous.
    """
    lowered = relative.lower()
    if re.search(r"fake", lowered):
        return 1
    if re.search(r"real", lowered):
        return 0
    raise ValueError(f"Cannot infer label from path: {relative}")


def build_manifest(
    data_dir: str | Path,
    output_csv: str | Path,
    split: str = "train",
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Path:
    """Scan ``data_dir`` and write a train/val manifest CSV.

    Args:
        data_dir: Root folder of the dataset.
        output_csv: Destination CSV path.
        split: Default split if ``val_fraction == 0``.
        val_fraction: Fraction of clips to reserve for validation.
        seed: Seed for the deterministic split.

    Returns:
        Path of the written manifest.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    root = Path(data_dir)
    records: list[dict[str, object]] = []
    for video in sorted(root.rglob("*")):
        if video.suffix.lower() not in VIDEO_EXTS:
            continue
        relative = video.relative_to(root).as_posix()
        try:
            label = infer_label(relative)
        except ValueError:
            continue
        records.append({
            "clip_id": video.stem,
            "video_path": str(video),
            "label": label,
        })

    if not records:
        raise ValueError(f"No video files found under {root}; check config paths.")

    clip_ids = [r["clip_id"] for r in records]
    rng.shuffle(clip_ids)
    n_val = int(len(clip_ids) * val_fraction)
    val_ids = set(clip_ids[:n_val])

    for record in records:
        record["split"] = "val" if record["clip_id"] in val_ids else split

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)

    counts = df.groupby(["split", "label"]).size().to_dict()
    print(f"Manifest written to {out_path} with {len(df)} clips.")
    for (s, label), count in counts.items():
        print(f"  split={s} label={label}: {count}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for manifest generation.

    Args:
        argv: Optional command line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Build manifest.csv from raw dataset.")
    parser.add_argument("data_dir", help="Dataset root (e.g. FakeAVCeleb).")
    parser.add_argument("--output", help="Output CSV (defaults to config manifest_file).")
    parser.add_argument("--split", default="train", help="Split column value for non-val clips.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction.")
    args = parser.parse_args(argv)

    config = load_config()
    output = args.output or config["paths"]["manifest_file"]
    build_manifest(args.data_dir, output, args.split, args.val_fraction)
    return 0


if __name__ == "__main__":
    sys.exit(main())