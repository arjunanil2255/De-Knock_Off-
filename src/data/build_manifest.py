"""Build the training manifest from raw dataset folders (Phase 0/3).

Scans a dataset root for video files, infers their split and label from the
directory layout, and writes ``data/processed/manifest.csv`` with columns
``{clip_id, video_path, label, source_group, split}``. A generalization corpus
must be built into a separate manifest with ``--split generalization`` so it is
never mixed with train/validation data (see ``rules.md``).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

from src.config import load_config

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def make_clip_id(relative_path: str, namespace: str = "") -> str:
    """Create a cache-safe ID unique across folders and dataset roots."""
    path = Path(relative_path)
    readable = re.sub(r"[^A-Za-z0-9]+", "-", path.stem).strip("-") or "clip"
    digest = hashlib.sha1(f"{namespace}/{relative_path}".encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def infer_source_group(relative_path: str, source_group_pattern: str) -> str:
    """Extract an auditable source group used to prevent split leakage.

    FakeAVCeleb-style names usually begin with an identity/source token before
    ``_`` or ``-``. Projects with a different layout should supply a regex
    containing one capture group through ``--source-group-pattern``.
    """
    stem = Path(relative_path).stem
    match = re.search(source_group_pattern, stem)
    if match and match.groups():
        return match.group(1)
    return stem


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
    source_group_pattern: str = r"^([^_-]+)",
) -> Path:
    """Scan ``data_dir`` and write a train/val manifest CSV.

    Args:
        data_dir: Root folder of the dataset.
        output_csv: Destination CSV path.
        split: Default split if ``val_fraction == 0``.
        val_fraction: Fraction of clips to reserve for validation.
        seed: Seed for the deterministic split.
        source_group_pattern: Regex whose first capture group identifies clips
            from the same underlying source/identity.

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
            "clip_id": make_clip_id(relative, namespace=str(root.resolve())),
            "video_path": str(video),
            "label": label,
            "source_group": infer_source_group(relative, source_group_pattern),
        })

    if not records:
        raise ValueError(f"No video files found under {root}; check config paths.")

    if split == "generalization":
        # A held-out generalization corpus must never be partitioned into the
        # training validation split by accident.
        val_fraction = 0.0
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")

    groups: dict[tuple[str, tuple[int, ...]], list[dict[str, object]]] = {}
    for record in records:
        group = str(record["source_group"])
        # Labels are included in the bucket after the first pass below, so
        # mixed-label source groups stay intact and are still auditable.
        groups.setdefault((group, ()), []).append(record)

    buckets: dict[tuple[int, ...], list[str]] = {}
    for (group, _), group_rows in groups.items():
        label_signature = tuple(sorted({int(row["label"]) for row in group_rows}))
        buckets.setdefault(label_signature, []).append(group)

    val_groups: set[str] = set()
    for group_names in buckets.values():
        rng.shuffle(group_names)
        # Do not place a group in validation when it is the only source group
        # available for that label signature.
        count = int(round(len(group_names) * val_fraction))
        if val_fraction > 0.0 and len(group_names) > 1:
            count = max(1, count)
        val_groups.update(group_names[:count])

    for record in records:
        record["split"] = "val" if record["source_group"] in val_groups else split

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
    parser.add_argument(
        "--source-group-pattern",
        default=r"^([^_-]+)",
        help="Regex with capture group 1 identifying clips from one source/identity.",
    )
    args = parser.parse_args(argv)

    config = load_config()
    output = args.output or config["paths"]["manifest_file"]
    build_manifest(
        args.data_dir,
        output,
        args.split,
        args.val_fraction,
        source_group_pattern=args.source_group_pattern,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
