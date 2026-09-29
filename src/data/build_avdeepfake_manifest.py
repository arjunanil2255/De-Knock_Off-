"""Build a held-out binary-classification manifest from AV-Deepfake1M metadata.

AV-Deepfake1M labels clips through ``modify_type`` in JSON metadata rather
than through ``real``/``fake`` directory names, so it must not use the generic
folder-scanning manifest builder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.data.build_manifest import make_clip_id


REAL_MODIFY_TYPE = "real"


def build_avdeepfake_manifest(
    data_dir: str | Path, output_csv: str | Path, metadata_files: list[str | Path] | None = None
) -> Path:
    """Create a ``generalization`` manifest from AV-Deepfake1M JSON files.

    ``real`` entries get label 0; all documented modification modes get label
    1. The original-video field is retained as ``source_group`` for auditability
    but no split is created: this corpus remains entirely held out.
    """
    root = Path(data_dir)
    files = [Path(path) for path in metadata_files] if metadata_files else sorted(root.glob("*_metadata.json"))
    if not files:
        raise FileNotFoundError(f"No *_metadata.json files found under {root}.")

    records: list[dict[str, Any]] = []
    for metadata_path in files:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list in {metadata_path}, got {type(payload).__name__}.")
        for entry in payload:
            if not isinstance(entry, dict) or "file" not in entry or "modify_type" not in entry:
                raise ValueError(f"Invalid AV-Deepfake1M metadata entry in {metadata_path}: {entry!r}")
            relative = str(entry["file"]).replace("\\", "/")
            video_path = root / relative
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"Metadata references missing video: {video_path}. "
                    "Extract the complete official dataset before building its manifest."
                )
            modify_type = str(entry["modify_type"])
            records.append({
                "clip_id": make_clip_id(relative, namespace=str(root.resolve())),
                "video_path": str(video_path),
                "label": 0 if modify_type == REAL_MODIFY_TYPE else 1,
                "split": "generalization",
                "source_group": str(entry.get("original") or Path(relative).stem),
                "modify_type": modify_type,
                "metadata_file": metadata_path.name,
            })

    if not records:
        raise ValueError("No AV-Deepfake1M records found in the supplied metadata.")
    clip_ids = [str(record["clip_id"]) for record in records]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("Duplicate clip IDs in AV-Deepfake1M metadata.")

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    counts = frame.groupby(["modify_type", "label"]).size().to_dict()
    print(f"AV-Deepfake1M manifest written to {output} with {len(frame)} clips.")
    for (modify_type, label), count in counts.items():
        print(f"  modify_type={modify_type} label={label}: {count}")
    return output


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for AV-Deepfake1M held-out manifest generation."""
    parser = argparse.ArgumentParser(description="Build an AV-Deepfake1M held-out manifest.")
    parser.add_argument("data_dir", help="Official AV-Deepfake1M dataset root.")
    parser.add_argument("--output", help="Output CSV (defaults to data/processed).")
    parser.add_argument(
        "--metadata", action="append", help="Metadata JSON path; repeat for multiple files."
    )
    args = parser.parse_args(argv)
    config = load_config()
    output = args.output or Path(config["paths"]["processed_data_dir"]) / "av_deepfake1m_manifest.csv"
    build_avdeepfake_manifest(args.data_dir, output, args.metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
