"""Extract both modalities for every row in a training manifest.

This is the bridge between manifest generation and training. Training never
decodes raw clips, so invoke this command before ``src.train``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.extraction.extract_audio import process_audio
from src.extraction.extract_video import process_video

logger = logging.getLogger("extract_manifest")


def extract_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    splits: set[str],
    device: str,
    ctx_id: int,
    force: bool = False,
) -> tuple[int, int]:
    """Cache both feature modalities for selected manifest rows.

    Returns ``(successful, failed)``. One bad clip is logged without
    abandoning the remaining extraction batch.
    """
    manifest = pd.read_csv(manifest_path)
    rows = manifest[manifest["split"].isin(splits)]
    succeeded = 0
    failed = 0
    out = Path(output_dir)
    for row in rows.itertuples(index=False):
        clip_id = str(row.clip_id)
        video_path = Path(row.video_path)
        try:
            process_video(video_path, clip_id, out, ctx_id=ctx_id, force=force)
            process_audio(video_path, clip_id, out, device=device, force=force)
            succeeded += 1
        except Exception as exc:  # noqa: BLE001 - extraction must be batch resilient
            failed += 1
            logger.exception("Extraction failed for %s: %s", video_path, exc)
    return succeeded, failed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for manifest-driven feature extraction."""
    parser = argparse.ArgumentParser(description="Extract cached features for a manifest.")
    parser.add_argument("--manifest", help="Manifest CSV (defaults to config).")
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits.")
    parser.add_argument("--device", default="cpu", help="Audio extractor device.")
    parser.add_argument("--ctx-id", type=int, default=0, help="InsightFace device index.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing feature caches.")
    args = parser.parse_args(argv)

    config = load_config()
    manifest = args.manifest or config["paths"]["manifest_file"]
    successful, failed = extract_manifest(
        manifest,
        config["paths"]["processed_data_dir"],
        {part.strip() for part in args.splits.split(",") if part.strip()},
        args.device,
        args.ctx_id,
        args.force,
    )
    print(f"Extraction complete: successful={successful} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
