"""Model evaluation and generalization test (Phase 5).

Runs a saved checkpoint over a manifest split (FakeAVCeleb val or the held-out
AV-Deepfake1M / LAV-DF generalization split) and reports accuracy,
balanced accuracy, AUROC and F1. When both a baseline and a generalization
metric are available the generalization drop is printed for comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from src.config import load_config
from src.data.dataset import SyncVerityDataset
from src.models.model import SyncVerityModel


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    metric_names: list[str],
) -> dict[str, float]:
    """Compute classification metrics over a dataloader.

    Args:
        model: Model in eval mode.
        loader: DataLoader to evaluate.
        device: Torch device.
        metric_names: Metrics to compute (subset of ``accuracy``,
            ``balanced_accuracy``, ``auroc``, ``f1``).

    Returns:
        Dictionary of metric name to value.
    """
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            logits = model(input_batch)["logits"]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            all_probs.append(probs[:, 1])
            all_labels.extend(batch["label"].tolist())

    if not all_probs:
        return {name: float("nan") for name in metric_names}

    probs = np.concatenate(all_probs)
    labels = np.asarray(all_labels)
    predictions = (probs >= 0.5).astype(int)

    results: dict[str, float] = {}
    for name in metric_names:
        if name == "accuracy":
            results[name] = accuracy_score(labels, predictions)
        elif name == "balanced_accuracy":
            results[name] = balanced_accuracy_score(labels, predictions)
        elif name == "auroc":
            # AUROC is undefined when a small/dev split contains one class.
            results[name] = (
                roc_auc_score(labels, probs) if np.unique(labels).size == 2 else float("nan")
            )
        elif name == "f1":
            results[name] = f1_score(labels, predictions, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {name}")
    return results


def evaluate(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str,
    device: torch.device,
    manifest_path: str | Path | None = None,
) -> dict[str, float]:
    """Evaluate a checkpoint on one split and return per-metric scores.

    Args:
        config: Project configuration.
        checkpoint_path: Path to a training checkpoint.
        split: Manifest split to evaluate.
        device: Torch device.
        manifest_path: Optional manifest override, required when evaluating a
            physically separate held-out corpus.

    Returns:
        Dictionary mapping metric name to value.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    mode = checkpoint.get("mode", "combined")
    flags = {
        "fusion": {"enable_fusion": True, "enable_video_artifact": False, "enable_audio_artifact": False},
        "artifact": {"enable_fusion": False, "enable_video_artifact": True, "enable_audio_artifact": True},
        "combined": {"enable_fusion": True, "enable_video_artifact": True, "enable_audio_artifact": True},
    }[mode]

    model = SyncVerityModel(config, **flags)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    dataset = SyncVerityDataset(
        processed_dir=config["paths"]["processed_data_dir"],
        manifest_path=manifest_path or config["paths"]["manifest_file"],
        split=split,
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)

    print(f"Evaluating {checkpoint_path} on split={split} ({len(dataset)} clips)")
    metrics = evaluate_split(model, loader, device, list(config["evaluation"]["metrics"]))
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for evaluation.

    Args:
        argv: Optional command line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Evaluate a SyncVerity checkpoint.")
    parser.add_argument("checkpoint", help="Path to a saved checkpoint file.")
    parser.add_argument(
        "--split", default="val", help="Manifest split to evaluate (val / generalization)."
    )
    parser.add_argument("--manifest", help="Optional manifest CSV for the target split.")
    parser.add_argument(
        "--compare-split",
        help="Reference split used to calculate metric drop (for example: val).",
    )
    parser.add_argument(
        "--compare-manifest",
        help="Optional manifest CSV containing the reference split.",
    )
    parser.add_argument("--device", default=None, help="Torch device (defaults to auto).")
    args = parser.parse_args(argv)

    config = load_config()
    training_cfg = config["training"]
    device_name = args.device or training_cfg.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)

    target = evaluate(config, args.checkpoint, args.split, device, args.manifest)
    report: dict[str, Any] = {"target_split": args.split, "target_metrics": target}
    if args.compare_split:
        reference = evaluate(
            config, args.checkpoint, args.compare_split, device, args.compare_manifest
        )
        drop = {
            metric: reference[metric] - target[metric]
            for metric in target
            if metric in reference and not (np.isnan(reference[metric]) or np.isnan(target[metric]))
        }
        report.update({
            "reference_split": args.compare_split,
            "reference_metrics": reference,
            "generalization_drop": drop,
        })
        print("Generalization drop (reference - target):")
        for metric, value in drop.items():
            print(f"  {metric}: {value:.4f}")

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / f"evaluation_{Path(args.checkpoint).stem}_{args.split}.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Saved evaluation report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
