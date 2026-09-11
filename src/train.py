"""End-to-end training loop for SyncVerity (Phase 3/4).

Supports three modes so the ablation table can be produced honestly:
``fusion`` trains the cross-attention consistency path alone,
``artifact`` trains the per-modality CNN baselines alone, and ``combined``
trains everything jointly. Checkpoints are written every N epochs and the
split sizes / class balance are logged at the start of every run.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.dataset import SyncVerityDataset
from src.models.model import SyncVerityModel

BRANCH_FLAGS: dict[str, dict[str, bool]] = {
    "fusion": {"enable_fusion": True, "enable_video_artifact": False, "enable_audio_artifact": False},
    "artifact": {"enable_fusion": False, "enable_video_artifact": True, "enable_audio_artifact": True},
    "combined": {"enable_fusion": True, "enable_video_artifact": True, "enable_audio_artifact": True},
}


def set_seed(seed: int) -> None:
    """Seed all randomness sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_split_summary(dataset: SyncVerityDataset, name: str) -> None:
    """Log split size and class balance for one dataset."""
    if len(dataset) == 0:
        print(f"[{name}] empty split")
        return
    labels = [dataset[i]["label"] for i in range(len(dataset))]
    real = sum(1 for label in labels if label == 0)
    fake = sum(1 for label in labels if label == 1)
    print(f"[{name}] clips={len(dataset)} real={real} fake={fake}")


def build_dataloader(
    config: dict[str, Any], split: str, batch_size: int, shuffle: bool
) -> DataLoader:
    """Create a dataloader for a manifest split."""
    dataset = SyncVerityDataset(
        processed_dir=config["paths"]["processed_data_dir"],
        manifest_path=config["paths"]["manifest_file"],
        split=split,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=True if torch.cuda.is_available() else False,
    )


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Return (loss, accuracy) over a dataloader without gradients.

    Args:
        model: Model in eval mode.
        loader: DataLoader to evaluate.
        device: Torch device.

    Returns:
        Average loss and accuracy.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            output = model({key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)})
            loss = criterion(output["logits"], labels)
            total_loss += loss.item() * labels.size(0)
            correct += (output["logits"].argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch, returning (loss, accuracy)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="train", leave=False):
        labels = batch["label"].to(device)
        input_batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}

        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output["logits"], labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (output["logits"].argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train(
    config: dict[str, Any],
    mode: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> SyncVerityModel:
    """Run the full training procedure for a given mode.

    Args:
        config: Project configuration.
        mode: ``fusion``, ``artifact``, or ``combined``.
        epochs: Number of epochs to train.
        batch_size: Batch size.
        learning_rate: Optimizer learning rate.
        device: Torch device.

    Returns:
        The trained model.
    """
    set_seed(int(config["training"]["seed"]))

    train_loader = build_dataloader(config, "train", batch_size, shuffle=True)
    val_loader = build_dataloader(config, "val", batch_size, shuffle=False)
    log_split_summary(train_loader.dataset, "train")
    log_split_summary(val_loader.dataset, "val")

    flags = BRANCH_FLAGS[mode]
    model = SyncVerityModel(config, **flags).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=float(config["training"]["weight_decay"]))

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        print(f"epoch {epoch}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        checkpoint_every = int(config["training"]["checkpoint_every"])
        if epoch % checkpoint_every == 0 or epoch == epochs:
            state = {
                "epoch": epoch,
                "mode": mode,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            torch.save(state, checkpoint_dir / f"syncverity_{mode}_epoch{epoch}.pt")

    return model


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for training.

    Args:
        argv: Optional command line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Train SyncVerity model.")
    parser.add_argument(
        "--mode",
        choices=["fusion", "artifact", "combined"],
        default="fusion",
        help="Which branches to train (ablation mode).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", default=None, help="Torch device (defaults to auto).")
    args = parser.parse_args(argv)

    config = load_config()
    training_cfg = config["training"]

    device_name = args.device or training_cfg.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    epochs = args.epochs if args.epochs is not None else int(training_cfg["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(training_cfg["batch_size"])
    learning_rate = args.lr if args.lr is not None else float(training_cfg["learning_rate"])

    print(f"Training mode={args.mode} device={device} epochs={epochs} batch={batch_size} lr={learning_rate}")
    train(config, args.mode, epochs, batch_size, learning_rate, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())