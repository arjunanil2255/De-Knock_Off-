"""Centralised configuration loading.

Every module reads its settings through ``load_config`` so that paths and
hyperparameters live in one place (``configs/config.yaml``) and never in code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the YAML configuration as a nested dictionary.

    Args:
        path: Optional explicit path to the config file. Defaults to
            ``configs/config.yaml`` relative to the package root.

    Returns:
        The parsed configuration mapping.
    """
    config_file = Path(path) if path is not None else CONFIG_PATH
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config