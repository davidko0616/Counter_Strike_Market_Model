"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml_config(name: str) -> dict[str, Any]:
    """Load a YAML config from the repository configs directory."""
    import yaml

    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of config: {path}")
    return data


def data_path(*parts: str) -> Path:
    """Return a path inside the local data directory."""
    return PROJECT_ROOT.joinpath("data", *parts)


def reports_path(*parts: str) -> Path:
    """Return a path inside the local reports directory."""
    return PROJECT_ROOT.joinpath("reports", *parts)

