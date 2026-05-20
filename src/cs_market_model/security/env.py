"""Environment loading and secret-safe helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from cs_market_model.config import PROJECT_ROOT


def load_local_env(env_path: Path | None = None) -> None:
    """Load local .env values without overriding existing environment variables."""
    load_dotenv(env_path or PROJECT_ROOT / ".env", override=False)


def require_env(name: str) -> str:
    """Return a required environment variable with a clear error message."""
    load_local_env()
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is required. Set it in your shell or in the local .env file."
        )
    return value


def redact_secret(value: str | None, visible_prefix: int = 4, visible_suffix: int = 2) -> str:
    """Return a display-safe version of a secret."""
    if not value:
        return "<unset>"
    if len(value) <= visible_prefix + visible_suffix:
        return "<redacted>"
    return f"{value[:visible_prefix]}...{value[-visible_suffix:]}"

