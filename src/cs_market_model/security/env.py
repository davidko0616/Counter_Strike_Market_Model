"""Environment loading and secret-safe helpers."""

from __future__ import annotations

import os
from pathlib import Path

from cs_market_model.config import PROJECT_ROOT


def load_local_env(env_path: Path | None = None) -> None:
    """Load local .env values without overriding existing environment variables."""
    path = env_path or PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv(path, override=False)
        return

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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

