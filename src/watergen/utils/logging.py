"""Logging helper for consistent console messages."""

from __future__ import annotations

from datetime import datetime, timezone


def info(message: str) -> None:
    """Print an informational log line with UTC timestamp."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"[{ts}] [info] {message}")
