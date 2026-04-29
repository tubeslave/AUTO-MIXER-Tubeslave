"""Shared artifact locations for offline mixing outputs.

Offline renders should keep listenable audio separate from diagnostic logs:
audio goes to the operator's Desktop/Ai MIXING folder, while reports and JSONL
logs go to Desktop/Ai LOGS. Live mixer control is intentionally unaffected.
"""

from __future__ import annotations

from pathlib import Path


AI_MIXING_DIR_NAME = "Ai MIXING"
AI_LOGS_DIR_NAME = "Ai LOGS"


def desktop_dir() -> Path:
    """Return the current user's Desktop directory."""

    return Path.home() / "Desktop"


def ai_mixing_dir() -> Path:
    """Return the shared directory for rendered audio artifacts."""

    return desktop_dir() / AI_MIXING_DIR_NAME


def ai_logs_dir() -> Path:
    """Return the shared directory for reports and JSONL logs."""

    return desktop_dir() / AI_LOGS_DIR_NAME


def ensure_ai_output_dirs() -> tuple[Path, Path]:
    """Create the shared audio/log directories and return them."""

    mixing_dir = ai_mixing_dir()
    logs_dir = ai_logs_dir()
    mixing_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return mixing_dir, logs_dir


def ai_mixing_path(filename: str | Path) -> Path:
    """Return a path inside Desktop/Ai MIXING."""

    return ai_mixing_dir() / Path(filename)


def ai_logs_path(filename: str | Path) -> Path:
    """Return a path inside Desktop/Ai LOGS."""

    return ai_logs_dir() / Path(filename)


def ensure_parent_dir(path: str | Path) -> Path:
    """Expand a path and create its parent directory."""

    expanded = Path(path).expanduser()
    expanded.parent.mkdir(parents=True, exist_ok=True)
    return expanded
