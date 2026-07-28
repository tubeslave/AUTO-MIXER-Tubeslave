#!/usr/bin/env python3
"""Repository CLI entrypoint.

Phase 0 exposes only the read-only Inspector command family here.
"""

from __future__ import annotations

try:
    from backend.system_inspector.cli import main
except ImportError:  # pragma: no cover - PYTHONPATH=backend script execution
    from system_inspector.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
