"""JSON report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_report(payload: Any, path: str | Path) -> Path:
    """Write a JSON-safe report."""
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "to_dict"):
        data = payload.to_dict()
    else:
        data = payload
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
