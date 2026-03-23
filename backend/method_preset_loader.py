"""Shared loader for method-specific preset bases.

Unified JSON schema (for gain/eq/compressor):
{
  "schema_version": 1,
  "method": "gain|eq|compressor",
  "defaults": {...},
  "instruments": [
    {
      "id": "kick",
      "aliases": ["kick_in", "kick_out"],
      "params": {...}
    }
  ]
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_value: str, fallback: Path) -> Path:
    if not path_value:
        return fallback
    p = Path(path_value)
    if p.is_absolute():
        return p
    return _workspace_root() / p


def build_preset_index(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build id/alias -> instrument entry index."""
    out: Dict[str, Dict[str, Any]] = {}
    for item in data.get("instruments", []):
        if not isinstance(item, dict):
            continue
        preset_id = str(item.get("id", "")).strip()
        if not preset_id:
            continue
        normalized_id = preset_id.lower().replace("-", "_").replace(" ", "_")
        out[normalized_id] = item
        for alias in item.get("aliases", []) or []:
            alias_key = str(alias).strip().lower().replace("-", "_").replace(" ", "_")
            if alias_key:
                out[alias_key] = item
    return out


def load_method_preset_base(path: Path, expected_method: str) -> Dict[str, Any]:
    """Load and minimally validate a method preset base."""
    defaults = {
        "schema_version": 1,
        "method": expected_method,
        "defaults": {},
        "instruments": [],
        "index": {},
    }
    try:
        if not path.exists():
            logger.warning("Preset base file not found for %s: %s", expected_method, path)
            return defaults
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            logger.warning("Preset base file has invalid root object: %s", path)
            return defaults
        method_name = str(raw.get("method", "")).strip().lower()
        if method_name and method_name != expected_method:
            logger.warning(
                "Preset base method mismatch. expected=%s actual=%s path=%s",
                expected_method,
                method_name,
                path,
            )
        instruments = raw.get("instruments", [])
        if not isinstance(instruments, list):
            instruments = []
        normalized = {
            "schema_version": int(raw.get("schema_version", 1)),
            "method": expected_method,
            "defaults": raw.get("defaults", {}) if isinstance(raw.get("defaults"), dict) else {},
            "instruments": instruments,
        }
        normalized["index"] = build_preset_index(normalized)
        return normalized
    except Exception as exc:
        logger.warning("Failed to load preset base %s from %s: %s", expected_method, path, exc)
        return defaults

