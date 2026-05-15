"""Normalize critic/evaluation payloads for Decision Engine v2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List


def normalize_critic_evaluations(payload: Any) -> Dict[str, Any]:
    """Return critic data with optional global and per-channel sections."""
    if payload is None:
        return {"global": {}, "channels": []}
    if isinstance(payload, Mapping):
        channels = payload.get("channels", [])
        if isinstance(channels, Mapping):
            normalized_channels: List[Dict[str, Any]] = []
            for key, value in channels.items():
                item = dict(value or {})
                try:
                    item.setdefault("channel_id", int(key))
                except (TypeError, ValueError):
                    item.setdefault("target", str(key))
                normalized_channels.append(item)
            channels = normalized_channels
        elif not isinstance(channels, list):
            channels = []
        return {
            "global": dict(payload.get("global", {})),
            "channels": [dict(item) for item in channels if isinstance(item, Mapping)],
        }
    if isinstance(payload, list):
        return {"global": {}, "channels": [dict(item) for item in payload if isinstance(item, Mapping)]}
    return {"global": {}, "channels": []}
