"""Small adapters that normalize analyzer outputs for Decision Engine v2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List


def normalize_analyzer_output(payload: Any) -> Dict[str, Any]:
    """Return a stable analyzer payload with a ``channels`` list.

    Existing code exposes metrics in several shapes: backend ``ChannelInfo``
    objects, dicts keyed by channel id, and already-normalized lists. The v2
    Decision Engine accepts this small common shape so it can stay independent
    from the legacy live engine.
    """
    if payload is None:
        return {"channels": [], "source_module": "unknown"}

    if isinstance(payload, Mapping):
        if "channels" in payload:
            channels = payload.get("channels") or []
            if isinstance(channels, Mapping):
                normalized = []
                for key, value in channels.items():
                    item = dict(value or {})
                    item.setdefault("channel_id", _coerce_channel_id(key))
                    normalized.append(item)
                channels = normalized
            return {
                **dict(payload),
                "channels": [dict(channel) for channel in channels if isinstance(channel, Mapping)],
            }
        return {"channels": [dict(payload)], "source_module": str(payload.get("source_module", "dict"))}

    if isinstance(payload, list):
        return {"channels": [dict(item) for item in payload if isinstance(item, Mapping)]}

    return {"channels": [], "source_module": payload.__class__.__name__}


def _coerce_channel_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
