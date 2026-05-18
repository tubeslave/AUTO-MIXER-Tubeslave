"""Bridge analysis recommendation artifacts into the operator proposal queue.

The bridge is intentionally write-free. It converts existing analyzer output
into the common product contract used by the frontend, then the queue/apply
handlers decide what is allowed by the current operator mode.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


SAFE_GAIN_SOURCE = "safe_gain_calibrator"
SOUNDCHECK_SOURCE = "soundcheck_recommendation_bundle"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 1) -> Optional[float]:
    numeric = _coerce_float(value)
    return round(numeric, digits) if numeric is not None else None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mapping_lookup(mapping: Any, channel: int) -> Optional[int]:
    if not isinstance(mapping, dict):
        return None
    for key in (channel, str(channel)):
        mapped = mapping.get(key)
        mapped_int = _coerce_int(mapped)
        if mapped_int is not None:
            return mapped_int
    return None


def _cached_trim_from_state(client: Any, channel: int) -> Optional[float]:
    state = getattr(client, "state", None)
    if not isinstance(state, dict):
        return None
    addresses = (
        f"/ch/{channel}/in/set/trim",
        f"/ch/{channel:02d}/in/set/trim",
        f"/ch/{channel}/gain",
    )
    for address in addresses:
        value = _coerce_float(state.get(address))
        if value is not None:
            return value
    return None


def _safe_current_trim(client: Any, channel: int) -> Optional[float]:
    if client is not None and hasattr(client, "get_channel_gain"):
        try:
            value = _coerce_float(client.get_channel_gain(channel))
            if value is not None:
                return value
        except Exception:
            pass
    return _cached_trim_from_state(client, channel)


def _sorted_suggestion_items(suggestions: Dict[Any, Any]) -> Iterable[Tuple[Any, Any]]:
    return sorted(
        suggestions.items(),
        key=lambda item: (_coerce_int(item[0]) is None, _coerce_int(item[0]) or str(item[0])),
    )


def _safe_gain_severity(suggestion: Dict[str, Any], gain_db: float) -> str:
    limited_by = str(suggestion.get("limited_by") or "").lower()
    peak_db = _coerce_float(suggestion.get("peak_db"))
    if limited_by == "peak" or (peak_db is not None and peak_db >= -1.0) or abs(gain_db) >= 8.0:
        return "high"
    if abs(gain_db) >= 4.0:
        return "medium"
    return "low"


def build_safe_gain_operator_proposals(
    suggestions: Dict[Any, Dict[str, Any]],
    *,
    channel_mapping: Optional[Dict[Any, Any]] = None,
    mixer_client: Any = None,
    source: str = SAFE_GAIN_SOURCE,
) -> List[Dict[str, Any]]:
    """Convert SafeGain suggestions into operator proposals.

    The proposals are applyable only when the current cached trim is known.
    Silent/zero-gain suggestions are skipped because there is no operator action.
    """

    if not isinstance(suggestions, dict):
        return []

    proposals: List[Dict[str, Any]] = []
    for raw_audio_channel, raw_suggestion in _sorted_suggestion_items(suggestions):
        if not isinstance(raw_suggestion, dict):
            continue
        audio_channel = _coerce_int(raw_suggestion.get("channel", raw_audio_channel))
        if audio_channel is None:
            continue

        gain_db = _coerce_float(raw_suggestion.get("suggested_gain_db"))
        if gain_db is None or abs(gain_db) < 0.05:
            continue

        mixer_channel = _mapping_lookup(channel_mapping, audio_channel) or audio_channel
        current_trim = _safe_current_trim(mixer_client, mixer_channel)
        target_trim = _clip((current_trim or 0.0) + gain_db, -24.0, 24.0)
        requested_change = None
        if current_trim is not None:
            requested_change = {
                "value_type": "gain",
                "channel": mixer_channel,
                "value": round(target_trim, 1),
                "current_value": round(current_trim, 1),
                "delta_db": round(gain_db, 1),
            }

        sign = "+" if gain_db >= 0 else ""
        proposals.append(
            {
                "id": f"safe_gain:{audio_channel}:{mixer_channel}",
                "title": f"Ch {mixer_channel}: SafeGain {sign}{gain_db:.1f} dB trim",
                "target": mixer_channel,
                "channel": mixer_channel,
                "kind": "gain" if requested_change else "analysis",
                "severity": _safe_gain_severity(raw_suggestion, gain_db),
                "confidence": 0.86 if current_trim is not None else 0.7,
                "source": source,
                "reason": (
                    "SafeGain measured channel loudness and suggests an input trim adjustment."
                    if requested_change
                    else "SafeGain has a trim suggestion, but current cached trim is unknown."
                ),
                "requested_change": requested_change,
                "raw": {
                    "audio_channel": audio_channel,
                    "mixer_channel": mixer_channel,
                    "suggestion": dict(raw_suggestion),
                    "current_trim_db": _round(current_trim),
                    "target_trim_db": _round(target_trim),
                },
            }
        )
    return proposals


def _source_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    payload = item.get("source_payload")
    return payload if isinstance(payload, dict) else item


def _operator_summary(item: Dict[str, Any]) -> str:
    payload = _source_payload(item)
    envelope = payload.get("action_envelope") or item.get("action_envelope") or {}
    if isinstance(envelope, dict):
        summary = envelope.get("operator_summary")
        if summary:
            return str(summary)
    rationale = payload.get("rationale") or item.get("rationale")
    if rationale:
        return str(rationale)
    return "Soundcheck recommendation imported into the operator proposal queue."


def _soundcheck_kind_and_change(item: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    action_type = str(item.get("action_type") or "").strip()
    kind = str(item.get("kind") or action_type or "soundcheck").strip()
    target = item.get("target") if isinstance(item.get("target"), dict) else {}
    requested_state = item.get("requested_state") if isinstance(item.get("requested_state"), dict) else {}
    current_state = item.get("current_state") if isinstance(item.get("current_state"), dict) else {}
    channel = _coerce_int(target.get("mixer_channel") or item.get("channel"))

    value_type = None
    value = None
    current_value = None
    if action_type == "set_input_trim_db" or kind == "input_gain":
        value_type = "gain"
        value = _coerce_float(requested_state.get("trim_db"))
        current_value = _coerce_float(current_state.get("trim_db"))
    elif action_type == "set_fader_db" or kind in {"fader", "feedback_fader_reduce"}:
        value_type = "fader"
        value = _coerce_float(requested_state.get("fader_db"))
        current_value = _coerce_float(current_state.get("fader_db"))

    if value_type and channel is not None and value is not None:
        change = {
            "value_type": value_type,
            "channel": channel,
            "value": round(value, 1),
        }
        if current_value is not None:
            change["current_value"] = round(current_value, 1)
            change["delta_db"] = round(value - current_value, 1)
        return value_type, change
    return kind or "soundcheck", None


def _proposal_title(item: Dict[str, Any], kind: str, requested_change: Optional[Dict[str, Any]]) -> str:
    target = item.get("target") if isinstance(item.get("target"), dict) else {}
    channel = target.get("mixer_channel") or item.get("channel") or "?"
    if requested_change:
        value_type = requested_change["value_type"]
        value = requested_change["value"]
        return f"Ch {channel}: Soundcheck {value_type} -> {value:.1f} dB"
    return f"Ch {channel}: Soundcheck {kind}"


def build_soundcheck_operator_proposals(
    bundle: Optional[Dict[str, Any]],
    *,
    source: str = SOUNDCHECK_SOURCE,
) -> List[Dict[str, Any]]:
    """Convert no-write soundcheck recommendations into operator proposals."""

    if not isinstance(bundle, dict):
        return []
    items = bundle.get("proposals") or bundle.get("recommendations") or []
    if not isinstance(items, list):
        return []

    proposals: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        kind, requested_change = _soundcheck_kind_and_change(item)
        target_dict = item.get("target") if isinstance(item.get("target"), dict) else {}
        item_id = item.get("proposal_id") or item.get("recommendation_id") or item.get("id") or f"{index + 1}"
        mode = str(item.get("mode") or _source_payload(item).get("mode") or "")
        severity = "high" if mode == "approval_required" else "medium"
        if mode == "observe_only":
            severity = "low"
        proposals.append(
            {
                "id": f"soundcheck:{item_id}",
                "title": _proposal_title(item, kind, requested_change),
                "target": (requested_change or {}).get("channel")
                or target_dict.get("mixer_channel")
                or item.get("channel"),
                "channel": (requested_change or {}).get("channel"),
                "kind": (requested_change or {}).get("value_type") or kind,
                "severity": severity,
                "confidence": _coerce_float(item.get("confidence") or _source_payload(item).get("confidence")),
                "source": source,
                "reason": _operator_summary(item),
                "requested_change": requested_change,
                "raw": {
                    "bundle_schema_version": bundle.get("schema_version"),
                    "bundle_generated_at": bundle.get("generated_at"),
                    "soundcheck_item": dict(item),
                },
            }
        )
    return proposals


def import_operator_proposals(
    queue: Any,
    proposals: List[Dict[str, Any]],
    operator_mode: Dict[str, Any],
    *,
    source: str,
) -> Dict[str, Any]:
    """Import proposals into the queue and summarize create/exists/blocked."""

    if queue is None:
        return {
            "type": "operator_proposals_imported",
            "status": "unavailable",
            "success": False,
            "source": source,
            "reason": "operator_proposal_queue_unavailable",
            "operator_mode": operator_mode,
            "imported_count": 0,
            "created_count": 0,
            "existing_count": 0,
            "blocked_count": 0,
            "results": [],
            "generated_at": _utc_now(),
        }

    results = []
    created_count = 0
    existing_count = 0
    blocked_count = 0
    for proposal in proposals:
        result = queue.create(proposal, operator_mode)
        results.append(result)
        if result.get("status") == "created":
            created_count += 1
        elif result.get("status") == "exists":
            existing_count += 1
        elif result.get("status") == "blocked":
            blocked_count += 1

    return {
        "type": "operator_proposals_imported",
        "status": "ok",
        "success": blocked_count < len(proposals) if proposals else True,
        "source": source,
        "operator_mode": operator_mode,
        "imported_count": len(proposals),
        "created_count": created_count,
        "existing_count": existing_count,
        "blocked_count": blocked_count,
        "results": results,
        "generated_at": _utc_now(),
    }
