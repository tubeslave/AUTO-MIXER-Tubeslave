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
CACHE_FRESHNESS_STALE_AFTER_SEC = 30.0


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


def _coerce_epoch_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, datetime):
        timestamp = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return timestamp.timestamp()
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return _coerce_epoch_seconds(float(text))
        except ValueError:
            pass
        try:
            return _coerce_epoch_seconds(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def _iso_from_epoch(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def _state_cache_timestamp(client: Any, address: str) -> Optional[float]:
    for attr in ("state_timestamps", "_state_timestamps", "state_updated_at", "_state_updated_at"):
        timestamps = getattr(client, attr, None)
        if not isinstance(timestamps, dict):
            continue
        raw = timestamps.get(address)
        if isinstance(raw, dict):
            raw = raw.get("updated_at") or raw.get("timestamp") or raw.get("ts")
        epoch = _coerce_epoch_seconds(raw)
        if epoch is not None:
            return epoch
    return None


def _cached_value_address(client: Any, channel: int, value_type: str) -> Optional[str]:
    state = getattr(client, "state", None)
    if not isinstance(state, dict):
        return None
    candidates = ()
    if value_type == "gain":
        candidates = (
            f"/ch/{channel}/in/set/trim",
            f"/ch/{channel:02d}/in/set/trim",
            f"/ch/{channel}/gain",
        )
    elif value_type == "fader":
        candidates = (
            f"/ch/{channel}/fdr",
            f"/ch/{channel:02d}/fdr",
        )
    for address in candidates:
        if state.get(address) is not None:
            return address
    return None


def _cache_freshness_for_change(client: Any, change: Dict[str, Any]) -> Dict[str, Any]:
    channel = _coerce_int(change.get("channel"))
    value_type = str(change.get("value_type") or "").strip().lower()
    address = _cached_value_address(client, channel, value_type) if channel is not None else None
    updated_epoch = _state_cache_timestamp(client, address) if address else None
    now = datetime.now(timezone.utc).timestamp()
    age_sec = round(max(0.0, now - updated_epoch), 3) if updated_epoch is not None else None
    if value_type not in {"gain", "fader"} or channel is None:
        status = "unsupported"
    elif address is None:
        status = "missing"
    elif updated_epoch is None:
        status = "unknown"
    elif age_sec is not None and age_sec > CACHE_FRESHNESS_STALE_AFTER_SEC:
        status = "stale"
    else:
        status = "fresh"
    return {
        "status": status,
        "address": address,
        "value_type": value_type or None,
        "channel": channel,
        "source": "state_timestamps" if updated_epoch is not None else "unavailable",
        "updated_at": _iso_from_epoch(updated_epoch),
        "age_sec": age_sec,
        "stale_after_sec": CACHE_FRESHNESS_STALE_AFTER_SEC,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _append_cache_freshness_evidence(change: Dict[str, Any], freshness: Dict[str, Any]) -> None:
    change["cache_freshness"] = freshness
    change.setdefault("safety_evidence", []).append({
        "label": "Cache freshness",
        "value": freshness.get("status"),
        "address": freshness.get("address"),
        "age_sec": freshness.get("age_sec"),
        "stale_after_sec": freshness.get("stale_after_sec"),
    })


def _gate_proposal_cache_freshness(proposal: Dict[str, Any], client: Any) -> Dict[str, Any]:
    if client is None:
        return proposal
    if not isinstance(getattr(client, "state", None), dict):
        return proposal
    change = proposal.get("requested_change")
    if not isinstance(change, dict):
        return proposal
    freshness = _cache_freshness_for_change(client, change)
    proposal.setdefault("raw", {})["cache_freshness"] = [freshness]
    proposal["cache_freshness"] = [freshness]
    if freshness["status"] == "fresh":
        _append_cache_freshness_evidence(change, freshness)
        proposal["raw"]["apply_status"] = proposal["raw"].get("apply_status", "cache_freshness_verified")
        proposal["apply_status"] = proposal["raw"]["apply_status"]
        return proposal

    proposal["raw"]["apply_status"] = "blocked_cache_freshness_not_fresh"
    proposal["raw"]["apply_blockers"] = [f"cache_freshness_{freshness['status']}:{freshness.get('address') or 'unknown'}"]
    proposal["raw"]["blocked_requested_change"] = dict(change)
    proposal["apply_status"] = proposal["raw"]["apply_status"]
    proposal["apply_blockers"] = proposal["raw"]["apply_blockers"]
    proposal["blocked_requested_change"] = proposal["raw"]["blocked_requested_change"]
    proposal["kind"] = "analysis"
    proposal["requested_change"] = None
    proposal["reason"] = (
        f"{proposal.get('reason', '').rstrip()} "
        "Current cached console value is not fresh enough for an applyable proposal."
    ).strip()
    proposal["title"] = f"{proposal.get('title', 'Operator recommendation')}: cache freshness not verified"
    proposal["confidence"] = min(_coerce_float(proposal.get("confidence")) or 0.7, 0.7)
    return proposal


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
    character = suggestion.get("input_gain_character") or {}
    distortion_risk = str(character.get("distortion_risk") or "").lower() if isinstance(character, dict) else ""
    if limited_by in {"peak", "peak_headroom"} or distortion_risk == "high" or (peak_db is not None and peak_db >= -1.0) or abs(gain_db) >= 8.0:
        return "high"
    if abs(gain_db) >= 4.0:
        return "medium"
    return "low"


def _safe_gain_evidence(suggestion: Dict[str, Any], gain_db: float) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = [
        {"label": "Method", "value": suggestion.get("method_id") or "safe_gain"},
        {"label": "Role", "value": suggestion.get("input_trim_role") or "input_headroom"},
        {"label": "Direction", "value": "gain_down" if gain_db < 0 else "gain_up"},
    ]
    target_peak = _round(suggestion.get("target_peak_db"))
    peak = _round(suggestion.get("peak_db"))
    if peak is not None:
        evidence.append({"label": "Peak", "value": peak, "target": target_peak})
    character = suggestion.get("input_gain_character")
    if isinstance(character, dict):
        evidence.extend([
            {"label": "Preamp character", "value": character.get("character")},
            {
                "label": "Preamp zone",
                "value": f"{character.get('current_zone')} -> {character.get('target_zone')}",
            },
            {"label": "Distortion risk", "value": character.get("distortion_risk")},
        ])
        ensemble = character.get("soundcheck_ensemble_context")
        if isinstance(ensemble, dict) and (ensemble.get("provided") or ensemble.get("reason")):
            reason = str(ensemble.get("reason") or "unknown")
            ensemble_enabled = ensemble.get("enabled") is not False and reason != "disabled"
            time_coherence_enabled = bool(ensemble.get("time_coherence_enabled")) and ensemble_enabled
            evidence.append({
                "label": "Ensemble context",
                "value": "active" if ensemble.get("active") else reason,
                "reason": reason,
            })
            if time_coherence_enabled:
                if not ensemble.get("time_coherence_evaluable", True):
                    time_value = "missing"
                elif ensemble.get("time_coherent"):
                    time_value = "coherent"
                else:
                    time_value = "stale"
                evidence.append({
                    "label": "Ensemble time coherence",
                    "value": time_value,
                    "reason": reason,
                })
                coherent_sources = _coerce_int(ensemble.get("time_coherent_loud_sources"))
                required_sources = _coerce_int(ensemble.get("required_time_coherent_sources"))
                if coherent_sources is not None and required_sources is not None:
                    evidence.append({
                        "label": "Ensemble time sources",
                        "value": coherent_sources,
                        "target": required_sources,
                    })
                reference_lag = _coerce_int(ensemble.get("reference_max_lag_samples"))
                max_lag = _coerce_int(ensemble.get("max_context_lag_samples"))
                if reference_lag is not None and max_lag is not None:
                    evidence.append({
                        "label": "Ensemble lag samples",
                        "value": reference_lag,
                        "target": max_lag,
                    })
            if ensemble.get("operator_review_only") is not None:
                evidence.append({
                    "label": "Ensemble apply policy",
                    "value": "operator_review_only" if ensemble.get("operator_review_only") else "not_restricted",
                })
    note = suggestion.get("operator_note")
    if note:
        evidence.append({"label": "Operator note", "value": str(note)})
    return evidence


def _safe_gain_ensemble_context_summary(suggestion: Dict[str, Any]) -> Optional[str]:
    character = suggestion.get("input_gain_character")
    if not isinstance(character, dict):
        return None
    ensemble = character.get("soundcheck_ensemble_context")
    if not isinstance(ensemble, dict) or not (ensemble.get("provided") or ensemble.get("reason")):
        return None

    reason = str(ensemble.get("reason") or "unknown")
    context_state = "active" if ensemble.get("active") else "inactive"
    parts = [f"Ensemble context {context_state} ({reason})"]
    ensemble_enabled = ensemble.get("enabled") is not False and reason != "disabled"
    time_coherence_enabled = bool(ensemble.get("time_coherence_enabled")) and ensemble_enabled
    if time_coherence_enabled:
        if not ensemble.get("time_coherence_evaluable", True):
            time_value = "missing"
        elif ensemble.get("time_coherent"):
            time_value = "coherent"
        else:
            time_value = "stale"
        parts.append(f"time {time_value}")

        coherent_sources = _coerce_int(ensemble.get("time_coherent_loud_sources"))
        required_sources = _coerce_int(ensemble.get("required_time_coherent_sources"))
        if coherent_sources is not None and required_sources is not None:
            parts.append(f"sources {coherent_sources}/{required_sources}")

        reference_lag = _coerce_int(ensemble.get("reference_max_lag_samples"))
        max_lag = _coerce_int(ensemble.get("max_context_lag_samples"))
        if reference_lag is not None and max_lag is not None:
            parts.append(f"lag {reference_lag}/{max_lag} samples")

    if ensemble.get("operator_review_only"):
        parts.append("operator review only")
    return "; ".join(parts) + "."


def _safe_gain_reason(suggestion: Dict[str, Any], requested_change: Optional[Dict[str, Any]]) -> str:
    if requested_change:
        reason = (
            suggestion.get("operator_note")
            or "SafeGain measured input headroom and suggests an input trim adjustment."
        )
    else:
        reason = "SafeGain has a trim suggestion, but current cached trim is unknown."

    ensemble_summary = _safe_gain_ensemble_context_summary(suggestion)
    if ensemble_summary:
        return f"{reason} {ensemble_summary}"
    return str(reason)


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
        suggested_target = _coerce_float(raw_suggestion.get("target_trim_db"))
        target_trim = suggested_target if suggested_target is not None else (current_trim or 0.0) + gain_db
        target_trim = _clip(target_trim, -18.0, 18.0)
        requested_change = None
        if current_trim is not None:
            requested_change = {
                "value_type": "gain",
                "channel": mixer_channel,
                "value": round(target_trim, 1),
                "current_value": round(current_trim, 1),
                "delta_db": round(gain_db, 1),
                "old_value": round(current_trim, 1),
                "proposed_value": round(target_trim, 1),
                "rollback_value": round(current_trim, 1),
                "readback_required": True,
                "safety_evidence": _safe_gain_evidence(raw_suggestion, gain_db),
            }

        sign = "+" if gain_db >= 0 else ""
        confidence = _coerce_float(raw_suggestion.get("confidence"))
        if confidence is None:
            confidence = 0.86 if current_trim is not None else 0.7
        proposal = {
            "id": f"safe_gain:{audio_channel}:{mixer_channel}",
            "title": f"Ch {mixer_channel}: SafeGain {sign}{gain_db:.1f} dB trim",
            "target": mixer_channel,
            "channel": mixer_channel,
            "kind": "gain" if requested_change else "analysis",
            "severity": _safe_gain_severity(raw_suggestion, gain_db),
            "confidence": confidence if current_trim is not None else min(confidence, 0.7),
            "source": source,
            "reason": _safe_gain_reason(raw_suggestion, requested_change),
            "requested_change": requested_change,
            "raw": {
                "audio_channel": audio_channel,
                "mixer_channel": mixer_channel,
                "suggestion": dict(raw_suggestion),
                "current_trim_db": _round(current_trim),
                "target_trim_db": _round(target_trim),
            },
        }
        proposals.append(_gate_proposal_cache_freshness(proposal, mixer_client))
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
    mixer_client: Any = None,
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
        proposal = {
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
        proposals.append(_gate_proposal_cache_freshness(proposal, mixer_client))
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
