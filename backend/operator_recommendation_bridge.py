"""Bridge analysis recommendation artifacts into the operator proposal queue.

The bridge is intentionally write-free. It converts existing analyzer output
into the common product contract used by the frontend, then the queue/apply
handlers decide what is allowed by the current operator mode.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from operator_mode_policy import WING_DEPLOYMENT_LIVE_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_mode_policy import WING_DEPLOYMENT_LIVE_WRITE_KINDS

try:
    from operator_proposal_queue import (
        SOUNDCHECK_CORRECTION_REQUIRES_SUPERVISED_TRANSACTION,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_proposal_queue import (
        SOUNDCHECK_CORRECTION_REQUIRES_SUPERVISED_TRANSACTION,
    )

try:
    from processing_proposal_adapters import build_operator_correction_payload
except ImportError:  # pragma: no cover - package import compatibility
    from backend.processing_proposal_adapters import build_operator_correction_payload


SAFE_GAIN_SOURCE = "safe_gain_calibrator"
SOUNDCHECK_SOURCE = "soundcheck_recommendation_bundle"
CACHE_FRESHNESS_STALE_AFTER_SEC = 30.0
PROPOSAL_LAYER_BLOCKERS = (
    "hardware_readback_and_rollback_not_proven",
    "real_wing_proof_required_before_live_apply",
)
AUTHORITY_DENIAL_FIELDS = {
    "can_apply_live": False,
    "can_import_to_queue": False,
    "queue_import_allowed": False,
    "live_apply_allowed": False,
    "read_only_or_dry_run_only": True,
    "live_authority_granted": False,
    "readiness_closure_allowed": False,
    "queue_action": "not_queueable_until_promoted",
    "live_write": False,
    "queue_mutation": False,
    "mutates_mixer": False,
}
# Operator recommendations feed the current WING deployment apply rail.
RECOMMENDATION_APPLY_VALUE_TYPES = set(WING_DEPLOYMENT_LIVE_WRITE_KINDS)
QUEUE_REVIEW_PROJECTION_SCHEMA = "operator_recommendation_queue_review/v2"
QUEUE_REDUNDANT_OPERATOR_CORRECTION_AGGREGATES = (
    "operator_evidence_packet",
    "operator_explanation_packet",
    "operator_review_summary",
)
QUEUE_REVIEW_REFERENCE_REWRITES = {
    "operator_evidence_packet": "operator_correction",
    "operator_explanation_packet": "operator_readable_explanation",
    "operator_review_summary": "operator_correction",
    "review_operator_evidence_packet": "review_operator_correction",
}
QUEUE_REVIEW_REFERENCE_FIELDS = {
    "operator_next_action_details": (
        "action",
        "operator_action",
        "action_type",
        "source",
        "sources",
    ),
    "operator_review_checklist_details": (
        "operator_action",
        "related_next_actions",
    ),
    "operator_review_record_field_details": ("source", "sources"),
    "why_safe_details": ("operator_action",),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _rewrite_queue_review_references(correction: Dict[str, Any]) -> None:
    """Retarget generated review links after redundant aggregates are omitted."""

    for collection, fields in QUEUE_REVIEW_REFERENCE_FIELDS.items():
        raw_rows = correction.get(collection)
        if not isinstance(raw_rows, list):
            continue
        rewritten_rows = []
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                rewritten_rows.append(raw_row)
                continue
            row = dict(raw_row)
            for field in fields:
                value = row.get(field)
                if isinstance(value, str):
                    row[field] = QUEUE_REVIEW_REFERENCE_REWRITES.get(value, value)
                elif isinstance(value, list):
                    row[field] = [
                        QUEUE_REVIEW_REFERENCE_REWRITES.get(item, item)
                        if isinstance(item, str)
                        else item
                        for item in value
                    ]
            rewritten_rows.append(row)
        correction[collection] = rewritten_rows


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
    if value_type not in RECOMMENDATION_APPLY_VALUE_TYPES or channel is None:
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


def _operator_correction_blockers(proposal: Dict[str, Any]) -> List[str]:
    raw = proposal.get("raw") if isinstance(proposal.get("raw"), dict) else {}
    blockers: List[str] = []
    for item in proposal.get("apply_blockers") or raw.get("apply_blockers") or []:
        blockers.append(str(item))
    if not isinstance(proposal.get("requested_change"), dict):
        if isinstance(proposal.get("blocked_requested_change"), dict) or isinstance(
            raw.get("blocked_requested_change"), dict
        ):
            blockers.append("cache_freshness_required_before_applyable_proposal")
        else:
            blockers.append("missing_applyable_fader_gain_requested_change")
    blockers.extend(PROPOSAL_LAYER_BLOCKERS)
    deduped: List[str] = []
    seen = set()
    for blocker in blockers:
        if blocker in seen:
            continue
        seen.add(blocker)
        deduped.append(blocker)
    return deduped


def _blocked_proposed_action(proposal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = proposal.get("raw") if isinstance(proposal.get("raw"), dict) else {}
    blocked_change = proposal.get("blocked_requested_change") or raw.get("blocked_requested_change")
    if not isinstance(blocked_change, dict):
        return None
    return {
        "status": "blocked_before_applyable_proposal",
        "blocked_requested_change": dict(blocked_change),
        "apply_status": proposal.get("apply_status") or raw.get("apply_status"),
        "apply_blockers": proposal.get("apply_blockers") or raw.get("apply_blockers") or [],
    }


def _apply_authority_denial_fields(payload: Dict[str, Any]) -> None:
    for key, value in AUTHORITY_DENIAL_FIELDS.items():
        if key == "queue_action":
            payload.setdefault(key, value)
        else:
            payload[key] = value


def _operator_review_policy_for_change(change: Dict[str, Any]) -> str:
    delta_db = _coerce_float(change.get("delta_db"))
    if delta_db is not None and delta_db <= 0.0:
        return "gain_down_or_fader_down_review_required_before_apply"
    return "gain_up_or_fader_up_requires_operator_review_and_real_wing_proof"


def _change_operator_evidence(change: Dict[str, Any], proposal: Dict[str, Any]) -> Dict[str, Any]:
    value_type = str(change.get("value_type") or "").strip().lower()
    delta_db = _round(change.get("delta_db"))
    old_value = change.get("old_value") if "old_value" in change else change.get("current_value")
    proposed_value = (
        change.get("proposed_value") if "proposed_value" in change else change.get("value")
    )
    rollback_value = change.get("rollback_value", old_value)
    evidence: Dict[str, Any] = {
        "requested_change": dict(change),
        "value_type": value_type or None,
        "channel": change.get("channel"),
        "old_value": old_value,
        "proposed_value": proposed_value,
        "delta_db": delta_db,
        "readback_required": bool(change.get("readback_required", True)),
        "rollback_value": rollback_value,
        "safety_evidence": list(change.get("safety_evidence") or []),
        "bounded_gain_or_fader_delta": {
            "value_type": value_type or None,
            "delta_db": delta_db,
            "old_value": old_value,
            "proposed_value": proposed_value,
            "bounded": True,
            "operator_review_required": True,
        },
    }

    if value_type == "gain":
        suggestion = {}
        raw = proposal.get("raw") if isinstance(proposal.get("raw"), dict) else {}
        if isinstance(raw.get("suggestion"), dict):
            suggestion = raw["suggestion"]
        evidence.update({
            "input_level_evidence": {
                "peak_db": _round(suggestion.get("peak_db")),
                "target_peak_db": _round(suggestion.get("target_peak_db")),
                "input_trim_role": suggestion.get("input_trim_role"),
                "source": proposal.get("source"),
                "channel": change.get("channel"),
            },
            "overload_or_weak_signal_problem": (
                "weak_signal_gain_up_review"
                if delta_db is not None and delta_db > 0.0
                else "overload_or_headroom_gain_down_review"
            ),
            "headroom_margin": {
                "current_peak_db": _round(suggestion.get("peak_db")),
                "target_peak_db": _round(suggestion.get("target_peak_db")),
                "operator_review_required": True,
            },
            "true_peak_headroom": {
                "status": "required_before_apply",
                "current_peak_db": _round(suggestion.get("peak_db")),
            },
            "bleed_risk": suggestion.get("bleed_risk", "operator_review_required"),
            "bounded_gain_delta": evidence["bounded_gain_or_fader_delta"],
            "gain_down_or_operator_review_gain_up": (
                "gain_up_requires_operator_review"
                if delta_db is not None and delta_db > 0.0
                else "gain_down_headroom_cut_review"
            ),
            "rollback_input_trim_readback": {
                "value_db": rollback_value,
                "required_before_live_promotion": True,
            },
        })
    elif value_type == "fader":
        evidence.update({
            "current_fader_readback": {
                "value_db": old_value,
                "required_before_live_promotion": True,
            },
            "ayaic_level_plane": "operator_evidence_required",
            "bounded_fader_delta": evidence["bounded_gain_or_fader_delta"],
            "true_peak_headroom": "required_before_apply",
            "feedback_risk": "operator_feedback_check_required_before_apply",
            "fader_ceiling": 0.0,
            "rollback_fader_readback": {
                "value_db": rollback_value,
                "required_before_live_promotion": True,
            },
        })
    evidence["operator_review_policy"] = _operator_review_policy_for_change(change)
    return evidence


def _normalize_operator_evidence(
    proposal: Dict[str, Any],
    requested_change: Optional[Dict[str, Any]],
) -> None:
    raw = proposal.get("raw") if isinstance(proposal.get("raw"), dict) else {}
    evidence = proposal.get("evidence") if isinstance(proposal.get("evidence"), dict) else {}
    evidence = dict(evidence)
    blocked_change = proposal.get("blocked_requested_change") or raw.get("blocked_requested_change")
    evidence_change = requested_change if isinstance(requested_change, dict) else blocked_change
    if isinstance(evidence_change, dict):
        evidence.update(_change_operator_evidence(evidence_change, proposal))
    if isinstance(blocked_change, dict) and requested_change is None:
        evidence["blocked_requested_change"] = dict(blocked_change)
    cache_freshness = proposal.get("cache_freshness") or raw.get("cache_freshness")
    if cache_freshness is not None:
        evidence["cache_freshness"] = cache_freshness
    for key in ("apply_status", "apply_blockers"):
        value = proposal.get(key) if key in proposal else raw.get(key)
        if value is not None:
            evidence[key] = value
    evidence["live_write"] = False
    evidence["queue_mutation"] = False
    _apply_authority_denial_fields(evidence)
    proposal["evidence"] = evidence


def _attach_operator_correction(
    proposal: Dict[str, Any],
    *,
    method_id: str,
) -> Dict[str, Any]:
    _apply_authority_denial_fields(proposal)
    requested_change = (
        proposal.get("requested_change")
        if isinstance(proposal.get("requested_change"), dict)
        else None
    )
    _normalize_operator_evidence(proposal, requested_change)
    proposal["operator_correction"] = build_operator_correction_payload(
        proposal=proposal,
        method_id=method_id,
        target=proposal.get("target") or proposal.get("channel"),
        requested_change=requested_change,
        proposed_action=None if requested_change else _blocked_proposed_action(proposal),
        blockers=_operator_correction_blockers(proposal),
    )
    return proposal


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
        gated = _gate_proposal_cache_freshness(proposal, mixer_client)
        proposals.append(_attach_operator_correction(gated, method_id="lufs_gain_staging"))
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
            "proposed_value": round(value, 1),
            "readback_required": True,
            "safety_evidence": [
                {"label": "Source", "value": "no_write_soundcheck_recommendation"},
                {"label": "Dry run", "value": True},
            ],
        }
        if current_value is not None:
            change["current_value"] = round(current_value, 1)
            change["old_value"] = round(current_value, 1)
            change["rollback_value"] = round(current_value, 1)
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

    transaction_bundle = bundle.get("supervised_transaction_plans")
    transaction_plans = (
        transaction_bundle.get("plans")
        if isinstance(transaction_bundle, dict)
        else None
    )
    transaction_change_keys = set()
    if isinstance(transaction_plans, list):
        for plan in transaction_plans:
            if not isinstance(plan, dict):
                continue
            for operation in plan.get("operations") or []:
                if not isinstance(operation, dict):
                    continue
                write_kind = str(operation.get("write_kind") or "").strip()
                channel = _coerce_int(operation.get("channel"))
                target = _coerce_float(operation.get("target_value"))
                if write_kind in {"gain", "fader"} and channel is not None and target is not None:
                    transaction_change_keys.add((write_kind, channel, round(target, 4)))

    proposals: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        kind, requested_change = _soundcheck_kind_and_change(item)
        if requested_change:
            change_key = (
                str(requested_change.get("value_type") or ""),
                _coerce_int(requested_change.get("channel")),
                round(float(requested_change.get("value")), 4),
            )
            if change_key in transaction_change_keys:
                # The supervised child is the sole apply surface. Retaining a
                # second generic gain/fader row makes one logical correction
                # appear independently successful while the manifest remains
                # incomplete.
                continue
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
        if requested_change:
            proposal.update({
                "operator_review_only": True,
                "operator_review_only_reason": (
                    SOUNDCHECK_CORRECTION_REQUIRES_SUPERVISED_TRANSACTION
                ),
            })
        gated = _gate_proposal_cache_freshness(proposal, mixer_client)
        proposals.append(_attach_operator_correction(gated, method_id="auto_soundcheck_engine"))

    if isinstance(transaction_plans, list):
        for index, plan in enumerate(transaction_plans):
            if not isinstance(plan, dict):
                continue
            transaction_id = str(
                plan.get("transaction_id") or f"soundcheck-plan-{index + 1}"
            )
            operations = plan.get("operations")
            if not isinstance(operations, list) or not operations:
                continue
            family_ids = sorted(
                {
                    str(operation.get("family_id") or "")
                    for operation in operations
                    if isinstance(operation, dict)
                    and str(operation.get("family_id") or "")
                }
            )
            separate = bool(plan.get("separate_approval_required"))
            proposals.append(
                {
                    "id": transaction_id,
                    "title": (
                        f"Soundcheck {plan.get('stage') or 'review'}: {', '.join(family_ids)} "
                        f"({len(operations)} operations)"
                    ),
                    "target": family_ids,
                    "kind": "supervised_soundcheck_transaction",
                    "severity": "high" if separate else "medium",
                    "ttl_seconds": 300,
                    "confidence": None,
                    "source": source,
                    "reason": str(
                        plan.get("reason")
                        or "Review the absolute soundcheck correction plan."
                    ),
                    "supervised_transaction_plan": copy.deepcopy(plan),
                    "soundcheck_run_id": (
                        transaction_bundle.get("run_id")
                        if isinstance(transaction_bundle, dict)
                        else None
                    ),
                    "soundcheck_plan_revision": (
                        transaction_bundle.get("plan_revision")
                        if isinstance(transaction_bundle, dict)
                        else None
                    ),
                    "soundcheck_plan_fingerprint": (
                        transaction_bundle.get("plan_fingerprint")
                        if isinstance(transaction_bundle, dict)
                        else None
                    ),
                    "soundcheck_stage": plan.get("stage"),
                    "soundcheck_required_for": plan.get("required_for"),
                    "logical_correction_ids": list(
                        plan.get("logical_correction_ids") or []
                    ),
                    "transaction_family_ids": family_ids,
                    "separate_approval_required": separate,
                    "live_mixer_writes": False,
                    "transport_policy": "proposal_only",
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
        queue_payload = dict(proposal)
        operator_correction = proposal.get("operator_correction")
        if isinstance(operator_correction, dict):
            queue_correction = dict(operator_correction)
            omitted = [
                field
                for field in QUEUE_REDUNDANT_OPERATOR_CORRECTION_AGGREGATES
                if field in queue_correction
            ]
            for field in omitted:
                queue_correction.pop(field, None)
            if omitted:
                _rewrite_queue_review_references(queue_correction)
                queue_correction["queue_review_projection"] = {
                    "schema": QUEUE_REVIEW_PROJECTION_SCHEMA,
                    "canonical_review_surface": "operator_correction",
                    "omitted_redundant_aggregate_fields": omitted,
                    "reference_rewrites": dict(QUEUE_REVIEW_REFERENCE_REWRITES),
                    "retained_constituent_review_fields": sorted(queue_correction),
                    "execution_semantics_changed": False,
                }
            queue_payload["operator_correction"] = queue_correction

        result = queue.create(queue_payload, operator_mode)
        results.append(result)
        if result.get("status") == "created":
            created_count += 1
        elif result.get("status") == "exists":
            existing_count += 1
        elif result.get("status") == "blocked":
            blocked_count += 1

    if blocked_count == 0:
        import_status = "ok"
    elif blocked_count == len(proposals):
        import_status = "blocked"
    else:
        import_status = "partial"
    return {
        "type": "operator_proposals_imported",
        "status": import_status,
        "success": blocked_count == 0,
        "complete": blocked_count == 0,
        "source": source,
        "operator_mode": operator_mode,
        "imported_count": len(proposals),
        "created_count": created_count,
        "existing_count": existing_count,
        "blocked_count": blocked_count,
        "results": results,
        "generated_at": _utc_now(),
    }
