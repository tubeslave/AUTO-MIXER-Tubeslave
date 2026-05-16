"""Replay-safe soundcheck autonomy contract helpers.

This module defines the dry-run-only autonomy vocabulary approved for TUB-289.
It intentionally does not include any live transport behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional


def _compact(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


class AutonomyMode(str, Enum):
    OBSERVE_ONLY = "observe_only"
    ASSISTED = "assisted"
    SEMI_AUTONOMOUS_CALIBRATION = "semi_autonomous_calibration"
    SUPERVISED_AUTONOMOUS_TUNING = "supervised_autonomous_tuning"
    EMERGENCY_HUMAN_ONLY_FALLBACK = "emergency_human_only_fallback"


class ReviewMode(str, Enum):
    OBSERVE_ONLY = "observe_only"
    ASSISTED = "assisted"
    APPROVAL_REQUIRED = "approval_required"


class ApprovalLevel(str, Enum):
    NONE = "none"
    OPERATOR_REVIEW = "operator_review"
    OPERATOR_REQUIRED = "operator_required"
    HUMAN_REQUIRED = "human_required"


class TrustGateState(str, Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
    UNKNOWN = "unknown"


BASELINE_AUTONOMY_MODE_BY_KIND: Dict[str, AutonomyMode] = {
    "input_gain": AutonomyMode.SEMI_AUTONOMOUS_CALIBRATION,
    "hpf": AutonomyMode.ASSISTED,
    "eq": AutonomyMode.ASSISTED,
    "compressor": AutonomyMode.ASSISTED,
    "pan": AutonomyMode.ASSISTED,
    "fader": AutonomyMode.SEMI_AUTONOMOUS_CALIBRATION,
    "fx_send": AutonomyMode.ASSISTED,
    "feedback_notch": AutonomyMode.OBSERVE_ONLY,
    "feedback_fader_reduce": AutonomyMode.OBSERVE_ONLY,
    "gain": AutonomyMode.SEMI_AUTONOMOUS_CALIBRATION,
}


@dataclass(frozen=True)
class EnvelopeEvaluation:
    autonomy_mode: AutonomyMode
    review_mode: ReviewMode
    approval_level: ApprovalLevel
    trust_gates: Dict[str, Any]
    risk: Dict[str, Any]


def _normalize_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "known", "ready", "clear"}:
        return True
    if normalized in {"0", "false", "no", "off", "unknown", "missing", "blocked"}:
        return False
    return None


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_confidence_bucket(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"low", "high", "unknown"}:
            return normalized
        try:
            value = float(normalized)
        except ValueError:
            return "unknown"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unknown"
    return "high" if numeric >= 0.8 else "low"


def _gate(
    *,
    name: str,
    state: TrustGateState,
    value: Any,
    reason: str,
) -> Dict[str, Any]:
    return {
        "name": name,
        "state": state.value,
        "value": value,
        "reason": reason,
    }


def _review_mode_for(autonomy_mode: AutonomyMode) -> ReviewMode:
    if autonomy_mode == AutonomyMode.OBSERVE_ONLY:
        return ReviewMode.OBSERVE_ONLY
    if autonomy_mode == AutonomyMode.ASSISTED:
        return ReviewMode.ASSISTED
    return ReviewMode.APPROVAL_REQUIRED


def _approval_level_for(autonomy_mode: AutonomyMode) -> ApprovalLevel:
    if autonomy_mode == AutonomyMode.OBSERVE_ONLY:
        return ApprovalLevel.NONE
    if autonomy_mode == AutonomyMode.ASSISTED:
        return ApprovalLevel.OPERATOR_REVIEW
    if autonomy_mode == AutonomyMode.EMERGENCY_HUMAN_ONLY_FALLBACK:
        return ApprovalLevel.HUMAN_REQUIRED
    return ApprovalLevel.OPERATOR_REQUIRED


def evaluate_soundcheck_contract(
    *,
    kind: str,
    confidence: float,
    context_metadata: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EnvelopeEvaluation:
    """Resolve effective autonomy mode, trust gates, and risk metadata."""
    baseline_mode = BASELINE_AUTONOMY_MODE_BY_KIND.get(
        kind,
        AutonomyMode.ASSISTED,
    )
    context_metadata = dict(context_metadata or {})
    metadata = dict(metadata or {})

    venue_value = _first_non_null(
        metadata.get("venue_familiarity"),
        context_metadata.get("venue_familiarity"),
    )
    performer_value = _first_non_null(
        metadata.get("performer_familiarity"),
        context_metadata.get("performer_familiarity"),
    )
    confidence_value = _first_non_null(
        metadata.get("calibration_confidence"),
        context_metadata.get("calibration_confidence"),
        confidence,
    )
    telemetry_value = _first_non_null(
        metadata.get("telemetry_readback_available"),
        context_metadata.get("telemetry_readback_available"),
    )
    rollback_value = _first_non_null(
        metadata.get("rollback_path_available"),
        context_metadata.get("rollback_path_available"),
    )
    panic_value = _first_non_null(
        metadata.get("panic_stop_clear"),
        context_metadata.get("panic_stop_clear"),
    )

    venue_state = TrustGateState.UNKNOWN
    if venue_value == "known_venue":
        venue_state = TrustGateState.PASS
    elif venue_value == "first_time_venue":
        venue_state = TrustGateState.WARN

    performer_state = TrustGateState.UNKNOWN
    if performer_value == "known_performer":
        performer_state = TrustGateState.PASS
    elif performer_value == "unknown_performer":
        performer_state = TrustGateState.WARN

    confidence_bucket = _coerce_confidence_bucket(confidence_value)
    confidence_state = {
        "high": TrustGateState.PASS,
        "low": TrustGateState.WARN,
    }.get(confidence_bucket, TrustGateState.UNKNOWN)

    telemetry_bool = _normalize_bool(telemetry_value)
    rollback_bool = _normalize_bool(rollback_value)
    panic_bool = _normalize_bool(panic_value)

    telemetry_state = (
        TrustGateState.PASS if telemetry_bool is True
        else TrustGateState.BLOCK if telemetry_bool is False
        else TrustGateState.UNKNOWN
    )
    rollback_state = (
        TrustGateState.PASS if rollback_bool is True
        else TrustGateState.BLOCK if rollback_bool is False
        else TrustGateState.UNKNOWN
    )
    panic_state = (
        TrustGateState.PASS if panic_bool is True
        else TrustGateState.BLOCK if panic_bool is False
        else TrustGateState.UNKNOWN
    )

    trust_gates = {
        "venue_familiarity": _gate(
            name="venue_familiarity",
            state=venue_state,
            value=venue_value or "unspecified",
            reason="First-time venues stay conservative until venue-specific evidence exists."
            if venue_value == "first_time_venue"
            else "Venue familiarity not provided."
            if venue_value is None
            else "Known venue evidence present.",
        ),
        "performer_familiarity": _gate(
            name="performer_familiarity",
            state=performer_state,
            value=performer_value or "unspecified",
            reason="Unknown performers stay conservative until performer-specific evidence exists."
            if performer_value == "unknown_performer"
            else "Performer familiarity not provided."
            if performer_value is None
            else "Known performer evidence present.",
        ),
        "calibration_confidence": _gate(
            name="calibration_confidence",
            state=confidence_state,
            value=confidence_bucket,
            reason="Low-confidence calibration must not escalate autonomy."
            if confidence_bucket == "low"
            else "Calibration confidence not provided."
            if confidence_bucket == "unknown"
            else "High-confidence calibration evidence present.",
        ),
        "telemetry_readback_available": _gate(
            name="telemetry_readback_available",
            state=telemetry_state,
            value=telemetry_bool if telemetry_bool is not None else "unspecified",
            reason="Explicitly missing telemetry/readback support."
            if telemetry_bool is False
            else "Telemetry/readback capability not provided."
            if telemetry_bool is None
            else "Telemetry/readback capability declared available.",
        ),
        "rollback_path_available": _gate(
            name="rollback_path_available",
            state=rollback_state,
            value=rollback_bool if rollback_bool is not None else "unspecified",
            reason="Explicitly missing rollback path."
            if rollback_bool is False
            else "Rollback path availability not provided."
            if rollback_bool is None
            else "Rollback path declared available.",
        ),
        "panic_stop_clear": _gate(
            name="panic_stop_clear",
            state=panic_state,
            value=panic_bool if panic_bool is not None else "unspecified",
            reason="Panic stop is not clear."
            if panic_bool is False
            else "Panic-stop state not provided."
            if panic_bool is None
            else "Panic-stop state is clear.",
        ),
    }

    effective_mode = baseline_mode
    risk_reasons: list[str] = []

    if kind in {"feedback_notch", "feedback_fader_reduce"}:
        # Current WING dry-run safety posture keeps feedback actions non-live.
        effective_mode = AutonomyMode.OBSERVE_ONLY
        risk_reasons.append("Feedback actions stay recommendation-only on current WING dry-run paths.")

    if venue_state == TrustGateState.WARN:
        risk_reasons.append("First-time venue evidence requires conservative autonomy.")
        if effective_mode in {
            AutonomyMode.SEMI_AUTONOMOUS_CALIBRATION,
            AutonomyMode.SUPERVISED_AUTONOMOUS_TUNING,
        }:
            effective_mode = AutonomyMode.ASSISTED

    if performer_state == TrustGateState.WARN:
        risk_reasons.append("Unknown performer evidence requires conservative autonomy.")
        if effective_mode == AutonomyMode.SUPERVISED_AUTONOMOUS_TUNING:
            effective_mode = AutonomyMode.ASSISTED

    if confidence_state == TrustGateState.WARN:
        risk_reasons.append("Low-confidence calibration must not escalate autonomy.")
        if effective_mode == AutonomyMode.SUPERVISED_AUTONOMOUS_TUNING:
            effective_mode = AutonomyMode.ASSISTED

    if any(state == TrustGateState.BLOCK for state in (telemetry_state, rollback_state, panic_state)):
        effective_mode = AutonomyMode.EMERGENCY_HUMAN_ONLY_FALLBACK
        if telemetry_state == TrustGateState.BLOCK:
            risk_reasons.append("Missing telemetry/readback explicitly blocks autonomy escalation.")
        if rollback_state == TrustGateState.BLOCK:
            risk_reasons.append("Missing rollback path explicitly blocks autonomy escalation.")
        if panic_state == TrustGateState.BLOCK:
            risk_reasons.append("Panic-stop state explicitly blocks autonomy escalation.")

    review_mode = _review_mode_for(effective_mode)
    approval_level = _approval_level_for(effective_mode)
    severity = (
        "critical" if effective_mode == AutonomyMode.EMERGENCY_HUMAN_ONLY_FALLBACK
        else "medium" if risk_reasons
        else "low"
    )
    trust_gates["summary"] = {
        "baseline_autonomy_mode": baseline_mode.value,
        "effective_autonomy_mode": effective_mode.value,
        "has_unknowns": any(
            gate["state"] == TrustGateState.UNKNOWN.value
            for gate in trust_gates.values()
            if isinstance(gate, dict) and gate.get("name")
        ),
        "blocking_gate_names": [
            gate["name"]
            for gate in trust_gates.values()
            if isinstance(gate, dict) and gate.get("state") == TrustGateState.BLOCK.value
        ],
    }
    risk = {
        "severity": severity,
        "reasons": risk_reasons,
    }
    return EnvelopeEvaluation(
        autonomy_mode=effective_mode,
        review_mode=review_mode,
        approval_level=approval_level,
        trust_gates=trust_gates,
        risk=risk,
    )


def build_action_envelope(
    *,
    envelope_id: str,
    source_system: str,
    kind: str,
    family: str,
    action_type: str,
    target: Mapping[str, Any],
    current_state: Mapping[str, Any],
    requested_state: Mapping[str, Any],
    confidence: float,
    replay_correlation_id: str,
    context_metadata: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    rationale: Optional[Mapping[str, Any]] = None,
    audit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    evaluation = evaluate_soundcheck_contract(
        kind=kind,
        confidence=confidence,
        context_metadata=context_metadata,
        metadata=metadata,
    )
    operator_summary = (
        f"{kind} recommendation for channel {target.get('mixer_channel', 'unknown')} "
        f"stays {evaluation.autonomy_mode.value}."
    )
    operator_explanation = (
        "This is a replay-safe dry-run recommendation only. "
        f"Trust gates resolved to {evaluation.autonomy_mode.value}, "
        f"approval level {evaluation.approval_level.value}."
    )
    rollback_handle = {
        "type": "dry_run_reference",
        "replay_correlation_id": replay_correlation_id,
        "restores_live_console": False,
    }
    return {
        "schema_version": "soundcheck_action_envelope/v1",
        "envelope_id": envelope_id,
        "source_system": str(source_system or "auto_soundcheck_engine"),
        "kind": str(kind),
        "autonomy_mode": evaluation.autonomy_mode.value,
        "review_mode": evaluation.review_mode.value,
        "action_type": str(action_type),
        "family": str(family),
        "target": dict(target),
        "current_state": dict(current_state),
        "requested_state": dict(requested_state),
        "confidence": float(confidence),
        "risk": dict(evaluation.risk),
        "trust_gates": dict(evaluation.trust_gates),
        "required_approval_level": evaluation.approval_level.value,
        "operator_explanation": operator_explanation,
        "operator_summary": operator_summary,
        "rollback_handle": rollback_handle,
        "replay_correlation_id": replay_correlation_id,
        "transport_policy": "dry_run_only",
        "live_mixer_writes": False,
        "auto_apply_blocked": True,
        "metadata": dict(metadata or {}),
        "rationale": dict(rationale or {}),
        "audit": dict(audit or {}),
    }
