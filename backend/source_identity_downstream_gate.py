"""Shared gate for replay-calibrated source identity trust.

The gate is transport-free and read-only. It lets downstream proposal modules
consume source identity calibration evidence without mutating classifiers,
memory, queues, or live console state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Mapping, Optional, Tuple


TRUSTED_STATES = {
    "trusted",
    "trusted_for_downstream_priority",
    "operator_accepted",
}
LEGACY_ALLOWED_STATES = {
    "",
    "not_provided",
    "unknown",
}
HOLD_STATES = {
    "hold",
    "hold_for_operator_review",
    "needs_more_replay_outcomes",
}
BLOCKED_STATES = {
    "blocked",
    "blocked_from_downstream_trust",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _clip01(value: Any) -> float:
    return _clip(_safe_float(value), 0.0, 1.0)


def _normalize_token(value: Any, default: str = "not_provided") -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text or default


def _freshness_status(payload: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(payload, Mapping):
        return "not_provided"
    return str(payload.get("status") or "unknown").strip().lower() or "unknown"


def _freshness_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"status": "not_provided", "provided": False}
    allowed_keys = (
        "status",
        "source",
        "field_count",
        "fresh_field_count",
        "stale_field_count",
        "unknown_field_count",
        "missing_field_count",
        "stale_fields",
        "unknown_fields",
        "missing_fields",
        "max_age_seconds",
        "oldest_age_seconds",
        "read_only",
        "live_write",
        "queries_console",
    )
    normalized = {key: payload[key] for key in allowed_keys if key in payload}
    normalized.setdefault("status", _freshness_status(payload))
    normalized.setdefault("provided", True)
    return normalized


@dataclass(frozen=True)
class SourceIdentityDownstreamTrustDecision:
    source_id: str
    source_type: str
    trust_state: str
    status: str
    usable_for_downstream: bool
    confidence: float
    replay_outcome_count: int
    blockers: Tuple[str, ...] = ()
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["blockers"] = list(self.blockers)
        return payload


def evaluate_source_identity_downstream_trust(
    *,
    source_id: str,
    source_type: str,
    trust_state: Any = "not_provided",
    trust_confidence: Any = 0.0,
    replay_outcome_count: Any = 0,
    current_state_cache_freshness: Optional[Mapping[str, Any]] = None,
) -> SourceIdentityDownstreamTrustDecision:
    """Evaluate whether downstream proposals may trust a source identity label."""

    normalized_state = _normalize_token(trust_state)
    confidence = _clip01(trust_confidence)
    outcomes = max(0, _safe_int(replay_outcome_count))
    freshness = _freshness_payload(current_state_cache_freshness)
    freshness_status = str(freshness.get("status") or "unknown")

    blockers = []
    status = "legacy_unverified"
    usable = True

    if freshness_status not in {"fresh", "not_provided"}:
        blockers.append("source_identity_replay_context_not_fresh")
    if normalized_state in TRUSTED_STATES:
        status = "trusted"
    elif normalized_state in HOLD_STATES:
        status = "held_for_operator_review"
        blockers.append("source_identity_not_trusted_for_downstream")
    elif normalized_state in BLOCKED_STATES:
        status = "blocked"
        blockers.append("source_identity_blocked_from_downstream_trust")
    elif normalized_state in LEGACY_ALLOWED_STATES:
        status = "legacy_unverified"
    else:
        status = "unknown_trust_state"
        blockers.append("source_identity_trust_state_unknown")

    blockers_tuple = tuple(sorted(set(blockers)))
    if blockers_tuple:
        usable = False

    evidence = {
        "source": "source_identity_downstream_gate_v1",
        "source_id": str(source_id),
        "source_type": str(source_type),
        "trust_state": normalized_state,
        "trust_status": status,
        "trust_confidence": round(confidence, 3),
        "replay_outcome_count": outcomes,
        "current_source_identity_replay_context_cache_freshness": freshness_status,
        "current_state_cache_freshness": freshness,
        "usable_for_downstream": usable,
        "legacy_missing_calibration_allowed": normalized_state in LEGACY_ALLOWED_STATES,
        "operator_review_required_before_trust_change": True,
        "transport_free": True,
        "live_write": False,
        "queue_mutation": False,
        "model_mutation": False,
        "memory_mutation": False,
    }
    return SourceIdentityDownstreamTrustDecision(
        source_id=str(source_id),
        source_type=str(source_type),
        trust_state=normalized_state,
        status=status,
        usable_for_downstream=usable,
        confidence=round(confidence, 3),
        replay_outcome_count=outcomes,
        blockers=blockers_tuple,
        evidence=evidence,
    )
