"""Proposal-only frequency separation recommendations.

This module is transport-free. It turns source/stem band-energy contribution
evidence into operator-facing frequency-lane proposals, but it never creates a
live console write request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:
    from source_identity_downstream_gate import evaluate_source_identity_downstream_trust
    from processing_proposal_adapters import build_operator_correction_payload
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_downstream_gate import evaluate_source_identity_downstream_trust
    from backend.processing_proposal_adapters import build_operator_correction_payload


FREQUENCY_SEPARATION_METHOD_ID = "frequency_separation_proposal"
FREQUENCY_SEPARATION_VERSION = "frequency_separation_proposal_v1"
READ_REQUIRED_SOURCE = "cached_or_physical_eq_band_and_stem_spectrum_read_required"
MIN_BAND_LEVEL_DB = -70.0
DEFAULT_CONFLICT_TOLERANCE_DB = 5.0
MIN_CONFIDENCE = 0.55
MAX_SUPPORT_CUT_DB = 3.0

BAND_CENTERS_HZ = {
    "sub": 40.0,
    "bass": 120.0,
    "low_mid": 375.0,
    "mid": 1250.0,
    "high_mid": 3000.0,
    "high": 6000.0,
    "air": 12000.0,
}

ROLE_BAND_AFFINITY = {
    "low_anchor": {"sub": 1.0, "bass": 1.0, "low_mid": 0.35},
    "rhythmic_anchor": {"bass": 0.55, "low_mid": 0.75, "mid": 0.65, "high_mid": 0.45},
    "lead": {"mid": 1.0, "high_mid": 1.0, "high": 0.65, "low_mid": 0.45},
    "featured": {"mid": 0.9, "high_mid": 0.9, "high": 0.65},
    "backing": {"mid": 0.55, "high_mid": 0.55, "high": 0.35},
    "harmonic_support": {"low_mid": 0.65, "mid": 0.55, "high_mid": 0.45},
    "texture": {"high_mid": 0.45, "high": 0.75, "air": 0.65},
    "effect": {"high": 0.4, "air": 0.5},
}

SOURCE_TYPE_ROLE_HINTS = {
    "kick": "rhythmic_anchor",
    "snare": "rhythmic_anchor",
    "drums": "rhythmic_anchor",
    "bass": "low_anchor",
    "lead_vocal": "lead",
    "vocal": "lead",
    "backing_vocal": "backing",
    "guitar": "harmonic_support",
    "keys": "harmonic_support",
    "synth": "texture",
    "fx_return": "effect",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _clip01(value: Any) -> float:
    return _clip(_safe_float(value), 0.0, 1.0)


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


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


def _freshness_is_no_authority(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("read_only") is True
        and payload.get("live_write") is False
        and payload.get("queries_console") is False
    )


def _role_for(source_type: str, arrangement_role: str) -> str:
    role = _normalize_token(arrangement_role)
    if role and role != "unknown":
        return role
    return SOURCE_TYPE_ROLE_HINTS.get(_normalize_token(source_type), "unknown")


def _role_band_affinity(source_type: str, arrangement_role: str, band: str) -> float:
    role = _role_for(source_type, arrangement_role)
    affinity = ROLE_BAND_AFFINITY.get(role, {}).get(band, 0.0)
    if affinity:
        return affinity
    return 0.2 if band in {"mid", "high_mid"} and role == "unknown" else 0.0


def _lane_score(source: "FrequencySeparationSource", band: str) -> float:
    priority_component = _clip((5.0 - _safe_float(source.priority, 3.0)) / 5.0, 0.0, 1.0)
    confidence_component = _clip01(source.confidence) * 0.25
    contribution_component = _clip01(source.stem_contribution.get(band, 0.0)) * 0.2
    return (
        _role_band_affinity(source.source_type, source.arrangement_role, band)
        + priority_component * 0.35
        + confidence_component
        + contribution_component
    )


def _band_level(source: "FrequencySeparationSource", band: str) -> float:
    return _safe_float(source.band_energy_db.get(band), -120.0)


@dataclass(frozen=True)
class FrequencySeparationSource:
    source_id: str
    channel_id: Optional[int] = None
    label: str = ""
    source_type: str = "unknown"
    arrangement_role: str = "unknown"
    stem_id: str = ""
    priority: int = 3
    band_energy_db: Dict[str, float] = field(default_factory=dict)
    stem_contribution: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.75
    active: bool = True
    muted: bool = False
    source_identity_trust_state: str = "not_provided"
    source_identity_trust_confidence: float = 0.0
    source_identity_replay_outcome_count: int = 0
    source_identity_calibration_freshness: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class FrequencySeparationRecommendation:
    id: str
    status: str
    action_type: str
    target_source_id: str
    protected_source_id: str
    stem_id: str
    band: str
    center_hz: float
    target_role: str
    protected_role: str
    suggested_eq_delta_db: float
    confidence: float
    severity: str
    reason: str
    blockers: Tuple[str, ...] = ()
    evidence: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["blockers"] = list(self.blockers)
        return payload

    def frequency_lane_safety_summary(self) -> Dict[str, Any]:
        source_stem_contribution = dict(
            self.evidence.get("source_stem_contribution") or {}
        )
        lane_conflict_pair = dict(self.evidence.get("lane_conflict_pair") or {})
        band_energy_db = dict(self.evidence.get("band_energy_db") or {})
        role_lane_affinity = dict(self.evidence.get("role_lane_affinity") or {})
        return {
            "summary_type": "frequency_lane_separation_review",
            "status": "passed" if not self.blockers else "blocked",
            "target_source_id": self.target_source_id,
            "protected_source_id": self.protected_source_id,
            "stem_id": self.stem_id,
            "band": self.band,
            "center_hz": self.center_hz,
            "target_role": self.target_role,
            "protected_role": self.protected_role,
            "suggested_eq_delta_db": self.suggested_eq_delta_db,
            "bounded_subtractive_delta": self.evidence.get(
                "bounded_subtractive_delta"
            ),
            "source_stem_contribution": source_stem_contribution,
            "lane_conflict_pair": lane_conflict_pair,
            "band_energy_db": band_energy_db,
            "role_lane_affinity": role_lane_affinity,
            "current_frequency_context_cache_freshness": self.evidence.get(
                "current_frequency_context_cache_freshness"
            ),
            "current_frequency_context_cache_no_live_authority": self.evidence.get(
                "current_frequency_context_cache_no_live_authority"
            ),
            "sum_bus_check": self.evidence.get("sum_bus_check"),
            "feedback_safety": "required_before_apply",
            "true_peak_headroom": "required_before_apply",
            "eq_band_readback": self.evidence.get("eq_band_readback"),
            "readback_required_before_promotion": True,
            "rollback_required_before_promotion": True,
            "operator_review_required": True,
            "read_only_or_dry_run_only": True,
            "live_write": False,
            "queue_mutation": False,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "can_apply_live": False,
            "can_import_to_queue": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "queue_action": "not_queueable_until_promoted",
            "can_authorize_live": False,
            "mutates_mixer": False,
            "blockers": list(self.blockers),
        }

    def to_operator_proposal(self) -> Dict[str, Any]:
        """Return a Product Layer style proposal without live-apply fields."""

        frequency_lane_safety_summary = self.frequency_lane_safety_summary()
        blockers = [
            "unsupported_value_type:frequency_split",
            "not_in_supervised_fader_gain_executor",
            "hardware_readback_and_rollback_not_proven",
            *self.blockers,
        ]
        proposed_action = {
            "action_type": self.action_type,
            "target_source_id": self.target_source_id,
            "protected_source_id": self.protected_source_id,
            "stem_id": self.stem_id,
            "band": self.band,
            "center_hz": self.center_hz,
            "suggested_eq_delta_db": self.suggested_eq_delta_db,
            "frequency_lane_safety_summary": frequency_lane_safety_summary,
        }
        evidence = {
            **self.evidence,
            "frequency_lane_safety_summary": frequency_lane_safety_summary,
            "live_write": False,
            "queue_mutation": False,
            "can_apply_live": False,
            "can_import_to_queue": False,
            "queue_import_allowed": False,
            "live_apply_allowed": False,
            "read_only_or_dry_run_only": True,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "queue_action": "not_queueable_until_promoted",
            "mutates_mixer": False,
            "readback_required_before_promotion": True,
            "rollback_required_before_promotion": True,
        }
        proposal = {
            "id": self.id,
            "method_id": FREQUENCY_SEPARATION_METHOD_ID,
            "source": FREQUENCY_SEPARATION_METHOD_ID,
            "title": f"{self.target_source_id}: frequency separation proposal",
            "reason": self.reason,
            "target": self.target_source_id,
            "kind": "frequency_split",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "queue_import_allowed": False,
            "live_apply_allowed": False,
            "read_only_or_dry_run_only": True,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "mutates_mixer": False,
            "blockers": blockers,
            "proposed_action": proposed_action,
            "evidence": evidence,
        }
        proposal["operator_correction"] = build_operator_correction_payload(
            proposal=proposal,
            method_id=FREQUENCY_SEPARATION_METHOD_ID,
            target=self.target_source_id,
            proposed_action=proposed_action,
            blockers=blockers,
        )
        return proposal


@dataclass(frozen=True)
class FrequencySeparationResult:
    method_id: str
    version: str
    status: str
    recommendations: Tuple[FrequencySeparationRecommendation, ...]
    warnings: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_id": self.method_id,
            "version": self.version,
            "status": self.status,
            "recommendations": [item.to_dict() for item in self.recommendations],
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def _usable_sources(
    sources: Iterable[FrequencySeparationSource],
) -> Tuple[FrequencySeparationSource, ...]:
    return tuple(
        source
        for source in sources
        if source.active
        and not source.muted
        and _clip01(source.confidence) >= MIN_CONFIDENCE
        and any(_band_level(source, band) >= MIN_BAND_LEVEL_DB for band in BAND_CENTERS_HZ)
    )


def _source_identity_gate(source: FrequencySeparationSource):
    return evaluate_source_identity_downstream_trust(
        source_id=source.source_id,
        source_type=source.source_type,
        trust_state=source.source_identity_trust_state,
        trust_confidence=source.source_identity_trust_confidence,
        replay_outcome_count=source.source_identity_replay_outcome_count,
        current_state_cache_freshness=source.source_identity_calibration_freshness,
    )


def _source_identity_usable_sources(
    sources: Iterable[FrequencySeparationSource],
) -> Tuple[Tuple[FrequencySeparationSource, ...], Tuple[Dict[str, Any], ...], Tuple[str, ...]]:
    usable = []
    blocked = []
    blockers = set()
    for source in sources:
        decision = _source_identity_gate(source)
        if decision.usable_for_downstream:
            usable.append(source)
            continue
        blocked.append(decision.to_dict())
        blockers.update(decision.blockers)
    return tuple(usable), tuple(blocked), tuple(sorted(blockers))


def _candidate_recommendation(
    protected: FrequencySeparationSource,
    target: FrequencySeparationSource,
    band: str,
    *,
    level_gap_db: float,
    overlap_strength: float,
    current_state_freshness_status: str,
    current_state_no_authority: bool,
) -> FrequencySeparationRecommendation:
    confidence = round(
        _clip01(
            0.45
            + 0.25 * overlap_strength
            + 0.15 * _clip01(protected.confidence)
            + 0.15 * _clip01(target.confidence)
        ),
        3,
    )
    suggested_cut = -round(_clip(0.8 + 2.2 * overlap_strength, 0.8, MAX_SUPPORT_CUT_DB), 1)
    target_role = _role_for(target.source_type, target.arrangement_role)
    protected_role = _role_for(protected.source_type, protected.arrangement_role)
    source_identity_trust = {
        "target_source": _source_identity_gate(target).to_dict(),
        "protected_source": _source_identity_gate(protected).to_dict(),
    }
    severity = "medium" if confidence >= 0.72 or abs(suggested_cut) >= 2.2 else "low"
    stem_id = target.stem_id or protected.stem_id or "unknown"
    reason = (
        f"Separate {target.label or target.source_id} from "
        f"{protected.label or protected.source_id} "
        f"in the {band.replace('_', '-')} lane before applying EQ."
    )
    return FrequencySeparationRecommendation(
        id=(
            f"{FREQUENCY_SEPARATION_METHOD_ID}:{stem_id}:{band}:"
            f"{target.source_id}:from:{protected.source_id}"
        ),
        status="proposal_only",
        action_type="frequency_lane_separation_candidate",
        target_source_id=target.source_id,
        protected_source_id=protected.source_id,
        stem_id=stem_id,
        band=band,
        center_hz=BAND_CENTERS_HZ[band],
        target_role=target_role,
        protected_role=protected_role,
        suggested_eq_delta_db=suggested_cut,
        confidence=confidence,
        severity=severity,
        reason=reason,
        evidence={
            "source_stem_contribution": {
                "target": round(_clip01(target.stem_contribution.get(band, 0.0)), 3),
                "protected": round(_clip01(protected.stem_contribution.get(band, 0.0)), 3),
            },
            "lane_conflict_pair": {
                "target_source_id": target.source_id,
                "protected_source_id": protected.source_id,
                "band": band,
                "center_hz": BAND_CENTERS_HZ[band],
            },
            "band_energy_db": {
                "target": round(_band_level(target, band), 2),
                "protected": round(_band_level(protected, band), 2),
                "level_gap_db": round(level_gap_db, 2),
            },
            "role_lane_affinity": {
                "target": round(
                    _role_band_affinity(target.source_type, target.arrangement_role, band),
                    3,
                ),
                "protected": round(
                    _role_band_affinity(protected.source_type, protected.arrangement_role, band),
                    3,
                ),
            },
            "bounded_subtractive_delta": True,
            "current_frequency_context_cache_freshness": current_state_freshness_status,
            "current_frequency_context_cache_no_live_authority": current_state_no_authority,
            "sum_bus_check": "required_before_apply",
            "eq_band_readback": "required_before_apply",
            "source_identity_downstream_trust": source_identity_trust,
            "rollback_eq_band_readback": {
                "target_source_id": target.source_id,
                "protected_source_id": protected.source_id,
                "stem_id": stem_id,
                "band": band,
                "center_hz": BAND_CENTERS_HZ[band],
                "suggested_eq_delta_db": suggested_cut,
                "source": READ_REQUIRED_SOURCE,
                "required_before_live_promotion": True,
                "live_write": False,
            },
            "old_value_source": READ_REQUIRED_SOURCE,
            "rollback_value_source": READ_REQUIRED_SOURCE,
        },
    )


def analyze_frequency_separation_need(
    sources: Sequence[FrequencySeparationSource],
    *,
    current_state_cache_freshness: Optional[Mapping[str, Any]] = None,
    conflict_tolerance_db: float = DEFAULT_CONFLICT_TOLERANCE_DB,
    max_recommendations: int = 8,
) -> FrequencySeparationResult:
    """Analyze source/stem band lanes and emit proposal-only split candidates."""

    current_state_freshness = _freshness_payload(current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
    metadata: Dict[str, Any] = {
        "method_id": FREQUENCY_SEPARATION_METHOD_ID,
        "version": FREQUENCY_SEPARATION_VERSION,
        "proposal_only": True,
        "live_write_allowed": False,
        "can_import_to_queue": False,
        "can_apply_live": False,
        "queue_import_allowed": False,
        "live_apply_allowed": False,
        "read_only_or_dry_run_only": True,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "queue_action": "not_queueable_until_promoted",
        "mutates_mixer": False,
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "current_frequency_context_cache_no_live_authority": current_state_no_authority,
        "old_value_source": READ_REQUIRED_SOURCE,
        "readback_required": True,
        "rollback_required": True,
        "blockers": (),
    }
    source_identity_usable, blocked_source_identity, source_identity_blockers = (
        _source_identity_usable_sources(sources)
    )
    metadata["source_identity_blocked_sources"] = list(blocked_source_identity)
    metadata["source_identity_blocked_source_count"] = len(blocked_source_identity)

    if current_state_freshness_status != "fresh":
        metadata["blockers"] = ("frequency_separation_context_cache_not_fresh",)
        metadata["recommendation_count"] = 0
        return FrequencySeparationResult(
            method_id=FREQUENCY_SEPARATION_METHOD_ID,
            version=FREQUENCY_SEPARATION_VERSION,
            status="analysis_only",
            recommendations=(),
            warnings=(
                "frequency separation context cache is not fresh; "
                "collect fresh source/stem spectrum evidence before proposals",
            ),
            metadata=metadata,
        )
    if not current_state_no_authority:
        metadata["blockers"] = ("frequency_separation_context_cache_live_authority_present",)
        metadata["recommendation_count"] = 0
        return FrequencySeparationResult(
            method_id=FREQUENCY_SEPARATION_METHOD_ID,
            version=FREQUENCY_SEPARATION_VERSION,
            status="analysis_only",
            recommendations=(),
            warnings=(
                "frequency separation context cache has live authority; "
                "collect read-only source/stem spectrum evidence before proposals",
            ),
            metadata=metadata,
        )
    if source_identity_blockers:
        metadata["blockers"] = source_identity_blockers
        metadata["recommendation_count"] = 0
        return FrequencySeparationResult(
            method_id=FREQUENCY_SEPARATION_METHOD_ID,
            version=FREQUENCY_SEPARATION_VERSION,
            status="analysis_only",
            recommendations=(),
            warnings=(
                "source identity replay calibration blocks frequency separation; "
                "review source identity trust before lane proposals",
            ),
            metadata=metadata,
        )

    usable = _usable_sources(source_identity_usable)
    metadata["source_count"] = len(sources)
    metadata["usable_source_count"] = len(usable)
    if len(usable) < 2:
        metadata["recommendation_count"] = 0
        return FrequencySeparationResult(
            method_id=FREQUENCY_SEPARATION_METHOD_ID,
            version=FREQUENCY_SEPARATION_VERSION,
            status="analysis_only",
            recommendations=(),
            warnings=("not enough active sources for frequency separation",),
            metadata=metadata,
        )

    recommendations = []
    tolerance = max(0.5, _safe_float(conflict_tolerance_db, DEFAULT_CONFLICT_TOLERANCE_DB))
    for index, first in enumerate(usable):
        for second in usable[index + 1:]:
            for band in BAND_CENTERS_HZ:
                first_level = _band_level(first, band)
                second_level = _band_level(second, band)
                if first_level < MIN_BAND_LEVEL_DB or second_level < MIN_BAND_LEVEL_DB:
                    continue
                level_gap = abs(first_level - second_level)
                if level_gap > tolerance:
                    continue

                first_score = _lane_score(first, band)
                second_score = _lane_score(second, band)
                if abs(first_score - second_score) < 0.12:
                    continue
                protected, target = (
                    (first, second) if first_score > second_score else (second, first)
                )
                overlap_strength = 1.0 - (level_gap / tolerance)
                recommendations.append(
                    _candidate_recommendation(
                        protected,
                        target,
                        band,
                        level_gap_db=level_gap,
                        overlap_strength=overlap_strength,
                        current_state_freshness_status=current_state_freshness_status,
                        current_state_no_authority=current_state_no_authority,
                    )
                )

    recommendations = sorted(
        recommendations,
        key=lambda item: (item.confidence, abs(item.suggested_eq_delta_db)),
        reverse=True,
    )[: max(0, int(max_recommendations))]
    status = "proposal_only" if recommendations else "analysis_only"
    metadata["recommendation_count"] = len(recommendations)
    metadata["conflict_tolerance_db"] = tolerance
    metadata["operator_review_required"] = True
    return FrequencySeparationResult(
        method_id=FREQUENCY_SEPARATION_METHOD_ID,
        version=FREQUENCY_SEPARATION_VERSION,
        status=status,
        recommendations=tuple(recommendations),
        metadata=metadata,
    )


def build_frequency_separation_proposals(
    sources: Sequence[FrequencySeparationSource],
    *,
    current_state_cache_freshness: Optional[Mapping[str, Any]] = None,
    conflict_tolerance_db: float = DEFAULT_CONFLICT_TOLERANCE_DB,
    max_recommendations: int = 8,
) -> Tuple[Dict[str, Any], ...]:
    result = analyze_frequency_separation_need(
        sources,
        current_state_cache_freshness=current_state_cache_freshness,
        conflict_tolerance_db=conflict_tolerance_db,
        max_recommendations=max_recommendations,
    )
    return tuple(item.to_operator_proposal() for item in result.recommendations)
