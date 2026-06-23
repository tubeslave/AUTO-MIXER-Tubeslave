"""Proposal-only vocal de-esser recommendations.

The module is intentionally transport-free. It turns vocal sibilance evidence
into operator-facing proposals, but it never creates a live mixer write request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

try:
    from source_identity_downstream_gate import evaluate_source_identity_downstream_trust
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_downstream_gate import evaluate_source_identity_downstream_trust

try:
    from processing_proposal_adapters import build_operator_correction_payload
except ImportError:  # pragma: no cover - package import compatibility
    from backend.processing_proposal_adapters import build_operator_correction_payload


DEESSER_METHOD_ID = "deesser_vocal_sibilance"
DEESSER_VERSION = "deesser_vocal_sibilance_v1"
DEESSER_READ_REQUIRED_SOURCE = (
    "cached_or_physical_dynamics_or_dynamic_eq_read_required_before_promotion"
)
EPSILON = 1e-12
VOCAL_ROLE_TOKENS = ("vocal", "vox", "voice", "speech", "lead", "backing")
SIBILANCE_BAND = (6000.0, 10000.0)
DEFAULT_CENTER_HZ = 7500.0
MIN_CENTER_HZ = 5500.0
MAX_CENTER_HZ = 10500.0
MIN_THRESHOLD_DB = -42.0
MAX_THRESHOLD_DB = -12.0
MIN_RATIO = 1.5
MAX_RATIO = 6.0
MIN_MAX_GAIN_REDUCTION_DB = 1.0
MAX_MAX_GAIN_REDUCTION_DB = 8.0
DEESSER_MAKEUP_GAIN_DB = 0.0


def _ordered_unique(values: Iterable[str]) -> Tuple[str, ...]:
    ordered = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _deesser_required_safety_checks() -> Dict[str, str]:
    return {
        "source_identity": "required_before_live_promotion",
        "arrangement_role": "required_before_live_promotion",
        "speech_intelligibility": "required_before_live_promotion",
        "bounded_attenuation": "required_before_live_promotion",
        "sibilance_persistence": "required_before_live_promotion",
        "true_peak_headroom": "required_before_live_promotion",
        "dynamic_eq_or_deesser_bypass_state": "required_before_live_promotion",
    }


def _deesser_rollback_expectation() -> Dict[str, Any]:
    return {
        "required_before_live_promotion": True,
        "restore_previous_dynamics_or_dynamic_eq_values_after_readback": True,
        "restore_previous_bypass_state_after_physical_readback": True,
        "manual_operator_rollback_required_if_promoted": True,
    }


def _deesser_future_wing_readback_contract() -> Dict[str, Any]:
    return {
        "required": True,
        "required_before_live_promotion": True,
        "target_kind": "channel_deesser_or_dynamic_eq",
        "old_value_source": DEESSER_READ_REQUIRED_SOURCE,
        "readback_targets": [
            "deesser_band",
            "dynamic_eq_band",
            "threshold",
            "ratio",
            "range",
            "max_gain_reduction",
            "bypass_state",
        ],
        "blocked_live_actions": [
            "deesser_parameter_write",
            "dynamic_eq_band_write",
            "threshold_range_write",
        ],
        "requires_real_wing_proof": True,
        "requires_operator_approval": True,
        "requires_separate_non_fader_adr": True,
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


def _db_to_power(db_value: Any) -> float:
    numeric = _safe_float(db_value, -120.0)
    return 10.0 ** (numeric / 10.0)


def _normalize_band_map(band_energy_db: Mapping[str, Any]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    aliases = {
        "sibilance": "sibilance",
        "sibilanceindex": "sibilance",
        "6_10khz": "sibilance",
        "6000_10000": "sibilance",
        "presence": "presence",
        "harshness": "harshness",
        "air": "air",
        "mid": "mid",
        "low_mid": "low_mid",
    }
    for key, value in (band_energy_db or {}).items():
        text = str(key).strip().lower().replace("-", "_").replace(" ", "_")
        text = text.replace("/", "_").replace(".", "")
        canonical = aliases.get(text, text)
        normalized[canonical] = _safe_float(value, -120.0)
    return normalized


def is_vocal_role(source_role: str) -> bool:
    text = str(source_role or "").strip().lower().replace("-", "_").replace(" ", "_")
    return any(token in text for token in VOCAL_ROLE_TOKENS)


def estimate_sibilance_ratio(band_energy_db: Mapping[str, Any]) -> float:
    bands = _normalize_band_map(band_energy_db)
    sibilance_power = _db_to_power(bands.get("sibilance", bands.get("air", -120.0)))
    reference_power = sum(
        _db_to_power(bands.get(key, -120.0))
        for key in ("low_mid", "mid", "presence", "harshness", "sibilance", "air")
    )
    return _clip01(sibilance_power / max(reference_power, EPSILON))


@dataclass(frozen=True)
class DeesserAnalysisInput:
    channel_id: int
    source_role: str
    channel_label: str = ""
    arrangement_role: str = ""
    band_energy_db: Mapping[str, Any] = field(default_factory=dict)
    sibilance_ratio: Optional[float] = None
    sibilance_excess_db: float = 0.0
    peak_dbfs: float = -18.0
    contribution_share: float = 0.0
    persistence_cycles: int = 1
    true_peak_headroom_db: float = 6.0
    current_state_cache_freshness: Optional[Dict[str, Any]] = None
    arrangement_context_cache_freshness: Optional[Dict[str, Any]] = None
    source_identity_trust_state: str = "not_provided"
    source_identity_trust_confidence: float = 0.0
    source_identity_replay_outcome_count: int = 0
    source_identity_calibration_freshness: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class DeesserRecommendation:
    id: str
    status: str
    action_type: str
    channel_id: int
    channel_label: str
    source_role: str
    frequency_hz: float
    threshold_db: float
    ratio: float
    max_gain_reduction_db: float
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

    def operator_safety_blockers(self) -> Tuple[str, ...]:
        return _ordered_unique(
            (
                "deesser_live_apply_quarantined",
                "unsupported_value_type:deesser",
                "not_in_supervised_fader_gain_executor",
                "real_wing_proof_required_before_live_apply",
                "non_fader_live_expansion_requires_separate_adr",
                "operator_approval_required_before_supervised_deesser_apply",
                "hardware_readback_and_rollback_not_proven",
                *self.blockers,
            )
        )

    def deesser_safety_summary(self) -> Dict[str, Any]:
        source_identity = dict(
            self.evidence.get("source_identity_downstream_trust") or {}
        )
        return {
            "status": "passed" if not self.blockers else "blocked",
            "channel_id": self.channel_id,
            "source_role": self.source_role,
            "arrangement_role": self.evidence.get("arrangement_role"),
            "arrangement_deesser_profile": self.evidence.get(
                "arrangement_deesser_profile"
            ),
            "frequency_hz": self.frequency_hz,
            "threshold_db": self.threshold_db,
            "ratio": self.ratio,
            "max_gain_reduction_db": self.max_gain_reduction_db,
            "bounded_attenuation_only": self.evidence.get(
                "bounded_attenuation_only"
            ),
            "makeup_gain_db": DEESSER_MAKEUP_GAIN_DB,
            "boost_allowed": False,
            "sibilance_ratio": self.evidence.get("sibilance_ratio"),
            "sibilance_excess_db": self.evidence.get("sibilance_excess_db"),
            "true_peak_headroom_db": self.evidence.get("true_peak_headroom_db"),
            "current_dynamics_or_dynamic_eq_cache_freshness_status": (
                self.evidence.get("current_state_cache_freshness_status")
            ),
            "current_dynamics_or_dynamic_eq_cache_no_live_authority": (
                self.evidence.get(
                    "current_dynamics_or_dynamic_eq_cache_no_live_authority"
                )
            ),
            "arrangement_context_cache_freshness_status": self.evidence.get(
                "arrangement_context_cache_freshness_status"
            ),
            "current_arrangement_context_cache_no_live_authority": (
                self.evidence.get(
                    "current_arrangement_context_cache_no_live_authority"
                )
            ),
            "source_identity_trust_status": source_identity.get("status"),
            "source_identity_usable_for_downstream": source_identity.get(
                "usable_for_downstream"
            ),
            "sibilance_persistence": self.evidence.get("sibilance_persistence"),
            "speech_intelligibility": self.evidence.get("speech_intelligibility"),
            "quiet_words_tail_attack_preservation": self.evidence.get(
                "quiet_words_tail_attack_preservation"
            ),
            "required_safety_checks": _deesser_required_safety_checks(),
            "safety_blockers": list(self.operator_safety_blockers()),
            "operator_explanation": (
                "De-esser evidence is offline-only. Review the vocal identity, "
                "arrangement role, sibilance persistence, bounded attenuation "
                "and speech-intelligibility risk before any future WING "
                "promotion."
            ),
            "rollback_expectation": _deesser_rollback_expectation(),
            "future_wing_readback_contract": _deesser_future_wing_readback_contract(),
            "readback_required_before_promotion": True,
            "rollback_required_before_promotion": True,
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
            "blockers": list(self.blockers),
        }

    def to_operator_proposal(self) -> Dict[str, Any]:
        """Return a Product Layer style proposal without live-apply fields."""

        deesser_safety_summary = self.deesser_safety_summary()
        blockers = list(self.operator_safety_blockers())
        proposed_action = {
            "action_type": self.action_type,
            "frequency_hz": self.frequency_hz,
            "threshold_db": self.threshold_db,
            "ratio": self.ratio,
            "max_gain_reduction_db": self.max_gain_reduction_db,
            "makeup_gain_db": DEESSER_MAKEUP_GAIN_DB,
            "boost_allowed": False,
            "required_safety_checks": deesser_safety_summary[
                "required_safety_checks"
            ],
            "rollback_expectation": deesser_safety_summary[
                "rollback_expectation"
            ],
            "future_wing_readback_contract": deesser_safety_summary[
                "future_wing_readback_contract"
            ],
            "deesser_safety_summary": deesser_safety_summary,
        }
        evidence = {
            **self.evidence,
            "deesser_safety_summary": deesser_safety_summary,
            "required_safety_checks": deesser_safety_summary[
                "required_safety_checks"
            ],
            "operator_explanation": deesser_safety_summary[
                "operator_explanation"
            ],
            "rollback_expectation": deesser_safety_summary[
                "rollback_expectation"
            ],
            "future_wing_readback_contract": deesser_safety_summary[
                "future_wing_readback_contract"
            ],
            "old_value_source": DEESSER_READ_REQUIRED_SOURCE,
            "rollback_value_source": DEESSER_READ_REQUIRED_SOURCE,
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
            "method_id": DEESSER_METHOD_ID,
            "source": DEESSER_METHOD_ID,
            "title": f"{self.channel_label or f'Ch {self.channel_id}'}: de-esser proposal",
            "reason": self.reason,
            "channel": self.channel_id,
            "target": self.channel_id,
            "kind": "deesser",
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
            "live_write": False,
            "queue_mutation": False,
            "mutates_mixer": False,
            "blockers": blockers,
            "safety_blockers": deesser_safety_summary["safety_blockers"],
            "operator_explanation": deesser_safety_summary[
                "operator_explanation"
            ],
            "required_safety_checks": deesser_safety_summary[
                "required_safety_checks"
            ],
            "rollback_expectation": deesser_safety_summary[
                "rollback_expectation"
            ],
            "future_wing_readback_contract": deesser_safety_summary[
                "future_wing_readback_contract"
            ],
            "proposed_action": proposed_action,
            "deesser_safety_summary": deesser_safety_summary,
            "old_value": {
                "source": DEESSER_READ_REQUIRED_SOURCE,
                "required_before_apply": True,
            },
            "rollback_value": {
                "source": DEESSER_READ_REQUIRED_SOURCE,
                "required_before_apply": True,
            },
            "evidence": evidence,
        }
        proposal["operator_correction"] = build_operator_correction_payload(
            proposal=proposal,
            method_id=DEESSER_METHOD_ID,
            target=self.channel_id,
            proposed_action=proposed_action,
            blockers=blockers,
        )
        return proposal


def _severity_from_score(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _severity(blockers: Iterable[str], score: float) -> str:
    blocker_set = set(blockers)
    if (
        "source_identity_blocked_from_downstream_trust" in blocker_set
        or "source_identity_replay_context_not_fresh" in blocker_set
    ):
        return "high"
    if "source_identity_not_trusted_for_downstream" in blocker_set:
        return "medium"
    return _severity_from_score(score)


def _frequency_from_bands(band_energy_db: Mapping[str, Any]) -> float:
    bands = _normalize_band_map(band_energy_db)
    air = bands.get("air", -120.0)
    sibilance = bands.get("sibilance", -120.0)
    harshness = bands.get("harshness", -120.0)
    if air > sibilance + 1.5:
        return 9000.0
    if harshness > sibilance + 2.0:
        return 6200.0
    return DEFAULT_CENTER_HZ


def _arrangement_deesser_profile(arrangement_role: str, source_role: str) -> Tuple[str, float, float, float]:
    text = f"{arrangement_role} {source_role}".strip().lower().replace("-", "_")
    if any(token in text for token in ("backing", "background", "tucked", "double", "harmony")):
        return "tucked_or_backing_vocal", 0.8, 1.0, 2.0
    if any(token in text for token in ("lead", "main", "anchor", "solo")):
        return "lead_vocal", 1.0, 0.0, 0.0
    return "general_vocal", 0.9, 0.5, 1.0


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
        "missing_fields",
        "untimestamped_present_fields",
        "stale_after_sec",
        "read_only",
        "live_write",
        "queries_console",
    )
    normalized = {key: payload.get(key) for key in allowed_keys if key in payload}
    normalized["provided"] = True
    normalized.setdefault("status", _freshness_status(payload))
    return normalized


def _freshness_is_no_authority(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("read_only") is True
        and payload.get("live_write") is False
        and payload.get("queries_console") is False
    )


def analyze_deesser_need(
    analysis: DeesserAnalysisInput,
    *,
    min_ratio: float = 0.34,
    min_excess_db: float = 2.0,
    min_confidence: float = 0.45,
) -> DeesserRecommendation:
    ratio = (
        _clip01(analysis.sibilance_ratio)
        if analysis.sibilance_ratio is not None
        else estimate_sibilance_ratio(analysis.band_energy_db)
    )
    excess_db = max(0.0, _safe_float(analysis.sibilance_excess_db))
    contribution = _clip01(analysis.contribution_share)
    persistence = max(1, int(analysis.persistence_cycles or 1))
    is_vocal = is_vocal_role(analysis.source_role)
    current_state_freshness = _freshness_payload(analysis.current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
    arrangement_context_freshness = _freshness_payload(
        analysis.arrangement_context_cache_freshness
    )
    arrangement_context_freshness_status = str(
        arrangement_context_freshness.get("status") or "unknown"
    )
    arrangement_context_no_authority = _freshness_is_no_authority(
        arrangement_context_freshness
    )
    arrangement_profile, score_scale, excess_floor_offset, threshold_offset_db = (
        _arrangement_deesser_profile(analysis.arrangement_role, analysis.source_role)
    )
    source_identity_trust = evaluate_source_identity_downstream_trust(
        source_id=f"ch:{int(analysis.channel_id)}",
        source_type=str(analysis.source_role or "unknown"),
        trust_state=analysis.source_identity_trust_state,
        trust_confidence=analysis.source_identity_trust_confidence,
        replay_outcome_count=analysis.source_identity_replay_outcome_count,
        current_state_cache_freshness=analysis.source_identity_calibration_freshness,
    )

    evidence = {
        "source": DEESSER_VERSION,
        "source_role": analysis.source_role,
        "arrangement_role": analysis.arrangement_role,
        "arrangement_deesser_profile": arrangement_profile,
        "sibilance_ratio": round(ratio, 3),
        "sibilance_excess_db": round(excess_db, 2),
        "sibilance_persistence": {
            "persistence_cycles": persistence,
            "minimum_cycles_before_proposal": 2,
            "status": "persistent" if persistence >= 2 else "single_cycle_review",
        },
        "contribution_share": round(contribution, 3),
        "persistence_cycles": persistence,
        "peak_dbfs": round(_safe_float(analysis.peak_dbfs), 2),
        "true_peak_headroom_db": round(_safe_float(analysis.true_peak_headroom_db), 2),
        "true_peak_headroom": round(_safe_float(analysis.true_peak_headroom_db), 2),
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "current_dynamics_or_dynamic_eq_cache_freshness": current_state_freshness,
        "current_dynamics_or_dynamic_eq_cache_no_live_authority": (
            current_state_no_authority
        ),
        "arrangement_context_cache_freshness_status": (
            arrangement_context_freshness_status
        ),
        "arrangement_context_cache_freshness": arrangement_context_freshness,
        "current_arrangement_context_cache_freshness": (
            arrangement_context_freshness
        ),
        "current_arrangement_context_cache_no_live_authority": (
            arrangement_context_no_authority
        ),
        "source_identity_downstream_trust": source_identity_trust.to_dict(),
        "transport_free": True,
        "vocal_source_role": analysis.source_role,
    }

    blockers = []
    if not is_vocal:
        blockers.append("source_not_vocal")
    effective_min_excess_db = min_excess_db + excess_floor_offset
    if ratio < min_ratio and excess_db < effective_min_excess_db:
        blockers.append("sibilance_below_threshold")
    if _safe_float(analysis.true_peak_headroom_db) < 1.0:
        blockers.append("insufficient_true_peak_headroom_for_processing")
    if current_state_freshness_status != "fresh":
        blockers.append("deesser_current_state_cache_not_fresh")
    if current_state_freshness_status == "fresh" and not current_state_no_authority:
        blockers.append("deesser_current_state_cache_live_authority_present")
    if arrangement_context_freshness_status != "fresh":
        blockers.append("deesser_arrangement_context_cache_not_fresh")
    if (
        arrangement_context_freshness_status == "fresh"
        and not arrangement_context_no_authority
    ):
        blockers.append("deesser_arrangement_context_cache_live_authority_present")
    blockers.extend(source_identity_trust.blockers)

    score = _clip01(
        (
            0.35 * (ratio / max(min_ratio, EPSILON))
            + 0.35 * (excess_db / 8.0)
            + 0.15 * contribution
            + 0.15 * min(persistence / 3.0, 1.0)
        )
        * score_scale
    )
    confidence = _clip01(score if is_vocal else score * 0.35)
    status = "proposal_only" if not blockers and confidence >= min_confidence else "analysis_only"

    frequency_hz = _clip(_frequency_from_bands(analysis.band_energy_db), MIN_CENTER_HZ, MAX_CENTER_HZ)
    ratio_value = _clip(MIN_RATIO + score * 4.0, MIN_RATIO, MAX_RATIO)
    max_gain_reduction_db = _clip(
        max(MIN_MAX_GAIN_REDUCTION_DB, excess_db * 0.75 + score * 2.0),
        MIN_MAX_GAIN_REDUCTION_DB,
        MAX_MAX_GAIN_REDUCTION_DB,
    )
    threshold = _clip(
        _safe_float(analysis.peak_dbfs) - (8.0 + score * 8.0) + threshold_offset_db,
        MIN_THRESHOLD_DB,
        MAX_THRESHOLD_DB,
    )
    label = analysis.channel_label or f"Ch {analysis.channel_id}"
    if status == "proposal_only":
        reason = (
            f"{label} has sustained vocal sibilance "
            f"(ratio {ratio:.2f}, excess {excess_db:.1f} dB)."
        )
    else:
        reason = f"{label} does not meet safe de-esser proposal criteria."

    evidence.update(
        {
            "sibilance_band_hz": list(SIBILANCE_BAND),
            "bounded_attenuation_only": True,
            "max_gain_reduction_db": round(max_gain_reduction_db, 1),
            "makeup_gain_db": DEESSER_MAKEUP_GAIN_DB,
            "boost_allowed": False,
            "speech_intelligibility": {
                "status": "operator_review_required_before_live_promotion",
                "bounded_attenuation_only": True,
                "boost_allowed": False,
                "max_gain_reduction_db": round(max_gain_reduction_db, 1),
                "arrangement_deesser_profile": arrangement_profile,
            },
            "quiet_words_tail_attack_preservation": {
                "status": "operator_review_required_before_live_promotion",
                "scope": "sibilance_band_attenuation_only",
                "broadband_gate_or_expander": False,
                "bounded_attenuation_only": True,
                "boost_allowed": False,
                "max_gain_reduction_db": round(max_gain_reduction_db, 1),
                "speech_intelligibility_required": True,
            },
            "dynamic_eq_or_deesser_readback": DEESSER_READ_REQUIRED_SOURCE,
            "rollback_dynamics_or_dynamic_eq_readback": DEESSER_READ_REQUIRED_SOURCE,
            "bypass_state": "readback_required_before_live_promotion",
        }
    )

    return DeesserRecommendation(
        id=f"{DEESSER_METHOD_ID}:ch{int(analysis.channel_id)}",
        status=status,
        action_type="deesser_proposal",
        channel_id=int(analysis.channel_id),
        channel_label=label,
        source_role=str(analysis.source_role or "unknown"),
        frequency_hz=round(frequency_hz, 1),
        threshold_db=round(threshold, 1),
        ratio=round(ratio_value, 2),
        max_gain_reduction_db=round(max_gain_reduction_db, 1),
        confidence=round(confidence, 3),
        severity=_severity(blockers, score),
        reason=reason,
        blockers=tuple(sorted(set(blockers))),
        evidence=evidence,
    )


def build_deesser_proposals(
    analyses: Iterable[DeesserAnalysisInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_deesser_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
