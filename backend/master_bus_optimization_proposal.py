"""Proposal-only master bus optimization recommendations.

This module is transport-free. It turns master bus loudness, true-peak,
dynamics, tonal balance, and stereo evidence into operator-facing proposals, but
it never creates a live console write request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


MASTER_BUS_METHOD_ID = "master_bus_optimization_proposal"
MASTER_BUS_VERSION = "master_bus_optimization_proposal_v1"
MIN_CONFIDENCE = 0.45
MIN_TRIM_DELTA_DB = 0.3
MIN_EQ_DELTA_DB = 0.4
MASTER_STATIC_UP_GAIN_CAP_DB = 1.0
MASTER_STATIC_DOWN_GAIN_CAP_DB = 3.0
MASTER_TRUE_PEAK_RESERVE_DB = 0.5
REFERENCE_FEED_ROLE_TOKENS = ("reference", "measurement", "capture", "analysis")


@dataclass(frozen=True)
class MasterBusTargetProfile:
    id: str
    target_integrated_lufs: float
    max_true_peak_dbfs: float
    target_crest_factor_db: float
    max_lra_db: float
    max_static_loudness_up_db: float


TARGET_PROFILES: Dict[str, MasterBusTargetProfile] = {
    "live": MasterBusTargetProfile("live", -18.0, -2.0, 11.0, 8.0, 1.0),
    "studio": MasterBusTargetProfile("studio", -16.0, -1.0, 10.0, 7.0, 1.0),
    "streaming": MasterBusTargetProfile("streaming", -14.0, -1.0, 9.0, 6.0, 0.8),
    "broadcast": MasterBusTargetProfile("broadcast", -23.0, -1.0, 12.0, 10.0, 1.0),
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


def resolve_target_profile(target_profile: str, reference_lufs: Optional[float] = None) -> MasterBusTargetProfile:
    profile_id = str(target_profile or "live").strip().lower()
    profile = TARGET_PROFILES.get(profile_id, TARGET_PROFILES["live"])
    if reference_lufs is None:
        return profile

    reference = _safe_float(reference_lufs, profile.target_integrated_lufs)
    target = _clip(reference, -24.0, -12.0)
    return MasterBusTargetProfile(
        id=profile.id,
        target_integrated_lufs=round(target, 1),
        max_true_peak_dbfs=profile.max_true_peak_dbfs,
        target_crest_factor_db=profile.target_crest_factor_db,
        max_lra_db=profile.max_lra_db,
        max_static_loudness_up_db=profile.max_static_loudness_up_db,
    )


@dataclass(frozen=True)
class MasterBusAnalysisInput:
    bus_id: str = "main_lr"
    bus_label: str = "Main L/R"
    target_profile: str = "live"
    reference_lufs: Optional[float] = None
    integrated_lufs: float = -18.0
    short_term_lufs: float = -18.0
    true_peak_dbfs: float = -4.0
    crest_factor_db: float = 10.0
    lra_db: float = 5.0
    spectral_tilt_db: float = 0.0
    low_end_excess_db: float = 0.0
    harshness_db: float = 0.0
    stereo_correlation: float = 0.2
    current_limiter_ceiling_dbfs: Optional[float] = None
    master_headroom_db: Optional[float] = None
    limiter_or_dynamics_proof: bool = False
    output_feed_role: str = "pa_main"
    output_feed_label: str = "PA main"
    pa_feed_readback_available: bool = False
    pa_feed_safe_for_master_change: bool = False
    pa_feed_cache_freshness: Optional[Dict[str, Any]] = None
    is_reference_feed: bool = False
    confidence: float = 0.75
    persistence_cycles: int = 1
    current_state_cache_freshness: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class MasterBusRecommendation:
    id: str
    status: str
    action_type: str
    bus_id: str
    bus_label: str
    target_profile: str
    target_integrated_lufs: float
    current_integrated_lufs: float
    short_term_lufs: float
    current_true_peak_dbfs: float
    max_true_peak_dbfs: float
    true_peak_headroom_db: float
    suggested_trim_db: float
    limiter_ceiling_dbfs: float
    broad_eq_tilt_db: float
    low_shelf_db: float
    high_shelf_db: float
    bus_compression_ratio: float
    bus_compression_threshold_db: float
    target_gain_reduction_db: float
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

    def to_operator_proposal(self) -> Dict[str, Any]:
        """Return a Product Layer style proposal without live-apply fields."""

        return {
            "id": self.id,
            "method_id": MASTER_BUS_METHOD_ID,
            "source": MASTER_BUS_METHOD_ID,
            "title": f"{self.bus_label}: master bus proposal",
            "reason": self.reason,
            "target": self.bus_id,
            "kind": "master_bus",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:master_bus",
                "not_in_supervised_fader_gain_executor",
                "hardware_readback_and_rollback_not_proven",
                *self.blockers,
            ],
            "proposed_action": {
                "action_type": self.action_type,
                "target_integrated_lufs": self.target_integrated_lufs,
                "max_true_peak_dbfs": self.max_true_peak_dbfs,
                "suggested_trim_db": self.suggested_trim_db,
                "limiter_ceiling_dbfs": self.limiter_ceiling_dbfs,
                "broad_eq_tilt_db": self.broad_eq_tilt_db,
                "low_shelf_db": self.low_shelf_db,
                "high_shelf_db": self.high_shelf_db,
                "bus_compression_ratio": self.bus_compression_ratio,
                "bus_compression_threshold_db": self.bus_compression_threshold_db,
                "target_gain_reduction_db": self.target_gain_reduction_db,
            },
            "evidence": {
                **self.evidence,
                "live_write": False,
                "queue_mutation": False,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": True,
            },
        }


def _severity(
    loudness_gap_db: float,
    true_peak_headroom_db: float,
    blockers: Iterable[str],
    eq_magnitude_db: float,
    target_gain_reduction_db: float,
) -> str:
    blocker_set = set(blockers)
    if (
        "true_peak_headroom_insufficient" in blocker_set
        or "master_reference_feed_excluded" in blocker_set
        or "pa_feed_safety_not_confirmed" in blocker_set
        or abs(loudness_gap_db) >= 5.0
    ):
        return "high"
    if (
        abs(loudness_gap_db) >= 1.5
        or true_peak_headroom_db <= 1.0
        or eq_magnitude_db >= 1.2
        or target_gain_reduction_db >= 2.0
    ):
        return "medium"
    return "low"


def _safe_static_trim(loudness_gap_db: float, true_peak_headroom_db: float, profile: MasterBusTargetProfile) -> float:
    if loudness_gap_db > 0.0:
        available = max(0.0, true_peak_headroom_db - MASTER_TRUE_PEAK_RESERVE_DB)
        max_up = min(available, MASTER_STATIC_UP_GAIN_CAP_DB, profile.max_static_loudness_up_db)
        return round(_clip(loudness_gap_db, 0.0, max_up), 1)
    return round(_clip(loudness_gap_db, -MASTER_STATIC_DOWN_GAIN_CAP_DB, 0.0), 1)


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


def _is_reference_feed(analysis: MasterBusAnalysisInput) -> bool:
    role = str(analysis.output_feed_role or "").strip().lower().replace("-", "_").replace(" ", "_")
    return bool(analysis.is_reference_feed or any(token in role for token in REFERENCE_FEED_ROLE_TOKENS))


def _tonal_moves(analysis: MasterBusAnalysisInput) -> Tuple[float, float, float]:
    tilt = _safe_float(analysis.spectral_tilt_db)
    low_excess = _safe_float(analysis.low_end_excess_db)
    harshness = _safe_float(analysis.harshness_db)

    broad_eq_tilt = 0.0
    if abs(tilt) >= 1.5:
        broad_eq_tilt = round(_clip(-0.25 * tilt, -1.5, 1.5), 1)

    low_shelf = 0.0
    if low_excess >= 1.2:
        low_shelf = round(-_clip(0.35 * low_excess, 0.4, 2.5), 1)

    high_shelf = 0.0
    if harshness >= 1.5:
        high_shelf = round(-_clip(0.30 * harshness, 0.4, 2.0), 1)

    return broad_eq_tilt, low_shelf, high_shelf


def _dynamics_moves(
    analysis: MasterBusAnalysisInput,
    profile: MasterBusTargetProfile,
) -> Tuple[float, float, float, bool]:
    crest = max(0.0, _safe_float(analysis.crest_factor_db))
    lra = max(0.0, _safe_float(analysis.lra_db))
    crest_excess = max(0.0, crest - profile.target_crest_factor_db)
    lra_excess = max(0.0, lra - profile.max_lra_db)
    needs_glue = crest_excess >= 2.0 or lra_excess >= 1.5
    if not needs_glue:
        return 1.0, 0.0, 0.0, False

    target_gr = round(_clip(1.0 + 0.35 * crest_excess + 0.25 * lra_excess, 1.0, 3.5), 1)
    ratio = round(_clip(1.2 + 0.12 * crest_excess + 0.08 * lra_excess, 1.2, 2.5), 2)
    threshold = round(_clip(_safe_float(analysis.short_term_lufs, analysis.integrated_lufs) + 3.0, -28.0, -8.0), 1)
    return ratio, threshold, target_gr, True


def analyze_master_bus_need(
    analysis: MasterBusAnalysisInput,
    *,
    min_confidence: float = MIN_CONFIDENCE,
) -> MasterBusRecommendation:
    profile = resolve_target_profile(analysis.target_profile, analysis.reference_lufs)
    integrated_lufs = _safe_float(analysis.integrated_lufs, -100.0)
    short_term_lufs = _safe_float(analysis.short_term_lufs, integrated_lufs)
    true_peak = _safe_float(analysis.true_peak_dbfs, -120.0)
    measured_headroom = (
        profile.max_true_peak_dbfs - true_peak
        if analysis.master_headroom_db is None
        else _safe_float(analysis.master_headroom_db)
    )
    headroom = round(measured_headroom, 2)
    loudness_gap = round(profile.target_integrated_lufs - integrated_lufs, 2)
    suggested_trim = _safe_static_trim(loudness_gap, headroom, profile)
    current_ceiling = (
        profile.max_true_peak_dbfs
        if analysis.current_limiter_ceiling_dbfs is None
        else _safe_float(analysis.current_limiter_ceiling_dbfs, profile.max_true_peak_dbfs)
    )
    limiter_ceiling = round(min(current_ceiling, profile.max_true_peak_dbfs), 1)
    broad_eq_tilt, low_shelf, high_shelf = _tonal_moves(analysis)
    ratio, threshold, target_gr, needs_glue = _dynamics_moves(analysis, profile)
    confidence_input = _clip01(analysis.confidence)
    persistence = max(1, int(analysis.persistence_cycles or 1))
    stereo_correlation = _clip(_safe_float(analysis.stereo_correlation), -1.0, 1.0)
    current_state_freshness = _freshness_payload(analysis.current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
    pa_feed_freshness = _freshness_payload(analysis.pa_feed_cache_freshness)
    pa_feed_freshness_status = str(pa_feed_freshness.get("status") or "unknown")
    pa_feed_no_authority = _freshness_is_no_authority(pa_feed_freshness)
    output_feed_role = str(analysis.output_feed_role or "unknown").strip().lower() or "unknown"

    blockers = []
    if integrated_lufs <= -70.0 or true_peak <= -90.0:
        blockers.append("master_signal_below_analysis_floor")
    if _is_reference_feed(analysis):
        blockers.append("master_reference_feed_excluded")
    if not analysis.pa_feed_readback_available:
        blockers.append("pa_feed_readback_missing")
    if not analysis.pa_feed_safe_for_master_change:
        blockers.append("pa_feed_safety_not_confirmed")
    if pa_feed_freshness_status != "fresh":
        blockers.append("pa_feed_cache_not_fresh")
    if pa_feed_freshness_status == "fresh" and not pa_feed_no_authority:
        blockers.append("pa_feed_cache_live_authority_present")
    if loudness_gap > 0.0 and headroom <= MASTER_TRUE_PEAK_RESERVE_DB:
        blockers.append("true_peak_headroom_insufficient")
    if loudness_gap > suggested_trim + 0.2 and not analysis.limiter_or_dynamics_proof:
        blockers.append("master_loudness_requires_approved_limiter_or_dynamics_stage")
    if needs_glue and not analysis.limiter_or_dynamics_proof and loudness_gap > 1.5:
        blockers.append("master_loudness_requires_approved_limiter_or_dynamics_stage")
    if stereo_correlation < -0.15:
        blockers.append("stereo_correlation_phase_risk")
    if current_state_freshness_status != "fresh":
        blockers.append("master_bus_current_state_cache_not_fresh")
    if current_state_freshness_status == "fresh" and not current_state_no_authority:
        blockers.append("master_bus_current_state_cache_live_authority_present")

    eq_magnitude = max(abs(broad_eq_tilt), abs(low_shelf), abs(high_shelf))
    ceiling_move_needed = current_ceiling > profile.max_true_peak_dbfs + 0.1
    action_needed = (
        abs(suggested_trim) >= MIN_TRIM_DELTA_DB
        or eq_magnitude >= MIN_EQ_DELTA_DB
        or target_gr > 0.0
        or ceiling_move_needed
    )
    if not action_needed:
        blockers.append("master_bus_change_not_needed")

    confidence = _clip01(
        0.55 * confidence_input
        + 0.15 * min(persistence / 3.0, 1.0)
        + 0.10 * min(abs(loudness_gap) / 4.0, 1.0)
        + 0.10 * min(max(headroom, 0.0) / 6.0, 1.0)
        + 0.10 * min(max(eq_magnitude, target_gr) / 2.5, 1.0)
    )

    blockers_tuple = tuple(sorted(set(blockers)))
    status = "proposal_only" if not blockers_tuple and confidence >= min_confidence else "analysis_only"
    label = analysis.bus_label or analysis.bus_id
    if status == "proposal_only":
        reason = (
            f"{label} can target {profile.target_integrated_lufs:.1f} LUFS with "
            f"{suggested_trim:+.1f} dB trim and bounded master-bus tone/dynamics settings "
            f"after PA feed safety is confirmed."
        )
    else:
        reason = f"{label} does not meet safe master-bus proposal criteria."

    evidence = {
        "source": MASTER_BUS_VERSION,
        "target_profile": profile.id,
        "integrated_lufs": round(integrated_lufs, 2),
        "short_term_lufs": round(short_term_lufs, 2),
        "target_integrated_lufs": profile.target_integrated_lufs,
        "loudness_gap_db": loudness_gap,
        "true_peak_dbfs": round(true_peak, 2),
        "max_true_peak_dbfs": profile.max_true_peak_dbfs,
        "true_peak_headroom_db": headroom,
        "crest_factor_db": round(_safe_float(analysis.crest_factor_db), 2),
        "lra_db": round(_safe_float(analysis.lra_db), 2),
        "spectral_tilt_db": round(_safe_float(analysis.spectral_tilt_db), 2),
        "low_end_excess_db": round(_safe_float(analysis.low_end_excess_db), 2),
        "harshness_db": round(_safe_float(analysis.harshness_db), 2),
        "stereo_correlation": round(stereo_correlation, 3),
        "current_limiter_ceiling_dbfs": round(current_ceiling, 2),
        "limiter_or_dynamics_proof": bool(analysis.limiter_or_dynamics_proof),
        "output_feed_role": output_feed_role,
        "output_feed_label": str(analysis.output_feed_label or ""),
        "pa_feed_readback_available": bool(analysis.pa_feed_readback_available),
        "pa_feed_safe_for_master_change": bool(analysis.pa_feed_safe_for_master_change),
        "pa_feed_cache_freshness_status": pa_feed_freshness_status,
        "pa_feed_cache_freshness": pa_feed_freshness,
        "pa_feed_cache_no_live_authority": pa_feed_no_authority,
        "is_reference_feed": bool(_is_reference_feed(analysis)),
        "persistence_cycles": persistence,
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "current_main_bus_processing_cache_no_live_authority": current_state_no_authority,
        "transport_free": True,
        "lufs_only_loudness_up_is_not_listening_master": True,
    }

    return MasterBusRecommendation(
        id=f"{MASTER_BUS_METHOD_ID}:{str(analysis.bus_id or 'main_lr')}",
        status=status,
        action_type="master_bus_proposal",
        bus_id=str(analysis.bus_id or "main_lr"),
        bus_label=label,
        target_profile=profile.id,
        target_integrated_lufs=profile.target_integrated_lufs,
        current_integrated_lufs=round(integrated_lufs, 2),
        short_term_lufs=round(short_term_lufs, 2),
        current_true_peak_dbfs=round(true_peak, 2),
        max_true_peak_dbfs=profile.max_true_peak_dbfs,
        true_peak_headroom_db=headroom,
        suggested_trim_db=suggested_trim,
        limiter_ceiling_dbfs=limiter_ceiling,
        broad_eq_tilt_db=broad_eq_tilt,
        low_shelf_db=low_shelf,
        high_shelf_db=high_shelf,
        bus_compression_ratio=ratio,
        bus_compression_threshold_db=threshold,
        target_gain_reduction_db=target_gr,
        confidence=round(confidence, 3),
        severity=_severity(loudness_gap, headroom, blockers_tuple, eq_magnitude, target_gr),
        reason=reason,
        blockers=blockers_tuple,
        evidence=evidence,
    )


def build_master_bus_proposals(
    analyses: Iterable[MasterBusAnalysisInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_master_bus_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
