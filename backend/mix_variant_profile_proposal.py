"""Proposal-only mix variant profile recommendations.

This module is transport-free. It turns genre, task, arrangement, reference,
and safety evidence into operator-facing mix profile proposals, but it never
creates live console writes or named-artist style clone requests.
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


MIX_VARIANT_METHOD_ID = "mix_variant_profile_proposal"
MIX_VARIANT_VERSION = "mix_variant_profile_proposal_v1"
SUPPORTED_GENRES = {
    "pop_rock",
    "rock",
    "metal",
    "jazz",
    "acoustic",
    "electronic",
    "worship",
    "speech",
    "broadcast",
    "custom",
}
SUPPORTED_TASKS = {
    "live_foh",
    "studio_mix",
    "streaming_reference",
    "broadcast",
    "rehearsal",
    "soundcheck",
}
SAFE_VARIANT_WINDOWS = {"offline", "studio", "soundcheck", "rehearsal"}
MIN_CONFIDENCE = 0.5
MIN_GENRE_CONFIDENCE = 0.55
MIN_TASK_CONFIDENCE = 0.55
MIN_REFERENCE_CONFIDENCE = 0.45
MIN_ARRANGEMENT_CONFIDENCE = 0.5
STYLE_CLONE_SIMILARITY_LIMIT = 0.82
TRUE_PEAK_RESERVE_DB = 1.0
MAX_FEEDBACK_RISK_FOR_AMBIENT_LIVE_PROFILE = 0.72


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


def normalize_genre(value: Any) -> str:
    text = _normalize_token(value)
    aliases = {
        "pop": "pop_rock",
        "poprock": "pop_rock",
        "pop_rock_live": "pop_rock",
        "hard_rock": "rock",
        "heavy": "metal",
        "edm": "electronic",
        "dance": "electronic",
        "folk": "acoustic",
        "singer_songwriter": "acoustic",
        "church": "worship",
        "spoken": "speech",
        "voice": "speech",
        "podcast": "speech",
        "tv": "broadcast",
    }
    return aliases.get(text, text or "custom")


def normalize_task(value: Any) -> str:
    text = _normalize_token(value)
    aliases = {
        "live": "live_foh",
        "foh": "live_foh",
        "concert": "live_foh",
        "studio": "studio_mix",
        "mix": "studio_mix",
        "streaming": "streaming_reference",
        "reference": "streaming_reference",
        "broadcast_mix": "broadcast",
        "rehearsal_monitor": "rehearsal",
        "line_check": "soundcheck",
    }
    return aliases.get(text, text or "live_foh")


@dataclass(frozen=True)
class MixVariantTargetProfile:
    id: str
    genre: str
    task: str
    target_integrated_lufs: float
    lead_plane_lufs: float
    strong_plane_lufs: float
    support_plane_lufs: float
    fx_return_plane_lufs: float
    vocal_focus: float
    drum_punch: float
    bass_weight: float
    stereo_width: float
    ambience: float
    density: float
    notes: Tuple[str, ...] = ()

    def target_planes(self) -> Dict[str, float]:
        return {
            "lead_main": self.lead_plane_lufs,
            "strong_sources": self.strong_plane_lufs,
            "support_sources": self.support_plane_lufs,
            "fx_returns": self.fx_return_plane_lufs,
        }

    def intent(self) -> Dict[str, float]:
        return {
            "vocal_focus": self.vocal_focus,
            "drum_punch": self.drum_punch,
            "bass_weight": self.bass_weight,
            "stereo_width": self.stereo_width,
            "ambience": self.ambience,
            "density": self.density,
        }


GENRE_BASES: Dict[str, Dict[str, float]] = {
    "pop_rock": {"vocal": 0.78, "drums": 0.72, "bass": 0.68, "width": 0.56, "ambience": 0.34, "density": 0.62},
    "rock": {"vocal": 0.68, "drums": 0.82, "bass": 0.72, "width": 0.52, "ambience": 0.28, "density": 0.72},
    "metal": {"vocal": 0.62, "drums": 0.88, "bass": 0.78, "width": 0.58, "ambience": 0.22, "density": 0.82},
    "jazz": {"vocal": 0.58, "drums": 0.46, "bass": 0.58, "width": 0.48, "ambience": 0.44, "density": 0.34},
    "acoustic": {"vocal": 0.82, "drums": 0.28, "bass": 0.42, "width": 0.42, "ambience": 0.48, "density": 0.26},
    "electronic": {"vocal": 0.54, "drums": 0.78, "bass": 0.9, "width": 0.7, "ambience": 0.38, "density": 0.76},
    "worship": {"vocal": 0.84, "drums": 0.58, "bass": 0.55, "width": 0.62, "ambience": 0.58, "density": 0.54},
    "speech": {"vocal": 0.95, "drums": 0.0, "bass": 0.12, "width": 0.12, "ambience": 0.08, "density": 0.16},
    "broadcast": {"vocal": 0.84, "drums": 0.42, "bass": 0.38, "width": 0.34, "ambience": 0.18, "density": 0.35},
    "custom": {"vocal": 0.65, "drums": 0.55, "bass": 0.55, "width": 0.45, "ambience": 0.32, "density": 0.45},
}


def resolve_mix_variant_profile(
    genre: str,
    task: str,
    *,
    reference_lufs: Optional[float] = None,
    lead_role_present: bool = True,
    dense_arrangement: bool = False,
    feedback_risk: float = 0.0,
) -> MixVariantTargetProfile:
    normalized_genre = normalize_genre(genre)
    normalized_task = normalize_task(task)
    base = GENRE_BASES.get(normalized_genre, GENRE_BASES["custom"])
    task_target_lufs = {
        "live_foh": -18.0,
        "studio_mix": -16.0,
        "streaming_reference": -14.0,
        "broadcast": -23.0,
        "rehearsal": -20.0,
        "soundcheck": -20.0,
    }.get(normalized_task, -18.0)
    target_lufs = task_target_lufs
    if reference_lufs is not None:
        target_lufs = _clip(_safe_float(reference_lufs, task_target_lufs), -24.0, -12.0)

    vocal_focus = _clip01(base["vocal"] + (0.06 if lead_role_present else -0.08))
    drum_punch = _clip01(base["drums"] + (0.06 if dense_arrangement else 0.0))
    bass_weight = _clip01(base["bass"] + (0.04 if dense_arrangement else 0.0))
    stereo_width = _clip01(base["width"])
    ambience = _clip01(base["ambience"])
    density = _clip01(base["density"] + (0.08 if dense_arrangement else 0.0))

    notes = []
    if normalized_task in {"live_foh", "soundcheck", "rehearsal"}:
        stereo_width = min(stereo_width, 0.56)
        ambience = min(ambience, 0.38)
        notes.append("live_feedback_reserve")
    if feedback_risk >= 0.55:
        ambience = min(ambience, 0.22)
        stereo_width = min(stereo_width, 0.42)
        notes.append("feedback_risk_ambience_limited")
    if normalized_task == "broadcast":
        stereo_width = min(stereo_width, 0.38)
        ambience = min(ambience, 0.22)
        notes.append("broadcast_translation")
    if normalized_task == "streaming_reference":
        stereo_width = min(0.74, stereo_width + 0.08)
        notes.append("offline_reference_translation")

    lead_plane = -22.0 + (0.8 if vocal_focus >= 0.78 else 0.0)
    strong_plane = -25.0 + (0.7 if max(drum_punch, bass_weight) >= 0.78 else 0.0)
    support_plane = -30.0 + (0.4 if density >= 0.7 else 0.0)
    fx_plane = -35.0 + (0.6 if ambience >= 0.5 and normalized_task not in {"live_foh", "soundcheck"} else 0.0)

    return MixVariantTargetProfile(
        id=f"{normalized_genre}:{normalized_task}",
        genre=normalized_genre,
        task=normalized_task,
        target_integrated_lufs=round(target_lufs, 1),
        lead_plane_lufs=round(lead_plane, 1),
        strong_plane_lufs=round(strong_plane, 1),
        support_plane_lufs=round(support_plane, 1),
        fx_return_plane_lufs=round(fx_plane, 1),
        vocal_focus=round(vocal_focus, 3),
        drum_punch=round(drum_punch, 3),
        bass_weight=round(bass_weight, 3),
        stereo_width=round(stereo_width, 3),
        ambience=round(ambience, 3),
        density=round(density, 3),
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class MixVariantProfileAnalysisInput:
    variant_id: str
    genre: str
    task: str
    current_profile_id: str = ""
    reference_id: str = ""
    reference_lufs: Optional[float] = None
    reference_metrics_available: bool = False
    arrangement_context_available: bool = False
    arrangement_roles: Tuple[str, ...] = ()
    detected_source_count: int = 0
    stem_count: int = 0
    operator_intent: str = ""
    mix_window: str = "soundcheck"
    active_performance: bool = False
    current_integrated_lufs: float = -18.0
    current_true_peak_dbfs: float = -6.0
    main_bus_headroom_db: float = 4.0
    feedback_risk: float = 0.0
    style_clone_requested: bool = False
    reference_similarity_target: float = 0.55
    genre_confidence: float = 0.75
    task_confidence: float = 0.75
    reference_confidence: float = 0.65
    arrangement_confidence: float = 0.7
    persistence_cycles: int = 1
    current_state_cache_freshness: Optional[Dict[str, Any]] = None
    source_identity_trust_state: str = "not_provided"
    source_identity_trust_confidence: float = 0.0
    source_identity_replay_outcome_count: int = 0
    source_identity_calibration_freshness: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class MixVariantProfileRecommendation:
    id: str
    status: str
    action_type: str
    variant_id: str
    genre: str
    task: str
    target_profile: Dict[str, Any]
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
            "method_id": MIX_VARIANT_METHOD_ID,
            "source": MIX_VARIANT_METHOD_ID,
            "title": f"{self.variant_id}: mix variant profile proposal",
            "reason": self.reason,
            "target": self.variant_id,
            "kind": "mix_variant_profile",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:mix_variant_profile",
                "not_in_supervised_fader_gain_executor",
                "variant_profile_requires_operator_review",
                *self.blockers,
            ],
            "proposed_action": {
                "action_type": self.action_type,
                "genre": self.genre,
                "task": self.task,
                "target_profile": self.target_profile,
                "followup_proposal_methods": [
                    "bus_group_balance_proposal",
                    "fx_send_proposal",
                    "master_bus_optimization_proposal",
                    "scene_preset_management_proposal",
                ],
            },
            "evidence": {
                **self.evidence,
                "live_write": False,
                "queue_mutation": False,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": True,
            },
        }


def _profile_payload(profile: MixVariantTargetProfile) -> Dict[str, Any]:
    return {
        "id": profile.id,
        "genre": profile.genre,
        "task": profile.task,
        "target_integrated_lufs": profile.target_integrated_lufs,
        "target_planes_lufs": profile.target_planes(),
        "mix_intent": profile.intent(),
        "notes": list(profile.notes),
    }


def _severity(blockers: Iterable[str], profile: MixVariantTargetProfile, feedback_risk: float) -> str:
    blocker_set = set(blockers)
    if (
        "source_identity_blocked_from_downstream_trust" in blocker_set
        or "source_identity_replay_context_not_fresh" in blocker_set
        or "active_performance_variant_blocked" in blocker_set
        or "style_clone_request_blocked" in blocker_set
    ):
        return "high"
    if (
        "source_identity_not_trusted_for_downstream" in blocker_set
        or "source_identity_trust_state_unknown" in blocker_set
    ):
        return "medium"
    if feedback_risk >= MAX_FEEDBACK_RISK_FOR_AMBIENT_LIVE_PROFILE and profile.ambience >= 0.2:
        return "high"
    if abs(profile.target_integrated_lufs + 18.0) >= 4.0 or profile.density >= 0.75:
        return "medium"
    return "low"


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


def analyze_mix_variant_profile_need(
    analysis: MixVariantProfileAnalysisInput,
    *,
    min_confidence: float = MIN_CONFIDENCE,
) -> MixVariantProfileRecommendation:
    genre = normalize_genre(analysis.genre)
    task = normalize_task(analysis.task)
    roles = tuple(str(role).strip().lower() for role in analysis.arrangement_roles if str(role).strip())
    lead_present = any("lead" in role or "vocal" in role or "main" in role for role in roles)
    dense_arrangement = max(int(analysis.detected_source_count or 0), int(analysis.stem_count or 0)) >= 12
    feedback_risk = _clip01(analysis.feedback_risk)
    profile = resolve_mix_variant_profile(
        genre,
        task,
        reference_lufs=analysis.reference_lufs if analysis.reference_metrics_available else None,
        lead_role_present=lead_present,
        dense_arrangement=dense_arrangement,
        feedback_risk=feedback_risk,
    )
    current_profile = str(analysis.current_profile_id or "").strip().lower()
    current_lufs = _safe_float(analysis.current_integrated_lufs, -18.0)
    true_peak = _safe_float(analysis.current_true_peak_dbfs, -6.0)
    headroom = max(0.0, _safe_float(analysis.main_bus_headroom_db, -true_peak))
    genre_confidence = _clip01(analysis.genre_confidence)
    task_confidence = _clip01(analysis.task_confidence)
    reference_confidence = _clip01(analysis.reference_confidence)
    arrangement_confidence = _clip01(analysis.arrangement_confidence)
    similarity_target = _clip01(analysis.reference_similarity_target)
    persistence = max(1, int(analysis.persistence_cycles or 1))
    window = _normalize_token(analysis.mix_window)
    operator_intent = str(analysis.operator_intent or "").strip().lower()
    lufs_delta = profile.target_integrated_lufs - current_lufs
    current_state_freshness = _freshness_payload(analysis.current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
    source_identity_trust = evaluate_source_identity_downstream_trust(
        source_id=str(analysis.variant_id),
        source_type=",".join(roles) or str(profile.genre),
        trust_state=analysis.source_identity_trust_state,
        trust_confidence=analysis.source_identity_trust_confidence,
        replay_outcome_count=analysis.source_identity_replay_outcome_count,
        current_state_cache_freshness=analysis.source_identity_calibration_freshness,
    )

    blockers = []
    if genre not in SUPPORTED_GENRES:
        blockers.append("unsupported_genre_profile")
    if task not in SUPPORTED_TASKS:
        blockers.append("unsupported_variant_task")
    if not operator_intent:
        blockers.append("operator_intent_required")
    if analysis.active_performance or window not in SAFE_VARIANT_WINDOWS:
        blockers.append("active_performance_variant_blocked")
    if not analysis.arrangement_context_available:
        blockers.append("arrangement_context_missing")
    if analysis.reference_id and not analysis.reference_metrics_available:
        blockers.append("reference_metrics_missing")
    if analysis.style_clone_requested or similarity_target > STYLE_CLONE_SIMILARITY_LIMIT:
        blockers.append("style_clone_request_blocked")
    if genre_confidence < MIN_GENRE_CONFIDENCE:
        blockers.append("genre_confidence_low")
    if task_confidence < MIN_TASK_CONFIDENCE:
        blockers.append("task_confidence_low")
    if analysis.reference_id and reference_confidence < MIN_REFERENCE_CONFIDENCE:
        blockers.append("reference_confidence_low")
    if arrangement_confidence < MIN_ARRANGEMENT_CONFIDENCE:
        blockers.append("arrangement_confidence_low")
    if lufs_delta > 0.5 and headroom < TRUE_PEAK_RESERVE_DB + lufs_delta:
        blockers.append("variant_loudness_headroom_insufficient")
    if (
        task in {"live_foh", "soundcheck", "rehearsal"}
        and feedback_risk >= MAX_FEEDBACK_RISK_FOR_AMBIENT_LIVE_PROFILE
        and profile.ambience >= 0.2
    ):
        blockers.append("feedback_risk_too_high_for_variant_ambience")
    if current_profile == profile.id and abs(lufs_delta) < 0.2:
        blockers.append("mix_variant_profile_change_not_needed")
    if current_state_freshness_status != "fresh":
        blockers.append("mix_variant_context_cache_not_fresh")
    if current_state_freshness_status == "fresh" and not current_state_no_authority:
        blockers.append("mix_variant_context_cache_live_authority_present")
    blockers.extend(source_identity_trust.blockers)

    confidence = _clip01(
        0.22 * genre_confidence
        + 0.18 * task_confidence
        + 0.18 * arrangement_confidence
        + 0.14 * (reference_confidence if analysis.reference_id else 0.7)
        + 0.12 * min(persistence / 3.0, 1.0)
        + 0.08 * (1.0 if operator_intent else 0.0)
        + 0.05 * min(headroom / 6.0, 1.0)
        + 0.03 * (1.0 - feedback_risk)
    )
    blockers_tuple = tuple(sorted(set(blockers)))
    status = "proposal_only" if not blockers_tuple and confidence >= min_confidence else "analysis_only"
    if status == "proposal_only":
        reason = f"{analysis.variant_id} can be proposed as {profile.genre}/{profile.task} mix profile."
    else:
        reason = f"{analysis.variant_id} does not meet safe mix-variant proposal criteria."

    evidence = {
        "source": MIX_VARIANT_VERSION,
        "current_profile_id": analysis.current_profile_id,
        "reference_id": analysis.reference_id,
        "reference_lufs": None if analysis.reference_lufs is None else round(_safe_float(analysis.reference_lufs), 2),
        "reference_metrics_available": bool(analysis.reference_metrics_available),
        "arrangement_context_available": bool(analysis.arrangement_context_available),
        "arrangement_roles": list(roles),
        "detected_source_count": int(analysis.detected_source_count or 0),
        "stem_count": int(analysis.stem_count or 0),
        "operator_intent": operator_intent,
        "mix_window": window,
        "active_performance": bool(analysis.active_performance),
        "current_integrated_lufs": round(current_lufs, 2),
        "current_true_peak_dbfs": round(true_peak, 2),
        "main_bus_headroom_db": round(headroom, 2),
        "feedback_risk": round(feedback_risk, 3),
        "style_clone_requested": bool(analysis.style_clone_requested),
        "reference_similarity_target": round(similarity_target, 3),
        "genre_confidence": round(genre_confidence, 3),
        "task_confidence": round(task_confidence, 3),
        "reference_confidence": round(reference_confidence, 3),
        "arrangement_confidence": round(arrangement_confidence, 3),
        "persistence_cycles": persistence,
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "current_mix_variant_context_cache_no_live_authority": current_state_no_authority,
        "source_identity_downstream_trust": source_identity_trust.to_dict(),
        "transport_free": True,
        "no_named_artist_style_clone": True,
        "ayaic_level_planes": True,
    }

    return MixVariantProfileRecommendation(
        id=f"{MIX_VARIANT_METHOD_ID}:{analysis.variant_id}",
        status=status,
        action_type="mix_variant_profile_proposal",
        variant_id=str(analysis.variant_id or profile.id),
        genre=profile.genre,
        task=profile.task,
        target_profile=_profile_payload(profile),
        confidence=round(confidence, 3),
        severity=_severity(blockers_tuple, profile, feedback_risk),
        reason=reason,
        blockers=blockers_tuple,
        evidence=evidence,
    )


def build_mix_variant_profile_proposals(
    analyses: Iterable[MixVariantProfileAnalysisInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_mix_variant_profile_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
