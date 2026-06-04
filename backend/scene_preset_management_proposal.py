"""Proposal-only scene and preset management recommendations.

This module is transport-free. It turns scene/preset intent, readback state,
rollback evidence, scope checks, and activity checks into operator proposals,
but it never creates a live console write request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Tuple


SCENE_PRESET_METHOD_ID = "scene_preset_management_proposal"
SCENE_PRESET_VERSION = "scene_preset_management_proposal_v1"
SUPPORTED_ARTIFACT_TYPES = {
    "scene",
    "snapshot",
    "user_preset",
    "channel_preset",
    "fx_preset",
}
SUPPORTED_ACTION_TYPES = {
    "scene_recall",
    "scene_save",
    "preset_recall",
    "preset_save",
}
SUPPORTED_TARGET_SCOPES = {
    "channel",
    "channel_strip",
    "fx_slot",
    "bus",
    "matrix",
    "scoped_scene",
    "preset_slot",
}
SAFE_SCENE_PRESET_WINDOWS = {"offline", "soundcheck", "rehearsal", "line_check"}
MIN_CONFIDENCE = 0.5
MIN_TARGET_CONFIDENCE = 0.65
MIN_SCOPE_CONFIDENCE = 0.65
MIN_ROLLBACK_CONFIDENCE = 0.65
ACTIVE_SIGNAL_DBFS = -60.0
MAX_DESTRUCTIVE_SCOPE_SCORE = 0.35
MAX_AFFECTED_CHANNELS = 2
MAX_AFFECTED_BUSES = 1
MAX_AFFECTED_FX_SLOTS = 1


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


def normalize_artifact_type(artifact_type: Any) -> str:
    text = _normalize_token(artifact_type)
    aliases = {
        "snap": "snapshot",
        "snap_shot": "snapshot",
        "user": "user_preset",
        "preset": "user_preset",
        "channel": "channel_preset",
        "strip": "channel_preset",
        "fx": "fx_preset",
        "effect": "fx_preset",
    }
    return aliases.get(text, text)


def normalize_scene_preset_action(action_type: Any, artifact_type: Any = "") -> str:
    text = _normalize_token(action_type)
    artifact = normalize_artifact_type(artifact_type)
    aliases = {
        "recall": "preset_recall" if "preset" in artifact else "scene_recall",
        "restore": "preset_recall" if "preset" in artifact else "scene_recall",
        "apply": "preset_recall" if "preset" in artifact else "scene_recall",
        "save": "preset_save" if "preset" in artifact else "scene_save",
        "store": "preset_save" if "preset" in artifact else "scene_save",
        "capture": "preset_save" if "preset" in artifact else "scene_save",
        "scene": "scene_recall",
        "snapshot": "scene_recall",
        "user_preset": "preset_recall",
        "channel_preset": "preset_recall",
        "fx_preset": "preset_recall",
    }
    return aliases.get(text, text)


def normalize_target_scope(scope: Any) -> str:
    text = _normalize_token(scope)
    aliases = {
        "ch": "channel",
        "channel_preset": "channel_strip",
        "strip": "channel_strip",
        "fx": "fx_slot",
        "effect": "fx_slot",
        "scene": "scoped_scene",
        "snapshot": "scoped_scene",
        "preset": "preset_slot",
    }
    return aliases.get(text, text)


@dataclass(frozen=True)
class ScenePresetManagementAnalysisInput:
    artifact_id: str
    artifact_name: str = ""
    artifact_type: str = "scene"
    action_type: str = "recall"
    target_scope: str = "channel"
    target_label: str = ""
    active_scene_id: str = ""
    active_scene_name: str = ""
    artifact_exists: bool = False
    target_name_match: bool = False
    current_state_readback_available: bool = False
    target_preview_available: bool = False
    scoped_diff_available: bool = False
    rollback_snapshot_available: bool = False
    rollback_snapshot_id: str = ""
    operator_intent: str = ""
    change_window: str = "soundcheck"
    active_performance: bool = False
    affected_channel_count: int = 1
    affected_bus_count: int = 0
    affected_fx_slot_count: int = 0
    estimated_changed_parameter_count: int = 1
    destructive_scope_score: float = 0.0
    current_mix_signal_dbfs: float = -100.0
    affected_sources_active: bool = False
    target_slot_occupied: bool = False
    allow_overwrite: bool = False
    target_confidence: float = 0.75
    scope_confidence: float = 0.75
    rollback_confidence: float = 0.75
    persistence_cycles: int = 1


@dataclass(frozen=True)
class ScenePresetManagementRecommendation:
    id: str
    status: str
    action_type: str
    artifact_id: str
    artifact_name: str
    artifact_type: str
    target_scope: str
    target_label: str
    active_scene_id: str
    active_scene_name: str
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
            "method_id": SCENE_PRESET_METHOD_ID,
            "source": SCENE_PRESET_METHOD_ID,
            "title": f"{self.target_label or self.artifact_name or self.artifact_id}: scene/preset proposal",
            "reason": self.reason,
            "target": self.artifact_id,
            "kind": "scene_preset",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:scene_preset",
                "not_in_supervised_fader_gain_executor",
                "broad_console_state_write_not_proven",
                *self.blockers,
            ],
            "proposed_action": {
                "action_type": self.action_type,
                "artifact_id": self.artifact_id,
                "artifact_name": self.artifact_name,
                "artifact_type": self.artifact_type,
                "target_scope": self.target_scope,
                "active_scene_id": self.active_scene_id,
                "active_scene_name": self.active_scene_name,
            },
            "evidence": {
                **self.evidence,
                "live_write": False,
                "queue_mutation": False,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": True,
            },
        }


def _severity(blockers: Iterable[str], destructive_scope_score: float, affected_total: int) -> str:
    blocker_set = set(blockers)
    if (
        "active_performance_scene_preset_blocked" in blocker_set
        or "scene_preset_scope_too_broad" in blocker_set
        or destructive_scope_score >= 0.65
        or affected_total > 4
    ):
        return "high"
    if destructive_scope_score >= 0.2 or affected_total > 1:
        return "medium"
    return "low"


def _action_needs_existing_target(action_type: str) -> bool:
    return action_type in {"scene_recall", "preset_recall"}


def _action_is_save(action_type: str) -> bool:
    return action_type in {"scene_save", "preset_save"}


def analyze_scene_preset_management_need(
    analysis: ScenePresetManagementAnalysisInput,
    *,
    min_confidence: float = MIN_CONFIDENCE,
) -> ScenePresetManagementRecommendation:
    artifact_type = normalize_artifact_type(analysis.artifact_type)
    action_type = normalize_scene_preset_action(analysis.action_type, artifact_type)
    target_scope = normalize_target_scope(analysis.target_scope)
    artifact_id = str(analysis.artifact_id or "").strip()
    artifact_name = str(analysis.artifact_name or "").strip()
    target_label = analysis.target_label or artifact_name or artifact_id
    current_scene_id = str(analysis.active_scene_id or "").strip()
    current_scene_name = str(analysis.active_scene_name or "").strip()
    window = _normalize_token(analysis.change_window)
    operator_intent = str(analysis.operator_intent or "").strip().lower()
    changed_parameters = max(0, int(analysis.estimated_changed_parameter_count or 0))
    affected_channels = max(0, int(analysis.affected_channel_count or 0))
    affected_buses = max(0, int(analysis.affected_bus_count or 0))
    affected_fx = max(0, int(analysis.affected_fx_slot_count or 0))
    affected_total = affected_channels + affected_buses + affected_fx
    destructive_scope = _clip01(analysis.destructive_scope_score)
    current_signal = _safe_float(analysis.current_mix_signal_dbfs, -100.0)
    target_confidence = _clip01(analysis.target_confidence)
    scope_confidence = _clip01(analysis.scope_confidence)
    rollback_confidence = _clip01(analysis.rollback_confidence)
    persistence = max(1, int(analysis.persistence_cycles or 1))

    blockers = []
    if not artifact_id and not artifact_name:
        blockers.append("scene_preset_target_missing")
    if artifact_type not in SUPPORTED_ARTIFACT_TYPES:
        blockers.append("unsupported_scene_preset_artifact_type")
    if action_type not in SUPPORTED_ACTION_TYPES:
        blockers.append("unsupported_scene_preset_action")
    if target_scope not in SUPPORTED_TARGET_SCOPES:
        blockers.append("unsupported_scene_preset_scope")
    if not analysis.current_state_readback_available:
        blockers.append("current_console_state_readback_missing")
    if not analysis.target_preview_available:
        blockers.append("scene_preset_preview_missing")
    if not analysis.scoped_diff_available:
        blockers.append("scene_preset_scoped_diff_missing")
    if not analysis.rollback_snapshot_available or not analysis.rollback_snapshot_id:
        blockers.append("rollback_snapshot_missing")
    if not operator_intent:
        blockers.append("operator_intent_required")
    if analysis.active_performance or window not in SAFE_SCENE_PRESET_WINDOWS:
        blockers.append("active_performance_scene_preset_blocked")
    if (
        destructive_scope > MAX_DESTRUCTIVE_SCOPE_SCORE
        or affected_channels > MAX_AFFECTED_CHANNELS
        or affected_buses > MAX_AFFECTED_BUSES
        or affected_fx > MAX_AFFECTED_FX_SLOTS
    ):
        blockers.append("scene_preset_scope_too_broad")
    if current_signal >= ACTIVE_SIGNAL_DBFS or analysis.affected_sources_active:
        blockers.append("active_signal_requires_manual_scene_preset_change")
    if _action_needs_existing_target(action_type):
        if not analysis.artifact_exists:
            blockers.append("scene_preset_artifact_missing")
        if not analysis.target_name_match:
            blockers.append("scene_preset_name_mismatch")
    if _action_is_save(action_type) and analysis.target_slot_occupied and not analysis.allow_overwrite:
        blockers.append("overwrite_requires_explicit_operator_intent")
    if (
        action_type == "scene_recall"
        and artifact_id
        and current_scene_id
        and artifact_id == current_scene_id
        and changed_parameters == 0
    ):
        blockers.append("scene_change_not_needed")
    if changed_parameters == 0:
        blockers.append("scene_preset_change_not_needed")
    if target_confidence < MIN_TARGET_CONFIDENCE:
        blockers.append("target_confidence_low")
    if scope_confidence < MIN_SCOPE_CONFIDENCE:
        blockers.append("scope_confidence_low")
    if rollback_confidence < MIN_ROLLBACK_CONFIDENCE:
        blockers.append("rollback_confidence_low")

    confidence = _clip01(
        0.22 * target_confidence
        + 0.22 * scope_confidence
        + 0.18 * rollback_confidence
        + 0.12 * min(persistence / 3.0, 1.0)
        + 0.10 * (1.0 if analysis.current_state_readback_available else 0.0)
        + 0.08 * (1.0 if analysis.target_preview_available else 0.0)
        + 0.05 * (1.0 - destructive_scope)
        + 0.03 * (1.0 if operator_intent else 0.0)
    )
    blockers_tuple = tuple(sorted(set(blockers)))
    status = "proposal_only" if not blockers_tuple and confidence >= min_confidence else "analysis_only"
    if status == "proposal_only":
        reason = f"{target_label} can be proposed as a scoped {action_type.replace('_', ' ')}."
    else:
        reason = f"{target_label or 'Scene/preset target'} does not meet safe scene/preset proposal criteria."

    evidence = {
        "source": SCENE_PRESET_VERSION,
        "artifact_type": artifact_type,
        "action_type": action_type,
        "target_scope": target_scope,
        "current_state_readback_available": bool(analysis.current_state_readback_available),
        "target_preview_available": bool(analysis.target_preview_available),
        "scoped_diff_available": bool(analysis.scoped_diff_available),
        "rollback_snapshot_available": bool(analysis.rollback_snapshot_available),
        "rollback_snapshot_id": analysis.rollback_snapshot_id,
        "operator_intent": operator_intent,
        "change_window": window,
        "active_performance": bool(analysis.active_performance),
        "affected_channel_count": affected_channels,
        "affected_bus_count": affected_buses,
        "affected_fx_slot_count": affected_fx,
        "estimated_changed_parameter_count": changed_parameters,
        "destructive_scope_score": round(destructive_scope, 3),
        "current_mix_signal_dbfs": round(current_signal, 2),
        "affected_sources_active": bool(analysis.affected_sources_active),
        "target_slot_occupied": bool(analysis.target_slot_occupied),
        "allow_overwrite": bool(analysis.allow_overwrite),
        "target_confidence": round(target_confidence, 3),
        "scope_confidence": round(scope_confidence, 3),
        "rollback_confidence": round(rollback_confidence, 3),
        "persistence_cycles": persistence,
        "transport_free": True,
        "destructive_console_state": True,
    }

    return ScenePresetManagementRecommendation(
        id=f"{SCENE_PRESET_METHOD_ID}:{artifact_id or artifact_name}",
        status=status,
        action_type=action_type,
        artifact_id=artifact_id,
        artifact_name=artifact_name,
        artifact_type=artifact_type,
        target_scope=target_scope,
        target_label=target_label,
        active_scene_id=current_scene_id,
        active_scene_name=current_scene_name,
        confidence=round(confidence, 3),
        severity=_severity(blockers_tuple, destructive_scope, affected_total),
        reason=reason,
        blockers=blockers_tuple,
        evidence=evidence,
    )


def build_scene_preset_management_proposals(
    analyses: Iterable[ScenePresetManagementAnalysisInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_scene_preset_management_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
