"""Proposal-only scene and preset management recommendations.

This module is transport-free. It turns scene/preset intent, readback state,
rollback evidence, scope checks, and activity checks into operator proposals,
but it never creates a live console write request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


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
    current_state_cache_freshness: Optional[Dict[str, Any]] = None
    current_snapshots_scenes_snapshot: Optional[Dict[str, Any]] = None
    current_user_presets_snapshot: Optional[Dict[str, Any]] = None
    scoped_diff_evidence: Optional[Dict[str, Any]] = None
    rollback_evidence: Optional[Dict[str, Any]] = None


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
        "snapshot_type",
        "snapshot_source",
        "snapshot_read_coverage_status",
        "snapshot_live_write",
        "snapshot_queries_console",
        "snapshot_live_apply_allowed",
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
        and payload.get("snapshot_live_write", False) is False
        and payload.get("snapshot_queries_console", False) is False
        and payload.get("snapshot_live_apply_allowed", False) is False
    )


def scene_preset_cache_freshness_from_snapshots(
    artifact_type: Any,
    *,
    current_snapshots_scenes_snapshot: Optional[Mapping[str, Any]] = None,
    current_user_presets_snapshot: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return freshness evidence from read-only Product Layer snapshots."""

    normalized_artifact_type = normalize_artifact_type(artifact_type)
    if normalized_artifact_type in {"scene", "snapshot"}:
        snapshot = current_snapshots_scenes_snapshot
    elif normalized_artifact_type in {"user_preset", "channel_preset", "fx_preset"}:
        snapshot = current_user_presets_snapshot
    else:
        snapshot = current_snapshots_scenes_snapshot or current_user_presets_snapshot
    if not isinstance(snapshot, Mapping):
        return None

    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), Mapping) else {}
    safety = snapshot.get("safety") if isinstance(snapshot.get("safety"), Mapping) else {}
    cache_freshness = snapshot.get("cache_freshness")
    if isinstance(cache_freshness, Mapping):
        payload = _freshness_payload(cache_freshness)
    else:
        payload = {
            "status": str(summary.get("cache_freshness_status") or "unknown").strip().lower(),
            "provided": True,
        }
    payload.update(
        {
            "snapshot_type": snapshot.get("type", "unknown"),
            "snapshot_source": snapshot.get("source", "unknown"),
            "snapshot_read_coverage_status": summary.get("read_coverage_status", "unknown"),
            "snapshot_live_write": bool(safety.get("live_write")),
            "snapshot_queries_console": bool(safety.get("queries_console")),
            "snapshot_live_apply_allowed": bool(summary.get("live_apply_allowed")),
        }
    )
    return payload


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return default
    return numeric if numeric >= 0 else default


def scene_preset_scoped_diff_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize read-only scoped scene/preset diff evidence."""

    if not isinstance(payload, Mapping):
        return {
            "provided": False,
            "available": False,
            "read_only": True,
            "live_write": False,
            "queries_console": False,
            "mutates_console_state": False,
        }
    target_preview_available = bool(
        payload.get("target_preview_available")
        if "target_preview_available" in payload
        else payload.get("preview_available")
    )
    rollback_snapshot_id = str(
        payload.get("rollback_snapshot_id")
        or payload.get("rollback_id")
        or ""
    ).strip()
    rollback_available = bool(
        payload.get("rollback_snapshot_available")
        if "rollback_snapshot_available" in payload
        else payload.get("rollback_available")
    ) or bool(rollback_snapshot_id)
    changed_parameter_count = _safe_int(
        payload.get("estimated_changed_parameter_count")
        if "estimated_changed_parameter_count" in payload
        else payload.get("changed_parameter_count"),
        0,
    )
    affected_channel_count = _safe_int(payload.get("affected_channel_count"), 0)
    affected_bus_count = _safe_int(payload.get("affected_bus_count"), 0)
    affected_fx_slot_count = _safe_int(payload.get("affected_fx_slot_count"), 0)
    destructive_scope_score = _clip01(payload.get("destructive_scope_score", 0.0))
    scope_confidence = _clip01(payload.get("scope_confidence", 0.0))
    rollback_confidence = _clip01(payload.get("rollback_confidence", 0.0))
    available = bool(
        payload.get("available")
        if "available" in payload
        else payload.get("scoped_diff_available")
    )
    if not available and changed_parameter_count > 0:
        available = True
    if not available and any((affected_channel_count, affected_bus_count, affected_fx_slot_count)):
        available = True

    return {
        "provided": True,
        "available": available,
        "target_preview_available": target_preview_available,
        "rollback_snapshot_available": rollback_available,
        "rollback_snapshot_id": rollback_snapshot_id,
        "estimated_changed_parameter_count": changed_parameter_count,
        "affected_channel_count": affected_channel_count,
        "affected_bus_count": affected_bus_count,
        "affected_fx_slot_count": affected_fx_slot_count,
        "destructive_scope_score": round(destructive_scope_score, 3),
        "scope_confidence": round(scope_confidence, 3),
        "rollback_confidence": round(rollback_confidence, 3),
        "diff_source": str(payload.get("source") or payload.get("diff_source") or "unknown"),
        "diff_id": str(payload.get("id") or payload.get("diff_id") or "").strip(),
        "read_only": bool(payload.get("read_only", True)),
        "live_write": bool(payload.get("live_write", False)),
        "queries_console": bool(payload.get("queries_console", False)),
        "mutates_console_state": bool(payload.get("mutates_console_state", False)),
        "destructive_change_marker": str(payload.get("destructive_change_marker") or "").strip(),
    }


def scene_preset_rollback_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize read-only rollback evidence for scene/preset changes."""

    if not isinstance(payload, Mapping):
        return {
            "provided": False,
            "available": False,
            "read_only": True,
            "live_write": False,
            "queries_console": False,
            "mutates_console_state": False,
            "target_scope_matches": True,
        }
    rollback_snapshot_id = str(
        payload.get("rollback_snapshot_id")
        or payload.get("snapshot_id")
        or payload.get("rollback_id")
        or ""
    ).strip()
    available = bool(
        payload.get("available")
        if "available" in payload
        else payload.get("rollback_snapshot_available")
    ) or bool(rollback_snapshot_id)
    target_scope_matches = bool(
        payload.get("target_scope_matches")
        if "target_scope_matches" in payload
        else payload.get("covers_target_scope", True)
    )
    return {
        "provided": True,
        "available": available,
        "rollback_snapshot_id": rollback_snapshot_id,
        "rollback_confidence": round(_clip01(payload.get("rollback_confidence", payload.get("confidence", 0.0))), 3),
        "rollback_source": str(payload.get("source") or payload.get("rollback_source") or "unknown"),
        "rollback_evidence_id": str(payload.get("id") or payload.get("evidence_id") or "").strip(),
        "target_scope_matches": target_scope_matches,
        "target_scope": str(payload.get("target_scope") or "").strip(),
        "read_only": bool(payload.get("read_only", True)),
        "live_write": bool(payload.get("live_write", False)),
        "queries_console": bool(payload.get("queries_console", False)),
        "mutates_console_state": bool(payload.get("mutates_console_state", False)),
        "destructive_change_marker": str(payload.get("destructive_change_marker") or "").strip(),
    }


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
    scoped_diff = scene_preset_scoped_diff_payload(analysis.scoped_diff_evidence)
    rollback = scene_preset_rollback_payload(analysis.rollback_evidence)
    scope_confidence = max(_clip01(analysis.scope_confidence), _clip01(scoped_diff.get("scope_confidence")))
    rollback_confidence = max(
        _clip01(analysis.rollback_confidence),
        _clip01(scoped_diff.get("rollback_confidence")),
        _clip01(rollback.get("rollback_confidence")),
    )
    persistence = max(1, int(analysis.persistence_cycles or 1))
    current_state_cache_freshness = (
        analysis.current_state_cache_freshness
        if isinstance(analysis.current_state_cache_freshness, Mapping)
        else scene_preset_cache_freshness_from_snapshots(
            artifact_type,
            current_snapshots_scenes_snapshot=analysis.current_snapshots_scenes_snapshot,
            current_user_presets_snapshot=analysis.current_user_presets_snapshot,
        )
    )
    current_state_freshness = _freshness_payload(current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    current_state_no_authority = _freshness_is_no_authority(current_state_freshness)

    target_preview_available = bool(
        analysis.target_preview_available or scoped_diff.get("target_preview_available")
    )
    scoped_diff_available = bool(analysis.scoped_diff_available or scoped_diff.get("available"))
    rollback_snapshot_id = str(
        analysis.rollback_snapshot_id
        or rollback.get("rollback_snapshot_id")
        or scoped_diff.get("rollback_snapshot_id")
        or ""
    ).strip()
    rollback_snapshot_available = bool(
        analysis.rollback_snapshot_available
        or rollback.get("available")
        or scoped_diff.get("rollback_snapshot_available")
        or rollback_snapshot_id
    )
    changed_parameters = max(changed_parameters, _safe_int(scoped_diff.get("estimated_changed_parameter_count"), 0))
    affected_channels = max(affected_channels, _safe_int(scoped_diff.get("affected_channel_count"), 0))
    affected_buses = max(affected_buses, _safe_int(scoped_diff.get("affected_bus_count"), 0))
    affected_fx = max(affected_fx, _safe_int(scoped_diff.get("affected_fx_slot_count"), 0))
    affected_total = affected_channels + affected_buses + affected_fx
    destructive_scope = max(destructive_scope, _clip01(scoped_diff.get("destructive_scope_score")))

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
    if not target_preview_available:
        blockers.append("scene_preset_preview_missing")
    if not scoped_diff_available:
        blockers.append("scene_preset_scoped_diff_missing")
    if not rollback_snapshot_available or not rollback_snapshot_id:
        blockers.append("rollback_snapshot_missing")
    if scoped_diff.get("provided") is True:
        if scoped_diff.get("read_only") is not True:
            blockers.append("scene_preset_scoped_diff_not_read_only")
        if scoped_diff.get("live_write") is True:
            blockers.append("scene_preset_scoped_diff_live_write_claim")
        if scoped_diff.get("queries_console") is True:
            blockers.append("scene_preset_scoped_diff_queries_console")
        if scoped_diff.get("mutates_console_state") is True:
            blockers.append("scene_preset_scoped_diff_mutates_console_state")
    if rollback.get("provided") is True:
        if rollback.get("read_only") is not True:
            blockers.append("scene_preset_rollback_not_read_only")
        if rollback.get("live_write") is True:
            blockers.append("scene_preset_rollback_live_write_claim")
        if rollback.get("queries_console") is True:
            blockers.append("scene_preset_rollback_queries_console")
        if rollback.get("mutates_console_state") is True:
            blockers.append("scene_preset_rollback_mutates_console_state")
        if rollback.get("target_scope_matches") is False:
            blockers.append("scene_preset_rollback_target_scope_mismatch")
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
    if current_state_freshness_status != "fresh":
        blockers.append("scene_preset_current_state_cache_not_fresh")
    if current_state_freshness_status == "fresh" and not current_state_no_authority:
        blockers.append("scene_preset_current_state_cache_live_authority_present")

    confidence = _clip01(
        0.22 * target_confidence
        + 0.22 * scope_confidence
        + 0.18 * rollback_confidence
        + 0.12 * min(persistence / 3.0, 1.0)
        + 0.10 * (1.0 if analysis.current_state_readback_available else 0.0)
        + 0.08 * (1.0 if target_preview_available else 0.0)
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
        "target_preview_available": target_preview_available,
        "scoped_diff_available": scoped_diff_available,
        "rollback_snapshot_available": rollback_snapshot_available,
        "rollback_snapshot_id": rollback_snapshot_id,
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
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "current_scene_preset_cache_no_live_authority": current_state_no_authority,
        "scoped_diff_evidence": scoped_diff,
        "rollback_evidence": rollback,
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
    *,
    current_snapshots_scenes_snapshot: Optional[Mapping[str, Any]] = None,
    current_user_presets_snapshot: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        enriched_analysis = analysis
        if (
            analysis.current_state_cache_freshness is None
            and analysis.current_snapshots_scenes_snapshot is None
            and analysis.current_user_presets_snapshot is None
            and (
                current_snapshots_scenes_snapshot is not None
                or current_user_presets_snapshot is not None
            )
        ):
            enriched_analysis = replace(
                analysis,
                current_snapshots_scenes_snapshot=dict(current_snapshots_scenes_snapshot)
                if isinstance(current_snapshots_scenes_snapshot, Mapping)
                else None,
                current_user_presets_snapshot=dict(current_user_presets_snapshot)
                if isinstance(current_user_presets_snapshot, Mapping)
                else None,
            )
        recommendation = analyze_scene_preset_management_need(enriched_analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
