"""Proposal-only FX send and return recommendations.

This module is transport-free. It turns source role, tempo, arrangement, return
headroom, feedback risk, and intelligibility evidence into operator-facing FX
send proposals, but it never creates a live console write request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
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


FX_METHOD_ID = "fx_send_proposal"
FX_VERSION = "fx_send_proposal_v1"
FX_READ_REQUIRED_SOURCE = "cached_or_physical_fx_read_required_before_promotion"
MIN_SEND_DB = -36.0
MAX_SEND_DB = -12.0
MAX_RETURN_LEVEL_DB = -3.0
MIN_RETURN_HEADROOM_DB = 3.0
HIGH_FEEDBACK_RISK = 0.55
HIGH_INTELLIGIBILITY_RISK = 0.65
HIGH_MONITOR_SAFETY_RISK = 0.35
MIN_SEND_DELTA_DB = 1.5

DRY_ANCHOR_FAMILIES = {"kick", "bass", "sub", "master", "reference"}
FX_TYPE_SCHEMA_FAMILIES = {
    "reverb": "reverb",
    "delay": "delay",
    "chorus": "modulation",
}
ROLE_TOKENS = (
    ("lead_vocal", ("lead_vocal", "lead vocal", "main_vocal", "main vocal", "vox lead")),
    ("backing_vocal", ("backing_vocal", "backing vocal", "back_vocal", "bgv", "bv")),
    ("kick", ("kick", "bd", "bass_drum")),
    ("snare", ("snare", "sn")),
    ("bass", ("bass", "di bass", "bass_di")),
    ("sub", ("sub", "subkick")),
    ("tom", ("tom", "floor_tom", "floor tom")),
    ("overhead", ("overhead", "oh", "cymbal")),
    ("guitar", ("gtr", "guitar")),
    ("keys", ("keys", "key", "piano", "synth", "organ")),
    ("percussion", ("perc", "clap", "shaker", "tamb")),
    ("playback", ("playback", "track", "stem", "loop")),
    ("master", ("master", "main", "reference")),
)


def _ordered_unique(values: Iterable[str]) -> Tuple[str, ...]:
    ordered = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _fx_required_safety_checks() -> Dict[str, str]:
    return {
        "source_identity": "required_before_live_promotion",
        "send_level_limit": "required_before_live_promotion",
        "fx_return_headroom": "required_before_live_promotion",
        "return_level_safety": "required_before_live_promotion",
        "feedback_risk": "required_before_live_promotion",
        "intelligibility_risk": "required_before_live_promotion",
        "monitor_safety": "required_before_live_promotion",
        "dry_wet_bounds": "required_before_live_promotion",
        "fx_parameter_schema": "required_before_live_promotion",
    }


def _fx_rollback_expectation() -> Dict[str, Any]:
    return {
        "required_before_live_promotion": True,
        "restore_previous_send_level_after_physical_readback": True,
        "restore_previous_fx_return_level_after_physical_readback": True,
        "restore_previous_fx_slot_parameters_after_physical_readback": True,
        "manual_operator_rollback_required_if_promoted": True,
    }


def _fx_future_wing_readback_contract() -> Dict[str, Any]:
    return {
        "required": True,
        "required_before_live_promotion": True,
        "target_kind": "send_or_fx_parameter",
        "old_value_source": FX_READ_REQUIRED_SOURCE,
        "readback_targets": [
            "send_level",
            "fx_return_level",
            "fx_slot_parameters",
            "monitor_send",
            "rollback_send_level",
            "rollback_fx_return_level",
        ],
        "blocked_live_actions": [
            "fx_send_level_write",
            "fx_parameter_write",
            "monitor_send_write",
        ],
        "requires_real_wing_proof": True,
        "requires_operator_approval": True,
        "requires_separate_non_fader_adr": True,
    }


@dataclass(frozen=True)
class FXPreset:
    fx_type: str
    bus_name: str
    target_send_db: float
    return_level_db: float
    hpf_hz: float
    lpf_hz: float
    predelay_ms: float = 0.0
    decay_s: float = 0.0
    delay_note: str = ""
    ducking_db: float = 0.0


ROLE_PRESETS: Dict[str, Tuple[FXPreset, ...]] = {
    "lead_vocal": (
        FXPreset("reverb", "Vocal Plate", -18.0, -5.0, 220.0, 7800.0, predelay_ms=42.0, decay_s=1.55, ducking_db=1.8),
        FXPreset("delay", "Tempo Delay", -24.0, -5.0, 260.0, 5200.0, delay_note="dotted_eighth", ducking_db=4.0),
    ),
    "backing_vocal": (
        FXPreset("reverb", "Vocal Plate", -15.0, -6.0, 220.0, 7600.0, predelay_ms=36.0, decay_s=1.45, ducking_db=1.0),
        FXPreset("chorus", "Mod Doubler", -23.0, -11.0, 220.0, 8500.0),
    ),
    "snare": (
        FXPreset("reverb", "Drum Room", -17.0, -5.5, 180.0, 5600.0, predelay_ms=10.0, decay_s=0.72),
    ),
    "tom": (
        FXPreset("reverb", "Drum Room", -19.0, -5.5, 160.0, 5400.0, predelay_ms=8.0, decay_s=0.78),
    ),
    "overhead": (
        FXPreset("reverb", "Drum Room", -30.0, -8.0, 240.0, 5400.0, predelay_ms=4.0, decay_s=0.55),
    ),
    "guitar": (
        FXPreset("reverb", "Shared Plate", -25.0, -7.0, 220.0, 7200.0, predelay_ms=24.0, decay_s=1.2),
        FXPreset("chorus", "Mod Doubler", -26.0, -11.0, 240.0, 8200.0),
    ),
    "keys": (
        FXPreset("reverb", "Shared Plate", -24.0, -7.0, 220.0, 7600.0, predelay_ms=28.0, decay_s=1.35),
    ),
    "percussion": (
        FXPreset("reverb", "Small Room", -24.0, -8.0, 220.0, 6000.0, predelay_ms=6.0, decay_s=0.65),
    ),
    "playback": (
        FXPreset("reverb", "Shared Plate", -31.0, -9.0, 260.0, 7000.0, predelay_ms=20.0, decay_s=1.0),
    ),
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


def classify_source_family(source_role: str) -> str:
    text = str(source_role or "").strip().lower().replace("-", "_")
    spaced = text.replace("_", " ")
    for family, tokens in ROLE_TOKENS:
        if any(token in text or token in spaced for token in tokens):
            return family
    return "default"


def tempo_delay_ms(tempo_bpm: Any, note: str = "dotted_eighth") -> float:
    bpm = _clip(_safe_float(tempo_bpm, 120.0), 40.0, 240.0)
    quarter = 60000.0 / bpm
    if note == "eighth":
        return round(quarter * 0.5, 1)
    if note == "quarter":
        return round(quarter, 1)
    return round(quarter * 0.75, 1)


@dataclass(frozen=True)
class FXSendAnalysisInput:
    channel_id: int
    source_role: str
    channel_label: str = ""
    current_send_db: Optional[float] = None
    tempo_bpm: float = 120.0
    arrangement_role: str = ""
    vocal_priority: bool = True
    contribution_share: float = 0.0
    return_headroom_db: float = 6.0
    feedback_risk: float = 0.0
    intelligibility_risk: float = 0.0
    monitor_send_exposure: bool = False
    monitor_safety_risk: float = 0.0
    monitor_safety_freshness: Optional[Dict[str, Any]] = None
    source_confidence: float = 0.75
    persistence_cycles: int = 1
    requested_fx_type: Optional[str] = None
    current_state_cache_freshness: Optional[Dict[str, Any]] = None
    current_fx_parameter_schema: Optional[Dict[str, Any]] = None
    current_fx_inserts_snapshot: Optional[Dict[str, Any]] = None
    source_identity_trust_state: str = "not_provided"
    source_identity_trust_confidence: float = 0.0
    source_identity_replay_outcome_count: int = 0
    source_identity_calibration_freshness: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class FXSendRecommendation:
    id: str
    status: str
    action_type: str
    channel_id: int
    channel_label: str
    source_role: str
    source_family: str
    fx_type: str
    bus_name: str
    current_send_db: Optional[float]
    target_send_db: float
    delta_send_db: Optional[float]
    return_level_db: float
    hpf_hz: float
    lpf_hz: float
    predelay_ms: float
    decay_s: float
    delay_ms: Optional[float]
    ducking_db: float
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
                "fx_send_live_apply_quarantined",
                "unsupported_value_type:fx_send",
                "not_in_supervised_fader_gain_executor",
                "real_wing_proof_required_before_live_apply",
                "non_fader_live_expansion_requires_separate_adr",
                "operator_approval_required_before_supervised_fx_apply",
                "hardware_readback_and_rollback_not_proven",
                *self.blockers,
            )
        )

    def fx_send_safety_summary(self) -> Dict[str, Any]:
        return_level_safety = dict(self.evidence.get("return_level_safety") or {})
        monitor_safety = dict(self.evidence.get("monitor_safety") or {})
        parameter_schema = dict(
            self.evidence.get("current_fx_parameter_schema") or {}
        )
        source_identity = dict(
            self.evidence.get("source_identity_downstream_trust") or {}
        )
        return {
            "status": "passed" if not self.blockers else "blocked",
            "channel_id": self.channel_id,
            "source_family": self.source_family,
            "fx_type": self.fx_type,
            "bus_name": self.bus_name,
            "current_send_db": self.current_send_db,
            "target_send_db": self.target_send_db,
            "delta_send_db": self.delta_send_db,
            "return_level_db": self.return_level_db,
            "return_headroom_db": self.evidence.get("return_headroom_db"),
            "send_level_limit": self.evidence.get("send_level_limit"),
            "send_level_limit_status": (
                dict(self.evidence.get("send_level_limit") or {}).get("status")
            ),
            "fx_return_headroom": self.evidence.get("fx_return_headroom"),
            "fx_return_headroom_status": (
                dict(self.evidence.get("fx_return_headroom") or {}).get("status")
            ),
            "return_level_safety_status": return_level_safety.get("status"),
            "feedback_risk": self.evidence.get("feedback_risk"),
            "intelligibility_risk": self.evidence.get("intelligibility_risk"),
            "monitor_safety_status": monitor_safety.get("status"),
            "monitor_send_exposure": monitor_safety.get("monitor_send_exposure"),
            "monitor_safety_risk": monitor_safety.get("monitor_safety_risk"),
            "dry_wet_bounds": self.evidence.get("dry_wet_bounds"),
            "dry_wet_bounds_status": (
                dict(self.evidence.get("dry_wet_bounds") or {}).get("status")
            ),
            "post_fader_send_required_before_promotion": self.evidence.get(
                "post_fader_send_required_before_promotion"
            ),
            "current_fx_send_return_cache_freshness_status": self.evidence.get(
                "current_state_cache_freshness_status"
            ),
            "current_fx_send_return_cache_no_live_authority": self.evidence.get(
                "current_fx_send_return_cache_no_live_authority"
            ),
            "current_fx_parameter_schema_status": self.evidence.get(
                "current_fx_parameter_schema_status"
            ),
            "current_fx_parameter_schema_family": parameter_schema.get(
                "schema_family"
            ),
            "expected_fx_schema_family": self.evidence.get(
                "expected_fx_schema_family"
            ),
            "fx_parameter_schema_no_live_authority": self.evidence.get(
                "fx_parameter_schema_no_live_authority"
            ),
            "source_identity_trust_status": source_identity.get("status"),
            "source_identity_usable_for_downstream": source_identity.get(
                "usable_for_downstream"
            ),
            "rollback_fx_readback": self.evidence.get("rollback_fx_readback"),
            "required_safety_checks": _fx_required_safety_checks(),
            "safety_blockers": list(self.operator_safety_blockers()),
            "operator_explanation": (
                "FX send evidence is offline-only. Review source identity, "
                "send bounds, FX return headroom, monitor exposure, feedback "
                "and intelligibility risk before any future WING promotion."
            ),
            "rollback_expectation": _fx_rollback_expectation(),
            "future_wing_readback_contract": _fx_future_wing_readback_contract(),
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

        fx_send_safety_summary = self.fx_send_safety_summary()
        blockers = list(self.operator_safety_blockers())
        proposed_action = {
            "action_type": self.action_type,
            "fx_type": self.fx_type,
            "bus_name": self.bus_name,
            "current_send_db": self.current_send_db,
            "target_send_db": self.target_send_db,
            "delta_send_db": self.delta_send_db,
            "return_level_db": self.return_level_db,
            "hpf_hz": self.hpf_hz,
            "lpf_hz": self.lpf_hz,
            "predelay_ms": self.predelay_ms,
            "decay_s": self.decay_s,
            "delay_ms": self.delay_ms,
            "ducking_db": self.ducking_db,
            "send_level_limit": fx_send_safety_summary["send_level_limit"],
            "fx_return_headroom": fx_send_safety_summary["fx_return_headroom"],
            "dry_wet_bounds": fx_send_safety_summary["dry_wet_bounds"],
            "rollback_fx_readback": fx_send_safety_summary["rollback_fx_readback"],
            "required_safety_checks": fx_send_safety_summary[
                "required_safety_checks"
            ],
            "rollback_expectation": fx_send_safety_summary[
                "rollback_expectation"
            ],
            "future_wing_readback_contract": fx_send_safety_summary[
                "future_wing_readback_contract"
            ],
            "fx_send_safety_summary": fx_send_safety_summary,
        }
        evidence = {
            **self.evidence,
            "fx_send_safety_summary": fx_send_safety_summary,
            "current_fx_parameter_schema_hints": self.evidence.get(
                "current_fx_parameter_schema"
            ),
            "required_safety_checks": fx_send_safety_summary[
                "required_safety_checks"
            ],
            "operator_explanation": fx_send_safety_summary[
                "operator_explanation"
            ],
            "rollback_expectation": fx_send_safety_summary[
                "rollback_expectation"
            ],
            "future_wing_readback_contract": fx_send_safety_summary[
                "future_wing_readback_contract"
            ],
            "old_value_source": FX_READ_REQUIRED_SOURCE,
            "rollback_value_source": FX_READ_REQUIRED_SOURCE,
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
            "method_id": FX_METHOD_ID,
            "source": FX_METHOD_ID,
            "title": f"{self.channel_label or f'Ch {self.channel_id}'}: FX send proposal",
            "reason": self.reason,
            "channel": self.channel_id,
            "target": self.channel_id,
            "kind": "fx_send",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "live_write": False,
            "queue_mutation": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
            "queue_import_allowed": False,
            "live_apply_allowed": False,
            "read_only_or_dry_run_only": True,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "mutates_mixer": False,
            "blockers": blockers,
            "safety_blockers": fx_send_safety_summary["safety_blockers"],
            "operator_explanation": fx_send_safety_summary[
                "operator_explanation"
            ],
            "required_safety_checks": fx_send_safety_summary[
                "required_safety_checks"
            ],
            "rollback_expectation": fx_send_safety_summary[
                "rollback_expectation"
            ],
            "future_wing_readback_contract": fx_send_safety_summary[
                "future_wing_readback_contract"
            ],
            "proposed_action": proposed_action,
            "fx_send_safety_summary": fx_send_safety_summary,
            "old_value": {
                "source": FX_READ_REQUIRED_SOURCE,
                "required_before_apply": True,
            },
            "rollback_value": {
                "source": FX_READ_REQUIRED_SOURCE,
                "required_before_apply": True,
            },
            "evidence": evidence,
        }
        proposal["operator_correction"] = build_operator_correction_payload(
            proposal=proposal,
            method_id=FX_METHOD_ID,
            target=self.channel_id,
            proposed_action=proposed_action,
            blockers=blockers,
        )
        return proposal


def _severity(
    delta: Optional[float],
    feedback_risk: float,
    intelligibility_risk: float,
    blockers: Iterable[str] = (),
) -> str:
    blocker_set = set(blockers)
    if (
        "source_identity_blocked_from_downstream_trust" in blocker_set
        or "source_identity_replay_context_not_fresh" in blocker_set
    ):
        return "high"
    if "source_identity_not_trusted_for_downstream" in blocker_set:
        return "medium"
    if blocker_set & {
        "fx_current_state_cache_live_authority_present",
        "fx_parameter_schema_live_authority_present",
        "fx_parameter_schema_family_mismatch",
        "fx_inserts_snapshot_cache_not_fresh",
        "fx_parameter_schema_not_mapped",
        "fx_parameter_schema_not_scoped_to_fx_type",
    }:
        return "medium"
    if blocker_set & {"monitor_safety_risk_high", "fx_monitor_safety_context_not_fresh"}:
        return "high"
    magnitude = abs(delta or 0.0)
    if feedback_risk >= 0.45 or intelligibility_risk >= 0.5 or magnitude >= 8.0:
        return "high"
    if magnitude >= 3.0:
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


def _schema_status(payload: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(payload, Mapping):
        return "not_provided"
    return str(payload.get("status") or "unknown").strip().lower() or "unknown"


def _fx_parameter_schema_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"status": "not_provided", "provided": False}
    allowed_keys = (
        "status",
        "schema_family",
        "schema_source",
        "mapped_parameter_count",
        "generic_parameter_count",
        "cached_parameter_count",
        "physical_model_schema_verified",
        "live_write_authority",
        "fx_slot",
        "fx_model",
        "snapshot_type",
        "snapshot_cache_freshness_status",
        "snapshot_read_coverage_status",
        "snapshot_queries_console",
        "snapshot_live_write",
    )
    normalized = {key: payload.get(key) for key in allowed_keys if key in payload}
    normalized["provided"] = bool(payload.get("provided", True))
    normalized.setdefault("status", _schema_status(payload))
    normalized.setdefault("schema_family", "unknown")
    return normalized


def _expected_schema_family(fx_type: str) -> str:
    return FX_TYPE_SCHEMA_FAMILIES.get(str(fx_type or "").strip().lower(), "unknown")


def fx_parameter_schema_from_snapshot(
    snapshot: Optional[Mapping[str, Any]],
    fx_type: str,
) -> Dict[str, Any]:
    """Extract scoped FX schema evidence from a Product Layer FX snapshot."""

    if not isinstance(snapshot, Mapping):
        return {
            "status": "not_provided",
            "provided": False,
            "schema_family": "unknown",
            "schema_source": "fx_inserts_snapshot_not_provided",
        }
    expected_family = _expected_schema_family(fx_type)
    safety = snapshot.get("safety") if isinstance(snapshot.get("safety"), Mapping) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), Mapping) else {}
    cache_freshness = (
        snapshot.get("cache_freshness")
        if isinstance(snapshot.get("cache_freshness"), Mapping)
        else {}
    )
    base = {
        "provided": True,
        "schema_family": "unknown",
        "schema_source": "fx_inserts_snapshot",
        "snapshot_type": snapshot.get("type"),
        "snapshot_cache_freshness_status": cache_freshness.get(
            "status",
            summary.get("cache_freshness_status", "unknown"),
        ),
        "snapshot_read_coverage_status": summary.get("read_coverage_status", "unknown"),
        "snapshot_queries_console": safety.get("queries_console"),
        "snapshot_live_write": safety.get("live_write"),
        "physical_model_schema_verified": False,
        "live_write_authority": False,
    }
    if snapshot.get("type") != "fx_inserts_snapshot":
        return {
            **base,
            "status": "unknown",
            "schema_source": "unexpected_snapshot_type",
        }
    for slot in snapshot.get("fx_slots") or ():
        if not isinstance(slot, Mapping):
            continue
        schema = slot.get("parameter_schema")
        if not isinstance(schema, Mapping):
            continue
        schema_family = str(schema.get("schema_family") or "unknown").strip().lower()
        if schema.get("status") == "mapped" and schema_family == expected_family:
            return {
                **base,
                "status": "mapped",
                "schema_family": schema_family,
                "schema_source": schema.get("schema_source", "fx_inserts_snapshot"),
                "mapped_parameter_count": schema.get("mapped_parameter_count", 0),
                "generic_parameter_count": schema.get("generic_parameter_count", 0),
                "cached_parameter_count": schema.get("cached_parameter_count", 0),
                "physical_model_schema_verified": bool(schema.get("physical_model_schema_verified")),
                "live_write_authority": bool(schema.get("live_write_authority")),
                "fx_slot": slot.get("slot"),
                "fx_model": slot.get("model"),
            }
    snapshot_schema = snapshot.get("parameter_schema") if isinstance(snapshot.get("parameter_schema"), Mapping) else {}
    return {
        **base,
        "status": snapshot_schema.get("status", "missing"),
        "schema_source": snapshot_schema.get("schema_source", "fx_inserts_snapshot"),
        "mapped_parameter_count": snapshot_schema.get("mapped_parameter_count", 0),
        "generic_parameter_count": snapshot_schema.get("generic_parameter_count", 0),
        "cached_parameter_count": snapshot_schema.get("cached_parameter_count", 0),
        "physical_model_schema_verified": bool(snapshot_schema.get("physical_model_schema_verified")),
        "live_write_authority": bool(snapshot_schema.get("live_write_authority")),
    }


def _select_preset(analysis: FXSendAnalysisInput, family: str) -> Optional[FXPreset]:
    presets = ROLE_PRESETS.get(family)
    if not presets:
        return None
    requested = str(analysis.requested_fx_type or "").strip().lower()
    if requested:
        return next((preset for preset in presets if preset.fx_type == requested), None)
    if family == "lead_vocal" and "space" in str(analysis.arrangement_role).lower():
        return presets[0]
    return presets[0]


def analyze_fx_send_need(
    analysis: FXSendAnalysisInput,
    *,
    min_confidence: float = 0.45,
) -> FXSendRecommendation:
    family = classify_source_family(analysis.source_role)
    preset = _select_preset(analysis, family)
    current = None if analysis.current_send_db is None else round(_safe_float(analysis.current_send_db), 1)
    feedback_risk = _clip01(analysis.feedback_risk)
    intelligibility_risk = _clip01(analysis.intelligibility_risk)
    monitor_safety_risk = _clip01(analysis.monitor_safety_risk)
    contribution = _clip01(analysis.contribution_share)
    source_confidence = _clip01(analysis.source_confidence)
    headroom = _safe_float(analysis.return_headroom_db)
    persistence = max(1, int(analysis.persistence_cycles or 1))
    current_state_freshness = _freshness_payload(analysis.current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
    monitor_safety_freshness = _freshness_payload(analysis.monitor_safety_freshness)
    monitor_safety_freshness_status = str(monitor_safety_freshness.get("status") or "unknown")
    source_identity_trust = evaluate_source_identity_downstream_trust(
        source_id=f"ch:{int(analysis.channel_id)}",
        source_type=str(analysis.source_role or "unknown"),
        trust_state=analysis.source_identity_trust_state,
        trust_confidence=analysis.source_identity_trust_confidence,
        replay_outcome_count=analysis.source_identity_replay_outcome_count,
        current_state_cache_freshness=analysis.source_identity_calibration_freshness,
    )

    blockers = []
    if family in DRY_ANCHOR_FAMILIES:
        blockers.append("dry_anchor_source")
    if preset is None:
        blockers.append("no_fx_preset_for_source")
        preset = FXPreset("reverb", "No FX", -36.0, -12.0, 250.0, 6500.0)
    if headroom < MIN_RETURN_HEADROOM_DB:
        blockers.append("insufficient_fx_return_headroom")
    if feedback_risk >= HIGH_FEEDBACK_RISK:
        blockers.append("feedback_risk_high")
    if preset.fx_type in {"delay", "reverb"} and intelligibility_risk >= HIGH_INTELLIGIBILITY_RISK:
        blockers.append("intelligibility_risk_high")
    if analysis.monitor_send_exposure and monitor_safety_freshness_status != "fresh":
        blockers.append("fx_monitor_safety_context_not_fresh")
    if monitor_safety_risk >= HIGH_MONITOR_SAFETY_RISK:
        blockers.append("monitor_safety_risk_high")
    if current_state_freshness_status != "fresh":
        blockers.append("fx_current_state_cache_not_fresh")
    if current_state_freshness_status == "fresh" and not current_state_no_authority:
        blockers.append("fx_current_state_cache_live_authority_present")
    expected_fx_schema_family = _expected_schema_family(preset.fx_type)
    schema_input = (
        analysis.current_fx_parameter_schema
        if isinstance(analysis.current_fx_parameter_schema, Mapping)
        else fx_parameter_schema_from_snapshot(analysis.current_fx_inserts_snapshot, preset.fx_type)
    )
    fx_parameter_schema = _fx_parameter_schema_payload(schema_input)
    fx_parameter_schema_status = str(fx_parameter_schema.get("status") or "unknown")
    provided_schema_family = str(fx_parameter_schema.get("schema_family") or "unknown").strip().lower()
    if fx_parameter_schema_status != "mapped":
        blockers.append("fx_parameter_schema_not_mapped")
    elif fx_parameter_schema_status == "mapped" and provided_schema_family == "unknown":
        blockers.append("fx_parameter_schema_not_scoped_to_fx_type")
    elif (
        fx_parameter_schema_status == "mapped"
        and expected_fx_schema_family != "unknown"
        and provided_schema_family != expected_fx_schema_family
    ):
        blockers.append("fx_parameter_schema_family_mismatch")
    if fx_parameter_schema.get("live_write_authority") is not False:
        blockers.append("fx_parameter_schema_live_authority_present")
    snapshot_schema_freshness_status = str(
        fx_parameter_schema.get("snapshot_cache_freshness_status") or "not_provided"
    ).strip().lower()
    if (
        fx_parameter_schema.get("provided") is True
        and fx_parameter_schema.get("snapshot_type") == "fx_inserts_snapshot"
        and snapshot_schema_freshness_status != "fresh"
    ):
        blockers.append("fx_inserts_snapshot_cache_not_fresh")

    target = _clip(preset.target_send_db, MIN_SEND_DB, MAX_SEND_DB)
    if intelligibility_risk >= 0.45:
        target = min(MIN_SEND_DB, target - 4.0)
    if feedback_risk >= 0.35:
        target = min(MIN_SEND_DB, target - 5.0)
    if monitor_safety_risk >= 0.2:
        target = min(MIN_SEND_DB, target - 3.0)
    target = round(_clip(target, MIN_SEND_DB, MAX_SEND_DB), 1)
    delta = None if current is None else round(target - current, 1)
    if delta is not None and abs(delta) < MIN_SEND_DELTA_DB:
        blockers.append("send_change_not_needed")
    blockers.extend(source_identity_trust.blockers)

    delay_ms = tempo_delay_ms(analysis.tempo_bpm, preset.delay_note) if preset.fx_type == "delay" else None
    confidence = _clip01(
        0.35 * source_confidence
        + 0.20 * min(persistence / 3.0, 1.0)
        + 0.15 * (1.0 - feedback_risk)
        + 0.15 * (1.0 - intelligibility_risk)
        + 0.10 * min(max(headroom, 0.0) / 8.0, 1.0)
        + 0.05 * contribution
        - 0.10 * monitor_safety_risk
    )
    return_level_safety = {
        "status": "passed" if headroom >= MIN_RETURN_HEADROOM_DB else "blocked",
        "target_return_level_db": round(_clip(preset.return_level_db, MIN_SEND_DB, MAX_RETURN_LEVEL_DB), 1),
        "return_headroom_db": round(headroom, 2),
        "max_return_level_db": MAX_RETURN_LEVEL_DB,
        "min_return_headroom_db": MIN_RETURN_HEADROOM_DB,
    }
    send_level_limit = {
        "status": "passed" if MIN_SEND_DB <= target <= MAX_SEND_DB else "blocked",
        "target_send_db": target,
        "current_send_db": current,
        "delta_send_db": delta,
        "min_send_db": MIN_SEND_DB,
        "max_send_db": MAX_SEND_DB,
        "post_fader_send_required": True,
    }
    fx_return_headroom = {
        "status": return_level_safety["status"],
        "return_headroom_db": round(headroom, 2),
        "target_return_level_db": return_level_safety["target_return_level_db"],
        "min_return_headroom_db": MIN_RETURN_HEADROOM_DB,
        "max_return_level_db": MAX_RETURN_LEVEL_DB,
    }
    monitor_safety = {
        "status": "blocked"
        if "fx_monitor_safety_context_not_fresh" in blockers
        or "monitor_safety_risk_high" in blockers
        else "passed",
        "monitor_send_exposure": bool(analysis.monitor_send_exposure),
        "monitor_safety_risk": round(monitor_safety_risk, 3),
        "monitor_safety_freshness_status": monitor_safety_freshness_status,
        "post_fader_send_required": True,
    }
    dry_wet_bounds = {
        "status": "passed"
        if send_level_limit["status"] == "passed"
        and return_level_safety["status"] == "passed"
        and monitor_safety["status"] == "passed"
        else "blocked",
        "post_fader_sends_only": True,
        "target_send_db": target,
        "target_return_level_db": return_level_safety["target_return_level_db"],
        "hpf_hz": round(preset.hpf_hz, 1),
        "lpf_hz": round(preset.lpf_hz, 1),
        "ducking_db": round(preset.ducking_db if analysis.vocal_priority else 0.0, 1),
        "fx_mix_percent": 100.0,
    }
    rollback_fx_readback = {
        "status": "required_before_live_promotion",
        "source": FX_READ_REQUIRED_SOURCE,
        "required": True,
        "send_level_readback_required": True,
        "return_level_readback_required": True,
        "fx_slot_parameter_readback_required": True,
        "manual_operator_rollback_required_if_promoted": True,
    }

    evidence = {
        "source": FX_VERSION,
        "source_role": analysis.source_role,
        "source_family": family,
        "tempo_bpm": round(_clip(_safe_float(analysis.tempo_bpm, 120.0), 40.0, 240.0), 1),
        "arrangement_role": analysis.arrangement_role,
        "vocal_priority": bool(analysis.vocal_priority),
        "contribution_share": round(contribution, 3),
        "return_headroom_db": round(headroom, 2),
        "feedback_risk": round(feedback_risk, 3),
        "intelligibility_risk": round(intelligibility_risk, 3),
        "monitor_safety_risk": round(monitor_safety_risk, 3),
        "monitor_send_exposure": bool(analysis.monitor_send_exposure),
        "monitor_safety_freshness_status": monitor_safety_freshness_status,
        "monitor_safety_freshness": monitor_safety_freshness,
        "monitor_safety": monitor_safety,
        "send_level_limit": send_level_limit,
        "fx_return_headroom": fx_return_headroom,
        "return_level_safety": return_level_safety,
        "dry_wet_bounds": dry_wet_bounds,
        "source_confidence": round(source_confidence, 3),
        "persistence_cycles": persistence,
        "current_send_db": current,
        "target_send_db": target,
        "delta_send_db": delta,
        "current_state_cache_freshness_status": current_state_freshness_status,
        "current_state_cache_freshness": current_state_freshness,
        "current_fx_send_return_cache_freshness": current_state_freshness,
        "current_fx_send_return_cache_no_live_authority": current_state_no_authority,
        "current_fx_parameter_schema_status": fx_parameter_schema_status,
        "current_fx_parameter_schema": fx_parameter_schema,
        "current_fx_parameter_schema_hints": fx_parameter_schema,
        "fx_parameter_schema_no_live_authority": (
            fx_parameter_schema.get("live_write_authority") is False
        ),
        "expected_fx_schema_family": expected_fx_schema_family,
        "source_identity_downstream_trust": source_identity_trust.to_dict(),
        "old_value_source": FX_READ_REQUIRED_SOURCE,
        "rollback_value_source": FX_READ_REQUIRED_SOURCE,
        "rollback_fx_readback": rollback_fx_readback,
        "readback_required_before_promotion": True,
        "rollback_required_before_promotion": True,
        "transport_free": True,
        "post_fader_send_required_before_promotion": True,
        "return_filtering_required": True,
    }

    status = "proposal_only" if not blockers and confidence >= min_confidence else "analysis_only"
    label = analysis.channel_label or f"Ch {analysis.channel_id}"
    if status == "proposal_only":
        reason = f"{label} can use {preset.bus_name} {preset.fx_type} send at {target:.1f} dB."
    else:
        reason = f"{label} does not meet safe FX send proposal criteria."

    return FXSendRecommendation(
        id=f"{FX_METHOD_ID}:ch{int(analysis.channel_id)}:{preset.fx_type}",
        status=status,
        action_type="fx_send_proposal",
        channel_id=int(analysis.channel_id),
        channel_label=label,
        source_role=str(analysis.source_role or "unknown"),
        source_family=family,
        fx_type=preset.fx_type,
        bus_name=preset.bus_name,
        current_send_db=current,
        target_send_db=target,
        delta_send_db=delta,
        return_level_db=round(_clip(preset.return_level_db, MIN_SEND_DB, MAX_RETURN_LEVEL_DB), 1),
        hpf_hz=round(preset.hpf_hz, 1),
        lpf_hz=round(preset.lpf_hz, 1),
        predelay_ms=round(preset.predelay_ms, 1),
        decay_s=round(preset.decay_s, 2),
        delay_ms=delay_ms,
        ducking_db=round(preset.ducking_db if analysis.vocal_priority else 0.0, 1),
        confidence=round(confidence, 3),
        severity=_severity(delta, feedback_risk, intelligibility_risk, blockers),
        reason=reason,
        blockers=tuple(sorted(set(blockers))),
        evidence=evidence,
    )


def build_fx_send_proposals(
    analyses: Iterable[FXSendAnalysisInput],
    *,
    current_fx_inserts_snapshot: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        enriched_analysis = analysis
        if (
            current_fx_inserts_snapshot is not None
            and analysis.current_fx_inserts_snapshot is None
            and analysis.current_fx_parameter_schema is None
        ):
            enriched_analysis = replace(
                analysis,
                current_fx_inserts_snapshot=current_fx_inserts_snapshot,
            )
        recommendation = analyze_fx_send_need(enriched_analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
