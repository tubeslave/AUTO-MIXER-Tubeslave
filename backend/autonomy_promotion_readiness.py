"""Read-only promotion readiness for autonomous Automixer capabilities.

This module combines the project goal matrix, safety proof matrix, read surface
inventory, and write surface inventory into a fail-closed view of what can be
considered for supervised promotion. It does not import mixer transports or
create live write requests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple

try:
    from project_goal_capability_matrix import get_project_goal_capability
    from read_surface_inventory import get_read_surface
    from safety_coverage_matrix import get_safety_coverage_row
    from wing_readonly_preflight_evidence import build_wing_readonly_preflight_evidence
    from write_surface_inventory import get_write_surface
except ImportError:  # pragma: no cover - package import compatibility
    from backend.project_goal_capability_matrix import get_project_goal_capability
    from backend.read_surface_inventory import get_read_surface
    from backend.safety_coverage_matrix import get_safety_coverage_row
    from backend.wing_readonly_preflight_evidence import build_wing_readonly_preflight_evidence
    from backend.write_surface_inventory import get_write_surface


READINESS_STATUSES = {
    "supervised_live_allowed",
    "proof_blocked",
    "read_coverage_blocked",
    "manual_only_blocked",
}

REQUIRED_FULL_PROOF_FIELDS = (
    "unit_tests",
    "offline_harness",
    "fake_mixer_proof",
    "real_wing_proof",
    "pre_read",
    "apply_readback",
    "rollback_readback",
    "emergency_stop_proof",
)

GAP_BUCKETS = (
    "capability",
    "safety",
    "proof",
    "read_surface",
    "write_surface",
    "autonomy_mode",
    "evidence_requirement",
    "other",
)

WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND = (
    "PYTHONPATH=backend",
    "python3",
    "tools/no_wing_release_check.py",
    "--quick",
)
WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FIELDS = (
    "listener_processes",
    "operator_next_step",
)


@dataclass(frozen=True)
class PromotionTarget:
    id: str
    label: str
    capability_ids: Tuple[str, ...]
    action_family: str
    safety_row_id: str = ""
    write_surface_ids: Tuple[str, ...] = ()
    read_surface_ids: Tuple[str, ...] = ()
    evidence_requirement_ids: Tuple[str, ...] = ()
    target_mode: str = "supervised_live_candidate"


@dataclass(frozen=True)
class PromotionReadinessRow:
    id: str
    label: str
    action_family: str
    target_mode: str
    status: str
    supervised_live_allowed: bool
    autonomous_ready: bool
    capability_ids: Tuple[str, ...]
    capability_statuses: Tuple[str, ...]
    safety_row_id: str
    safety_status: str
    write_surface_ids: Tuple[str, ...]
    write_surface_statuses: Tuple[str, ...]
    read_surface_ids: Tuple[str, ...]
    read_surface_statuses: Tuple[str, ...]
    evidence_requirement_ids: Tuple[str, ...]
    blockers: Tuple[str, ...]
    next_required_evidence: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in READINESS_STATUSES:
            raise ValueError(f"Unknown readiness status: {self.status}")
        if self.autonomous_ready:
            raise ValueError(f"{self.id} cannot be autonomous-ready while Full Autonomous mode is blocked")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "capability_ids",
            "capability_statuses",
            "write_surface_ids",
            "write_surface_statuses",
            "read_surface_ids",
            "read_surface_statuses",
            "evidence_requirement_ids",
            "blockers",
            "next_required_evidence",
        ):
            payload[key] = list(payload[key])
        return payload


PROMOTION_TARGETS: Tuple[PromotionTarget, ...] = (
    PromotionTarget(
        id="fader_supervised_promotion",
        label="Channel fader supervised promotion",
        capability_ids=("console_fader_control", "mix_balance", "mode_supervised_automix"),
        action_family="fader",
        safety_row_id="fader_supervised",
        write_surface_ids=("product_layer_fader_gain_apply", "gain_fader_proposal_sources"),
        read_surface_ids=("channel_identity_and_basic_controls",),
    ),
    PromotionTarget(
        id="gain_trim_supervised_promotion",
        label="Input gain/trim supervised promotion",
        capability_ids=("console_preamp_trim_control", "gain_staging", "mode_supervised_automix"),
        action_family="gain",
        safety_row_id="gain_supervised",
        write_surface_ids=("product_layer_fader_gain_apply", "gain_fader_proposal_sources"),
        read_surface_ids=("channel_identity_and_basic_controls",),
    ),
    PromotionTarget(
        id="eq_supervised_promotion",
        label="EQ supervised promotion",
        capability_ids=("console_eq_control", "equalization", "frequency_separation", "cross_channel_unmasking"),
        action_family="eq",
        safety_row_id="eq_autoeq",
        write_surface_ids=("eq_recommendation_sources", "eq_direct_apply_and_reset"),
        read_surface_ids=("channel_eq_dynamics_gate",),
    ),
    PromotionTarget(
        id="dynamics_supervised_promotion",
        label="Compressor/dynamics supervised promotion",
        capability_ids=("console_compressor_control", "compression", "deessing"),
        action_family="compressor",
        safety_row_id="dynamics_compressor",
        write_surface_ids=("dynamics_recommendation_sources", "compressor_direct_apply"),
        read_surface_ids=("channel_eq_dynamics_gate",),
    ),
    PromotionTarget(
        id="gate_supervised_promotion",
        label="Gate/expander supervised promotion",
        capability_ids=("console_gate_control", "gating"),
        action_family="gate",
        safety_row_id="gate_expander",
        write_surface_ids=("context_gate_bleed_recommendations", "gate_hpf_send_direct_apply"),
        read_surface_ids=("channel_eq_dynamics_gate",),
    ),
    PromotionTarget(
        id="pan_phase_supervised_promotion",
        label="Pan/phase/delay supervised promotion",
        capability_ids=("console_pan_control", "panning", "phase_correction"),
        action_family="pan_phase_delay",
        safety_row_id="pan_phase_delay",
        write_surface_ids=("pan_phase_delay_direct_apply",),
        read_surface_ids=("channel_identity_and_basic_controls",),
    ),
    PromotionTarget(
        id="fx_supervised_promotion",
        label="FX/send supervised promotion",
        capability_ids=("console_fx_control", "effects_management", "effect_sends"),
        action_family="fx_send_insert",
        safety_row_id="fx_sends_inserts",
        write_surface_ids=("fx_insert_send_direct_apply",),
        read_surface_ids=("fx_slots_and_inserts", "routing_and_sends"),
    ),
    PromotionTarget(
        id="bus_group_matrix_supervised_promotion",
        label="Bus/group/DCA/matrix supervised promotion",
        capability_ids=("console_dca_group_matrix_control", "mix_balance", "master_bus_optimization"),
        action_family="bus_group_matrix",
        safety_row_id="bus_group_matrix",
        write_surface_ids=("mixer_transport_clients",),
        read_surface_ids=("bus_group_main_dca_state", "matrix_state"),
        evidence_requirement_ids=(
            "bus_group_assignment_cache_freshness",
            "bus_group_fader_readback",
            "matrix_send_readback",
            "main_bus_true_peak_headroom",
            "bus_group_physical_fixture",
            "hardware_readback_and_rollback_proof",
        ),
    ),
    PromotionTarget(
        id="mute_supervised_promotion",
        label="Mute supervised promotion",
        capability_ids=("console_mute_control",),
        action_family="mute",
        safety_row_id="mute_supervised",
        write_surface_ids=("mixer_transport_clients", "voice_ai_direct_commands"),
        read_surface_ids=("channel_identity_and_basic_controls",),
        evidence_requirement_ids=(
            "current_mute_cache_freshness",
            "mute_owner_intent_evidence",
            "signal_return_or_idle_window_evidence",
            "manual_mute_authority_review",
            "mute_physical_fixture",
            "hardware_readback_rollback_and_emergency_proof",
        ),
    ),
    PromotionTarget(
        id="routing_supervised_promotion",
        label="Routing supervised promotion",
        capability_ids=("console_routing_control",),
        action_family="routing",
        safety_row_id="routing_snapshots_scenes",
        write_surface_ids=("routing_snapshot_scene_writes",),
        read_surface_ids=("routing_and_sends",),
        evidence_requirement_ids=(
            "current_route_cache_freshness",
            "route_diff_evidence",
            "source_identity_downstream_trust",
            "routing_physical_fixture",
            "hardware_readback_and_rollback_proof",
        ),
    ),
    PromotionTarget(
        id="scene_preset_supervised_promotion",
        label="Scene/preset supervised promotion",
        capability_ids=("console_scene_preset_control",),
        action_family="scene_preset",
        safety_row_id="routing_snapshots_scenes",
        write_surface_ids=("routing_snapshot_scene_writes",),
        read_surface_ids=("snapshots_scenes", "user_presets"),
        evidence_requirement_ids=(
            "current_scene_preset_cache_freshness",
            "snapshots_scenes_snapshot_cache_freshness",
            "user_presets_snapshot_cache_freshness",
            "read_only_scoped_diff_evidence",
            "read_only_rollback_evidence",
            "scene_preset_physical_fixture",
            "hardware_readback_and_rollback_proof",
        ),
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _capabilities(ids: Iterable[str]) -> Tuple[Any, ...]:
    return tuple(row for row in (get_project_goal_capability(row_id) for row_id in ids) if row is not None)


def _safety_full_proof_passed(row: Any) -> bool:
    return bool(row) and all(getattr(row, field_name) == "passed" for field_name in REQUIRED_FULL_PROOF_FIELDS)


def _gap_bucket(blocker: str) -> str:
    if blocker.startswith("capability_"):
        return "capability"
    if blocker.startswith("safety_"):
        return "safety"
    if blocker.startswith("read_surface_"):
        return "read_surface"
    if blocker.startswith("write_surface_") or blocker == "no_allowed_live_write_surface":
        return "write_surface"
    if blocker.startswith("mode_") or blocker == "not_autonomous_ready":
        return "autonomy_mode"
    if any(blocker.startswith(f"{field_name}_") for field_name in REQUIRED_FULL_PROOF_FIELDS):
        return "proof"
    return "other"


def _append_unique(target: Dict[str, list], key: str, values: Iterable[str]) -> None:
    bucket = target.setdefault(key, [])
    for value in values:
        if value and value not in bucket:
            bucket.append(value)


def _promotion_gap_summary(rows: Tuple["PromotionReadinessRow", ...]) -> Dict[str, Any]:
    bucket_counts: Dict[str, int] = {bucket: 0 for bucket in GAP_BUCKETS}
    blocked_target_ids_by_bucket: Dict[str, list] = {bucket: [] for bucket in GAP_BUCKETS}
    evidence_by_action_family: Dict[str, list] = {}
    next_by_action_family: Dict[str, list] = {}

    for row in rows:
        row_buckets = set()
        for blocker in row.blockers:
            bucket = _gap_bucket(blocker)
            bucket_counts[bucket] += 1
            row_buckets.add(bucket)
        if row.evidence_requirement_ids:
            bucket_counts["evidence_requirement"] += len(row.evidence_requirement_ids)
            row_buckets.add("evidence_requirement")
            _append_unique(evidence_by_action_family, row.action_family, row.evidence_requirement_ids)
        _append_unique(next_by_action_family, row.action_family, row.next_required_evidence)
        for bucket in row_buckets:
            _append_unique(blocked_target_ids_by_bucket, bucket, (row.id,))

    return {
        "read_only": True,
        "live_write": False,
        "full_autonomous_ready": False,
        "gap_bucket_counts": bucket_counts,
        "blocked_target_ids_by_bucket": {
            bucket: ids for bucket, ids in blocked_target_ids_by_bucket.items() if ids
        },
        "evidence_requirement_ids_by_action_family": evidence_by_action_family,
        "next_required_evidence_by_action_family": next_by_action_family,
        "claim": "promotion gaps are aggregated for planning only; this summary grants no live authority",
    }


def _wing_readonly_preflight_readiness_overlay(
    evidence_snapshot: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    evidence = evidence_snapshot or build_wing_readonly_preflight_evidence()
    summary = evidence.get("summary") if isinstance(evidence.get("summary"), dict) else {}
    matrix = (
        evidence.get("mvp_blocker_matrix")
        if isinstance(evidence.get("mvp_blocker_matrix"), list)
        else []
    )
    blocked_by = summary.get("blocked_by") if isinstance(summary.get("blocked_by"), list) else []
    violations = summary.get("violations") if isinstance(summary.get("violations"), list) else []
    valid_candidate_count = int(summary.get("valid_candidate_count") or 0)
    invalid_candidate_count = int(summary.get("invalid_candidate_count") or 0)
    update_allowed = summary.get("readiness_matrix_update_allowed") is True
    closure_allowed = summary.get("readiness_closure_allowed_by_evidence") is True
    live_authority = summary.get("live_authority_granted") is True
    full_auto = summary.get("full_autonomous_ready") is True
    candidate_available = (
        valid_candidate_count > 0
        and invalid_candidate_count == 0
        and update_allowed
        and not closure_allowed
        and not live_authority
        and not full_auto
    )
    matrix_rows = [row for row in matrix if isinstance(row, dict)]
    candidate_ids = [
        str(row.get("id"))
        for row in matrix_rows
        if row.get("status") == "candidate_available" and row.get("id")
    ]
    closed_ids = [
        str(row.get("id"))
        for row in matrix_rows
        if row.get("closed") is True and row.get("id")
    ]
    next_actions = (
        [
            "Manually review the read-only preflight candidate before updating readiness rows.",
            "Keep supervised apply, rollback, emergency-stop, and timeout proof as separate WING steps.",
        ]
        if candidate_available
        else [
            "Before starting the intended backend/WING session, run the no-WING quick gate and resolve only confirmed unexpected listeners.",
            "Run the read-only WING preflight runner and capture physical readback artifacts.",
            "Do not update readiness rows from missing or invalid preflight evidence.",
        ]
    )
    return {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "automatic_safety_update_allowed": False,
        "readiness_closure_allowed_by_evidence": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "evidence_status": evidence.get("status"),
        "artifact_count": int(summary.get("artifact_count") or 0),
        "valid_candidate_count": valid_candidate_count,
        "invalid_candidate_count": invalid_candidate_count,
        "readiness_update_candidate_available": candidate_available,
        "prestart_backend_port_check_required": True,
        "prestart_backend_port_check_must_pass_before_starting_backend": True,
        "prestart_backend_port_check_command": list(
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
        ),
        "prestart_backend_port_check_failure_detail_fields": list(
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FIELDS
        ),
        "candidate_blocker_ids": candidate_ids,
        "closed_blocker_ids": closed_ids,
        "latest_valid_candidate_run_id": summary.get("latest_valid_candidate_run_id"),
        "blocked_by": list(blocked_by),
        "violations": list(violations),
        "next_required_actions": next_actions,
        "claim": "read-only preflight evidence can inform manual readiness review only; it closes no blocker automatically",
    }


def _target_readiness(target: PromotionTarget, *, full_autonomous_blocked: bool) -> PromotionReadinessRow:
    capabilities = _capabilities(target.capability_ids)
    safety = get_safety_coverage_row(target.safety_row_id) if target.safety_row_id else None
    write_surfaces = tuple(
        surface for surface in (get_write_surface(surface_id) for surface_id in target.write_surface_ids) if surface is not None
    )
    read_surfaces = tuple(
        surface for surface in (get_read_surface(surface_id) for surface_id in target.read_surface_ids) if surface is not None
    )
    blockers = []
    next_required = []

    if len(capabilities) != len(target.capability_ids):
        blockers.append("capability_row_missing")
    if safety is None:
        blockers.append("safety_coverage_row_missing")
        next_required.append("Create a safety coverage row with unit, offline, fake-mixer, real-WING, readback, rollback, and emergency-stop proof states.")
    else:
        if not safety.live_allowed:
            blockers.append("safety_live_not_allowed")
        if safety.status in {"blocked", "quarantined"}:
            blockers.append(f"safety_status_{safety.status}")
        for field_name in REQUIRED_FULL_PROOF_FIELDS:
            if getattr(safety, field_name) != "passed":
                blockers.append(f"{field_name}_{getattr(safety, field_name)}")
        next_required.append(safety.next_required_evidence)

    if len(write_surfaces) != len(target.write_surface_ids):
        blockers.append("write_surface_row_missing")
    allowed_write_surfaces = [surface for surface in write_surfaces if surface.allowed_live]
    if not allowed_write_surfaces:
        blockers.append("no_allowed_live_write_surface")
    for surface in write_surfaces:
        if surface.status in {"blocked", "quarantined"}:
            blockers.append(f"write_surface_{surface.id}_{surface.status}")
        next_required.append(surface.next_action)

    if len(read_surfaces) != len(target.read_surface_ids):
        blockers.append("read_surface_row_missing")
    for surface in read_surfaces:
        if surface.status in {"partial", "planned", "missing"}:
            blockers.append(f"read_surface_{surface.id}_{surface.status}")
        next_required.append(surface.next_action)

    capability_live_kinds = tuple(
        kind
        for capability in capabilities
        for kind in capability.current_live_write_kinds
    )
    if capability_live_kinds and safety is not None and not safety.live_allowed:
        blockers.append("capability_declares_live_kind_without_safety_proof")

    supervised_live_allowed = bool(safety and safety.live_allowed and allowed_write_surfaces)
    autonomous_ready = bool(
        supervised_live_allowed
        and not full_autonomous_blocked
        and _safety_full_proof_passed(safety)
        and all(surface.status in {"cached", "queryable"} for surface in read_surfaces)
        and not any(surface.status in {"blocked", "quarantined"} for surface in write_surfaces)
    )
    if full_autonomous_blocked:
        blockers.append("mode_full_autonomous_blocked")
    if not autonomous_ready:
        blockers.append("not_autonomous_ready")

    if supervised_live_allowed:
        status = "supervised_live_allowed"
    elif safety is not None and safety.status in {"blocked", "quarantined"}:
        status = "manual_only_blocked"
    elif any(surface.status in {"partial", "planned", "missing"} for surface in read_surfaces):
        status = "read_coverage_blocked"
    else:
        status = "proof_blocked"

    return PromotionReadinessRow(
        id=target.id,
        label=target.label,
        action_family=target.action_family,
        target_mode=target.target_mode,
        status=status,
        supervised_live_allowed=supervised_live_allowed,
        autonomous_ready=autonomous_ready,
        capability_ids=tuple(capability.id for capability in capabilities),
        capability_statuses=tuple(capability.status for capability in capabilities),
        safety_row_id=safety.id if safety else "",
        safety_status=safety.status if safety else "missing",
        write_surface_ids=tuple(surface.id for surface in write_surfaces),
        write_surface_statuses=tuple(surface.status for surface in write_surfaces),
        read_surface_ids=tuple(surface.id for surface in read_surfaces),
        read_surface_statuses=tuple(surface.status for surface in read_surfaces),
        evidence_requirement_ids=target.evidence_requirement_ids,
        blockers=tuple(sorted(set(blockers))),
        next_required_evidence=tuple(dict.fromkeys(item for item in next_required if item)),
    )


def iter_promotion_readiness_rows() -> Iterable[PromotionReadinessRow]:
    full_auto = get_project_goal_capability("mode_full_autonomous")
    full_autonomous_blocked = bool(full_auto is None or full_auto.status == "blocked_live")
    return tuple(
        _target_readiness(target, full_autonomous_blocked=full_autonomous_blocked)
        for target in PROMOTION_TARGETS
    )


def build_autonomy_promotion_readiness_snapshot(
    *,
    wing_readonly_preflight_evidence_snapshot: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    rows = tuple(iter_promotion_readiness_rows())
    wing_preflight_overlay = _wing_readonly_preflight_readiness_overlay(
        wing_readonly_preflight_evidence_snapshot
    )
    status_counts: Dict[str, int] = {status: 0 for status in sorted(READINESS_STATUSES)}
    for row in rows:
        status_counts[row.status] += 1
    supervised_ids = [row.id for row in rows if row.supervised_live_allowed]
    autonomous_ids = [row.id for row in rows if row.autonomous_ready]
    blocked_ids = [row.id for row in rows if not row.autonomous_ready]

    return {
        "type": "autonomy_promotion_readiness",
        "generated_at": _utc_now(),
        "status": "ok",
        "safety": {
            "read_only": True,
            "live_write": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "supervised_only": True,
        },
        "summary": {
            "target_count": len(rows),
            "status_counts": status_counts,
            "supervised_live_allowed_target_ids": supervised_ids,
            "autonomous_ready_target_ids": autonomous_ids,
            "blocked_target_ids": blocked_ids,
            "global_blockers": [
                "mode_full_autonomous_blocked",
                "only_supervised_fader_gain_executor_is_registered",
                "read_coverage_is_partial_or_missing",
                "real_wing_and_emergency_stop_proof_incomplete",
            ],
            "next_global_step": "Run proof gates per action family before widening supervised or autonomous live writes.",
            "promotion_gap_summary": _promotion_gap_summary(rows),
            "wing_readonly_preflight_evidence": wing_preflight_overlay,
        },
        "rows": [row.to_dict() for row in rows],
    }
