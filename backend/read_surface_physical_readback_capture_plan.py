"""Operator-required capture plan for read-surface physical readback evidence.

The plan turns fixture/remediation metadata into a concrete checklist for the
next hardware proof target. It does not import mixer transports, query a
console, execute tests, or authorize live writes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from read_surface_fixture_evidence import build_read_surface_fixture_evidence_snapshot
    from read_surface_remediation_plan import build_read_surface_remediation_plan
except ImportError:  # pragma: no cover - package import compatibility
    from backend.read_surface_fixture_evidence import build_read_surface_fixture_evidence_snapshot
    from backend.read_surface_remediation_plan import build_read_surface_remediation_plan


READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PLAN_METHOD_ID = (
    "read_surface_physical_readback_capture_plan"
)
CAPTURE_PLAN_STATUSES = {
    "operator_capture_required",
    "read_surface_physical_readback_complete",
}
PHYSICAL_READBACK_CONTRACT_METHOD_BY_SURFACE = {
    "raw_mixer_state_cache": "raw_mixer_state_physical_readback_contract",
    "channel_identity_and_basic_controls": "channel_basic_controls_physical_readback_contract",
    "channel_eq_dynamics_gate": "channel_processing_physical_readback_contract",
    "bus_group_main_dca_state": "bus_group_main_dca_physical_readback_contract",
    "matrix_state": "matrix_state_physical_readback_contract",
    "fx_slots_and_inserts": "fx_slots_inserts_physical_readback_contract",
    "routing_and_sends": "routing_sends_physical_readback_contract",
    "snapshots_scenes": "snapshots_scenes_physical_readback_contract",
    "user_presets": "user_presets_physical_readback_contract",
    "dlive_state": "dlive_state_physical_readback_contract",
}
FORBIDDEN_CAPTURE_ACTIONS = (
    "do not apply fader, gain, trim, pan, mute, EQ, dynamics, gate, routing, DCA, matrix, FX, scene, or preset changes",
    "do not transmit OSC, MIDI, TCP, or websocket mixer-control messages",
    "do not recall, store, load, save, or overwrite scenes, snapshots, shows, or user presets",
    "do not mark cached state as physical proof",
    "do not change safety rows, operator mode, promotion readiness, or live-write policy from this plan",
)


@dataclass(frozen=True)
class ReadSurfacePhysicalReadbackCaptureStep:
    id: str
    surface_id: str
    remediation_action_id: str
    target_console_surface: str
    requires_connection: bool
    read_mode: str
    operator_required: bool
    execution_allowed_by_plan: bool
    live_write: bool
    mutates_console_state: bool
    queries_console: bool
    executes_tests: bool
    contract_method_id: str
    required_artifact_contracts: Tuple[str, ...]
    source_paths: Tuple[str, ...]
    entrypoints: Tuple[str, ...]
    expected_artifacts: Tuple[str, ...]
    prerequisites: Tuple[str, ...]
    capture_instructions: Tuple[str, ...]
    abort_conditions: Tuple[str, ...]
    forbidden_actions: Tuple[str, ...]
    blockers_to_clear: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.operator_required:
            raise ValueError("Physical readback capture steps require an operator")
        if self.execution_allowed_by_plan:
            raise ValueError("The capture plan must not authorize execution")
        if self.live_write or self.mutates_console_state or self.queries_console or self.executes_tests:
            raise ValueError("Capture steps must be non-executing plan metadata")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "required_artifact_contracts",
            "source_paths",
            "entrypoints",
            "expected_artifacts",
            "prerequisites",
            "capture_instructions",
            "abort_conditions",
            "forbidden_actions",
            "blockers_to_clear",
        ):
            payload[key] = list(payload[key])
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_tuple(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _distinct(items: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _actions_by_id(plan: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    actions = plan.get("actions")
    if not isinstance(actions, list):
        return {}
    return {
        action["id"]: action
        for action in actions
        if isinstance(action, Mapping) and isinstance(action.get("id"), str)
    }


def _physical_gap_ids(fixture_snapshot: Mapping[str, Any]) -> Tuple[str, ...]:
    summary = _mapping(fixture_snapshot.get("summary"))
    declared = _string_tuple(summary.get("physical_fixture_missing_surface_ids"))
    if declared:
        return declared
    surfaces = fixture_snapshot.get("surfaces")
    if not isinstance(surfaces, list):
        return ()
    return tuple(
        row["surface_id"]
        for row in surfaces
        if isinstance(row, Mapping)
        and isinstance(row.get("surface_id"), str)
        and "physical_readback_fixture_missing" in _string_tuple(row.get("remaining_gaps"))
    )


def _expected_next_target(fixture_snapshot: Mapping[str, Any]) -> str | None:
    summary = _mapping(fixture_snapshot.get("summary"))
    source_next = summary.get("next_physical_readback_surface_id")
    if isinstance(source_next, str) and source_next:
        return source_next
    gap_ids = _physical_gap_ids(fixture_snapshot)
    return gap_ids[0] if gap_ids else None


def _required_artifact_contracts(action: Mapping[str, Any]) -> Tuple[str, ...]:
    surface_id = str(action.get("surface_id") or "")
    contracts = _string_tuple(action.get("missing_contracts"))
    contract_method_id = PHYSICAL_READBACK_CONTRACT_METHOD_BY_SURFACE.get(surface_id, "")
    return _distinct(("physical_readback_fixture", contract_method_id) + contracts)


def _expected_artifacts(action: Mapping[str, Any], contract_method_id: str) -> Tuple[str, ...]:
    artifacts = _string_tuple(action.get("expected_artifacts"))
    if artifacts:
        return artifacts
    surface_id = str(action.get("surface_id") or "unknown_surface")
    return (
        f"Operator-captured physical readback artifact for {surface_id}.",
        f"Artifact validates against {contract_method_id or 'the matching physical readback contract'}.",
    )


def _step_from_action(action: Mapping[str, Any]) -> ReadSurfacePhysicalReadbackCaptureStep:
    surface_id = str(action.get("surface_id") or "")
    action_id = str(action.get("id") or f"remediate_{surface_id}")
    contract_method_id = PHYSICAL_READBACK_CONTRACT_METHOD_BY_SURFACE.get(surface_id, "")
    blockers = _distinct(
        _string_tuple(action.get("blockers"))
        + _string_tuple(action.get("fixture_remaining_gaps"))
        + _string_tuple(action.get("missing_contracts"))
    )
    return ReadSurfacePhysicalReadbackCaptureStep(
        id=f"capture_{surface_id}_physical_readback",
        surface_id=surface_id,
        remediation_action_id=action_id,
        target_console_surface=str(action.get("target_console_surface") or ""),
        requires_connection=bool(action.get("requires_connection")),
        read_mode=str(action.get("read_mode") or ""),
        operator_required=True,
        execution_allowed_by_plan=False,
        live_write=False,
        mutates_console_state=False,
        queries_console=False,
        executes_tests=False,
        contract_method_id=contract_method_id,
        required_artifact_contracts=_required_artifact_contracts(action),
        source_paths=_string_tuple(action.get("source_paths")),
        entrypoints=_string_tuple(action.get("entrypoints")),
        expected_artifacts=_expected_artifacts(action, contract_method_id),
        prerequisites=(
            "Operator explicitly chooses the target console and capture tooling outside this module.",
            "The capture tool is configured for readback only and cannot apply mixer changes.",
            "The console/show state is known, backed up, and safe to observe.",
            "The artifact path, timestamp, source, and operator approval are recorded before review.",
        ),
        capture_instructions=(
            f"Capture physical readback evidence for {surface_id}.",
            f"Validate the artifact with {contract_method_id or 'the matching contract validator'}.",
            "Attach the resulting artifact to the next no-WING release-check or proof review bundle.",
            "Review the artifact manually before closing the fixture gap or changing any autonomy ceiling.",
        ),
        abort_conditions=(
            "The capture path would transmit a mixer-control command.",
            "The source is cached, synthetic, simulated, or unavailable instead of physical readback.",
            "The artifact omits required false safety flags or mutation-evidence fields.",
            "The operator cannot confirm the intended console, show, or capture scope.",
        ),
        forbidden_actions=FORBIDDEN_CAPTURE_ACTIONS,
        blockers_to_clear=blockers,
    )


def build_read_surface_physical_readback_capture_plan(
    *,
    remediation_plan: Mapping[str, Any] | None = None,
    fixture_evidence_snapshot: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a non-executing operator checklist for the next readback proof."""

    fixture_snapshot = (
        fixture_evidence_snapshot
        if fixture_evidence_snapshot is not None
        else build_read_surface_fixture_evidence_snapshot()
    )
    plan = (
        remediation_plan
        if remediation_plan is not None
        else build_read_surface_remediation_plan(fixture_evidence_snapshot=fixture_snapshot)
    )
    fixture_summary = _mapping(fixture_snapshot.get("summary"))
    plan_summary = _mapping(plan.get("summary"))
    expected_next_target = _expected_next_target(fixture_snapshot)
    expected_next_action_id = f"remediate_{expected_next_target}" if expected_next_target else None
    next_surface_id = plan_summary.get("next_physical_readback_surface_id")
    if not isinstance(next_surface_id, str) or not next_surface_id:
        next_surface_id = expected_next_target
    next_action_id = plan_summary.get("next_physical_readback_action_id")
    if not isinstance(next_action_id, str) or not next_action_id:
        next_action_id = expected_next_action_id
    action_by_id = _actions_by_id(plan)
    next_action = action_by_id.get(next_action_id) if isinstance(next_action_id, str) else None
    target_action_consistency = (
        expected_next_target == next_surface_id
        and expected_next_action_id == next_action_id
        and isinstance(next_action, Mapping)
        and next_action.get("surface_id") == next_surface_id
        and next_action.get("physical_readback_gap") is True
        and next_action.get("next_physical_readback_target") is True
    )
    steps = (
        (_step_from_action(next_action),)
        if isinstance(next_action, Mapping) and next_surface_id
        else ()
    )
    physical_gap_ids = _physical_gap_ids(fixture_snapshot)
    blocked_by = []
    if physical_gap_ids:
        blocked_by.append("physical_readback_fixtures_incomplete")
    if not target_action_consistency and next_surface_id:
        blocked_by.append("read_surface_next_physical_readback_target_inconsistent")
    if steps:
        blocked_by.append("operator_must_capture_and_review_physical_readback_artifact")

    return {
        "type": "read_surface_physical_readback_capture_plan",
        "generated_at": _utc_now(),
        "method_id": READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PLAN_METHOD_ID,
        "status": (
            "operator_capture_required"
            if next_surface_id
            else "read_surface_physical_readback_complete"
        ),
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_plan": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "source_fixture_status": fixture_snapshot.get("status"),
            "source_remediation_status": plan.get("status"),
            "physical_fixture_missing_surface_ids": list(physical_gap_ids),
            "next_physical_readback_surface_id": next_surface_id,
            "next_physical_readback_action_id": next_action_id,
            "expected_next_physical_readback_surface_id": expected_next_target,
            "expected_next_physical_readback_action_id": expected_next_action_id,
            "target_action_consistency": target_action_consistency,
            "required_artifact_contracts": (
                list(steps[0].required_artifact_contracts) if steps else []
            ),
            "capture_step_count": len(steps),
            "blocked_by": blocked_by,
            "next_global_step": (
                f"Operator captures physical readback evidence for {next_surface_id}; "
                "review the artifact before closing the fixture gap."
                if next_surface_id
                else "No remaining physical readback fixture gaps are declared."
            ),
            "source_claim": fixture_summary.get("claim"),
        },
        "steps": [step.to_dict() for step in steps],
    }
