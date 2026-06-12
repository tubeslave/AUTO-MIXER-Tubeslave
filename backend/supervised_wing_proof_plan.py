"""Read-only plan for the next supervised WING proof run.

The plan is an operator checklist, not an executor. It may include commands for
the existing physical proof runner, but this module never imports transport
clients and never runs those commands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple

try:
    from autonomy_promotion_readiness import build_autonomy_promotion_readiness_snapshot
except ImportError:  # pragma: no cover - package import compatibility
    from backend.autonomy_promotion_readiness import build_autonomy_promotion_readiness_snapshot

try:
    from safety_coverage_matrix import iter_safety_coverage_rows
except ImportError:  # pragma: no cover - package import compatibility
    from backend.safety_coverage_matrix import iter_safety_coverage_rows

try:
    from wing_readonly_preflight_review_decision import (
        build_wing_readonly_preflight_review_decision_ledger,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.wing_readonly_preflight_review_decision import (
        build_wing_readonly_preflight_review_decision_ledger,
    )

try:
    from wing_readonly_preflight_contract import (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.wing_readonly_preflight_contract import (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    )


SUPERVISED_WING_PROOF_PLAN_METHOD_ID = "supervised_wing_proof_plan"
READONLY_PREFLIGHT_CANDIDATE_BLOCKER = (
    "readonly_preflight_candidate_required_before_supervised_proof"
)
READONLY_PREFLIGHT_REVIEW_DECISION_BLOCKER = (
    "accepted_readonly_preflight_review_decision_required_before_supervised_proof"
)
READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER = (
    "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
)
PLAN_STATUSES = {
    "operator_proof_required",
    "manual_review_ready",
    "blocked_by_invalid_artifacts",
    "blocked_by_readonly_preflight",
}
DEFAULT_CHANNELS_BY_VALUE_TYPE = {
    "gain": (3,),
    "fader": (2, 8, 16),
}
DEFAULT_DELTA_DB_BY_VALUE_TYPE = {
    "gain": -0.25,
    "fader": -0.25,
}
DEFAULT_ARTIFACT_ROOT = "artifacts/wing_telemetry"
DEFAULT_READONLY_PREFLIGHT_ARTIFACT_ROOT = "artifacts/wing_readonly_preflight"
DEFAULT_BACKEND_URL = "ws://localhost:8765"


@dataclass(frozen=True)
class WingProofPlanStep:
    id: str
    value_type: str
    channel: int
    delta_db: float
    live_execution_required: bool
    execution_allowed_by_plan: bool
    operator_approval_required: bool
    command: Tuple[str, ...]
    rationale: str
    prerequisites: Tuple[str, ...]
    abort_conditions: Tuple[str, ...]
    expected_artifacts: Tuple[str, ...]
    evidence_blockers_to_review: Tuple[str, ...]
    required_evidence_components: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.execution_allowed_by_plan:
            raise ValueError("The proof plan must not authorize execution")
        if not self.operator_approval_required:
            raise ValueError("Physical proof steps require explicit operator approval")
        if self.delta_db >= 0:
            raise ValueError("Physical proof plan only supports negative dB deltas")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "command",
            "prerequisites",
            "abort_conditions",
            "expected_artifacts",
            "evidence_blockers_to_review",
            "required_evidence_components",
        ):
            payload[key] = list(payload[key])
        return payload


@dataclass(frozen=True)
class WingReadonlyPreflightPlanStep:
    id: str
    live_execution_required: bool
    execution_allowed_by_plan: bool
    operator_approval_required: bool
    read_only: bool
    command: Tuple[str, ...]
    rationale: str
    prerequisites: Tuple[str, ...]
    abort_conditions: Tuple[str, ...]
    expected_artifacts: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.live_execution_required:
            raise ValueError("Read-only preflight step must not be a live write step")
        if self.execution_allowed_by_plan:
            raise ValueError("The proof plan must not authorize read-only preflight execution")
        if not self.operator_approval_required:
            raise ValueError("Read-only WING preflight still requires operator presence")
        forbidden = {
            "--execute",
            "tools/wing_operator_proposal_validation_runner.py",
            "apply_" + "operator_proposal",
            "create_" + "operator_proposal",
        }
        if any(token in self.command for token in forbidden):
            raise ValueError("Read-only preflight step must not include supervised apply commands")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in (
            "command",
            "prerequisites",
            "abort_conditions",
            "expected_artifacts",
        ):
            payload[key] = list(payload[key])
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _evidence_rows(evidence_snapshot: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    rows = evidence_snapshot.get("rows")
    if not isinstance(rows, list):
        return ()
    return tuple(row for row in rows if isinstance(row, Mapping))


def _invalid_artifacts_present(evidence_snapshot: Mapping[str, Any]) -> bool:
    rows = _evidence_rows(evidence_snapshot)
    return any("invalid_proof_summaries_present" in set(row.get("blockers") or []) for row in rows)


def _readonly_preflight_overlay(
    autonomy_readiness_snapshot: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    readiness = autonomy_readiness_snapshot or build_autonomy_promotion_readiness_snapshot()
    summary = readiness.get("summary") if isinstance(readiness.get("summary"), Mapping) else {}
    overlay = summary.get("wing_readonly_preflight_evidence")
    return overlay if isinstance(overlay, Mapping) else {}


def _readonly_preflight_candidate_available(overlay: Mapping[str, Any]) -> bool:
    return (
        overlay.get("readiness_update_candidate_available") is True
        and int(overlay.get("valid_candidate_count") or 0) > 0
        and int(overlay.get("invalid_candidate_count") or 0) == 0
        and overlay.get("readiness_closure_allowed_by_evidence") is not True
        and overlay.get("live_authority_granted") is not True
        and overlay.get("full_autonomous_ready") is not True
    )


def _readonly_preflight_review_decision_overlay(
    autonomy_readiness_snapshot: Mapping[str, Any] | None,
    explicit_ledger: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if isinstance(explicit_ledger, Mapping):
        return explicit_ledger
    readiness = autonomy_readiness_snapshot or {}
    summary = readiness.get("summary") if isinstance(readiness.get("summary"), Mapping) else {}
    overlay = summary.get("wing_readonly_preflight_review_decision_ledger")
    if isinstance(overlay, Mapping):
        return overlay
    return build_wing_readonly_preflight_review_decision_ledger()


def _review_decision_summary(overlay: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = overlay.get("summary")
    return summary if isinstance(summary, Mapping) else overlay


def _readonly_preflight_review_decision_accepted(overlay: Mapping[str, Any]) -> bool:
    summary = _review_decision_summary(overlay)
    accepted_source_ids = _string_list(summary.get("accepted_source_ids"))
    return (
        summary.get("manual_readiness_update_candidate_available") is True
        and int(summary.get("accepted_record_count") or 0) > 0
        and bool(accepted_source_ids)
        and _accepted_candidate_prestart_source_ids(summary) == accepted_source_ids
        and int(summary.get("invalid_record_count") or 0) == 0
        and int(summary.get("unreviewed_candidate_count") or 0) == 0
        and summary.get("readiness_matrix_update_allowed_by_decision") is not True
        and summary.get("readiness_closure_allowed_by_decision") is not True
        and summary.get("live_authority_granted") is not True
        and summary.get("full_autonomous_ready") is not True
    )


def _readonly_preflight_review_decision_status(overlay: Mapping[str, Any]) -> Any:
    return overlay.get("status") or _review_decision_summary(overlay).get("ledger_status")


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value if item] if isinstance(value, list) else []


def _accepted_candidate_prestart_source_ids(summary: Mapping[str, Any]) -> list[str]:
    accepted_source_ids = _string_list(summary.get("accepted_source_ids"))
    if not accepted_source_ids:
        return []
    rows = summary.get("candidate_reviews")
    if not isinstance(rows, list):
        return []
    rows_by_source = {
        str(row.get("source_id")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("source_id")
    }
    valid_source_ids: list[str] = []
    for source_id in accepted_source_ids:
        row = rows_by_source.get(source_id)
        if not isinstance(row, Mapping):
            continue
        if row.get("backend_port_prestart_check_required") is not True:
            continue
        if row.get("backend_port_prestart_check_must_pass_before_starting_backend") is not True:
            continue
        if _string_list(row.get("backend_port_prestart_check_command")) != list(
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
        ):
            continue
        if _string_list(row.get("backend_port_prestart_check_failure_detail_fields")) != list(
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
        ):
            continue
        valid_source_ids.append(source_id)
    return valid_source_ids


def _row_by_value_type(evidence_snapshot: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(row.get("value_type")): row
        for row in _evidence_rows(evidence_snapshot)
        if row.get("value_type")
    }


def _required_evidence_components_by_value_type() -> Dict[str, Tuple[str, ...]]:
    return {
        row.action_type: tuple(row.required_evidence_components)
        for row in iter_safety_coverage_rows()
        if row.required_evidence_components
    }


def _proof_command(
    *,
    backend_url: str,
    artifact_root: str,
    channel: int,
    value_type: str,
    delta_db: float,
) -> Tuple[str, ...]:
    return (
        "python3",
        "tools/wing_operator_proposal_validation_runner.py",
        "--url",
        backend_url,
        "--channel",
        str(channel),
        "--value-type",
        value_type,
        "--delta-db",
        f"{delta_db:.2f}",
        "--operator-confirm-channel",
        str(channel),
        "--operator-confirm-value-type",
        value_type,
        "--operator-confirm-delta-db",
        f"{delta_db:.2f}",
        "--execute",
        "--prove-blocked-apply",
        "--prove-emergency-stop",
        "--rollback-delay-sec",
        "0",
        "--artifact-root",
        artifact_root,
        "--reason",
        f"supervised_physical_{value_type}_proof",
    )


def _readonly_preflight_command(
    *,
    backend_url: str,
    artifact_root: str,
) -> Tuple[str, ...]:
    return (
        "python3",
        "tools/wing_readonly_preflight_runner.py",
        "--url",
        backend_url,
        "--channels",
        "3",
        "--value-types",
        "fader,gain",
        "--require-physical-readback",
        "--artifact-root",
        artifact_root,
    )


def _readonly_preflight_step(
    *,
    backend_url: str,
    artifact_root: str,
    blockers: Tuple[str, ...],
    next_actions: Tuple[str, ...],
) -> WingReadonlyPreflightPlanStep:
    prerequisites = (
        "Operator is physically present at the intended WING Rack.",
        "Backend is already connected in WING mode.",
        "No supervised apply, proposal, rollback, emergency-stop, connect, or disconnect request is sent by this step.",
        "Move or replace invalid preflight artifacts before trusting a readiness candidate.",
    )
    return WingReadonlyPreflightPlanStep(
        id="wing_readonly_preflight_required",
        live_execution_required=False,
        execution_allowed_by_plan=False,
        operator_approval_required=True,
        read_only=True,
        command=_readonly_preflight_command(
            backend_url=backend_url,
            artifact_root=artifact_root,
        ),
        rationale=(
            "Read-only WING connection and physical readback evidence is required before "
            "any supervised fader/gain proof step can be planned."
        ),
        prerequisites=prerequisites,
        abort_conditions=(
            "Backend is not connected to the intended WING Rack.",
            "Any response is not verified from physical readback.",
            "Any forbidden write, proposal, rollback, emergency, connect, or disconnect request is observed.",
            *blockers,
        ),
        expected_artifacts=(
            "wing_readonly_preflight.v1 summary.json",
            "events.jsonl with get_operator_mode/get_state/query_mixer_parameter only",
            *next_actions,
        ),
    )


def _common_prerequisites(value_type: str, blockers: Tuple[str, ...]) -> Tuple[str, ...]:
    prerequisites = [
        "A clean read-only WING preflight candidate exists and has been manually reviewed.",
        "Operator is physically present and explicitly approves a bounded proof run.",
        "Backend is connected to the intended WING Rack only.",
        "A current scene/console backup exists and the operator can restore manually.",
        "Monitor/PA risk is controlled before any supervised write.",
        "Proof uses the Product Layer proposal path and rollback gate only.",
    ]
    if value_type == "gain":
        prerequisites.append("Use gain-down proof only; gain-up remains out of scope for this plan.")
    if "blocked_artifacts_for_value_type" in blockers:
        prerequisites.append("Review blocked prior proof artifacts for this value type before trusting a new result.")
    return tuple(prerequisites)


def _abort_conditions() -> Tuple[str, ...]:
    return (
        "Pre-read is unavailable or not from physical readback.",
        "Requested delta is positive or exceeds 0.5 dB.",
        "Apply readback is missing, mismatched, or not physically verified.",
        "Rollback readback or rollback proof is missing.",
        "Emergency stop does not block a follow-up proposal.",
        "Any fader move would exceed 0 dBFS.",
        "Feedback, clipping, or operator concern appears at any point.",
    )


def _expected_artifacts(value_type: str, channel: int) -> Tuple[str, ...]:
    return (
        f"summary.json with value_type={value_type} and channel={channel}",
        "events.jsonl with proposal/apply/rollback/emergency-stop sequence",
        "proof report row with proof_mode=physical and status=supervised_candidate",
        "promotion evidence row marked safety_review_candidate only after manual artifact review",
    )


def _build_steps(
    evidence_snapshot: Mapping[str, Any],
    *,
    backend_url: str,
    artifact_root: str,
    channels_by_value_type: Mapping[str, Tuple[int, ...]],
    delta_db_by_value_type: Mapping[str, float],
) -> Tuple[WingProofPlanStep, ...]:
    rows = _row_by_value_type(evidence_snapshot)
    required_components = _required_evidence_components_by_value_type()
    steps: list[WingProofPlanStep] = []
    for value_type in ("gain", "fader"):
        row = rows.get(value_type, {})
        if row.get("safety_row_review_candidate") is True:
            continue
        blockers = tuple(str(item) for item in row.get("blockers", ()) if item)
        channels = channels_by_value_type.get(value_type, DEFAULT_CHANNELS_BY_VALUE_TYPE[value_type])
        delta_db = float(delta_db_by_value_type.get(value_type, DEFAULT_DELTA_DB_BY_VALUE_TYPE[value_type]))
        for channel in channels:
            steps.append(
                WingProofPlanStep(
                    id=f"{value_type}_physical_proof_ch{channel}",
                    value_type=value_type,
                    channel=int(channel),
                    delta_db=delta_db,
                    live_execution_required=True,
                    execution_allowed_by_plan=False,
                    operator_approval_required=True,
                    command=_proof_command(
                        backend_url=backend_url,
                        artifact_root=artifact_root,
                        channel=int(channel),
                        value_type=value_type,
                        delta_db=delta_db,
                    ),
                    rationale=(
                        "Physical proof candidate is missing or blocked; run a small supervised negative move "
                        "through the existing Product Layer proof runner."
                    ),
                    prerequisites=_common_prerequisites(value_type, blockers),
                    abort_conditions=_abort_conditions(),
                    expected_artifacts=_expected_artifacts(value_type, int(channel)),
                    evidence_blockers_to_review=blockers,
                    required_evidence_components=required_components.get(value_type, ()),
                )
            )
    return tuple(steps)


def build_supervised_wing_proof_plan(
    promotion_evidence_snapshot: Mapping[str, Any],
    *,
    autonomy_readiness_snapshot: Mapping[str, Any] | None = None,
    readonly_preflight_review_decision_ledger: Mapping[str, Any] | None = None,
    backend_url: str = DEFAULT_BACKEND_URL,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    channels_by_value_type: Mapping[str, Tuple[int, ...]] | None = None,
    delta_db_by_value_type: Mapping[str, float] | None = None,
    readonly_preflight_artifact_root: str = DEFAULT_READONLY_PREFLIGHT_ARTIFACT_ROOT,
) -> Dict[str, Any]:
    """Build a non-executing proof checklist from promotion evidence rows."""

    channels = dict(DEFAULT_CHANNELS_BY_VALUE_TYPE)
    if channels_by_value_type:
        channels.update({key: tuple(int(item) for item in value) for key, value in channels_by_value_type.items()})
    deltas = dict(DEFAULT_DELTA_DB_BY_VALUE_TYPE)
    if delta_db_by_value_type:
        deltas.update({key: float(value) for key, value in delta_db_by_value_type.items()})

    invalid = _invalid_artifacts_present(promotion_evidence_snapshot)
    preflight = _readonly_preflight_overlay(autonomy_readiness_snapshot)
    preflight_candidate = _readonly_preflight_candidate_available(preflight)
    review_decision = _readonly_preflight_review_decision_overlay(
        autonomy_readiness_snapshot,
        readonly_preflight_review_decision_ledger,
    )
    review_summary = _review_decision_summary(review_decision)
    review_decision_accepted = _readonly_preflight_review_decision_accepted(review_decision)
    accepted_candidate_prestart_source_ids = _accepted_candidate_prestart_source_ids(
        review_summary
    )
    accepted_source_ids = _string_list(review_summary.get("accepted_source_ids"))
    candidate_prestart_contract_valid = (
        bool(accepted_source_ids)
        and accepted_candidate_prestart_source_ids == accepted_source_ids
    )
    preflight_steps: Tuple[WingReadonlyPreflightPlanStep, ...] = ()
    if invalid:
        steps: Tuple[WingProofPlanStep, ...] = ()
        status = "blocked_by_invalid_artifacts"
        global_blockers = ["invalid_proof_summaries_present"]
    elif not preflight_candidate:
        steps = ()
        status = "blocked_by_readonly_preflight"
        preflight_steps = (
            _readonly_preflight_step(
                backend_url=backend_url,
                artifact_root=readonly_preflight_artifact_root,
                blockers=tuple(_string_list(preflight.get("blocked_by"))),
                next_actions=tuple(_string_list(preflight.get("next_required_actions"))),
            ),
        )
        global_blockers = [
            READONLY_PREFLIGHT_CANDIDATE_BLOCKER,
            *_string_list(preflight.get("blocked_by")),
        ]
    elif not review_decision_accepted:
        steps = ()
        status = "manual_review_ready"
        review_blockers = _string_list(review_summary.get("blocked_by"))
        if accepted_source_ids and not candidate_prestart_contract_valid:
            review_blockers = [
                READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER,
                *review_blockers,
            ]
        global_blockers = [
            READONLY_PREFLIGHT_REVIEW_DECISION_BLOCKER,
            *review_blockers,
        ]
    else:
        steps = _build_steps(
            promotion_evidence_snapshot,
            backend_url=backend_url,
            artifact_root=artifact_root,
            channels_by_value_type=channels,
            delta_db_by_value_type=deltas,
        )
        status = "manual_review_ready" if not steps else "operator_proof_required"
        global_blockers = [
            "operator_must_approve_and_execute_manually",
            "proof_plan_does_not_contact_mixer",
            "safety_matrix_review_required_after_artifacts",
        ]

    return {
        "type": "supervised_wing_proof_plan",
        "generated_at": _utc_now(),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "execution_allowed_by_plan": False,
            "live_widening_allowed": False,
            "automatic_safety_update_allowed": False,
            "full_autonomous_ready": False,
            "operator_required": True,
        },
        "summary": {
            "step_count": len(steps),
            "planned_value_types": sorted({step.value_type for step in steps}),
            "planned_channels": [step.channel for step in steps],
            "readonly_preflight_required": True,
            "readonly_preflight_candidate_available": preflight_candidate,
            "readonly_preflight_evidence_status": preflight.get("evidence_status"),
            "readonly_preflight_blocked_by": _string_list(preflight.get("blocked_by")),
            "readonly_preflight_prestart_check_required": (
                preflight.get("prestart_backend_port_check_required") is True
            ),
            "readonly_preflight_prestart_check_must_pass_before_starting_backend": (
                preflight.get(
                    "prestart_backend_port_check_must_pass_before_starting_backend"
                )
                is True
            ),
            "readonly_preflight_prestart_check_command": _string_list(
                preflight.get("prestart_backend_port_check_command")
            ),
            "readonly_preflight_prestart_check_failure_detail_fields": _string_list(
                preflight.get("prestart_backend_port_check_failure_detail_fields")
            ),
            "readonly_preflight_next_required_actions": _string_list(
                preflight.get("next_required_actions")
            ),
            "readonly_preflight_step_count": len(preflight_steps),
            "readonly_preflight_review_decision_required": True,
            "readonly_preflight_review_decision_accepted": review_decision_accepted,
            "readonly_preflight_review_decision_status": _readonly_preflight_review_decision_status(review_decision),
            "readonly_preflight_review_decision_blocked_by": _string_list(
                review_summary.get("blocked_by")
            ),
            "readonly_preflight_review_decision_accepted_record_count": int(
                review_summary.get("accepted_record_count") or 0
            ),
            "readonly_preflight_review_decision_accepted_source_ids": _string_list(
                review_summary.get("accepted_source_ids")
            ),
            "readonly_preflight_review_decision_candidate_prestart_contract_valid": (
                candidate_prestart_contract_valid
            ),
            "readonly_preflight_review_decision_candidate_prestart_source_ids": (
                accepted_candidate_prestart_source_ids
            ),
            "readonly_preflight_review_decision_rejected_source_ids": _string_list(
                review_summary.get("rejected_source_ids")
            ),
            "readonly_preflight_review_decision_invalid_record_count": int(
                review_summary.get("invalid_record_count") or 0
            ),
            "readonly_preflight_review_decision_invalid_record_ids": _string_list(
                review_summary.get("invalid_record_ids")
            ),
            "readonly_preflight_review_decision_duplicate_source_ids": _string_list(
                review_summary.get("duplicate_source_ids")
            ),
            "readonly_preflight_review_decision_conflicting_source_ids": _string_list(
                review_summary.get("conflicting_source_ids")
            ),
            "readonly_preflight_review_decision_unreviewed_candidate_count": int(
                review_summary.get("unreviewed_candidate_count") or 0
            ),
            "global_blockers": global_blockers,
        },
        "readonly_preflight_steps": [step.to_dict() for step in preflight_steps],
        "steps": [step.to_dict() for step in steps],
    }
