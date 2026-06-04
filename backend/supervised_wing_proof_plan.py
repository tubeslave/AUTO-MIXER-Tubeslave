"""Read-only plan for the next supervised WING proof run.

The plan is an operator checklist, not an executor. It may include commands for
the existing physical proof runner, but this module never imports transport
clients and never runs those commands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple


SUPERVISED_WING_PROOF_PLAN_METHOD_ID = "supervised_wing_proof_plan"
PLAN_STATUSES = {
    "operator_proof_required",
    "manual_review_ready",
    "blocked_by_invalid_artifacts",
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


def _row_by_value_type(evidence_snapshot: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(row.get("value_type")): row
        for row in _evidence_rows(evidence_snapshot)
        if row.get("value_type")
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
        "python",
        "tools/wing_operator_proposal_validation_runner.py",
        "--url",
        backend_url,
        "--channel",
        str(channel),
        "--value-type",
        value_type,
        "--delta-db",
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


def _common_prerequisites(value_type: str, blockers: Tuple[str, ...]) -> Tuple[str, ...]:
    prerequisites = [
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
                )
            )
    return tuple(steps)


def build_supervised_wing_proof_plan(
    promotion_evidence_snapshot: Mapping[str, Any],
    *,
    backend_url: str = DEFAULT_BACKEND_URL,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    channels_by_value_type: Mapping[str, Tuple[int, ...]] | None = None,
    delta_db_by_value_type: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Build a non-executing proof checklist from promotion evidence rows."""

    channels = dict(DEFAULT_CHANNELS_BY_VALUE_TYPE)
    if channels_by_value_type:
        channels.update({key: tuple(int(item) for item in value) for key, value in channels_by_value_type.items()})
    deltas = dict(DEFAULT_DELTA_DB_BY_VALUE_TYPE)
    if delta_db_by_value_type:
        deltas.update({key: float(value) for key, value in delta_db_by_value_type.items()})

    invalid = _invalid_artifacts_present(promotion_evidence_snapshot)
    if invalid:
        steps: Tuple[WingProofPlanStep, ...] = ()
        status = "blocked_by_invalid_artifacts"
    else:
        steps = _build_steps(
            promotion_evidence_snapshot,
            backend_url=backend_url,
            artifact_root=artifact_root,
            channels_by_value_type=channels,
            delta_db_by_value_type=deltas,
        )
        status = "manual_review_ready" if not steps else "operator_proof_required"

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
            "global_blockers": (
                ["invalid_proof_summaries_present"]
                if invalid
                else [
                    "operator_must_approve_and_execute_manually",
                    "proof_plan_does_not_contact_mixer",
                    "safety_matrix_review_required_after_artifacts",
                ]
            ),
        },
        "steps": [step.to_dict() for step in steps],
    }
