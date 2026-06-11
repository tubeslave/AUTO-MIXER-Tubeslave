"""Read-only remediation plan for blocked supervised proof artifacts.

The plan turns proof-report blockers into operator-facing follow-up actions.
It may include command tuples for rerunning the existing physical proof runner,
but it never imports transport clients and never executes commands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple


SUPERVISED_PROOF_REMEDIATION_METHOD_ID = "supervised_proof_remediation_plan"
REMEDIATION_STATUSES = {
    "no_artifacts",
    "manual_remediation_required",
    "manual_review_ready",
}
DEFAULT_ARTIFACT_ROOT = "artifacts/wing_telemetry"
DEFAULT_BACKEND_URL = "ws://localhost:8765"
DEFAULT_DELTA_DB_BY_VALUE_TYPE = {
    "gain": -0.25,
    "fader": -0.25,
}


@dataclass(frozen=True)
class ProofRemediationAction:
    id: str
    action_type: str
    proof_id: str
    value_type: str
    channel: int
    delta_db: float | None
    artifact_path: str
    blockers: Tuple[str, ...]
    execution_allowed_by_plan: bool
    live_execution_required: bool
    operator_approval_required: bool
    command: Tuple[str, ...]
    rationale: str
    expected_resolution: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.execution_allowed_by_plan:
            raise ValueError("Proof remediation plans must not authorize execution")
        if self.action_type == "rerun_current_physical_schema":
            if not self.operator_approval_required:
                raise ValueError("Physical proof reruns require explicit operator approval")
            if self.delta_db is None or self.delta_db >= 0:
                raise ValueError("Physical proof reruns must use negative dB deltas")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in ("blockers", "command", "expected_resolution"):
            payload[key] = list(payload[key])
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _rows(report: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    value = report.get("rows")
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _invalid_summaries(report: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    value = report.get("invalid_summaries")
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_delta(value_type: str, row_delta: Any) -> float:
    delta = _coerce_float(row_delta)
    if delta is not None and delta < 0:
        return max(delta, -0.5)
    return DEFAULT_DELTA_DB_BY_VALUE_TYPE.get(value_type, -0.25)


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
        f"proof_remediation_{value_type}_schema_rerun",
    )


def _blocker_counts(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        blockers = row.get("blockers")
        if not isinstance(blockers, list):
            continue
        for blocker in blockers:
            key = _text(blocker)
            if key:
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _value_type_counts(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value_type = _text(row.get("value_type") or "unknown")
        counts[value_type] = counts.get(value_type, 0) + 1
    return dict(sorted(counts.items()))


def _legacy_action(
    row: Mapping[str, Any],
    *,
    backend_url: str,
    artifact_root: str,
) -> ProofRemediationAction:
    value_type = _text(row.get("value_type") or "gain")
    channel = _coerce_int(row.get("channel"))
    evidence = row.get("evidence")
    row_delta = evidence.get("delta_db") if isinstance(evidence, Mapping) else None
    delta_db = _safe_delta(value_type, row_delta)
    blockers = tuple(_text(item) for item in row.get("blockers", ()) if item)
    return ProofRemediationAction(
        id=f"rerun_{_text(row.get('proof_id') or 'legacy_proof')}",
        action_type="rerun_current_physical_schema",
        proof_id=_text(row.get("proof_id") or "unnamed_proof"),
        value_type=value_type,
        channel=channel,
        delta_db=delta_db,
        artifact_path=_text(row.get("artifact_path")),
        blockers=blockers,
        execution_allowed_by_plan=False,
        live_execution_required=True,
        operator_approval_required=True,
        command=_proof_command(
            backend_url=backend_url,
            artifact_root=artifact_root,
            channel=channel,
            value_type=value_type,
            delta_db=delta_db,
        ),
        rationale=(
            "Historical WING proof artifact lacks the current explicit proof schema; "
            "rerun a bounded physical proof with current schema before promotion review."
        ),
        expected_resolution=(
            "New summary.json has proof_artifact_schema=wing_operator_proposal_validation.v2.",
            "New row evaluates as proof_mode=physical with physical pre-read, apply readback, and rollback readback.",
            "Old legacy artifact can remain as audit history but should no longer drive readiness.",
        ),
    )


def _blocked_review_action(row: Mapping[str, Any]) -> ProofRemediationAction:
    blockers = tuple(_text(item) for item in row.get("blockers", ()) if item)
    return ProofRemediationAction(
        id=f"review_{_text(row.get('proof_id') or 'blocked_proof')}",
        action_type="manual_blocker_review",
        proof_id=_text(row.get("proof_id") or "unnamed_proof"),
        value_type=_text(row.get("value_type") or "unknown"),
        channel=_coerce_int(row.get("channel")),
        delta_db=None,
        artifact_path=_text(row.get("artifact_path")),
        blockers=blockers,
        execution_allowed_by_plan=False,
        live_execution_required=False,
        operator_approval_required=False,
        command=(),
        rationale="Blocked proof artifact requires manual review before rerun planning.",
        expected_resolution=(
            "Resolve missing proof metadata or rerun the proof with current schema.",
            "Do not change safety coverage rows while this blocker is present.",
        ),
    )


def _invalid_action(index: int, summary: Mapping[str, Any]) -> ProofRemediationAction:
    reason = _text(summary.get("reason") or "invalid_summary")
    return ProofRemediationAction(
        id=f"fix_invalid_summary_{index}",
        action_type="fix_or_remove_invalid_summary",
        proof_id="invalid_summary",
        value_type="unknown",
        channel=0,
        delta_db=None,
        artifact_path=_text(summary.get("artifact_path")),
        blockers=(reason,),
        execution_allowed_by_plan=False,
        live_execution_required=False,
        operator_approval_required=False,
        command=(),
        rationale="Invalid JSON or non-object proof summary blocks trustworthy promotion review.",
        expected_resolution=(
            "Repair the summary if it is a proof artifact.",
            "Move unrelated corrupt output outside the proof artifact root.",
            "Regenerate the proof report after cleanup.",
        ),
    )


def _build_actions(
    report: Mapping[str, Any],
    *,
    backend_url: str,
    artifact_root: str,
    max_actions: int,
) -> Tuple[ProofRemediationAction, ...]:
    actions: list[ProofRemediationAction] = []
    for row in _rows(report):
        if row.get("status") != "blocked":
            continue
        blockers = set(_text(item) for item in row.get("blockers", ()) if item)
        if "legacy_physical_schema_missing" in blockers:
            actions.append(_legacy_action(row, backend_url=backend_url, artifact_root=artifact_root))
        else:
            actions.append(_blocked_review_action(row))
        if len(actions) >= max_actions:
            return tuple(actions)

    for index, summary in enumerate(_invalid_summaries(report), start=1):
        actions.append(_invalid_action(index, summary))
        if len(actions) >= max_actions:
            break
    return tuple(actions)


def build_supervised_proof_remediation_plan(
    proof_report: Mapping[str, Any],
    *,
    backend_url: str = DEFAULT_BACKEND_URL,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    max_actions: int = 25,
) -> Dict[str, Any]:
    """Build a non-executing remediation plan from a proof artifact report."""

    max_action_count = max(0, int(max_actions))
    rows = _rows(proof_report)
    blocked_rows = tuple(row for row in rows if row.get("status") == "blocked")
    invalid = _invalid_summaries(proof_report)
    actions = _build_actions(
        proof_report,
        backend_url=backend_url,
        artifact_root=artifact_root,
        max_actions=max_action_count,
    )
    total_remediations = len(blocked_rows) + len(invalid)
    if not rows and not invalid:
        status = "no_artifacts"
    elif actions:
        status = "manual_remediation_required"
    else:
        status = "manual_review_ready"

    return {
        "type": "supervised_proof_remediation_plan",
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
            "blocked_artifact_count": len(blocked_rows),
            "invalid_summary_count": len(invalid),
            "action_count": len(actions),
            "action_count_truncated": total_remediations > len(actions),
            "legacy_ambiguous_count": sum(1 for row in rows if row.get("proof_mode") == "legacy_ambiguous"),
            "blocker_counts": _blocker_counts(blocked_rows),
            "value_type_counts": _value_type_counts(blocked_rows),
            "global_blockers": [
                "proof_remediation_plan_does_not_contact_mixer",
                "operator_must_execute_reruns_manually",
                "safety_matrix_review_required_after_new_artifacts",
            ],
        },
        "actions": [action.to_dict() for action in actions],
    }
