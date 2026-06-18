"""Read-only plan for the next supervised WING proof run.

The plan is an operator checklist, not an executor. It may include commands for
the existing physical proof runner, but this module never imports transport
clients and never runs those commands.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple
from urllib.parse import urlparse

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
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_METHOD_ID,
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_SOURCE_AUTHORITY_FLAGS,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.wing_readonly_preflight_review_decision import (
        build_wing_readonly_preflight_review_decision_ledger,
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_METHOD_ID,
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_SOURCE_AUTHORITY_FLAGS,
    )

try:
    from wing_readonly_preflight_contract import (
        WING_READONLY_PREFLIGHT_ALLOWED_SUPERVISED_LIVE_VALUE_TYPES,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
        WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER,
        WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_ARTIFACT_ROOT_BLOCKER,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_REQUEST_SEQUENCE_BLOCKER,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_VERIFIER_WS_PROOF_BLOCKER,
        WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
        wing_readonly_preflight_accepted_candidate_artifact_root_contract,
        wing_readonly_preflight_accepted_candidate_request_sequence_source_ids,
        wing_readonly_preflight_accepted_candidate_request_sequence_violations_by_source,
        wing_readonly_preflight_accepted_candidate_verifier_ws_proof_source_ids,
        wing_readonly_preflight_accepted_candidate_verifier_ws_proof_violations_by_source,
        wing_readonly_preflight_accepted_candidate_prestart_violations_by_source,
        wing_readonly_preflight_accepted_candidate_prestart_source_ids,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.wing_readonly_preflight_contract import (
        WING_READONLY_PREFLIGHT_ALLOWED_SUPERVISED_LIVE_VALUE_TYPES,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
        WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER,
        WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_ARTIFACT_ROOT_BLOCKER,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_REQUEST_SEQUENCE_BLOCKER,
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_VERIFIER_WS_PROOF_BLOCKER,
        WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
        wing_readonly_preflight_accepted_candidate_artifact_root_contract,
        wing_readonly_preflight_accepted_candidate_request_sequence_source_ids,
        wing_readonly_preflight_accepted_candidate_request_sequence_violations_by_source,
        wing_readonly_preflight_accepted_candidate_verifier_ws_proof_source_ids,
        wing_readonly_preflight_accepted_candidate_verifier_ws_proof_violations_by_source,
        wing_readonly_preflight_accepted_candidate_prestart_violations_by_source,
        wing_readonly_preflight_accepted_candidate_prestart_source_ids,
    )


SUPERVISED_WING_PROOF_PLAN_METHOD_ID = "supervised_wing_proof_plan"
READONLY_PREFLIGHT_CANDIDATE_BLOCKER = (
    "readonly_preflight_candidate_required_before_supervised_proof"
)
READONLY_PREFLIGHT_REVIEW_DECISION_BLOCKER = (
    "accepted_readonly_preflight_review_decision_required_before_supervised_proof"
)
READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER = (
    WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER
)
READONLY_PREFLIGHT_REVIEW_CANDIDATE_ARTIFACT_ROOT_BLOCKER = (
    WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_ARTIFACT_ROOT_BLOCKER
)
READONLY_PREFLIGHT_REVIEW_CANDIDATE_REQUEST_SEQUENCE_BLOCKER = (
    WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_REQUEST_SEQUENCE_BLOCKER
)
READONLY_PREFLIGHT_REVIEW_CANDIDATE_VERIFIER_WS_PROOF_BLOCKER = (
    WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_VERIFIER_WS_PROOF_BLOCKER
)
READONLY_PREFLIGHT_REVIEW_CANDIDATE_EXPECTED_ARTIFACT_TRACE_BLOCKER = (
    "accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof"
)
READONLY_PREFLIGHT_REVIEW_CANDIDATE_CONTACT_TRACE_BLOCKER = (
    "accepted_readonly_preflight_candidate_contact_trace_required_before_supervised_proof"
)
READONLY_PREFLIGHT_EVIDENCE_AUTHORITY_FLAGS = (
    "readiness_closure_allowed_by_evidence",
    "live_authority_granted",
    "full_autonomous_ready",
)
READONLY_PREFLIGHT_REVIEW_DECISION_SUMMARY_AUTHORITY_FLAGS = (
    "readiness_matrix_update_allowed_by_decision",
    "readiness_closure_allowed_by_decision",
    "live_authority_granted",
    "full_autonomous_ready",
)
READONLY_PREFLIGHT_REVIEW_DECISION_SOURCE_AUTHORITY_FLAGS = (
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_SOURCE_AUTHORITY_FLAGS
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
MAX_SUPERVISED_PROOF_DELTA_DB = 0.5
MAX_SUPERVISED_PROOF_CHANNEL = 48
ALLOWED_SUPERVISED_WING_PROOF_VALUE_TYPES = frozenset(
    WING_READONLY_PREFLIGHT_ALLOWED_SUPERVISED_LIVE_VALUE_TYPES
)
DEFAULT_ARTIFACT_ROOT = "artifacts/wing_telemetry"
DEFAULT_READONLY_PREFLIGHT_ARTIFACT_ROOT = "artifacts/wing_readonly_preflight"
DEFAULT_READONLY_PREFLIGHT_EXPECTED_PLAN = WING_READONLY_PREFLIGHT_PLAN_OUTPUT
DEFAULT_READONLY_PREFLIGHT_PRESTART_SUMMARY = (
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
)
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
        if self.live_execution_required is not True:
            raise ValueError("Physical proof steps must be marked as live execution")
        if self.execution_allowed_by_plan is not False:
            raise ValueError("The proof plan must not authorize execution")
        if self.operator_approval_required is not True:
            raise ValueError("Physical proof steps require explicit operator approval")
        if self.value_type not in ALLOWED_SUPERVISED_WING_PROOF_VALUE_TYPES:
            raise ValueError(
                "Physical proof plan only supports supervised WING proof value types"
            )
        try:
            channel = int(self.channel)
        except (TypeError, ValueError):
            raise ValueError("Physical proof plan only supports WING channels 1..48") from None
        if (
            isinstance(self.channel, bool)
            or channel < 1
            or channel > MAX_SUPERVISED_PROOF_CHANNEL
            or self.channel != channel
        ):
            raise ValueError("Physical proof plan only supports WING channels 1..48")
        if self.id != f"{self.value_type}_physical_proof_ch{channel}":
            raise ValueError("Physical proof step id must match value type and channel")
        if not math.isfinite(self.delta_db):
            raise ValueError("Physical proof plan only supports finite dB deltas")
        if self.delta_db >= 0:
            raise ValueError("Physical proof plan only supports negative dB deltas")
        if abs(self.delta_db) > MAX_SUPERVISED_PROOF_DELTA_DB:
            raise ValueError("Physical proof plan only supports deltas up to 0.5 dB")
        command = tuple(self.command)
        proof_flags = (
            "--url",
            "--channel",
            "--operator-confirm-channel",
            "--value-type",
            "--operator-confirm-value-type",
            "--delta-db",
            "--operator-confirm-delta-db",
            "--execute",
            "--prove-blocked-apply",
            "--prove-emergency-stop",
            "--rollback-delay-sec",
            "--artifact-root",
            "--reason",
        )
        if _duplicate_command_flags(command, proof_flags):
            raise ValueError("Physical proof step command must not repeat proof flags")
        if _unknown_command_flags(command, proof_flags):
            raise ValueError("Physical proof step command must not include unknown flags")
        proof_value_flags = (
            "--url",
            "--channel",
            "--operator-confirm-channel",
            "--value-type",
            "--operator-confirm-value-type",
            "--delta-db",
            "--operator-confirm-delta-db",
            "--rollback-delay-sec",
            "--artifact-root",
            "--reason",
        )
        if _command_flags_with_flag_values(command, proof_value_flags):
            raise ValueError(
                "Physical proof step command must bind explicit proof flag values"
            )
        if _unexpected_command_arguments(command, proof_flags, proof_value_flags):
            raise ValueError(
                "Physical proof step command must not include stray arguments"
            )
        expected_delta = f"{self.delta_db:.2f}"
        if (
            _command_value(command, "--channel") != str(channel)
            or _command_value(command, "--operator-confirm-channel") != str(channel)
        ):
            raise ValueError(
                "Physical proof step command must bind matching channel confirmation"
            )
        if (
            _command_value(command, "--value-type") != self.value_type
            or _command_value(command, "--operator-confirm-value-type") != self.value_type
        ):
            raise ValueError(
                "Physical proof step command must bind matching value type confirmation"
            )
        if (
            _command_value(command, "--delta-db") != expected_delta
            or _command_value(command, "--operator-confirm-delta-db") != expected_delta
        ):
            raise ValueError(
                "Physical proof step command must bind matching delta confirmation"
            )
        for required_flag in (
            "--execute",
            "--prove-blocked-apply",
            "--prove-emergency-stop",
        ):
            if required_flag not in command:
                raise ValueError(
                    "Physical proof step command must include proof safety flags"
                )
        if command[:2] != (
            "python3",
            "tools/wing_operator_proposal_validation_runner.py",
        ):
            raise ValueError("Physical proof step command must use canonical proof runner")
        backend_url = _command_value(command, "--url")
        if not _is_ws_backend_url(backend_url):
            raise ValueError(
                "Physical proof step command must bind a WebSocket backend URL"
            )
        if _command_value(command, "--rollback-delay-sec") != "0":
            raise ValueError("Physical proof step command must bind zero rollback delay")
        artifact_root = _command_value(command, "--artifact-root")
        if not artifact_root:
            raise ValueError("Physical proof step command must bind artifact root")
        if not _is_artifacts_path(artifact_root):
            raise ValueError(
                "Physical proof step artifact root must stay under artifacts"
            )
        if _command_value(command, "--reason") != f"supervised_physical_{self.value_type}_proof":
            raise ValueError("Physical proof step command must bind supervised proof reason")
        if tuple(self.expected_artifacts) != _expected_artifacts(self.value_type, channel):
            raise ValueError(
                "Physical proof step expected artifacts must match value type and channel"
            )
        if tuple(self.abort_conditions) != _abort_conditions():
            raise ValueError("Physical proof step abort conditions must match safety contract")
        if tuple(self.prerequisites) != _common_prerequisites(
            self.value_type,
            tuple(self.evidence_blockers_to_review),
        ):
            raise ValueError("Physical proof step prerequisites must match safety contract")
        required_components = _required_evidence_components_by_value_type().get(
            self.value_type,
            (),
        )
        if tuple(self.required_evidence_components) != required_components:
            raise ValueError(
                "Physical proof step evidence components must match safety coverage matrix"
            )

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
    expected_plan_required_for_real_run: bool
    prestart_summary_required: bool
    expected_plan: str
    prestart_summary: str
    command: Tuple[str, ...]
    rationale: str
    prerequisites: Tuple[str, ...]
    abort_conditions: Tuple[str, ...]
    expected_artifacts: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.live_execution_required is not False:
            raise ValueError("Read-only preflight step must not be a live write step")
        if self.execution_allowed_by_plan is not False:
            raise ValueError("The proof plan must not authorize read-only preflight execution")
        if self.read_only is not True:
            raise ValueError("Read-only preflight step must be marked read-only")
        if self.operator_approval_required is not True:
            raise ValueError("Read-only WING preflight still requires operator presence")
        if self.expected_plan_required_for_real_run is not True:
            raise ValueError("Read-only preflight plan must require an expected plan")
        if self.prestart_summary_required is not True:
            raise ValueError("Read-only preflight plan must require a prestart summary")
        if self.id != "wing_readonly_preflight_required":
            raise ValueError("Read-only preflight step id must match safety contract")
        if tuple(self.prerequisites) != _readonly_preflight_prerequisites():
            raise ValueError(
                "Read-only preflight step prerequisites must match safety contract"
            )
        abort_conditions = tuple(self.abort_conditions)
        if abort_conditions[:3] != _readonly_preflight_base_abort_conditions():
            raise ValueError(
                "Read-only preflight step abort conditions must match safety contract"
            )
        if any(not str(condition).strip() for condition in abort_conditions[3:]):
            raise ValueError(
                "Read-only preflight step abort blockers must be explicit"
            )
        expected_artifacts = tuple(self.expected_artifacts)
        if expected_artifacts[:2] != _readonly_preflight_base_expected_artifacts():
            raise ValueError(
                "Read-only preflight step expected artifacts must match safety contract"
            )
        if any(not str(artifact).strip() for artifact in expected_artifacts[2:]):
            raise ValueError(
                "Read-only preflight step next-action artifacts must be explicit"
            )
        command = tuple(self.command)
        readonly_flags = (
            "--url",
            "--channels",
            "--operator-confirm-channels",
            "--operator-confirm-session-ready",
            "--value-types",
            "--require-physical-readback",
            "--artifact-root",
            "--prestart-summary",
            "--expected-plan",
        )
        if command[:2] != ("python3", "tools/wing_readonly_preflight_runner.py"):
            raise ValueError("Read-only preflight step must use canonical runner")
        if _duplicate_command_flags(command, readonly_flags):
            raise ValueError("Read-only preflight step command must not repeat flags")
        if _unknown_command_flags(command, readonly_flags):
            raise ValueError("Read-only preflight step command must not include unknown flags")
        readonly_value_flags = (
            "--url",
            "--channels",
            "--operator-confirm-channels",
            "--value-types",
            "--artifact-root",
            "--prestart-summary",
            "--expected-plan",
        )
        if _command_flags_with_flag_values(command, readonly_value_flags):
            raise ValueError(
                "Read-only preflight step command must bind explicit flag values"
            )
        if _unexpected_command_arguments(command, readonly_flags, readonly_value_flags):
            raise ValueError(
                "Read-only preflight step command must not include stray arguments"
            )
        if _command_value(command, "--expected-plan") != self.expected_plan:
            raise ValueError("Read-only preflight step must bind --expected-plan")
        if _command_value(command, "--prestart-summary") != self.prestart_summary:
            raise ValueError("Read-only preflight step must bind --prestart-summary")
        if not _is_artifacts_path(self.expected_plan):
            raise ValueError(
                "Read-only preflight step expected plan must stay under artifacts"
            )
        if not _is_artifacts_path(self.prestart_summary):
            raise ValueError(
                "Read-only preflight step prestart summary must stay under artifacts"
            )
        if "--require-physical-readback" not in command:
            raise ValueError(
                "Read-only preflight step must require physical readback"
            )
        backend_url = _command_value(command, "--url")
        if not _is_ws_backend_url(backend_url):
            raise ValueError(
                "Read-only preflight step must bind a WebSocket backend URL"
            )
        artifact_root = _command_value(command, "--artifact-root")
        if not artifact_root:
            raise ValueError("Read-only preflight step must bind artifact root")
        if not _is_artifacts_path(artifact_root):
            raise ValueError(
                "Read-only preflight step artifact root must stay under artifacts"
            )
        if _command_value(command, "--channels") != WING_READONLY_PREFLIGHT_RUNNER_CHANNELS:
            raise ValueError("Read-only preflight step must bind canonical --channels")
        if (
            _command_value(command, "--operator-confirm-channels")
            != WING_READONLY_PREFLIGHT_RUNNER_CHANNELS
        ):
            raise ValueError(
                "Read-only preflight step must require matching operator channel confirmation"
            )
        if "--operator-confirm-session-ready" not in command:
            raise ValueError(
                "Read-only preflight step must require session-ready operator confirmation"
            )
        if _command_value(command, "--value-types") != WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES:
            raise ValueError("Read-only preflight step must bind canonical --value-types")
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
    valid_candidate_count = _nonnegative_int_count(
        overlay.get("valid_candidate_count")
    )
    invalid_candidate_count = _nonnegative_int_count(
        overlay.get("invalid_candidate_count")
    )
    return (
        overlay.get("readiness_update_candidate_available") is True
        and valid_candidate_count is not None
        and valid_candidate_count > 0
        and invalid_candidate_count == 0
        and _readonly_preflight_evidence_denies_authority(overlay)
    )


def _readonly_preflight_evidence_denies_authority(overlay: Mapping[str, Any]) -> bool:
    return all(
        overlay.get(flag) is False
        for flag in READONLY_PREFLIGHT_EVIDENCE_AUTHORITY_FLAGS
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


def _readonly_preflight_review_decision_safety_allows_acceptance(
    overlay: Mapping[str, Any],
) -> bool:
    safety = overlay.get("safety")
    if not isinstance(safety, Mapping):
        return False
    required_false_flags = (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_decision",
        "readiness_matrix_update_allowed_by_decision",
        "readiness_closure_allowed_by_decision",
        "live_widening_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    )
    return (
        safety.get("read_only") is True
        and safety.get("operator_required") is True
        and all(safety.get(flag) is False for flag in required_false_flags)
    )


def _candidate_review_source_ids(summary: Mapping[str, Any]) -> list[str]:
    rows = summary.get("candidate_reviews")
    if not isinstance(rows, list):
        return []
    return [
        str(row.get("source_id"))
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("source_id"), str)
        and row.get("source_id")
    ]


def _review_decision_summary_denies_authority(summary: Mapping[str, Any]) -> bool:
    return all(
        summary.get(flag) is False
        for flag in READONLY_PREFLIGHT_REVIEW_DECISION_SUMMARY_AUTHORITY_FLAGS
    )


def _review_decision_source_authority_denied(summary: Mapping[str, Any]) -> bool:
    return (
        summary.get("source_authority_denied") is True
        and all(
            summary.get(flag) is False
            for flag in READONLY_PREFLIGHT_REVIEW_DECISION_SOURCE_AUTHORITY_FLAGS
        )
    )


def _readonly_preflight_review_decision_accepted(overlay: Mapping[str, Any]) -> bool:
    summary = _review_decision_summary(overlay)
    accepted_source_ids = _string_list(summary.get("accepted_source_ids"))
    rejected_source_ids = _string_list(summary.get("rejected_source_ids"))
    candidate_review_source_ids = _candidate_review_source_ids(summary)
    accepted_record_count = _nonnegative_int_count(
        summary.get("accepted_record_count")
    )
    rejected_record_count = _nonnegative_int_count(
        summary.get("rejected_record_count")
    )
    invalid_record_count = _nonnegative_int_count(summary.get("invalid_record_count"))
    invalid_record_ids = _string_list(summary.get("invalid_record_ids"))
    invalid_records = _invalid_review_decision_records(summary.get("invalid_records"))
    unreviewed_candidate_count = _nonnegative_int_count(
        summary.get("unreviewed_candidate_count")
    )
    unreviewed_candidate_source_ids = _string_list(
        summary.get("unreviewed_candidate_source_ids")
    )
    return (
        overlay.get("type") == "wing_readonly_preflight_review_decision_ledger"
        and overlay.get("method_id")
        == WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_METHOD_ID
        and overlay.get("source_method_id") == "wing_readonly_preflight_evidence"
        and _readonly_preflight_review_decision_safety_allows_acceptance(overlay)
        and _readonly_preflight_review_decision_status(overlay)
        == "operator_review_decision_records_available"
        and summary.get("manual_readiness_update_candidate_available") is True
        and accepted_record_count is not None
        and accepted_record_count > 0
        and rejected_record_count is not None
        and rejected_record_count == len(rejected_source_ids)
        and len(set(rejected_source_ids)) == len(rejected_source_ids)
        and all(_is_artifacts_path(source_id) for source_id in rejected_source_ids)
        and bool(accepted_source_ids)
        and accepted_record_count == len(accepted_source_ids)
        and len(set(accepted_source_ids)) == len(accepted_source_ids)
        and not set(accepted_source_ids).intersection(rejected_source_ids)
        and all(_is_artifacts_path(source_id) for source_id in accepted_source_ids)
        and all(
            candidate_review_source_ids.count(source_id) == 1
            for source_id in accepted_source_ids
        )
        and _accepted_candidate_prestart_source_ids(summary) == accepted_source_ids
        and _accepted_candidate_request_sequence_source_ids(summary) == accepted_source_ids
        and invalid_record_count == 0
        and not invalid_record_ids
        and not invalid_records
        and unreviewed_candidate_count == 0
        and not unreviewed_candidate_source_ids
        and not _string_list(summary.get("duplicate_source_ids"))
        and not _string_list(summary.get("conflicting_source_ids"))
        and _review_decision_summary_denies_authority(summary)
        and _review_decision_source_authority_denied(summary)
    )


def _readonly_preflight_review_decision_status(overlay: Mapping[str, Any]) -> Any:
    return overlay.get("status") or _review_decision_summary(overlay).get("ledger_status")


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value if item] if isinstance(value, list) else []


def _nonnegative_int_count(value: Any) -> int | None:
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    return None


def _invalid_review_decision_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping) or record.get("valid") is True:
            continue
        rows.append(
            {
                "review_id": str(record.get("review_id") or ""),
                "artifact_source_id": str(record.get("artifact_source_id") or ""),
                "decision": str(record.get("decision") or ""),
                "template_status": str(record.get("template_status") or ""),
                "violations": _string_list(record.get("violations")),
                "unsafe_authority_flags": _string_list(
                    record.get("unsafe_authority_flags")
                ),
            }
        )
    return rows


def _command_value(command: Tuple[str, ...], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def _is_ws_backend_url(value: str | None) -> bool:
    if not value:
        return False
    if value != value.strip():
        return False
    if any(char.isspace() for char in value):
        return False
    parsed = urlparse(value)
    try:
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in {"ws", "wss"}
        and bool(parsed.hostname)
        and port is not None
        and port > 0
        and not parsed.username
        and not parsed.password
        and not parsed.path
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
    )


def _is_artifacts_path(value: str | None) -> bool:
    if not value:
        return False
    if value != value.strip():
        return False
    if any(char.isspace() for char in value):
        return False
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    return parts[0] == "artifacts" and len(parts) > 1 and all(
        part and part == part.strip() and part not in {".", ".."}
        for part in parts[1:]
    )


def _duplicate_command_flags(command: Tuple[str, ...], flags: Tuple[str, ...]) -> Tuple[str, ...]:
    duplicates: list[str] = []
    for flag in flags:
        if command.count(flag) > 1:
            duplicates.append(flag)
    return tuple(duplicates)


def _unknown_command_flags(command: Tuple[str, ...], flags: Tuple[str, ...]) -> Tuple[str, ...]:
    allowed = set(flags)
    return tuple(token for token in command if token.startswith("--") and token not in allowed)


def _command_flags_with_flag_values(
    command: Tuple[str, ...], flags: Tuple[str, ...]
) -> Tuple[str, ...]:
    malformed: list[str] = []
    for flag in flags:
        if flag not in command:
            continue
        index = command.index(flag)
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            malformed.append(flag)
    return tuple(malformed)


def _unexpected_command_arguments(
    command: Tuple[str, ...],
    flags: Tuple[str, ...],
    value_flags: Tuple[str, ...],
) -> Tuple[str, ...]:
    allowed_flags = set(flags)
    flags_with_values = set(value_flags)
    unexpected: list[str] = []
    index = 2
    while index < len(command):
        token = command[index]
        if token not in allowed_flags:
            unexpected.append(token)
            index += 1
            continue
        index += 2 if token in flags_with_values else 1
    return tuple(unexpected)


def _unsupported_supervised_value_types(mapping: Mapping[str, Any] | None) -> list[str]:
    if not mapping:
        return []
    return sorted(
        str(key)
        for key in mapping
        if str(key) not in ALLOWED_SUPERVISED_WING_PROOF_VALUE_TYPES
    )


def _normalize_channel_overrides(
    channels_by_value_type: Mapping[str, Tuple[int, ...]] | None,
) -> dict[str, Tuple[int, ...]]:
    if not channels_by_value_type:
        return {}
    normalized: dict[str, Tuple[int, ...]] = {}
    for value_type, raw_channels in channels_by_value_type.items():
        if isinstance(raw_channels, (str, bytes)):
            raise ValueError(
                f"Invalid supervised WING proof channels for {value_type}: {raw_channels}"
            )
        try:
            raw_channel_values = tuple(raw_channels)
        except TypeError:
            raise ValueError(
                f"Invalid supervised WING proof channels for {value_type}: {raw_channels}"
            ) from None
        if not raw_channel_values:
            raise ValueError(
                f"Invalid supervised WING proof channels for {value_type}: empty"
            )
        channels: list[int] = []
        invalid_channels: list[str] = []
        duplicate_channels: list[str] = []
        seen_channels: set[int] = set()
        for raw_channel in raw_channel_values:
            if isinstance(raw_channel, bool):
                invalid_channels.append(str(raw_channel))
                continue
            try:
                channel = int(raw_channel)
            except (TypeError, ValueError):
                invalid_channels.append(str(raw_channel))
                continue
            if (
                channel < 1
                or channel > MAX_SUPERVISED_PROOF_CHANNEL
                or raw_channel != channel
            ):
                invalid_channels.append(str(raw_channel))
                continue
            if channel in seen_channels:
                duplicate_channels.append(str(channel))
                continue
            seen_channels.add(channel)
            channels.append(channel)
        if invalid_channels:
            raise ValueError(
                f"Invalid supervised WING proof channels for {value_type}: "
                + ", ".join(invalid_channels)
            )
        if duplicate_channels:
            raise ValueError(
                f"Duplicate supervised WING proof channels for {value_type}: "
                + ", ".join(duplicate_channels)
            )
        normalized[str(value_type)] = tuple(channels)
    return normalized


def _normalize_delta_overrides(
    delta_db_by_value_type: Mapping[str, float] | None,
) -> dict[str, float]:
    if not delta_db_by_value_type:
        return {}
    normalized: dict[str, float] = {}
    for value_type, raw_delta in delta_db_by_value_type.items():
        if isinstance(raw_delta, bool):
            raise ValueError(
                f"Invalid supervised WING proof delta for {value_type}: {raw_delta}"
            )
        try:
            delta_db = float(raw_delta)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid supervised WING proof delta for {value_type}: {raw_delta}"
            ) from None
        if (
            not math.isfinite(delta_db)
            or delta_db >= 0
            or abs(delta_db) > MAX_SUPERVISED_PROOF_DELTA_DB
        ):
            raise ValueError(
                f"Invalid supervised WING proof delta for {value_type}: {raw_delta}"
            )
        normalized[str(value_type)] = delta_db
    return normalized


def _accepted_candidate_prestart_source_ids(summary: Mapping[str, Any]) -> list[str]:
    accepted_source_ids = _string_list(summary.get("accepted_source_ids"))
    if not accepted_source_ids:
        return []
    rows = summary.get("candidate_reviews")
    if not isinstance(rows, list):
        return []
    return wing_readonly_preflight_accepted_candidate_prestart_source_ids(
        accepted_source_ids,
        tuple(row for row in rows if isinstance(row, Mapping)),
    )


def _accepted_candidate_prestart_violations_by_source(
    summary: Mapping[str, Any],
) -> dict[str, list[str]]:
    accepted_source_ids = _string_list(summary.get("accepted_source_ids"))
    if not accepted_source_ids:
        return {}
    rows = summary.get("candidate_reviews")
    if isinstance(rows, list):
        return wing_readonly_preflight_accepted_candidate_prestart_violations_by_source(
            accepted_source_ids,
            tuple(row for row in rows if isinstance(row, Mapping)),
        )
    declared_violations = summary.get(
        "accepted_candidate_prestart_contract_violations_by_source"
    )
    if isinstance(declared_violations, Mapping):
        return {
            str(source_id): _string_list(violations)
            for source_id, violations in declared_violations.items()
            if source_id
        }
    return {source_id: ["candidate_reviews_missing"] for source_id in accepted_source_ids}


def _accepted_candidate_verifier_ws_proof_source_ids(
    summary: Mapping[str, Any],
) -> list[str]:
    accepted_source_ids = _string_list(summary.get("accepted_source_ids"))
    if not accepted_source_ids:
        return []
    rows = summary.get("candidate_reviews")
    if not isinstance(rows, list):
        return []
    return wing_readonly_preflight_accepted_candidate_verifier_ws_proof_source_ids(
        accepted_source_ids,
        tuple(row for row in rows if isinstance(row, Mapping)),
    )


def _accepted_candidate_verifier_ws_proof_violations_by_source(
    summary: Mapping[str, Any],
) -> dict[str, list[str]]:
    accepted_source_ids = _string_list(summary.get("accepted_source_ids"))
    if not accepted_source_ids:
        return {}
    rows = summary.get("candidate_reviews")
    if not isinstance(rows, list):
        return {source_id: ["candidate_reviews_missing"] for source_id in accepted_source_ids}
    return wing_readonly_preflight_accepted_candidate_verifier_ws_proof_violations_by_source(
        accepted_source_ids,
        tuple(row for row in rows if isinstance(row, Mapping)),
    )


def _accepted_candidate_request_sequence_source_ids(
    summary: Mapping[str, Any],
) -> list[str]:
    accepted_source_ids = _string_list(summary.get("accepted_source_ids"))
    if not accepted_source_ids:
        return []
    rows = summary.get("candidate_reviews")
    if isinstance(rows, list):
        return wing_readonly_preflight_accepted_candidate_request_sequence_source_ids(
            accepted_source_ids,
            tuple(row for row in rows if isinstance(row, Mapping)),
        )
    return _string_list(summary.get("accepted_candidate_request_sequence_source_ids"))


def _accepted_candidate_request_sequence_violations_by_source(
    summary: Mapping[str, Any],
) -> dict[str, list[str]]:
    accepted_source_ids = _string_list(summary.get("accepted_source_ids"))
    if not accepted_source_ids:
        return {}
    rows = summary.get("candidate_reviews")
    if isinstance(rows, list):
        return (
            wing_readonly_preflight_accepted_candidate_request_sequence_violations_by_source(
                accepted_source_ids,
                tuple(row for row in rows if isinstance(row, Mapping)),
            )
        )
    violations = summary.get(
        "accepted_candidate_request_sequence_contract_violations_by_source"
    )
    if isinstance(violations, Mapping):
        return {
            str(source_id): _string_list(source_violations)
            for source_id, source_violations in violations.items()
        }
    return {source_id: ["candidate_reviews_missing"] for source_id in accepted_source_ids}


def _accepted_candidate_artifact_root_contract(
    summary: Mapping[str, Any],
) -> tuple[list[str], dict[str, list[str]]]:
    accepted_source_ids = _string_list(summary.get("accepted_source_ids"))
    if not accepted_source_ids:
        return [], {}
    rows = summary.get("candidate_reviews")
    if not isinstance(rows, list):
        return [], {
            source_id: ["candidate_reviews_missing"]
            for source_id in accepted_source_ids
        }
    return wing_readonly_preflight_accepted_candidate_artifact_root_contract(
        accepted_source_ids,
        tuple(row for row in rows if isinstance(row, Mapping)),
    )


def _accepted_candidate_expected_artifact_trace(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    accepted_source_ids = _string_list(summary.get("accepted_source_ids"))
    rows = summary.get("candidate_reviews")
    candidate_rows = (
        tuple(row for row in rows if isinstance(row, Mapping))
        if isinstance(rows, list)
        else ()
    )
    rows_by_source: dict[str, Mapping[str, Any]] = {
        str(row.get("source_id")): row
        for row in candidate_rows
        if isinstance(row.get("source_id"), str) and row.get("source_id")
    }
    expected_artifacts_by_source: dict[str, list[str]] = {}
    expected_summary_artifact_by_source: dict[str, str] = {}
    expected_events_artifact_by_source: dict[str, str] = {}
    summary_path_matches_by_source: dict[str, bool] = {}
    events_path_matches_by_source: dict[str, bool] = {}
    missing_source_ids: list[str] = []

    for source_id in accepted_source_ids:
        row = rows_by_source.get(source_id)
        if row is None:
            missing_source_ids.append(source_id)
            continue
        expected_artifacts = _string_list(row.get("expected_artifacts"))
        expected_summary_artifact = str(row.get("expected_summary_artifact") or "")
        expected_events_artifact = str(row.get("expected_events_artifact") or "")
        summary_path_matches = row.get("summary_path_matches_expected_artifact") is True
        events_path_matches = row.get("events_path_matches_expected_artifact") is True
        expected_artifacts_by_source[source_id] = expected_artifacts
        expected_summary_artifact_by_source[source_id] = expected_summary_artifact
        expected_events_artifact_by_source[source_id] = expected_events_artifact
        summary_path_matches_by_source[source_id] = summary_path_matches
        events_path_matches_by_source[source_id] = events_path_matches
        if (
            len(expected_artifacts) < 2
            or not expected_summary_artifact
            or not expected_events_artifact
            or not summary_path_matches
            or not events_path_matches
        ):
            missing_source_ids.append(source_id)

    return {
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids": (
            accepted_source_ids
        ),
        "readonly_preflight_review_decision_accepted_candidate_expected_artifacts_by_source": (
            expected_artifacts_by_source
        ),
        "readonly_preflight_review_decision_accepted_candidate_expected_summary_artifact_by_source": (
            expected_summary_artifact_by_source
        ),
        "readonly_preflight_review_decision_accepted_candidate_expected_events_artifact_by_source": (
            expected_events_artifact_by_source
        ),
        "readonly_preflight_review_decision_accepted_candidate_summary_path_matches_expected_artifact_by_source": (
            summary_path_matches_by_source
        ),
        "readonly_preflight_review_decision_accepted_candidate_events_path_matches_expected_artifact_by_source": (
            events_path_matches_by_source
        ),
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids": (
            missing_source_ids
        ),
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete": (
            bool(accepted_source_ids) and not missing_source_ids
        ),
    }


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _accepted_candidate_request_sequence_trace(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    accepted_source_ids = _string_list(summary.get("accepted_source_ids"))
    rows = summary.get("candidate_reviews")
    candidate_rows = (
        tuple(row for row in rows if isinstance(row, Mapping))
        if isinstance(rows, list)
        else ()
    )
    rows_by_source: dict[str, Mapping[str, Any]] = {
        str(row.get("source_id")): row
        for row in candidate_rows
        if isinstance(row.get("source_id"), str) and row.get("source_id")
    }
    request_sequence_by_source: dict[str, list[dict[str, Any]]] = {}
    plan_request_sequence_by_source: dict[str, list[dict[str, Any]]] = {}
    matches_plan_by_source: dict[str, bool] = {}
    missing_source_ids: list[str] = []

    for source_id in accepted_source_ids:
        row = rows_by_source.get(source_id)
        if row is None:
            missing_source_ids.append(source_id)
            continue
        request_sequence = _mapping_list(row.get("request_sequence"))
        plan_request_sequence = _mapping_list(row.get("plan_request_sequence"))
        matches_plan = row.get("request_sequence_matches_plan") is True
        request_sequence_by_source[source_id] = request_sequence
        plan_request_sequence_by_source[source_id] = plan_request_sequence
        matches_plan_by_source[source_id] = matches_plan
        if not request_sequence or not plan_request_sequence or not matches_plan:
            missing_source_ids.append(source_id)

    return {
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_source_ids": (
            accepted_source_ids
        ),
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_by_source": (
            request_sequence_by_source
        ),
        "readonly_preflight_review_decision_accepted_candidate_plan_request_sequence_by_source": (
            plan_request_sequence_by_source
        ),
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_matches_plan_by_source": (
            matches_plan_by_source
        ),
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids": (
            missing_source_ids
        ),
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete": (
            bool(accepted_source_ids) and not missing_source_ids
        ),
    }


def _accepted_candidate_contact_trace(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    accepted_source_ids = _string_list(summary.get("accepted_source_ids"))
    rows = summary.get("candidate_reviews")
    candidate_rows = (
        tuple(row for row in rows if isinstance(row, Mapping))
        if isinstance(rows, list)
        else ()
    )
    rows_by_source: dict[str, Mapping[str, Any]] = {
        str(row.get("source_id")): row
        for row in candidate_rows
        if isinstance(row.get("source_id"), str) and row.get("source_id")
    }
    contacts_backend_by_source: dict[str, bool] = {}
    queries_console_by_source: dict[str, bool] = {}
    missing_source_ids: list[str] = []

    for source_id in accepted_source_ids:
        row = rows_by_source.get(source_id)
        if row is None:
            missing_source_ids.append(source_id)
            continue
        contacts_backend = row.get("contacts_backend") is True
        queries_console = row.get("queries_console") is True
        contacts_backend_by_source[source_id] = contacts_backend
        queries_console_by_source[source_id] = queries_console
        if not contacts_backend or not queries_console:
            missing_source_ids.append(source_id)

    return {
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_source_ids": (
            accepted_source_ids
        ),
        "readonly_preflight_review_decision_accepted_candidate_contacts_backend_by_source": (
            contacts_backend_by_source
        ),
        "readonly_preflight_review_decision_accepted_candidate_queries_console_by_source": (
            queries_console_by_source
        ),
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids": (
            missing_source_ids
        ),
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete": (
            bool(accepted_source_ids) and not missing_source_ids
        ),
    }


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
    expected_plan: str,
    prestart_summary: str,
) -> Tuple[str, ...]:
    return (
        "python3",
        "tools/wing_readonly_preflight_runner.py",
        "--url",
        backend_url,
        "--channels",
        WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "--operator-confirm-channels",
        WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "--operator-confirm-session-ready",
        "--value-types",
        WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
        "--require-physical-readback",
        "--artifact-root",
        artifact_root,
        "--prestart-summary",
        prestart_summary,
        "--expected-plan",
        expected_plan,
    )


def _readonly_preflight_step(
    *,
    backend_url: str,
    artifact_root: str,
    expected_plan: str,
    prestart_summary: str,
    blockers: Tuple[str, ...],
    next_actions: Tuple[str, ...],
) -> WingReadonlyPreflightPlanStep:
    return WingReadonlyPreflightPlanStep(
        id="wing_readonly_preflight_required",
        live_execution_required=False,
        execution_allowed_by_plan=False,
        operator_approval_required=True,
        read_only=True,
        expected_plan_required_for_real_run=True,
        prestart_summary_required=True,
        expected_plan=expected_plan,
        prestart_summary=prestart_summary,
        command=_readonly_preflight_command(
            backend_url=backend_url,
            artifact_root=artifact_root,
            expected_plan=expected_plan,
            prestart_summary=prestart_summary,
        ),
        rationale=(
            "Read-only WING connection and physical readback evidence is required before "
            "any supervised fader/gain proof step can be planned."
        ),
        prerequisites=_readonly_preflight_prerequisites(),
        abort_conditions=(
            *_readonly_preflight_base_abort_conditions(),
            *blockers,
        ),
        expected_artifacts=(
            *_readonly_preflight_base_expected_artifacts(),
            *next_actions,
        ),
    )


def _readonly_preflight_prerequisites() -> Tuple[str, ...]:
    return (
        "Operator is physically present at the intended WING Rack.",
        "Backend is already connected in WING mode.",
        "No supervised apply, proposal, rollback, emergency-stop, connect, or disconnect request is sent by this step.",
        "Move or replace invalid preflight artifacts before trusting a readiness candidate.",
    )


def _readonly_preflight_base_abort_conditions() -> Tuple[str, ...]:
    return (
        "Backend is not connected to the intended WING Rack.",
        "Any response is not verified from physical readback.",
        "Any forbidden write, proposal, rollback, emergency, connect, or disconnect request is observed.",
    )


def _readonly_preflight_base_expected_artifacts() -> Tuple[str, ...]:
    return (
        WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT,
        (
            f"{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME} with "
            "get_operator_mode/get_state/query_mixer_parameter only"
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
    readonly_preflight_expected_plan: str = DEFAULT_READONLY_PREFLIGHT_EXPECTED_PLAN,
    readonly_preflight_prestart_summary: str = DEFAULT_READONLY_PREFLIGHT_PRESTART_SUMMARY,
) -> Dict[str, Any]:
    """Build a non-executing proof checklist from promotion evidence rows."""

    unsupported_channels = _unsupported_supervised_value_types(channels_by_value_type)
    if unsupported_channels:
        raise ValueError(
            "Unsupported supervised WING proof channel value types: "
            + ", ".join(unsupported_channels)
        )
    unsupported_deltas = _unsupported_supervised_value_types(delta_db_by_value_type)
    if unsupported_deltas:
        raise ValueError(
            "Unsupported supervised WING proof delta value types: "
            + ", ".join(unsupported_deltas)
        )

    channels = dict(DEFAULT_CHANNELS_BY_VALUE_TYPE)
    if channels_by_value_type:
        channels.update(_normalize_channel_overrides(channels_by_value_type))
    deltas = dict(DEFAULT_DELTA_DB_BY_VALUE_TYPE)
    if delta_db_by_value_type:
        deltas.update(_normalize_delta_overrides(delta_db_by_value_type))

    invalid = _invalid_artifacts_present(promotion_evidence_snapshot)
    preflight = _readonly_preflight_overlay(autonomy_readiness_snapshot)
    preflight_candidate = _readonly_preflight_candidate_available(preflight)
    review_decision = _readonly_preflight_review_decision_overlay(
        autonomy_readiness_snapshot,
        readonly_preflight_review_decision_ledger,
    )
    review_summary = _review_decision_summary(review_decision)
    review_invalid_records = _invalid_review_decision_records(
        review_summary.get("invalid_records")
    ) or _invalid_review_decision_records(review_decision.get("records"))
    accepted_candidate_prestart_source_ids = _accepted_candidate_prestart_source_ids(
        review_summary
    )
    accepted_source_ids = _string_list(review_summary.get("accepted_source_ids"))
    candidate_prestart_contract_valid = (
        bool(accepted_source_ids)
        and accepted_candidate_prestart_source_ids == accepted_source_ids
    )
    accepted_candidate_prestart_violations_by_source = (
        _accepted_candidate_prestart_violations_by_source(review_summary)
    )
    if accepted_candidate_prestart_violations_by_source:
        candidate_prestart_contract_valid = False
    accepted_candidate_verifier_ws_proof_source_ids = (
        _accepted_candidate_verifier_ws_proof_source_ids(review_summary)
    )
    accepted_candidate_verifier_ws_proof_violations_by_source = (
        _accepted_candidate_verifier_ws_proof_violations_by_source(
            review_summary
        )
    )
    accepted_candidate_verifier_ws_proof_complete = (
        bool(accepted_source_ids)
        and accepted_candidate_verifier_ws_proof_source_ids
        == accepted_source_ids
        and not accepted_candidate_verifier_ws_proof_violations_by_source
    )
    accepted_candidate_request_sequence_source_ids = (
        _accepted_candidate_request_sequence_source_ids(review_summary)
    )
    accepted_candidate_request_sequence_violations_by_source = (
        _accepted_candidate_request_sequence_violations_by_source(review_summary)
    )
    candidate_request_sequence_contract_valid = (
        bool(accepted_source_ids)
        and accepted_candidate_request_sequence_source_ids == accepted_source_ids
        and not accepted_candidate_request_sequence_violations_by_source
    )
    (
        accepted_candidate_artifact_roots,
        accepted_candidate_artifact_root_violations_by_source,
    ) = _accepted_candidate_artifact_root_contract(review_summary)
    accepted_candidate_expected_artifact_trace = (
        _accepted_candidate_expected_artifact_trace(review_summary)
    )
    accepted_candidate_expected_artifact_trace_complete = (
        accepted_candidate_expected_artifact_trace[
            "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete"
        ]
        is True
    )
    accepted_candidate_request_sequence_trace = (
        _accepted_candidate_request_sequence_trace(review_summary)
    )
    accepted_candidate_request_sequence_trace_complete = (
        accepted_candidate_request_sequence_trace[
            "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete"
        ]
        is True
    )
    accepted_candidate_contact_trace = _accepted_candidate_contact_trace(review_summary)
    accepted_candidate_contact_trace_complete = (
        accepted_candidate_contact_trace[
            "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete"
        ]
        is True
    )
    candidate_artifact_root_contract_valid = (
        bool(accepted_source_ids)
        and len(accepted_candidate_artifact_roots) == len(accepted_source_ids)
        and not accepted_candidate_artifact_root_violations_by_source
    )
    review_decision_accepted = (
        _readonly_preflight_review_decision_accepted(review_decision)
        and accepted_candidate_verifier_ws_proof_complete
        and candidate_request_sequence_contract_valid
        and candidate_artifact_root_contract_valid
        and accepted_candidate_expected_artifact_trace_complete
        and accepted_candidate_request_sequence_trace_complete
        and accepted_candidate_contact_trace_complete
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
                expected_plan=readonly_preflight_expected_plan,
                prestart_summary=readonly_preflight_prestart_summary,
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
        if accepted_source_ids and not candidate_artifact_root_contract_valid:
            review_blockers = [
                READONLY_PREFLIGHT_REVIEW_CANDIDATE_ARTIFACT_ROOT_BLOCKER,
                *review_blockers,
            ]
        if accepted_source_ids and not candidate_request_sequence_contract_valid:
            review_blockers = [
                READONLY_PREFLIGHT_REVIEW_CANDIDATE_REQUEST_SEQUENCE_BLOCKER,
                *review_blockers,
            ]
        if accepted_source_ids and not accepted_candidate_verifier_ws_proof_complete:
            review_blockers = [
                READONLY_PREFLIGHT_REVIEW_CANDIDATE_VERIFIER_WS_PROOF_BLOCKER,
                *review_blockers,
            ]
        if accepted_source_ids and not accepted_candidate_expected_artifact_trace_complete:
            review_blockers = [
                READONLY_PREFLIGHT_REVIEW_CANDIDATE_EXPECTED_ARTIFACT_TRACE_BLOCKER,
                *review_blockers,
            ]
        if accepted_source_ids and not accepted_candidate_contact_trace_complete:
            review_blockers = [
                READONLY_PREFLIGHT_REVIEW_CANDIDATE_CONTACT_TRACE_BLOCKER,
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
            "allowed_supervised_live_value_types": sorted(
                ALLOWED_SUPERVISED_WING_PROOF_VALUE_TYPES
            ),
            "step_count": len(steps),
            "planned_value_types": sorted({step.value_type for step in steps}),
            "planned_channels": [step.channel for step in steps],
            "readonly_preflight_required": True,
            "readonly_preflight_candidate_available": preflight_candidate,
            "readonly_preflight_evidence_status": preflight.get("evidence_status"),
            "readonly_preflight_evidence_authority_denied": (
                _readonly_preflight_evidence_denies_authority(preflight)
            ),
            "readonly_preflight_evidence_authority_flags": {
                flag: preflight.get(flag)
                for flag in READONLY_PREFLIGHT_EVIDENCE_AUTHORITY_FLAGS
            },
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
            "readonly_preflight_prestart_check_command_text": str(
                preflight.get("prestart_backend_port_check_command_text") or ""
            ),
            "readonly_preflight_prestart_check_failure_detail_fields": _string_list(
                preflight.get("prestart_backend_port_check_failure_detail_fields")
            ),
            "readonly_preflight_next_required_actions": _string_list(
                preflight.get("next_required_actions")
            ),
            "readonly_preflight_step_count": len(preflight_steps),
            "readonly_preflight_expected_plan_required_for_real_run": True,
            "readonly_preflight_prestart_summary_required": True,
            "readonly_preflight_command_channels": WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
            "readonly_preflight_command_operator_confirm_channels": (
                WING_READONLY_PREFLIGHT_RUNNER_CHANNELS
            ),
            "readonly_preflight_command_operator_confirm_session_ready": True,
            "readonly_preflight_command_value_types": (
                WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES
            ),
            "readonly_preflight_command_expected_plan": readonly_preflight_expected_plan,
            "readonly_preflight_command_prestart_summary": (
                readonly_preflight_prestart_summary
            ),
            "readonly_preflight_review_decision_required": True,
            "readonly_preflight_review_decision_accepted": review_decision_accepted,
            "readonly_preflight_review_decision_status": _readonly_preflight_review_decision_status(review_decision),
            "readonly_preflight_review_decision_blocked_by": _string_list(
                review_summary.get("blocked_by")
            ),
            "readonly_preflight_review_decision_accepted_record_count": (
                _nonnegative_int_count(review_summary.get("accepted_record_count"))
            ),
            "readonly_preflight_review_decision_accepted_source_ids": _string_list(
                review_summary.get("accepted_source_ids")
            ),
            "readonly_preflight_review_decision_candidate_review_source_ids": (
                _candidate_review_source_ids(review_summary)
            ),
            "readonly_preflight_review_decision_summary_authority_denied": (
                _review_decision_summary_denies_authority(review_summary)
            ),
            "readonly_preflight_review_decision_summary_authority_flags": {
                flag: review_summary.get(flag)
                for flag in READONLY_PREFLIGHT_REVIEW_DECISION_SUMMARY_AUTHORITY_FLAGS
            },
            "readonly_preflight_review_decision_source_authority_denied": (
                _review_decision_source_authority_denied(review_summary)
            ),
            "readonly_preflight_review_decision_source_authority_flags": {
                flag: review_summary.get(flag)
                for flag in READONLY_PREFLIGHT_REVIEW_DECISION_SOURCE_AUTHORITY_FLAGS
            },
            "readonly_preflight_review_decision_candidate_prestart_contract_valid": (
                candidate_prestart_contract_valid
            ),
            "readonly_preflight_review_decision_candidate_prestart_source_ids": (
                accepted_candidate_prestart_source_ids
            ),
            "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source": (
                accepted_candidate_prestart_violations_by_source
            ),
            "readonly_preflight_review_decision_expected_candidate_prestart_source_ids": (
                accepted_candidate_prestart_source_ids
            ),
            "readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source": (
                accepted_candidate_prestart_violations_by_source
            ),
            "readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids": (
                accepted_candidate_verifier_ws_proof_source_ids
            ),
            "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source": (
                accepted_candidate_verifier_ws_proof_violations_by_source
            ),
            "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete": (
                accepted_candidate_verifier_ws_proof_complete
            ),
            "readonly_preflight_review_decision_candidate_request_sequence_contract_valid": (
                candidate_request_sequence_contract_valid
            ),
            "readonly_preflight_review_decision_candidate_request_sequence_source_ids": (
                accepted_candidate_request_sequence_source_ids
            ),
            "readonly_preflight_review_decision_candidate_request_sequence_contract_violations_by_source": (
                accepted_candidate_request_sequence_violations_by_source
            ),
            "readonly_preflight_review_decision_expected_candidate_request_sequence_source_ids": (
                accepted_candidate_request_sequence_source_ids
            ),
            "readonly_preflight_review_decision_expected_candidate_request_sequence_contract_violations_by_source": (
                accepted_candidate_request_sequence_violations_by_source
            ),
            "readonly_preflight_review_decision_candidate_artifact_root_contract_valid": (
                candidate_artifact_root_contract_valid
            ),
            "readonly_preflight_review_decision_candidate_artifact_roots": (
                accepted_candidate_artifact_roots
            ),
            "readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source": (
                accepted_candidate_artifact_root_violations_by_source
            ),
            "readonly_preflight_review_decision_expected_candidate_artifact_roots": (
                accepted_candidate_artifact_roots
            ),
            "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source": (
                accepted_candidate_artifact_root_violations_by_source
            ),
            **accepted_candidate_expected_artifact_trace,
            **accepted_candidate_request_sequence_trace,
            **accepted_candidate_contact_trace,
            "readonly_preflight_review_decision_rejected_source_ids": _string_list(
                review_summary.get("rejected_source_ids")
            ),
            "readonly_preflight_review_decision_invalid_record_count": (
                _nonnegative_int_count(review_summary.get("invalid_record_count"))
            ),
            "readonly_preflight_review_decision_invalid_record_ids": _string_list(
                review_summary.get("invalid_record_ids")
            ),
            "readonly_preflight_review_decision_invalid_records": review_invalid_records,
            "readonly_preflight_review_decision_duplicate_source_ids": _string_list(
                review_summary.get("duplicate_source_ids")
            ),
            "readonly_preflight_review_decision_conflicting_source_ids": _string_list(
                review_summary.get("conflicting_source_ids")
            ),
            "readonly_preflight_review_decision_unreviewed_candidate_count": (
                _nonnegative_int_count(
                    review_summary.get("unreviewed_candidate_count")
                )
            ),
            "readonly_preflight_review_decision_unreviewed_candidate_source_ids": _string_list(
                review_summary.get("unreviewed_candidate_source_ids")
            ),
            "global_blockers": global_blockers,
        },
        "readonly_preflight_steps": [step.to_dict() for step in preflight_steps],
        "steps": [step.to_dict() for step in steps],
    }
