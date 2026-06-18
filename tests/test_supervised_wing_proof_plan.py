from pathlib import Path

import pytest

from backend.wing_readonly_preflight_contract import (
    WING_READONLY_PREFLIGHT_ALLOWED_SUPERVISED_LIVE_VALUE_TYPES,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE,
    WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT,
    WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_REQUEST_SEQUENCE_BLOCKER,
    WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
    WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
    WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME,
    wing_readonly_preflight_accepted_candidate_request_sequence_source_ids,
    wing_readonly_preflight_accepted_candidate_request_sequence_violations_by_source,
    wing_readonly_preflight_accepted_candidate_prestart_violations_by_source,
    wing_readonly_preflight_accepted_candidate_prestart_source_ids,
)
from backend.supervised_wing_proof_plan import (
    READONLY_PREFLIGHT_REVIEW_CANDIDATE_CONTACT_TRACE_BLOCKER,
    WingProofPlanStep,
    WingReadonlyPreflightPlanStep,
    build_supervised_wing_proof_plan,
)

PRESTART_BACKEND_PORT_CHECK_COMMAND = list(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
)
PRESTART_BACKEND_PORT_CHECK_COMMAND_TEXT = (
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
)
PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS = list(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
)
PRESTART_BACKEND_PORT_CHECK_REQUIRED_FALSE_DETAIL_FIELDS = list(
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
)
PRESTART_BACKEND_PORT_CHECK_REQUIRED_FALSE_PLAN_FIELDS = list(
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
)
PRESTART_BACKEND_PORT_CHECK_REQUIRED_FALSE_MATCH_FIELDS = list(
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS
)
READONLY_PREFLIGHT_EXPECTED_PLAN = WING_READONLY_PREFLIGHT_PLAN_OUTPUT
READONLY_PREFLIGHT_PRESTART_SUMMARY = (
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_PLACEHOLDER
)
READONLY_PREFLIGHT_CHANNELS = WING_READONLY_PREFLIGHT_RUNNER_CHANNELS
READONLY_PREFLIGHT_VALUE_TYPES = WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES
GAIN_REQUIRED_EVIDENCE_COMPONENTS = [
    "physical_pre_read",
    "bounded_gain_down_apply",
    "apply_readback",
    "rollback_readback",
    "emergency_stop_block",
    "unsafe_gain_up_followup_blocked",
    "gain_up_operator_authored_opt_in",
]


def proof_command(
    value_type="gain",
    channel=3,
    *,
    runner="tools/wing_operator_proposal_validation_runner.py",
    backend_url="ws://localhost:8765",
    delta="-0.25",
    artifact_root="artifacts/proof",
    operator_channel=None,
    include_safety_flags=True,
    tail=None,
):
    if operator_channel is None:
        operator_channel = channel
    command = (
        "python3",
        runner,
        "--url",
        backend_url,
        "--channel",
        str(channel),
        "--operator-confirm-channel",
        str(operator_channel),
        "--value-type",
        value_type,
        "--operator-confirm-value-type",
        value_type,
        "--delta-db",
        delta,
        "--operator-confirm-delta-db",
        delta,
    )
    if include_safety_flags:
        command += (
            "--execute",
            "--prove-blocked-apply",
            "--prove-emergency-stop",
        )
    if tail is None:
        tail = (
            "--rollback-delay-sec",
            "0",
            "--artifact-root",
            artifact_root,
            "--reason",
            f"supervised_physical_{value_type}_proof",
        )
    return command + tuple(tail)


def proof_expected_artifacts(value_type="gain", channel=3):
    return (
        f"summary.json with value_type={value_type} and channel={channel}",
        "events.jsonl with proposal/apply/rollback/emergency-stop sequence",
        "proof report row with proof_mode=physical and status=supervised_candidate",
        "promotion evidence row marked safety_review_candidate only after manual artifact review",
    )


def proof_abort_conditions():
    return (
        "Pre-read is unavailable or not from physical readback.",
        "Requested delta is positive or exceeds 0.5 dB.",
        "Apply readback is missing, mismatched, or not physically verified.",
        "Rollback readback or rollback proof is missing.",
        "Emergency stop does not block a follow-up proposal.",
        "Any fader move would exceed 0 dBFS.",
        "Feedback, clipping, or operator concern appears at any point.",
    )


def proof_prerequisites(value_type="gain"):
    prerequisites = (
        "A clean read-only WING preflight candidate exists and has been manually reviewed.",
        "Operator is physically present and explicitly approves a bounded proof run.",
        "Backend is connected to the intended WING Rack only.",
        "A current scene/console backup exists and the operator can restore manually.",
        "Monitor/PA risk is controlled before any supervised write.",
        "Proof uses the Product Layer proposal path and rollback gate only.",
    )
    if value_type == "gain":
        prerequisites += (
            "Use gain-down proof only; gain-up remains out of scope for this plan.",
        )
    return prerequisites


def readonly_preflight_command(
    *,
    runner="tools/wing_readonly_preflight_runner.py",
    backend_url="ws://localhost:8765",
    artifact_root="artifacts/wing_readonly_preflight",
    expected_plan=READONLY_PREFLIGHT_EXPECTED_PLAN,
    prestart_summary=READONLY_PREFLIGHT_PRESTART_SUMMARY,
    tail=(),
):
    return (
        "python3",
        runner,
        "--url",
        backend_url,
        "--channels",
        READONLY_PREFLIGHT_CHANNELS,
        "--operator-confirm-channels",
        READONLY_PREFLIGHT_CHANNELS,
        "--operator-confirm-session-ready",
        "--value-types",
        READONLY_PREFLIGHT_VALUE_TYPES,
        "--require-physical-readback",
        "--artifact-root",
        artifact_root,
        "--prestart-summary",
        prestart_summary,
        "--expected-plan",
        expected_plan,
        *tail,
    )


def readonly_preflight_step(
    command,
    *,
    id="wing_readonly_preflight_required",
    live_execution_required=False,
    execution_allowed_by_plan=False,
    operator_approval_required=True,
    read_only=True,
    expected_plan_required_for_real_run=True,
    prestart_summary_required=True,
    expected_plan=READONLY_PREFLIGHT_EXPECTED_PLAN,
    prestart_summary=READONLY_PREFLIGHT_PRESTART_SUMMARY,
    prerequisites=None,
    abort_conditions=None,
    expected_artifacts=None,
):
    if prerequisites is None:
        prerequisites = (
            "Operator is physically present at the intended WING Rack.",
            "Backend is already connected in WING mode.",
            "No supervised apply, proposal, rollback, emergency-stop, connect, or disconnect request is sent by this step.",
            "Move or replace invalid preflight artifacts before trusting a readiness candidate.",
        )
    if abort_conditions is None:
        abort_conditions = (
            "Backend is not connected to the intended WING Rack.",
            "Any response is not verified from physical readback.",
            "Any forbidden write, proposal, rollback, emergency, connect, or disconnect request is observed.",
        )
    if expected_artifacts is None:
        expected_artifacts = (
            WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT,
            (
                f"{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME} with "
                "get_operator_mode/get_state/query_mixer_parameter only"
            ),
        )
    return WingReadonlyPreflightPlanStep(
        id=id,
        live_execution_required=live_execution_required,
        execution_allowed_by_plan=execution_allowed_by_plan,
        operator_approval_required=operator_approval_required,
        read_only=read_only,
        expected_plan_required_for_real_run=expected_plan_required_for_real_run,
        prestart_summary_required=prestart_summary_required,
        expected_plan=expected_plan,
        prestart_summary=prestart_summary,
        command=command,
        rationale="unit regression",
        prerequisites=prerequisites,
        abort_conditions=abort_conditions,
        expected_artifacts=expected_artifacts,
    )


READONLY_PREFLIGHT_REQUEST_SEQUENCE = [
    {"type": "get_operator_mode"},
    {"type": "get_state"},
    {
        "type": "query_mixer_parameter",
        "request_id": "wing-readonly-preflight:ch3:fader",
        "channel": 3,
        "value_type": "fader",
        "address": "/ch/3/fdr",
    },
    {
        "type": "query_mixer_parameter",
        "request_id": "wing-readonly-preflight:ch3:gain",
        "channel": 3,
        "value_type": "gain",
        "address": "/ch/3/in/set/trim",
    },
    {
        "type": "query_mixer_parameter",
        "request_id": "wing-readonly-preflight:ch3:pan",
        "channel": 3,
        "value_type": "pan",
        "address": "/ch/3/pan",
    },
]


def assert_prestart_check(summary: dict) -> None:
    assert summary["readonly_preflight_prestart_check_required"] is True
    assert (
        summary["readonly_preflight_prestart_check_must_pass_before_starting_backend"]
        is True
    )
    assert (
        summary["readonly_preflight_prestart_check_command"]
        == PRESTART_BACKEND_PORT_CHECK_COMMAND
    )
    assert (
        summary["readonly_preflight_prestart_check_command_text"]
        == PRESTART_BACKEND_PORT_CHECK_COMMAND_TEXT
    )
    assert (
        summary["readonly_preflight_prestart_check_failure_detail_fields"]
        == PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert summary["readonly_preflight_expected_plan_required_for_real_run"] is True
    assert summary["readonly_preflight_prestart_summary_required"] is True
    assert (
        summary["readonly_preflight_command_expected_plan"]
        == READONLY_PREFLIGHT_EXPECTED_PLAN
    )
    assert (
        summary["readonly_preflight_command_prestart_summary"]
        == READONLY_PREFLIGHT_PRESTART_SUMMARY
    )
    assert summary["readonly_preflight_command_channels"] == READONLY_PREFLIGHT_CHANNELS
    assert (
        summary["readonly_preflight_command_operator_confirm_channels"]
        == READONLY_PREFLIGHT_CHANNELS
    )
    assert (
        summary["readonly_preflight_command_operator_confirm_session_ready"]
        is True
    )
    assert (
        summary["readonly_preflight_command_value_types"]
        == READONLY_PREFLIGHT_VALUE_TYPES
    )


def assert_readonly_preflight_command_bindings(step: dict) -> None:
    command = step["command"]
    assert step["expected_plan_required_for_real_run"] is True
    assert step["prestart_summary_required"] is True
    assert step["expected_plan"] == READONLY_PREFLIGHT_EXPECTED_PLAN
    assert step["prestart_summary"] == READONLY_PREFLIGHT_PRESTART_SUMMARY
    assert "--prestart-summary" in command
    assert command[command.index("--prestart-summary") + 1] == (
        READONLY_PREFLIGHT_PRESTART_SUMMARY
    )
    assert "--expected-plan" in command
    assert command[command.index("--expected-plan") + 1] == (
        READONLY_PREFLIGHT_EXPECTED_PLAN
    )
    assert "--channels" in command
    assert command[command.index("--channels") + 1] == READONLY_PREFLIGHT_CHANNELS
    assert "--operator-confirm-channels" in command
    assert command[command.index("--operator-confirm-channels") + 1] == (
        READONLY_PREFLIGHT_CHANNELS
    )
    assert "--operator-confirm-session-ready" in command
    assert "--value-types" in command
    assert command[command.index("--value-types") + 1] == READONLY_PREFLIGHT_VALUE_TYPES


def test_readonly_preflight_step_rejects_non_read_only_flag():
    with pytest.raises(
        ValueError,
        match="Read-only preflight step must be marked read-only",
    ):
        readonly_preflight_step(readonly_preflight_command(), read_only=False)


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        (
            {"live_execution_required": 0},
            "Read-only preflight step must not be a live write step",
        ),
        (
            {"execution_allowed_by_plan": 0},
            "The proof plan must not authorize read-only preflight execution",
        ),
        (
            {"operator_approval_required": 1},
            "Read-only WING preflight still requires operator presence",
        ),
        (
            {"expected_plan_required_for_real_run": 1},
            "Read-only preflight plan must require an expected plan",
        ),
        (
            {"prestart_summary_required": 1},
            "Read-only preflight plan must require a prestart summary",
        ),
    ],
)
def test_readonly_preflight_step_rejects_non_bool_safety_fields(
    override,
    expected_error,
):
    with pytest.raises(ValueError, match=expected_error):
        readonly_preflight_step(readonly_preflight_command(), **override)


def test_readonly_preflight_step_rejects_non_canonical_runner():
    with pytest.raises(
        ValueError,
        match="Read-only preflight step must use canonical runner",
    ):
        readonly_preflight_step(
            readonly_preflight_command(runner="tools/other_readonly_runner.py")
        )


def test_readonly_preflight_step_rejects_duplicate_command_flags():
    with pytest.raises(
        ValueError,
        match="Read-only preflight step command must not repeat flags",
    ):
        readonly_preflight_step(
            readonly_preflight_command(tail=("--channels", READONLY_PREFLIGHT_CHANNELS))
        )


def test_readonly_preflight_step_rejects_unknown_command_flags():
    with pytest.raises(
        ValueError,
        match="Read-only preflight step command must not include unknown flags",
    ):
        readonly_preflight_step(readonly_preflight_command(tail=("--dry-run",)))


def test_readonly_preflight_step_rejects_flag_used_as_value():
    with pytest.raises(
        ValueError,
        match="Read-only preflight step command must bind explicit flag values",
    ):
        readonly_preflight_step(
            (
                "python3",
                "tools/wing_readonly_preflight_runner.py",
                "--url",
                "ws://localhost:8765",
                "--channels",
                READONLY_PREFLIGHT_CHANNELS,
                "--operator-confirm-channels",
                READONLY_PREFLIGHT_CHANNELS,
                "--operator-confirm-session-ready",
                "--value-types",
                READONLY_PREFLIGHT_VALUE_TYPES,
                "--require-physical-readback",
                "--artifact-root",
                "artifacts/wing_readonly_preflight",
                "--prestart-summary",
                "--expected-plan",
                READONLY_PREFLIGHT_EXPECTED_PLAN,
            )
        )


def test_readonly_preflight_step_rejects_stray_command_arguments():
    with pytest.raises(
        ValueError,
        match="Read-only preflight step command must not include stray arguments",
    ):
        readonly_preflight_step(readonly_preflight_command(tail=("operator-note",)))


def test_readonly_preflight_step_rejects_swapped_plan_and_prestart_bindings():
    command = list(readonly_preflight_command())
    command[command.index("--prestart-summary") + 1] = READONLY_PREFLIGHT_EXPECTED_PLAN
    command[command.index("--expected-plan") + 1] = READONLY_PREFLIGHT_PRESTART_SUMMARY

    with pytest.raises(
        ValueError,
        match="Read-only preflight step must bind --expected-plan",
    ):
        readonly_preflight_step(tuple(command))


@pytest.mark.parametrize(
    "expected_plan",
    [
        "",
        "/tmp/wing_readonly_preflight_plan.json",
        "artifacts/",
        "artifacts//wing_readonly_preflight_plan/latest_plan.json",
        "artifacts/wing_readonly_preflight_plan/../latest_plan.json",
        "Docs/reports/WING_READONLY_PREFLIGHT_HANDOFF_PACKET_LATEST.md",
    ],
)
def test_readonly_preflight_step_rejects_unsafe_expected_plan(expected_plan):
    with pytest.raises(
        ValueError,
        match="Read-only preflight step expected plan must stay under artifacts",
    ):
        readonly_preflight_step(
            readonly_preflight_command(expected_plan=expected_plan),
            expected_plan=expected_plan,
        )


@pytest.mark.parametrize(
    "prestart_summary",
    [
        "",
        "/tmp/no_wing_release_check_summary.json",
        "artifacts/",
        "artifacts//no_wing_release_check/summary.json",
        "artifacts/no_wing_release_check/../summary.json",
        "Docs/reports/NO_WING_RELEASE_CHECK_LATEST.md",
    ],
)
def test_readonly_preflight_step_rejects_unsafe_prestart_summary(prestart_summary):
    with pytest.raises(
        ValueError,
        match="Read-only preflight step prestart summary must stay under artifacts",
    ):
        readonly_preflight_step(
            readonly_preflight_command(prestart_summary=prestart_summary),
            prestart_summary=prestart_summary,
        )


def test_readonly_preflight_step_requires_physical_readback_flag():
    command = tuple(
        token
        for token in readonly_preflight_command()
        if token != "--require-physical-readback"
    )

    with pytest.raises(
        ValueError,
        match="Read-only preflight step must require physical readback",
    ):
        readonly_preflight_step(command)


@pytest.mark.parametrize(
    "backend_url",
    ["", "http://localhost:8765", "localhost:8765", "ws://", "wss://"],
)
def test_readonly_preflight_step_rejects_non_websocket_backend_url(backend_url):
    with pytest.raises(
        ValueError,
        match="Read-only preflight step must bind a WebSocket backend URL",
    ):
        readonly_preflight_step(readonly_preflight_command(backend_url=backend_url))


@pytest.mark.parametrize(
    "artifact_root",
    [
        "",
        "/tmp/wing_readonly_preflight",
        "artifacts/",
        "artifacts//wing_readonly_preflight",
        "artifacts/wing_readonly_preflight/..",
        "artifacts\\wing_readonly_preflight\\..",
        "reports/wing_readonly_preflight",
    ],
)
def test_readonly_preflight_step_rejects_unsafe_artifact_root(artifact_root):
    with pytest.raises(
        ValueError,
        match=(
            "Read-only preflight step must bind artifact root"
            if not artifact_root
            else "Read-only preflight step artifact root must stay under artifacts"
        ),
    ):
        readonly_preflight_step(
            readonly_preflight_command(artifact_root=artifact_root)
        )


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        (
            {"id": "wing_readonly_preflight_optional"},
            "Read-only preflight step id must match safety contract",
        ),
        (
            {"prerequisites": ()},
            "Read-only preflight step prerequisites must match safety contract",
        ),
        (
            {"abort_conditions": ()},
            "Read-only preflight step abort conditions must match safety contract",
        ),
        (
            {"expected_artifacts": ()},
            "Read-only preflight step expected artifacts must match safety contract",
        ),
    ],
)
def test_readonly_preflight_step_rejects_truncated_operator_checklist(
    override,
    expected_error,
):
    with pytest.raises(ValueError, match=expected_error):
        readonly_preflight_step(readonly_preflight_command(), **override)


def test_readonly_preflight_step_rejects_empty_abort_blocker():
    with pytest.raises(
        ValueError,
        match="Read-only preflight step abort blockers must be explicit",
    ):
        readonly_preflight_step(
            readonly_preflight_command(),
            abort_conditions=(
                "Backend is not connected to the intended WING Rack.",
                "Any response is not verified from physical readback.",
                "Any forbidden write, proposal, rollback, emergency, connect, or disconnect request is observed.",
                "",
            ),
        )


def test_readonly_preflight_step_rejects_empty_next_action_artifact():
    with pytest.raises(
        ValueError,
        match="Read-only preflight step next-action artifacts must be explicit",
    ):
        readonly_preflight_step(
            readonly_preflight_command(),
            expected_artifacts=(
                WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT,
                (
                    f"{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME} with "
                    "get_operator_mode/get_state/query_mixer_parameter only"
                ),
                "",
            ),
        )


def evidence_snapshot(*rows):
    return {
        "type": "supervised_promotion_evidence",
        "safety": {
            "live_widening_allowed": False,
            "automatic_safety_update_allowed": False,
        },
        "rows": list(rows),
    }


def evidence_row(value_type: str, *, review_candidate: bool = False, blockers=None):
    return {
        "id": f"{value_type}_supervised_evidence",
        "value_type": value_type,
        "safety_row_review_candidate": review_candidate,
        "blockers": list(blockers or []),
    }


def accepted_review_decision_ledger() -> dict:
    return {
        "type": "wing_readonly_preflight_review_decision_ledger",
        "status": "operator_review_decision_records_available",
        "safety": {
            "read_only": True,
            "live_write": False,
            "execution_allowed_by_decision": False,
            "readiness_matrix_update_allowed_by_decision": False,
            "readiness_closure_allowed_by_decision": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "ledger_status": "operator_review_decision_records_available",
            "accepted_record_count": 1,
            "rejected_record_count": 0,
            "accepted_source_ids": [
                "artifacts/wing_readonly_preflight/run/summary.json"
            ],
            "accepted_candidate_prestart_contract_violations_by_source": {},
            "candidate_reviews": [
                {
                    "source_id": "artifacts/wing_readonly_preflight/run/summary.json",
                    "artifact_root": "artifacts/wing_readonly_preflight",
                    "status": "valid_readonly_preflight_artifact",
                    "summary_path": "artifacts/wing_readonly_preflight/run/summary.json",
                    "events_path": "artifacts/wing_readonly_preflight/run/events.jsonl",
                    "expected_artifacts": [
                        f"artifacts/wing_readonly_preflight/<run_id>/{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}",
                        f"artifacts/wing_readonly_preflight/<run_id>/{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}",
                    ],
                    "expected_summary_artifact": (
                        "artifacts/wing_readonly_preflight/run/summary.json"
                    ),
                    "expected_events_artifact": (
                        "artifacts/wing_readonly_preflight/run/events.jsonl"
                    ),
                    "summary_path_matches_expected_artifact": True,
                    "events_path_matches_expected_artifact": True,
                    "contacts_backend": True,
                    "queries_console": True,
                    "request_sequence": list(READONLY_PREFLIGHT_REQUEST_SEQUENCE),
                    "plan_request_sequence": list(READONLY_PREFLIGHT_REQUEST_SEQUENCE),
                    "request_sequence_matches_plan": True,
                    "backend_port_prestart_check_required": True,
                    "backend_port_prestart_check_must_pass_before_starting_backend": True,
                    "backend_port_prestart_check_command": list(
                        PRESTART_BACKEND_PORT_CHECK_COMMAND
                    ),
                    "backend_port_prestart_check_command_text": (
                        PRESTART_BACKEND_PORT_CHECK_COMMAND_TEXT
                    ),
                    "backend_port_prestart_check_failure_detail_fields": list(
                        PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS
                    ),
                    "backend_port_prestart_summary_required": True,
                    "backend_port_prestart_summary_path": (
                        "artifacts/no_wing_release_check/run/summary.json"
                    ),
                    "backend_port_prestart_summary_sha256": "3" * 64,
                    "backend_port_prestart_summary_type": (
                        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE
                    ),
                    "backend_port_prestart_summary_status": "passed",
                    "backend_port_prestart_summary_connection_policy": (
                        WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY
                    ),
                    "backend_port_prestart_summary_read_only": True,
                    "backend_port_prestart_summary_live_write": False,
                    "backend_port_prestart_summary_contacts_backend": False,
                    "backend_port_prestart_summary_contacts_wing": False,
                    "backend_port_prestart_summary_mutates_console_state": False,
                    "backend_port_prestart_summary_queries_console": False,
                    "backend_port_prestart_summary_executes_tests": False,
                    "backend_port_prestart_check_status": "passed",
                    "artifact_manifest_verifier_must_pass_before_websocket": True,
                    "artifact_manifest_verifier_passed_before_websocket": True,
                    "artifact_manifest_verifier_websocket_opened_after_pass": True,
                    **{
                        field: False
                        for field in PRESTART_BACKEND_PORT_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
                    },
                    **{
                        field: False
                        for field in PRESTART_BACKEND_PORT_CHECK_REQUIRED_FALSE_PLAN_FIELDS
                    },
                    **{
                        field: True
                        for field in PRESTART_BACKEND_PORT_CHECK_REQUIRED_FALSE_MATCH_FIELDS
                    },
                }
            ],
            "rejected_source_ids": [],
            "invalid_record_count": 0,
            "unreviewed_candidate_count": 0,
            "manual_readiness_update_candidate_available": True,
            "readiness_matrix_update_allowed_by_decision": False,
            "readiness_closure_allowed_by_decision": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": [
                "readiness_matrix_update_still_requires_separate_manual_action"
            ],
        },
    }


def preflight_ready_snapshot() -> dict:
    return {
        "type": "autonomy_promotion_readiness",
        "summary": {
            "wing_readonly_preflight_evidence": {
                "readiness_update_candidate_available": True,
                "valid_candidate_count": 1,
                "invalid_candidate_count": 0,
                "readiness_closure_allowed_by_evidence": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "evidence_status": "readonly_preflight_candidate_available",
                "blocked_by": [
                    "operator_must_review_readonly_preflight_candidate_before_readiness_update"
                ],
                "prestart_backend_port_check_required": True,
                "prestart_backend_port_check_must_pass_before_starting_backend": True,
                "prestart_backend_port_check_command": list(
                    PRESTART_BACKEND_PORT_CHECK_COMMAND
                ),
                "prestart_backend_port_check_command_text": (
                    PRESTART_BACKEND_PORT_CHECK_COMMAND_TEXT
                ),
                "prestart_backend_port_check_failure_detail_fields": list(
                    PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS
                ),
                "next_required_actions": [
                    "Manually review the read-only preflight candidate before updating readiness rows."
                ],
            },
            "wing_readonly_preflight_review_decision_ledger": accepted_review_decision_ledger(),
        },
    }


def preflight_candidate_without_review_decision_snapshot() -> dict:
    readiness = preflight_ready_snapshot()
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = {
        "type": "wing_readonly_preflight_review_decision_ledger",
        "status": "operator_review_required",
        "summary": {
            "ledger_status": "operator_review_required",
            "accepted_record_count": 0,
            "rejected_record_count": 0,
            "invalid_record_count": 0,
            "unreviewed_candidate_count": 1,
            "manual_readiness_update_candidate_available": False,
            "readiness_matrix_update_allowed_by_decision": False,
            "readiness_closure_allowed_by_decision": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": [
                "operator_must_record_readonly_preflight_review_decision"
            ],
        },
    }
    return readiness


def preflight_candidate_with_conflicting_review_decisions_snapshot() -> dict:
    readiness = preflight_ready_snapshot()
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = {
        "type": "wing_readonly_preflight_review_decision_ledger",
        "status": "operator_review_decision_replacement_required",
        "summary": {
            "ledger_status": "operator_review_decision_replacement_required",
            "accepted_record_count": 0,
            "rejected_record_count": 0,
            "invalid_record_count": 2,
            "invalid_record_ids": ["review-accept", "review-reject"],
            "invalid_records": [
                {
                    "review_id": "review-accept",
                    "artifact_source_id": (
                        "artifacts/wing_readonly_preflight/run/summary.json"
                    ),
                    "decision": "accepted",
                    "template_status": "",
                    "violations": ["review_decision_duplicate_artifact_source"],
                    "unsafe_authority_flags": [],
                },
                {
                    "review_id": "review-reject",
                    "artifact_source_id": (
                        "artifacts/wing_readonly_preflight/run/summary.json"
                    ),
                    "decision": "rejected",
                    "template_status": "",
                    "violations": [
                        "review_decision_duplicate_artifact_source",
                        "review_decision_conflicting_artifact_source_decisions",
                    ],
                    "unsafe_authority_flags": [],
                },
            ],
            "duplicate_source_ids": [
                "artifacts/wing_readonly_preflight/run/summary.json"
            ],
            "conflicting_source_ids": [
                "artifacts/wing_readonly_preflight/run/summary.json"
            ],
            "unreviewed_candidate_count": 0,
            "manual_readiness_update_candidate_available": False,
            "readiness_matrix_update_allowed_by_decision": False,
            "readiness_closure_allowed_by_decision": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": [
                "operator_must_replace_invalid_readonly_preflight_review_decisions"
            ],
        },
    }
    return readiness


def test_supervised_wing_proof_plan_module_is_transport_free():
    source = Path("backend/supervised_wing_proof_plan.py").read_text(encoding="utf-8")

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "apply_operator_proposal",
        "requested_change",
        "requested_changes",
    ]
    for token in forbidden:
        assert token not in source


def test_plan_builds_gain_and_fader_physical_proof_steps_without_execution_authority():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", blockers=["physical_candidate_missing"]),
            evidence_row("fader", blockers=["physical_candidate_missing", "blocked_artifacts_for_value_type"]),
        ),
        autonomy_readiness_snapshot=preflight_ready_snapshot(),
        backend_url="ws://proof-backend",
        artifact_root="artifacts/proof",
        channels_by_value_type={"gain": (3,), "fader": (2, 16)},
    )

    assert plan["type"] == "supervised_wing_proof_plan"
    assert plan["status"] == "operator_proof_required"
    assert plan["safety"] == {
        "read_only": True,
        "live_write": False,
        "execution_allowed_by_plan": False,
        "live_widening_allowed": False,
        "automatic_safety_update_allowed": False,
        "full_autonomous_ready": False,
        "operator_required": True,
    }
    assert plan["summary"]["step_count"] == 3
    assert plan["summary"]["allowed_supervised_live_value_types"] == sorted(
        WING_READONLY_PREFLIGHT_ALLOWED_SUPERVISED_LIVE_VALUE_TYPES
    )
    assert plan["summary"]["planned_value_types"] == ["fader", "gain"]
    assert plan["summary"]["planned_channels"] == [3, 2, 16]
    assert plan["summary"]["readonly_preflight_required"] is True
    assert plan["summary"]["readonly_preflight_candidate_available"] is True
    assert plan["summary"]["readonly_preflight_evidence_status"] == "readonly_preflight_candidate_available"
    assert_prestart_check(plan["summary"])
    assert plan["summary"]["readonly_preflight_step_count"] == 0
    assert plan["summary"]["readonly_preflight_review_decision_required"] is True
    assert plan["summary"]["readonly_preflight_review_decision_accepted"] is True
    assert (
        plan["summary"]["readonly_preflight_review_decision_status"]
        == "operator_review_decision_records_available"
    )
    assert plan["summary"]["readonly_preflight_review_decision_accepted_record_count"] == 1
    assert plan["summary"]["readonly_preflight_review_decision_accepted_source_ids"] == [
        "artifacts/wing_readonly_preflight/run/summary.json"
    ]
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is True
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] == wing_readonly_preflight_accepted_candidate_prestart_source_ids(
        plan["summary"]["readonly_preflight_review_decision_accepted_source_ids"],
        accepted_review_decision_ledger()["summary"]["candidate_reviews"],
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
    ] == wing_readonly_preflight_accepted_candidate_prestart_violations_by_source(
        plan["summary"]["readonly_preflight_review_decision_accepted_source_ids"],
        accepted_review_decision_ledger()["summary"]["candidate_reviews"],
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids"
    ] == ["artifacts/wing_readonly_preflight/run/summary.json"]
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source"
        ]
        == {}
    )
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete"
        ]
        is True
    )
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_request_sequence_contract_valid"
        ]
        is True
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_request_sequence_source_ids"
    ] == wing_readonly_preflight_accepted_candidate_request_sequence_source_ids(
        plan["summary"]["readonly_preflight_review_decision_accepted_source_ids"],
        accepted_review_decision_ledger()["summary"]["candidate_reviews"],
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_request_sequence_contract_violations_by_source"
    ] == wing_readonly_preflight_accepted_candidate_request_sequence_violations_by_source(
        plan["summary"]["readonly_preflight_review_decision_accepted_source_ids"],
        accepted_review_decision_ledger()["summary"]["candidate_reviews"],
    )
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_artifact_root_contract_valid"
        ]
        is True
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_artifact_roots"
    ] == ["artifacts/wing_readonly_preflight"]
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source"
    ] == {}
    assert plan["summary"][
        "readonly_preflight_review_decision_expected_candidate_artifact_roots"
    ] == ["artifacts/wing_readonly_preflight"]
    assert plan["summary"][
        "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source"
    ] == {}
    accepted_source_id = "artifacts/wing_readonly_preflight/run/summary.json"
    accepted_events_path = "artifacts/wing_readonly_preflight/run/events.jsonl"
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids"
    ] == [accepted_source_id]
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_expected_artifacts_by_source"
    ] == {
        accepted_source_id: [
            f"artifacts/wing_readonly_preflight/<run_id>/{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}",
            f"artifacts/wing_readonly_preflight/<run_id>/{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}",
        ]
    }
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_expected_summary_artifact_by_source"
    ] == {accepted_source_id: accepted_source_id}
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_expected_events_artifact_by_source"
    ] == {accepted_source_id: accepted_events_path}
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_summary_path_matches_expected_artifact_by_source"
    ] == {accepted_source_id: True}
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_events_path_matches_expected_artifact_by_source"
    ] == {accepted_source_id: True}
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids"
    ] == []
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete"
        ]
        is True
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_source_ids"
    ] == [accepted_source_id]
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_by_source"
    ] == {accepted_source_id: READONLY_PREFLIGHT_REQUEST_SEQUENCE}
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_plan_request_sequence_by_source"
    ] == {accepted_source_id: READONLY_PREFLIGHT_REQUEST_SEQUENCE}
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_matches_plan_by_source"
    ] == {accepted_source_id: True}
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids"
    ] == []
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete"
        ]
        is True
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_source_ids"
    ] == [accepted_source_id]
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_contacts_backend_by_source"
    ] == {accepted_source_id: True}
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_queries_console_by_source"
    ] == {accepted_source_id: True}
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids"
    ] == []
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete"
        ]
        is True
    )
    assert plan["summary"]["readonly_preflight_review_decision_rejected_source_ids"] == []
    assert plan["readonly_preflight_steps"] == []

    gain_step = plan["steps"][0]
    assert gain_step["id"] == "gain_physical_proof_ch3"
    assert gain_step["live_execution_required"] is True
    assert gain_step["execution_allowed_by_plan"] is False
    assert gain_step["operator_approval_required"] is True
    assert gain_step["delta_db"] == -0.25
    assert gain_step["command"][0] == "python3"
    assert "--execute" in gain_step["command"]
    assert "--operator-confirm-channel" in gain_step["command"]
    assert gain_step["command"][gain_step["command"].index("--operator-confirm-channel") + 1] == "3"
    assert "--operator-confirm-value-type" in gain_step["command"]
    assert (
        gain_step["command"][gain_step["command"].index("--operator-confirm-value-type") + 1]
        == "gain"
    )
    assert "--operator-confirm-delta-db" in gain_step["command"]
    assert (
        gain_step["command"][gain_step["command"].index("--operator-confirm-delta-db") + 1]
        == "-0.25"
    )
    assert "--prove-blocked-apply" in gain_step["command"]
    assert "--prove-emergency-stop" in gain_step["command"]
    assert gain_step["required_evidence_components"] == GAIN_REQUIRED_EVIDENCE_COMPONENTS
    assert "ws://proof-backend" in gain_step["command"]
    assert (
        "A clean read-only WING preflight candidate exists and has been manually reviewed."
        in gain_step["prerequisites"]
    )
    assert "Use gain-down proof only; gain-up remains out of scope for this plan." in gain_step["prerequisites"]

    fader_step = plan["steps"][1]
    assert "Review blocked prior proof artifacts for this value type before trusting a new result." in fader_step["prerequisites"]
    assert "blocked_artifacts_for_value_type" in fader_step["evidence_blockers_to_review"]


def test_plan_blocks_physical_proof_steps_until_review_decision_is_accepted():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", blockers=["physical_candidate_missing"]),
            evidence_row("fader", blockers=["physical_candidate_missing"]),
        ),
        autonomy_readiness_snapshot=preflight_candidate_without_review_decision_snapshot(),
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["readonly_preflight_candidate_available"] is True
    assert_prestart_check(plan["summary"])
    assert plan["summary"]["readonly_preflight_step_count"] == 0
    assert plan["summary"]["readonly_preflight_review_decision_required"] is True
    assert plan["summary"]["readonly_preflight_review_decision_accepted"] is False
    assert (
        plan["summary"]["readonly_preflight_review_decision_status"]
        == "operator_review_required"
    )
    assert (
        "accepted_readonly_preflight_review_decision_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert (
        "operator_must_record_readonly_preflight_review_decision"
        in plan["summary"]["global_blockers"]
    )
    assert plan["readonly_preflight_steps"] == []
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_when_accepted_candidate_lacks_prestart_contract():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    ledger["summary"]["candidate_reviews"] = [
        {
            "source_id": "artifacts/wing_readonly_preflight/run/summary.json",
            "status": "valid_readonly_preflight_artifact",
            "expected_artifacts": [
                f"artifacts/wing_readonly_preflight/<run_id>/{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}",
                f"artifacts/wing_readonly_preflight/<run_id>/{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}",
            ],
            "expected_summary_artifact": (
                "artifacts/wing_readonly_preflight/run/summary.json"
            ),
            "expected_events_artifact": (
                "artifacts/wing_readonly_preflight/run/events.jsonl"
            ),
            "summary_path_matches_expected_artifact": True,
            "events_path_matches_expected_artifact": True,
            "request_sequence": list(READONLY_PREFLIGHT_REQUEST_SEQUENCE),
            "plan_request_sequence": list(READONLY_PREFLIGHT_REQUEST_SEQUENCE),
            "request_sequence_matches_plan": True,
            "backend_port_prestart_check_required": False,
            "backend_port_prestart_check_must_pass_before_starting_backend": True,
            "backend_port_prestart_check_command": list(
                PRESTART_BACKEND_PORT_CHECK_COMMAND
            ),
            "backend_port_prestart_check_command_text": (
                PRESTART_BACKEND_PORT_CHECK_COMMAND_TEXT
            ),
            "backend_port_prestart_check_failure_detail_fields": list(
                PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS
            ),
            "backend_port_prestart_summary_required": True,
            "backend_port_prestart_summary_path": (
                "artifacts/no_wing_release_check/run/summary.json"
            ),
            "backend_port_prestart_summary_sha256": "3" * 64,
            "backend_port_prestart_summary_type": (
                WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE
            ),
            "backend_port_prestart_summary_status": "passed",
            "backend_port_prestart_summary_connection_policy": (
                WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY
            ),
            "backend_port_prestart_summary_read_only": True,
            "backend_port_prestart_summary_live_write": False,
            "backend_port_prestart_summary_contacts_backend": False,
            "backend_port_prestart_summary_contacts_wing": False,
            "backend_port_prestart_summary_mutates_console_state": False,
            "backend_port_prestart_summary_queries_console": False,
            "backend_port_prestart_summary_executes_tests": False,
            "backend_port_prestart_check_status": "passed",
            "artifact_manifest_verifier_must_pass_before_websocket": True,
            "artifact_manifest_verifier_passed_before_websocket": True,
            "artifact_manifest_verifier_websocket_opened_after_pass": True,
            **{
                field: False
                for field in PRESTART_BACKEND_PORT_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
            },
            **{
                field: False
                for field in PRESTART_BACKEND_PORT_CHECK_REQUIRED_FALSE_PLAN_FIELDS
            },
            **{
                field: True
                for field in PRESTART_BACKEND_PORT_CHECK_REQUIRED_FALSE_MATCH_FIELDS
            },
        }
    ]
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["readonly_preflight_review_decision_accepted"] is False
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is False
    )
    assert (
        plan["summary"]["readonly_preflight_review_decision_candidate_prestart_source_ids"]
        == []
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
    ] == {
        "artifacts/wing_readonly_preflight/run/summary.json": [
            "backend_port_prestart_check_required_not_true"
        ]
    }
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )


def test_plan_blocks_physical_proof_steps_when_accepted_candidate_lacks_request_sequence_trace():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    candidate = ledger["summary"]["candidate_reviews"][0]
    candidate["plan_request_sequence"] = []
    candidate["request_sequence_matches_plan"] = False
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    accepted_source_id = "artifacts/wing_readonly_preflight/run/summary.json"
    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["readonly_preflight_review_decision_accepted"] is False
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is True
    )
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_request_sequence_contract_valid"
        ]
        is False
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_request_sequence_source_ids"
    ] == []
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_request_sequence_contract_violations_by_source"
    ] == {
        accepted_source_id: [
            "plan_request_sequence_missing",
            "request_sequence_matches_plan_not_true",
        ]
    }
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids"
    ] == [accepted_source_id]
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete"
        ]
        is False
    )
    assert (
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_REQUEST_SEQUENCE_BLOCKER
        in plan["summary"]["global_blockers"]
    )
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_when_accepted_candidate_lacks_contact_trace():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    candidate = ledger["summary"]["candidate_reviews"][0]
    candidate["queries_console"] = False
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    accepted_source_id = "artifacts/wing_readonly_preflight/run/summary.json"
    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["readonly_preflight_review_decision_accepted"] is False
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_source_ids"
    ] == [accepted_source_id]
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_contacts_backend_by_source"
    ] == {accepted_source_id: True}
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_queries_console_by_source"
    ] == {accepted_source_id: False}
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids"
    ] == [accepted_source_id]
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete"
        ]
        is False
    )
    assert (
        READONLY_PREFLIGHT_REVIEW_CANDIDATE_CONTACT_TRACE_BLOCKER
        in plan["summary"]["global_blockers"]
    )
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_when_accepted_candidate_lacks_artifact_root():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    ledger["summary"]["candidate_reviews"][0]["artifact_root"] = ""
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["readonly_preflight_review_decision_accepted"] is False
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is True
    )
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_artifact_root_contract_valid"
        ]
        is False
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_artifact_roots"
    ] == []
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source"
    ] == {
        "artifacts/wing_readonly_preflight/run/summary.json": [
            "artifact_root_missing"
        ]
    }
    assert plan["summary"][
        "readonly_preflight_review_decision_expected_candidate_artifact_roots"
    ] == []
    assert plan["summary"][
        "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source"
    ] == {
        "artifacts/wing_readonly_preflight/run/summary.json": [
            "artifact_root_missing"
        ]
    }
    assert (
        "accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_when_candidate_artifact_root_contains_parent_traversal():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    candidate = ledger["summary"]["candidate_reviews"][0]
    candidate["artifact_root"] = "artifacts/wing_readonly_preflight/.."
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["readonly_preflight_review_decision_accepted"] is False
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is True
    )
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_artifact_root_contract_valid"
        ]
        is False
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_artifact_roots"
    ] == ["artifacts/wing_readonly_preflight/.."]
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source"
    ] == {
        "artifacts/wing_readonly_preflight/run/summary.json": [
            "artifact_root_contains_parent_traversal"
        ]
    }
    assert (
        "accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_when_candidate_summary_path_leaves_artifact_root():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    candidate = ledger["summary"]["candidate_reviews"][0]
    candidate["summary_path"] = "artifacts/stale_preflight/run/summary.json"
    candidate["expected_summary_artifact"] = (
        "artifacts/stale_preflight/run/summary.json"
    )
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["readonly_preflight_review_decision_accepted"] is False
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is True
    )
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_artifact_root_contract_valid"
        ]
        is False
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source"
    ] == {
        "artifacts/wing_readonly_preflight/run/summary.json": [
            "summary_path_not_under_artifact_root",
            "expected_summary_artifact_not_under_artifact_root",
        ]
    }
    assert (
        "accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_when_candidate_paths_escape_root_with_parent_traversal():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    candidate = ledger["summary"]["candidate_reviews"][0]
    candidate["summary_path"] = (
        "artifacts/wing_readonly_preflight/../stale_preflight/run/summary.json"
    )
    candidate["events_path"] = (
        "artifacts/wing_readonly_preflight/../stale_preflight/run/events.jsonl"
    )
    candidate["expected_summary_artifact"] = candidate["summary_path"]
    candidate["expected_events_artifact"] = candidate["events_path"]
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["readonly_preflight_review_decision_accepted"] is False
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is True
    )
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_artifact_root_contract_valid"
        ]
        is False
    )
    assert set(
        plan["summary"][
            "readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source"
        ]["artifacts/wing_readonly_preflight/run/summary.json"]
    ) == {
        "summary_path_contains_parent_traversal",
        "summary_path_not_under_artifact_root",
        "events_path_contains_parent_traversal",
        "events_path_not_under_artifact_root",
        "expected_summary_artifact_contains_parent_traversal",
        "expected_summary_artifact_not_under_artifact_root",
        "expected_events_artifact_contains_parent_traversal",
        "expected_events_artifact_not_under_artifact_root",
    }
    assert (
        "accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_when_accepted_candidate_lacks_prestart_summary_binding():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    ledger["summary"]["candidate_reviews"][0]["backend_port_prestart_summary_path"] = ""
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is False
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] == []
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
    ] == {
        "artifacts/wing_readonly_preflight/run/summary.json": [
            "backend_port_prestart_summary_path_missing"
        ]
    }
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_when_accepted_candidate_has_malformed_prestart_digest():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    ledger["summary"]["candidate_reviews"][0][
        "backend_port_prestart_summary_sha256"
    ] = "not-a-sha256"
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is False
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] == []
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
    ] == {
        "artifacts/wing_readonly_preflight/run/summary.json": [
            "backend_port_prestart_summary_sha256_invalid"
        ]
    }
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_when_accepted_candidate_has_unsafe_prestart_false_field():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    ledger["summary"]["candidate_reviews"][0][
        "backend_port_prestart_gate_live_authority_granted"
    ] = True
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is False
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] == []
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
    ] == {
        "artifacts/wing_readonly_preflight/run/summary.json": [
            "backend_port_prestart_gate_live_authority_granted_not_false"
        ]
    }
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_when_accepted_candidate_lacks_verifier_websocket_proof():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    ledger["summary"]["candidate_reviews"][0][
        "artifact_manifest_verifier_websocket_opened_after_pass"
    ] = False
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is False
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] == []
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
    ] == {
        "artifacts/wing_readonly_preflight/run/summary.json": [
            "artifact_manifest_verifier_websocket_opened_after_pass_not_true"
        ]
    }
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert plan["steps"] == []


def test_plan_reports_verifier_proof_blocker_when_verifier_proof_is_missing():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    ledger["summary"]["candidate_reviews"][0].pop(
        "artifact_manifest_verifier_must_pass_before_websocket"
    )
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is False
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids"
    ] == []
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source"
    ] == {
        "artifacts/wing_readonly_preflight/run/summary.json": [
            "artifact_manifest_verifier_must_pass_before_websocket_not_true"
        ]
    }
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete"
        ]
        is False
    )
    assert (
        "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_when_accepted_candidate_has_stale_prestart_match_flag():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    ledger["summary"]["candidate_reviews"][0][
        "backend_port_prestart_gate_live_authority_granted_matches_plan"
    ] = False
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is False
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] == []
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
    ] == {
        "artifacts/wing_readonly_preflight/run/summary.json": [
            "backend_port_prestart_gate_live_authority_granted_matches_plan_not_true"
        ]
    }
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_when_accepted_candidate_lacks_expected_artifact_binding():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    ledger["summary"]["candidate_reviews"][0][
        "summary_path_matches_expected_artifact"
    ] = False
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is False
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] == []
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
    ] == {
        "artifacts/wing_readonly_preflight/run/summary.json": [
            "summary_path_not_bound_to_expected_artifact"
        ]
    }
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert (
        "accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids"
    ] == ["artifacts/wing_readonly_preflight/run/summary.json"]
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete"
        ]
        is False
    )
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_for_conflicting_review_decisions():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", blockers=["physical_candidate_missing"]),
            evidence_row("fader", blockers=["physical_candidate_missing"]),
        ),
        autonomy_readiness_snapshot=preflight_candidate_with_conflicting_review_decisions_snapshot(),
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["readonly_preflight_candidate_available"] is True
    assert_prestart_check(plan["summary"])
    assert plan["summary"]["readonly_preflight_review_decision_accepted"] is False
    assert (
        plan["summary"]["readonly_preflight_review_decision_status"]
        == "operator_review_decision_replacement_required"
    )
    assert plan["summary"]["readonly_preflight_review_decision_invalid_record_count"] == 2
    assert plan["summary"]["readonly_preflight_review_decision_invalid_record_ids"] == [
        "review-accept",
        "review-reject",
    ]
    assert plan["summary"]["readonly_preflight_review_decision_invalid_records"] == [
        {
            "review_id": "review-accept",
            "artifact_source_id": "artifacts/wing_readonly_preflight/run/summary.json",
            "decision": "accepted",
            "template_status": "",
            "violations": ["review_decision_duplicate_artifact_source"],
            "unsafe_authority_flags": [],
        },
        {
            "review_id": "review-reject",
            "artifact_source_id": "artifacts/wing_readonly_preflight/run/summary.json",
            "decision": "rejected",
            "template_status": "",
            "violations": [
                "review_decision_duplicate_artifact_source",
                "review_decision_conflicting_artifact_source_decisions",
            ],
            "unsafe_authority_flags": [],
        },
    ]
    assert plan["summary"]["readonly_preflight_review_decision_duplicate_source_ids"] == [
        "artifacts/wing_readonly_preflight/run/summary.json"
    ]
    assert plan["summary"]["readonly_preflight_review_decision_conflicting_source_ids"] == [
        "artifacts/wing_readonly_preflight/run/summary.json"
    ]
    assert (
        "accepted_readonly_preflight_review_decision_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert (
        "operator_must_replace_invalid_readonly_preflight_review_decisions"
        in plan["summary"]["global_blockers"]
    )
    assert plan["readonly_preflight_steps"] == []
    assert plan["steps"] == []


def test_plan_skips_value_type_that_already_has_manual_review_candidate():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", review_candidate=True),
            evidence_row("fader", blockers=["physical_candidate_missing"]),
        ),
        autonomy_readiness_snapshot=preflight_ready_snapshot(),
        channels_by_value_type={"fader": (4,)},
    )

    assert plan["status"] == "operator_proof_required"
    assert [step["value_type"] for step in plan["steps"]] == ["fader"]
    assert plan["steps"][0]["channel"] == 4


def test_plan_rejects_unsupported_channel_value_type_override():
    with pytest.raises(
        ValueError,
        match="Unsupported supervised WING proof channel value types: pan",
    ):
        build_supervised_wing_proof_plan(
            evidence_snapshot(
                evidence_row("gain", blockers=["physical_candidate_missing"]),
                evidence_row("fader", blockers=["physical_candidate_missing"]),
            ),
            autonomy_readiness_snapshot=preflight_ready_snapshot(),
            channels_by_value_type={"pan": (3,)},
        )


def test_plan_rejects_non_positive_supervised_channel_override():
    with pytest.raises(
        ValueError,
        match="Invalid supervised WING proof channels for gain: 0, -1",
    ):
        build_supervised_wing_proof_plan(
            evidence_snapshot(
                evidence_row("gain", blockers=["physical_candidate_missing"]),
                evidence_row("fader", blockers=["physical_candidate_missing"]),
            ),
            autonomy_readiness_snapshot=preflight_ready_snapshot(),
            channels_by_value_type={"gain": (0, -1)},
        )


def test_plan_rejects_out_of_range_supervised_channel_override():
    with pytest.raises(
        ValueError,
        match="Invalid supervised WING proof channels for fader: 49",
    ):
        build_supervised_wing_proof_plan(
            evidence_snapshot(
                evidence_row("gain", blockers=["physical_candidate_missing"]),
                evidence_row("fader", blockers=["physical_candidate_missing"]),
            ),
            autonomy_readiness_snapshot=preflight_ready_snapshot(),
            channels_by_value_type={"fader": (49,)},
        )


def test_plan_rejects_fractional_supervised_channel_override():
    with pytest.raises(
        ValueError,
        match="Invalid supervised WING proof channels for fader: 3.5",
    ):
        build_supervised_wing_proof_plan(
            evidence_snapshot(
                evidence_row("gain", blockers=["physical_candidate_missing"]),
                evidence_row("fader", blockers=["physical_candidate_missing"]),
            ),
            autonomy_readiness_snapshot=preflight_ready_snapshot(),
            channels_by_value_type={"fader": (3.5,)},
        )


def test_plan_rejects_scalar_supervised_channel_override():
    with pytest.raises(
        ValueError,
        match="Invalid supervised WING proof channels for gain: 3",
    ):
        build_supervised_wing_proof_plan(
            evidence_snapshot(
                evidence_row("gain", blockers=["physical_candidate_missing"]),
                evidence_row("fader", blockers=["physical_candidate_missing"]),
            ),
            autonomy_readiness_snapshot=preflight_ready_snapshot(),
            channels_by_value_type={"gain": 3},
        )


def test_plan_rejects_empty_supervised_channel_override():
    with pytest.raises(
        ValueError,
        match="Invalid supervised WING proof channels for gain: empty",
    ):
        build_supervised_wing_proof_plan(
            evidence_snapshot(
                evidence_row("gain", blockers=["physical_candidate_missing"]),
                evidence_row("fader", blockers=["physical_candidate_missing"]),
            ),
            autonomy_readiness_snapshot=preflight_ready_snapshot(),
            channels_by_value_type={"gain": ()},
        )


def test_plan_rejects_duplicate_supervised_channel_override():
    with pytest.raises(
        ValueError,
        match="Duplicate supervised WING proof channels for fader: 2",
    ):
        build_supervised_wing_proof_plan(
            evidence_snapshot(
                evidence_row("gain", blockers=["physical_candidate_missing"]),
                evidence_row("fader", blockers=["physical_candidate_missing"]),
            ),
            autonomy_readiness_snapshot=preflight_ready_snapshot(),
            channels_by_value_type={"fader": (2, 2, 8)},
        )


def test_plan_rejects_unsupported_delta_value_type_override():
    with pytest.raises(
        ValueError,
        match="Unsupported supervised WING proof delta value types: pan",
    ):
        build_supervised_wing_proof_plan(
            evidence_snapshot(
                evidence_row("gain", blockers=["physical_candidate_missing"]),
                evidence_row("fader", blockers=["physical_candidate_missing"]),
            ),
            autonomy_readiness_snapshot=preflight_ready_snapshot(),
            delta_db_by_value_type={"pan": -0.25},
        )


def test_plan_rejects_positive_supervised_delta_override():
    with pytest.raises(
        ValueError,
        match="Invalid supervised WING proof delta for gain: 0.25",
    ):
        build_supervised_wing_proof_plan(
            evidence_snapshot(
                evidence_row("gain", blockers=["physical_candidate_missing"]),
                evidence_row("fader", blockers=["physical_candidate_missing"]),
            ),
            autonomy_readiness_snapshot=preflight_ready_snapshot(),
            delta_db_by_value_type={"gain": 0.25},
        )


def test_plan_rejects_too_large_supervised_delta_override():
    with pytest.raises(
        ValueError,
        match="Invalid supervised WING proof delta for fader: -0.75",
    ):
        build_supervised_wing_proof_plan(
            evidence_snapshot(
                evidence_row("gain", blockers=["physical_candidate_missing"]),
                evidence_row("fader", blockers=["physical_candidate_missing"]),
            ),
            autonomy_readiness_snapshot=preflight_ready_snapshot(),
            delta_db_by_value_type={"fader": -0.75},
        )


def test_plan_rejects_non_numeric_supervised_delta_override():
    with pytest.raises(
        ValueError,
        match="Invalid supervised WING proof delta for gain: loud",
    ):
        build_supervised_wing_proof_plan(
            evidence_snapshot(
                evidence_row("gain", blockers=["physical_candidate_missing"]),
                evidence_row("fader", blockers=["physical_candidate_missing"]),
            ),
            autonomy_readiness_snapshot=preflight_ready_snapshot(),
            delta_db_by_value_type={"gain": "loud"},
        )


def test_plan_rejects_nan_supervised_delta_override():
    with pytest.raises(
        ValueError,
        match="Invalid supervised WING proof delta for gain: nan",
    ):
        build_supervised_wing_proof_plan(
            evidence_snapshot(
                evidence_row("gain", blockers=["physical_candidate_missing"]),
                evidence_row("fader", blockers=["physical_candidate_missing"]),
            ),
            autonomy_readiness_snapshot=preflight_ready_snapshot(),
            delta_db_by_value_type={"gain": float("nan")},
        )


def test_plan_rejects_infinite_supervised_delta_override():
    with pytest.raises(
        ValueError,
        match="Invalid supervised WING proof delta for fader: -inf",
    ):
        build_supervised_wing_proof_plan(
            evidence_snapshot(
                evidence_row("gain", blockers=["physical_candidate_missing"]),
                evidence_row("fader", blockers=["physical_candidate_missing"]),
            ),
            autonomy_readiness_snapshot=preflight_ready_snapshot(),
            delta_db_by_value_type={"fader": float("-inf")},
        )


def test_proof_step_rejects_non_finite_delta():
    with pytest.raises(
        ValueError,
        match="Physical proof plan only supports finite dB deltas",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=float("nan"),
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=("python3", "tools/wing_operator_proposal_validation_runner.py"),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_non_live_execution_flag():
    with pytest.raises(
        ValueError,
        match="Physical proof steps must be marked as live execution",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=False,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=("python3", "tools/wing_operator_proposal_validation_runner.py"),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_non_bool_live_execution_flag():
    with pytest.raises(
        ValueError,
        match="Physical proof steps must be marked as live execution",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=1,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=("python3", "tools/wing_operator_proposal_validation_runner.py"),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_non_bool_execution_authority():
    with pytest.raises(
        ValueError,
        match="The proof plan must not authorize execution",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=0,
            operator_approval_required=True,
            command=("python3", "tools/wing_operator_proposal_validation_runner.py"),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_non_bool_operator_approval():
    with pytest.raises(
        ValueError,
        match="Physical proof steps require explicit operator approval",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=1,
            command=("python3", "tools/wing_operator_proposal_validation_runner.py"),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_out_of_range_channel():
    with pytest.raises(
        ValueError,
        match="Physical proof plan only supports WING channels 1..48",
    ):
        WingProofPlanStep(
            id="fader_physical_proof_ch49",
            value_type="fader",
            channel=49,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=("python3", "tools/wing_operator_proposal_validation_runner.py"),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_unsupported_value_type():
    with pytest.raises(
        ValueError,
        match="Physical proof plan only supports supervised WING proof value types",
    ):
        WingProofPlanStep(
            id="pan_physical_proof_ch3",
            value_type="pan",
            channel=3,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command("pan"),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_mismatched_id():
    with pytest.raises(
        ValueError,
        match="Physical proof step id must match value type and channel",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch4",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command(),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_mismatched_command_channel_confirmation():
    with pytest.raises(
        ValueError,
        match="Physical proof step command must bind matching channel confirmation",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command(operator_channel=4),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_duplicate_command_flags():
    with pytest.raises(
        ValueError,
        match="Physical proof step command must not repeat proof flags",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command(
                tail=(
                    "--rollback-delay-sec",
                    "0",
                    "--artifact-root",
                    "artifacts/proof",
                    "--reason",
                    "supervised_physical_gain_proof",
                    "--channel",
                    "4",
                ),
            ),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_command_flag_used_as_value():
    with pytest.raises(
        ValueError,
        match="Physical proof step command must bind explicit proof flag values",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command(
                tail=(
                    "--rollback-delay-sec",
                    "0",
                    "--artifact-root",
                    "--reason",
                    "supervised_physical_gain_proof",
                ),
            ),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_unknown_command_flags():
    with pytest.raises(
        ValueError,
        match="Physical proof step command must not include unknown flags",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command(
                tail=(
                    "--rollback-delay-sec",
                    "0",
                    "--artifact-root",
                    "artifacts/proof",
                    "--reason",
                    "supervised_physical_gain_proof",
                    "--dry-run",
                ),
            ),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_stray_command_arguments():
    with pytest.raises(
        ValueError,
        match="Physical proof step command must not include stray arguments",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command(
                tail=(
                    "--rollback-delay-sec",
                    "0",
                    "--artifact-root",
                    "artifacts/proof",
                    "--reason",
                    "supervised_physical_gain_proof",
                    "operator-note",
                ),
            ),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_command_without_safety_flags():
    with pytest.raises(
        ValueError,
        match="Physical proof step command must include proof safety flags",
    ):
        WingProofPlanStep(
            id="fader_physical_proof_ch2",
            value_type="fader",
            channel=2,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command("fader", 2, include_safety_flags=False),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_non_canonical_runner_command():
    with pytest.raises(
        ValueError,
        match="Physical proof step command must use canonical proof runner",
    ):
        WingProofPlanStep(
            id="fader_physical_proof_ch2",
            value_type="fader",
            channel=2,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command("fader", 2, runner="tools/other_runner.py"),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


@pytest.mark.parametrize(
    "backend_url",
    ["", "http://localhost:8765", "localhost:8765", "ws://", "wss://"],
)
def test_proof_step_rejects_non_websocket_backend_url(backend_url):
    with pytest.raises(
        ValueError,
        match="Physical proof step command must bind a WebSocket backend URL",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command(backend_url=backend_url),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


@pytest.mark.parametrize(
    "artifact_root",
    [
        "",
        "/tmp/wing_physical_proof",
        "artifacts/",
        "artifacts//proof",
        "artifacts/proof/..",
        "artifacts\\proof\\..",
        "reports/wing_physical_proof",
    ],
)
def test_proof_step_rejects_unsafe_artifact_root(artifact_root):
    with pytest.raises(
        ValueError,
        match=(
            "Physical proof step command must bind artifact root"
            if not artifact_root
            else "Physical proof step artifact root must stay under artifacts"
        ),
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command(artifact_root=artifact_root),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


@pytest.mark.parametrize(
    ("command_tail", "expected_error"),
    [
        (
            (
                "--artifact-root",
                "artifacts/proof",
                "--reason",
                "supervised_physical_fader_proof",
            ),
            "Physical proof step command must bind zero rollback delay",
        ),
        (
            (
                "--rollback-delay-sec",
                "0",
                "--reason",
                "supervised_physical_fader_proof",
            ),
            "Physical proof step command must bind artifact root",
        ),
        (
            (
                "--rollback-delay-sec",
                "0",
                "--artifact-root",
                "artifacts/proof",
                "--reason",
                "manual_fader_proof",
            ),
            "Physical proof step command must bind supervised proof reason",
        ),
    ],
)
def test_proof_step_rejects_incomplete_command_structure(command_tail, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        WingProofPlanStep(
            id="fader_physical_proof_ch2",
            value_type="fader",
            channel=2,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command("fader", 2, tail=command_tail),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_mismatched_expected_artifacts():
    with pytest.raises(
        ValueError,
        match="Physical proof step expected artifacts must match value type and channel",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command(),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=(),
            expected_artifacts=proof_expected_artifacts(channel=4),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_mismatched_abort_conditions():
    with pytest.raises(
        ValueError,
        match="Physical proof step abort conditions must match safety contract",
    ):
        WingProofPlanStep(
            id="fader_physical_proof_ch2",
            value_type="fader",
            channel=2,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command("fader", 2),
            rationale="unit regression",
            prerequisites=(),
            abort_conditions=("Feedback, clipping, or operator concern appears at any point.",),
            expected_artifacts=proof_expected_artifacts("fader", 2),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_mismatched_prerequisites():
    with pytest.raises(
        ValueError,
        match="Physical proof step prerequisites must match safety contract",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command(),
            rationale="unit regression",
            prerequisites=(
                "A clean read-only WING preflight candidate exists and has been manually reviewed.",
            ),
            abort_conditions=proof_abort_conditions(),
            expected_artifacts=proof_expected_artifacts(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_proof_step_rejects_mismatched_required_evidence_components():
    with pytest.raises(
        ValueError,
        match="Physical proof step evidence components must match safety coverage matrix",
    ):
        WingProofPlanStep(
            id="gain_physical_proof_ch3",
            value_type="gain",
            channel=3,
            delta_db=-0.25,
            live_execution_required=True,
            execution_allowed_by_plan=False,
            operator_approval_required=True,
            command=proof_command(),
            rationale="unit regression",
            prerequisites=proof_prerequisites(),
            abort_conditions=proof_abort_conditions(),
            expected_artifacts=proof_expected_artifacts(),
            evidence_blockers_to_review=(),
            required_evidence_components=(),
        )


def test_plan_is_manual_review_ready_when_all_value_types_have_candidates():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", review_candidate=True),
            evidence_row("fader", review_candidate=True),
        ),
        autonomy_readiness_snapshot=preflight_ready_snapshot(),
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["steps"] == []
    assert plan["safety"]["execution_allowed_by_plan"] is False


def test_plan_blocks_physical_proof_steps_until_readonly_preflight_candidate_exists():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", blockers=["physical_candidate_missing"]),
            evidence_row("fader", blockers=["physical_candidate_missing"]),
        ),
        autonomy_readiness_snapshot={
            "type": "autonomy_promotion_readiness",
            "summary": {
                "wing_readonly_preflight_evidence": {
                    "readiness_update_candidate_available": False,
                    "valid_candidate_count": 0,
                    "invalid_candidate_count": 0,
                    "evidence_status": "operator_preflight_required",
                    "blocked_by": ["operator_must_run_wing_readonly_preflight_runner"],
                    "prestart_backend_port_check_required": True,
                    "prestart_backend_port_check_must_pass_before_starting_backend": True,
                    "prestart_backend_port_check_command": list(
                        PRESTART_BACKEND_PORT_CHECK_COMMAND
                    ),
                    "prestart_backend_port_check_command_text": (
                        PRESTART_BACKEND_PORT_CHECK_COMMAND_TEXT
                    ),
                    "prestart_backend_port_check_failure_detail_fields": list(
                        PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS
                    ),
                    "next_required_actions": [
                        "Before starting the intended backend/WING session, run the no-WING quick gate and resolve only confirmed unexpected listeners.",
                        "Run the read-only WING preflight runner and capture physical readback artifacts."
                    ],
                },
            },
        },
    )

    assert plan["status"] == "blocked_by_readonly_preflight"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["planned_value_types"] == []
    assert plan["summary"]["readonly_preflight_required"] is True
    assert plan["summary"]["readonly_preflight_candidate_available"] is False
    assert plan["summary"]["readonly_preflight_evidence_status"] == "operator_preflight_required"
    assert_prestart_check(plan["summary"])
    assert plan["summary"]["readonly_preflight_step_count"] == 1
    assert "readonly_preflight_candidate_required_before_supervised_proof" in plan["summary"]["global_blockers"]
    assert "operator_must_run_wing_readonly_preflight_runner" in plan["summary"]["global_blockers"]
    preflight_step = plan["readonly_preflight_steps"][0]
    assert preflight_step["id"] == "wing_readonly_preflight_required"
    assert preflight_step["read_only"] is True
    assert preflight_step["live_execution_required"] is False
    assert preflight_step["execution_allowed_by_plan"] is False
    assert preflight_step["operator_approval_required"] is True
    assert preflight_step["command"][:2] == [
        "python3",
        "tools/wing_readonly_preflight_runner.py",
    ]
    assert "--require-physical-readback" in preflight_step["command"]
    assert_readonly_preflight_command_bindings(preflight_step)
    assert "--execute" not in preflight_step["command"]
    assert "tools/wing_operator_proposal_validation_runner.py" not in preflight_step["command"]
    assert (
        WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT
        in preflight_step["expected_artifacts"]
    )
    assert (
        f"{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME} with "
        "get_operator_mode/get_state/query_mixer_parameter only"
    ) in preflight_step["expected_artifacts"]
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_for_mixed_preflight_artifacts():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", blockers=["physical_candidate_missing"]),
            evidence_row("fader", blockers=["physical_candidate_missing"]),
        ),
        autonomy_readiness_snapshot={
            "type": "autonomy_promotion_readiness",
            "summary": {
                "wing_readonly_preflight_evidence": {
                    "readiness_update_candidate_available": False,
                    "valid_candidate_count": 1,
                    "invalid_candidate_count": 1,
                    "evidence_status": "readonly_preflight_artifact_replacement_required",
                    "blocked_by": [
                        "operator_must_replace_invalid_readonly_preflight_artifact"
                    ],
                    "prestart_backend_port_check_required": True,
                    "prestart_backend_port_check_must_pass_before_starting_backend": True,
                    "prestart_backend_port_check_command": list(
                        PRESTART_BACKEND_PORT_CHECK_COMMAND
                    ),
                    "prestart_backend_port_check_command_text": (
                        PRESTART_BACKEND_PORT_CHECK_COMMAND_TEXT
                    ),
                    "prestart_backend_port_check_failure_detail_fields": list(
                        PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS
                    ),
                    "next_required_actions": [
                        "Do not update readiness rows from missing or invalid preflight evidence."
                    ],
                },
            },
        },
    )

    assert plan["status"] == "blocked_by_readonly_preflight"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["readonly_preflight_candidate_available"] is False
    assert (
        plan["summary"]["readonly_preflight_evidence_status"]
        == "readonly_preflight_artifact_replacement_required"
    )
    assert_prestart_check(plan["summary"])
    assert plan["summary"]["readonly_preflight_blocked_by"] == [
        "operator_must_replace_invalid_readonly_preflight_artifact"
    ]
    assert plan["summary"]["readonly_preflight_step_count"] == 1
    assert "readonly_preflight_candidate_required_before_supervised_proof" in plan["summary"]["global_blockers"]
    assert "operator_must_replace_invalid_readonly_preflight_artifact" in plan["summary"]["global_blockers"]
    assert plan["steps"] == []

    preflight_step = plan["readonly_preflight_steps"][0]
    assert preflight_step["read_only"] is True
    assert preflight_step["execution_allowed_by_plan"] is False
    assert "operator_must_replace_invalid_readonly_preflight_artifact" in preflight_step["abort_conditions"]
    assert "--require-physical-readback" in preflight_step["command"]
    assert_readonly_preflight_command_bindings(preflight_step)
    assert "--execute" not in preflight_step["command"]


def test_invalid_artifacts_block_plan_generation():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", blockers=["invalid_proof_summaries_present"]),
            evidence_row("fader", blockers=["physical_candidate_missing"]),
        )
    )

    assert plan["status"] == "blocked_by_invalid_artifacts"
    assert plan["steps"] == []
    assert plan["summary"]["global_blockers"] == ["invalid_proof_summaries_present"]
