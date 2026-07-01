import json
from pathlib import Path

from backend.wing_readonly_preflight_evidence import (
    EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE_SUMMARY,
    build_wing_readonly_preflight_evidence,
)
from backend.wing_readonly_preflight_contract import (
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE,
    WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_ARTIFACT_ROOT_BLOCKER,
    WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER,
    WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_REQUEST_SEQUENCE_BLOCKER,
    WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_VERIFIER_WS_PROOF_BLOCKER,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_PRESTART_PROOF_FIELDS,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS,
    WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS,
    WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME,
    wing_readonly_preflight_accepted_candidate_artifact_root_contract,
    wing_readonly_preflight_accepted_candidate_request_sequence_source_ids,
    wing_readonly_preflight_accepted_candidate_request_sequence_violations_by_source,
    wing_readonly_preflight_accepted_candidate_verifier_ws_proof_source_ids,
    wing_readonly_preflight_accepted_candidate_verifier_ws_proof_violations_by_source,
    wing_readonly_preflight_accepted_candidate_prestart_violations_by_source,
    wing_readonly_preflight_accepted_candidate_prestart_source_ids,
)
from backend.wing_readonly_preflight_review_decision import (
    REQUIRED_REVIEW_CHECKLIST_IDS,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_METHOD_ID,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_SOURCE_AUTHORITY_FLAGS,
    build_wing_readonly_preflight_review_decision_ledger,
)
from tools import review_wing_readonly_preflight_decisions as decision_review_tool


def test_review_decision_checklist_ids_match_shared_contract():
    assert (
        REQUIRED_REVIEW_CHECKLIST_IDS
        == WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS
    )


def test_accepted_candidate_prestart_trace_uses_shared_contract_helper():
    source_id = (
        "artifacts/wing_readonly_preflight/"
        "20260606T120000Z_wing_readonly_preflight/summary.json"
    )
    candidate_review = {
        "source_id": source_id,
        "summary_path": source_id,
        "events_path": (
            "artifacts/wing_readonly_preflight/"
            "20260606T120000Z_wing_readonly_preflight/events.jsonl"
        ),
        "expected_artifacts": [
            f"artifacts/wing_readonly_preflight/<run_id>/{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}",
            f"artifacts/wing_readonly_preflight/<run_id>/{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}",
        ],
        "expected_summary_artifact": source_id,
        "expected_events_artifact": (
            "artifacts/wing_readonly_preflight/"
            "20260606T120000Z_wing_readonly_preflight/events.jsonl"
        ),
        "summary_path_matches_expected_artifact": True,
        "events_path_matches_expected_artifact": True,
        "backend_port_prestart_check_required": True,
        "backend_port_prestart_check_must_pass_before_starting_backend": True,
        "backend_port_prestart_check_command": list(
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
        ),
        "backend_port_prestart_check_command_text": (
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
        ),
        "backend_port_prestart_check_failure_detail_fields": list(
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
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
            for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
        },
        **{
            field: False
            for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
        },
        **{
            field: True
            for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS
        },
    }

    assert wing_readonly_preflight_accepted_candidate_prestart_source_ids(
        [source_id],
        [candidate_review],
    ) == [source_id]
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_violations_by_source(
            [source_id],
            [candidate_review],
        )
        == {}
    )

    forged_review = dict(candidate_review, backend_port_prestart_summary_sha256="")
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_source_ids(
            [source_id],
            [forged_review],
        )
        == []
    )
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_violations_by_source(
            [source_id],
            [forged_review],
        )
        == {source_id: ["backend_port_prestart_summary_sha256_missing"]}
    )
    malformed_digest_review = dict(
        candidate_review,
        backend_port_prestart_summary_sha256="not-a-sha256",
    )
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_source_ids(
            [source_id],
            [malformed_digest_review],
        )
        == []
    )
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_violations_by_source(
            [source_id],
            [malformed_digest_review],
        )
        == {source_id: ["backend_port_prestart_summary_sha256_invalid"]}
    )
    missing_verifier_pass_review = dict(
        candidate_review,
        artifact_manifest_verifier_passed_before_websocket=False,
    )
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_source_ids(
            [source_id],
            [missing_verifier_pass_review],
        )
        == []
    )
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_violations_by_source(
            [source_id],
            [missing_verifier_pass_review],
        )
        == {
            source_id: [
                "artifact_manifest_verifier_passed_before_websocket_not_true"
            ]
        }
    )
    missing_verifier_websocket_review = dict(
        candidate_review,
        artifact_manifest_verifier_websocket_opened_after_pass=False,
    )
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_source_ids(
            [source_id],
            [missing_verifier_websocket_review],
        )
        == []
    )
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_violations_by_source(
            [source_id],
            [missing_verifier_websocket_review],
        )
        == {
            source_id: [
                "artifact_manifest_verifier_websocket_opened_after_pass_not_true"
            ]
        }
    )
    unbound_review = dict(
        candidate_review,
        summary_path_matches_expected_artifact=False,
    )
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_source_ids(
            [source_id],
            [unbound_review],
        )
        == []
    )
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_violations_by_source(
            [source_id],
            [unbound_review],
        )
        == {source_id: ["summary_path_not_bound_to_expected_artifact"]}
    )
    mismatched_summary_review = dict(
        candidate_review,
        expected_summary_artifact="artifacts/wing_readonly_preflight/other/summary.json",
    )
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_source_ids(
            [source_id],
            [mismatched_summary_review],
        )
        == []
    )
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_violations_by_source(
            [source_id],
            [mismatched_summary_review],
        )
        == {source_id: ["expected_summary_artifact_mismatch"]}
    )
    mismatched_events_review = dict(
        candidate_review,
        expected_events_artifact="artifacts/wing_readonly_preflight/other/events.jsonl",
    )
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_source_ids(
            [source_id],
            [mismatched_events_review],
        )
        == []
    )
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_violations_by_source(
            [source_id],
            [mismatched_events_review],
        )
        == {source_id: ["expected_events_artifact_mismatch"]}
    )
    assert (
        wing_readonly_preflight_accepted_candidate_prestart_violations_by_source(
            ["missing-summary.json"],
            [candidate_review],
        )
        == {"missing-summary.json": ["candidate_review_missing"]}
    )


def test_accepted_candidate_artifact_root_contract_rejects_parent_traversal():
    source_id = (
        "artifacts/wing_readonly_preflight/"
        "20260606T120000Z_wing_readonly_preflight/summary.json"
    )
    traversal_summary = (
        "artifacts/wing_readonly_preflight/../stale_preflight/run/summary.json"
    )
    traversal_events = (
        "artifacts/wing_readonly_preflight/../stale_preflight/run/events.jsonl"
    )
    artifact_roots, violations = (
        wing_readonly_preflight_accepted_candidate_artifact_root_contract(
            [source_id],
            [
                {
                    "source_id": source_id,
                    "artifact_root": "artifacts/wing_readonly_preflight",
                    "summary_path": traversal_summary,
                    "events_path": traversal_events,
                    "expected_summary_artifact": traversal_summary,
                    "expected_events_artifact": traversal_events,
                }
            ],
        )
    )

    assert artifact_roots == ["artifacts/wing_readonly_preflight"]
    assert set(violations[source_id]) == {
        "summary_path_contains_parent_traversal",
        "summary_path_not_under_artifact_root",
        "events_path_contains_parent_traversal",
        "events_path_not_under_artifact_root",
        "expected_summary_artifact_contains_parent_traversal",
        "expected_summary_artifact_not_under_artifact_root",
        "expected_events_artifact_contains_parent_traversal",
        "expected_events_artifact_not_under_artifact_root",
    }


def test_accepted_candidate_artifact_root_contract_rejects_parent_traversal_in_root():
    source_id = (
        "artifacts/wing_readonly_preflight/"
        "20260606T120000Z_wing_readonly_preflight/summary.json"
    )
    events_path = (
        "artifacts/wing_readonly_preflight/"
        "20260606T120000Z_wing_readonly_preflight/events.jsonl"
    )
    artifact_roots, violations = (
        wing_readonly_preflight_accepted_candidate_artifact_root_contract(
            [source_id],
            [
                {
                    "source_id": source_id,
                    "artifact_root": "artifacts/wing_readonly_preflight/..",
                    "summary_path": source_id,
                    "events_path": events_path,
                    "expected_summary_artifact": source_id,
                    "expected_events_artifact": events_path,
                }
            ],
        )
    )

    assert artifact_roots == ["artifacts/wing_readonly_preflight/.."]
    assert violations == {
        source_id: ["artifact_root_contains_parent_traversal"]
    }


def _valid_summary() -> dict:
    return {
        "run_id": "20260606T120000Z_wing_readonly_preflight",
        "status": "passed",
        "reason": "wing_readonly_preflight_passed",
        "captured_at": "2026-06-07T01:00:00Z",
        "proof_artifact_schema": "wing_readonly_preflight.v1",
        "proof_runner": "wing_readonly_preflight_runner",
        "artifact_root": "artifacts/wing_readonly_preflight",
        "connection_mode": "wing",
        "channels": [3],
        "operator_channel_confirmation_required": True,
        "operator_confirmed_channels": [3],
        "operator_session_confirmation_required": True,
        "operator_confirmed_session_ready": True,
        "value_types": ["fader", "gain"],
        "require_physical_readback": True,
        "request_id_policy": "required_for_every_query_mixer_parameter_response",
        "expected_response_types_by_request_type": dict(
            EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE_SUMMARY
        ),
        "expected_plan_path": "artifacts/wing_readonly_preflight_plan/latest_plan.json",
        "expected_plan_sha256": "0" * 64,
        "expected_plan_match": True,
        "expected_plan_violations": [],
        "expected_plan_required_for_real_run": True,
        "physical_preflight_requires_existing_backend_connection": True,
        "backend_port_prestart_check_required": True,
        "backend_port_prestart_check_must_pass_before_starting_backend": True,
        "backend_port_prestart_check_command": list(
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
        ),
        "backend_port_prestart_check_command_text": (
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
        ),
        "backend_port_prestart_check_failure_detail_fields": list(
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
        ),
        "backend_port_prestart_summary_required": True,
        "backend_port_prestart_summary_path": "artifacts/no_wing_release_check/run/summary.json",
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
            for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
        },
        **{
            field: False
            for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
        },
        **{
            field: True
            for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS
        },
        "read_only": True,
        "live_write": False,
        "write_request_sent": False,
        "proposal_request_sent": False,
        "sent_request_types": [
            "get_operator_mode",
            "get_state",
            "query_mixer_parameter",
            "query_mixer_parameter",
        ],
        "contacts_backend": True,
        "queries_console": True,
        "forbidden_request_types_sent": [],
        "readback_count": 2,
        "physical_readback_count": 2,
        "readbacks": [
            {
                "type": "mixer_parameter_value",
                "request_id": "wing-readonly-preflight:ch3:fader",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "channel": 3,
                "value_type": "fader",
                "address": "/ch/3/fdr",
                "value": -12.0,
            },
            {
                "type": "mixer_parameter_value",
                "request_id": "wing-readonly-preflight:ch3:gain",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "channel": 3,
                "value_type": "gain",
                "address": "/ch/3/in/set/trim",
                "value": -0.25,
            },
        ],
        "summary_path": (
            "artifacts/wing_readonly_preflight/"
            "20260606T120000Z_wing_readonly_preflight/summary.json"
        ),
        "summary_sha256": "1" * 64,
        "events_path": (
            "artifacts/wing_readonly_preflight/"
            "20260606T120000Z_wing_readonly_preflight/events.jsonl"
        ),
        "events_sha256": "2" * 64,
    }


def _expected_request_sequence() -> list[dict]:
    return [
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
    ]


def _evidence_with_candidate() -> dict:
    evidence = build_wing_readonly_preflight_evidence(
        artifact_payloads=(_valid_summary(),),
        artifact_roots=(),
    )
    candidate = evidence["artifacts"][0]
    candidate["expected_artifacts"] = [
        "artifacts/wing_readonly_preflight/<run_id>/summary.json",
        "artifacts/wing_readonly_preflight/<run_id>/events.jsonl",
    ]
    candidate["expected_summary_artifact"] = (
        "artifacts/wing_readonly_preflight/"
        "20260606T120000Z_wing_readonly_preflight/summary.json"
    )
    candidate["expected_events_artifact"] = (
        "artifacts/wing_readonly_preflight/"
        "20260606T120000Z_wing_readonly_preflight/events.jsonl"
    )
    candidate["summary_path_matches_expected_artifact"] = True
    candidate["events_path_matches_expected_artifact"] = True
    candidate["event_log_run_id_matches_summary"] = True
    candidate["request_sequence"] = _expected_request_sequence()
    candidate["plan_request_sequence"] = _expected_request_sequence()
    candidate["request_sequence_matches_plan"] = True
    return evidence


def _accepted_record(**overrides) -> dict:
    valid_summary = _valid_summary()
    record = {
        "review_id": "review-wing-readonly-1",
        "artifact_source_id": "inline_payload_0",
        "run_id": "20260606T120000Z_wing_readonly_preflight",
        "artifact_captured_at": "2026-06-07T01:00:00Z",
        "artifact_root": "artifacts/wing_readonly_preflight",
        "summary_path": (
            "artifacts/wing_readonly_preflight/"
            "20260606T120000Z_wing_readonly_preflight/summary.json"
        ),
        "summary_sha256": "1" * 64,
        "events_path": (
            "artifacts/wing_readonly_preflight/"
            "20260606T120000Z_wing_readonly_preflight/events.jsonl"
        ),
        "events_sha256": "2" * 64,
        "channels": [3],
        "value_types": ["fader", "gain"],
        "operator_confirmed_channels": [3],
        "operator_confirmed_session_ready": True,
        "readback_count": 2,
        "physical_readback_count": 2,
        "request_id_policy": "required_for_every_query_mixer_parameter_response",
        "request_id_correlation_required": True,
        "event_log_run_id_matches_summary": True,
        "decision": "accepted",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-07T01:05:00Z",
        "checklist_ids": list(REQUIRED_REVIEW_CHECKLIST_IDS),
        **{
            field: valid_summary[field]
            for field in WING_READONLY_PREFLIGHT_REVIEW_DECISION_PRESTART_PROOF_FIELDS
        },
    }
    record.update(overrides)
    return record


def test_review_decision_ledger_requires_preflight_candidate_first():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence={"artifacts": []},
        decision_roots=(),
    )

    assert ledger["type"] == "wing_readonly_preflight_review_decision_ledger"
    assert ledger["method_id"] == WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_METHOD_ID
    assert ledger["status"] == "operator_review_required"
    assert ledger["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "operator_required": True,
        "execution_allowed_by_decision": False,
        "readiness_matrix_update_allowed_by_decision": False,
        "readiness_closure_allowed_by_decision": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }
    assert ledger["summary"]["review_candidate_count"] == 0
    assert ledger["summary"]["decision_record_count"] == 0
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    assert ledger["summary"]["source_authority_denied"] is True
    assert ledger["summary"]["source_authority_flags"] == {
        flag: False
        for flag in WING_READONLY_PREFLIGHT_REVIEW_DECISION_SOURCE_AUTHORITY_FLAGS
    }
    assert ledger["summary"]["readiness_matrix_update_allowed_by_decision"] is False
    assert ledger["summary"]["live_authority_granted"] is False


def test_review_decision_ledger_ignores_discovered_pending_template(tmp_path: Path):
    decision_root = tmp_path / "decisions"
    template_path = decision_root / Path(
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
    ).name
    template_path.parent.mkdir(parents=True)
    template_path.write_text(
        json.dumps(
            _accepted_record(
                decision="pending_operator_decision",
                reviewed_by="",
                reviewed_at="",
                template_status=WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_roots=(decision_root,),
    )

    assert ledger["status"] == "operator_review_required"
    assert ledger["summary"]["decision_record_count"] == 0
    assert ledger["summary"]["invalid_record_count"] == 0
    assert ledger["summary"]["unreviewed_candidate_count"] == 1

    explicit = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_paths=(template_path,),
    )
    assert explicit["status"] == "operator_review_decision_replacement_required"
    assert explicit["summary"]["invalid_record_count"] == 1
    assert "review_decision_unknown" in explicit["records"][0]["violations"]
    assert (
        "review_decision_template_status_unedited"
        in explicit["records"][0]["violations"]
    )


def test_review_decision_ledger_tracks_unreviewed_preflight_candidate():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate()
    )

    assert ledger["status"] == "operator_review_required"
    assert ledger["summary"]["review_candidate_count"] == 1
    assert ledger["summary"]["unreviewed_candidate_count"] == 1
    assert ledger["summary"]["unreviewed_candidate_source_ids"] == ["inline_payload_0"]
    assert ledger["summary"]["candidate_reviews"] == [
        {
            "source_id": "inline_payload_0",
            "status": "valid_readonly_preflight_artifact",
            "run_id": "20260606T120000Z_wing_readonly_preflight",
            "captured_at": "2026-06-07T01:00:00Z",
            "artifact_root": "artifacts/wing_readonly_preflight",
            "summary_path": (
                "artifacts/wing_readonly_preflight/"
                "20260606T120000Z_wing_readonly_preflight/summary.json"
            ),
            "summary_sha256": "1" * 64,
            "events_path": (
                "artifacts/wing_readonly_preflight/"
                "20260606T120000Z_wing_readonly_preflight/events.jsonl"
            ),
            "events_sha256": "2" * 64,
            "expected_artifacts": [
                "artifacts/wing_readonly_preflight/<run_id>/summary.json",
                "artifacts/wing_readonly_preflight/<run_id>/events.jsonl",
            ],
            "expected_summary_artifact": (
                "artifacts/wing_readonly_preflight/"
                "20260606T120000Z_wing_readonly_preflight/summary.json"
            ),
            "expected_events_artifact": (
                "artifacts/wing_readonly_preflight/"
                "20260606T120000Z_wing_readonly_preflight/events.jsonl"
            ),
            "summary_path_matches_expected_artifact": True,
            "events_path_matches_expected_artifact": True,
            "contacts_backend": True,
            "queries_console": True,
            "request_sequence": _expected_request_sequence(),
            "plan_request_sequence": _expected_request_sequence(),
            "request_sequence_matches_plan": True,
            "backend_port_prestart_check_required": True,
            "backend_port_prestart_check_must_pass_before_starting_backend": True,
            "backend_port_prestart_check_command": [
                "PYTHONPATH=backend",
                "python3",
                "tools/no_wing_release_check.py",
                "--quick",
            ],
            "backend_port_prestart_check_command_text": (
                WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
            ),
            "backend_port_prestart_check_failure_detail_fields": list(
                WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
            ),
            "backend_port_prestart_summary_required": True,
            "backend_port_prestart_summary_path": (
                "artifacts/no_wing_release_check/run/summary.json"
            ),
            "backend_port_prestart_summary_sha256": "3" * 64,
            "backend_port_prestart_summary_type": "no_wing_release_check",
            "backend_port_prestart_summary_status": "passed",
            "backend_port_prestart_summary_connection_policy": (
                "no_hardware_no_network_no_live_writes"
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
                for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
            },
            **{
                field: False
                for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
            },
            **{
                field: True
                for field in WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS
            },
        }
    ]
    assert "operator_must_record_readonly_preflight_review_decision" in ledger[
        "summary"
    ]["blocked_by"]


def test_review_decision_ledger_records_accepted_candidate_without_live_authority():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[_accepted_record()],
    )

    assert ledger["status"] == "operator_review_decision_records_available"
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["rejected_record_count"] == 0
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is True
    assert ledger["summary"]["source_authority_denied"] is True
    assert ledger["summary"]["source_authority_flags"] == {
        flag: False
        for flag in WING_READONLY_PREFLIGHT_REVIEW_DECISION_SOURCE_AUTHORITY_FLAGS
    }
    assert ledger["summary"]["accepted_candidate_prestart_contract_valid"] is True
    assert ledger["summary"]["accepted_candidate_prestart_source_ids"] == [
        "inline_payload_0"
    ]
    assert (
        ledger["summary"]["accepted_candidate_prestart_contract_violations_by_source"]
        == {}
    )
    assert ledger["summary"]["accepted_candidate_verifier_ws_proof_valid"] is True
    assert ledger["summary"]["accepted_candidate_verifier_ws_proof_source_ids"] == [
        "inline_payload_0"
    ] == wing_readonly_preflight_accepted_candidate_verifier_ws_proof_source_ids(
        ["inline_payload_0"],
        ledger["summary"]["candidate_reviews"],
    )
    assert (
        ledger["summary"]["accepted_candidate_verifier_ws_proof_violations_by_source"]
        == wing_readonly_preflight_accepted_candidate_verifier_ws_proof_violations_by_source(
            ["inline_payload_0"],
            ledger["summary"]["candidate_reviews"],
        )
    )
    assert ledger["summary"]["accepted_candidate_artifact_root_contract_valid"] is True
    assert ledger["summary"]["accepted_candidate_artifact_roots"] == [
        "artifacts/wing_readonly_preflight"
    ]
    assert (
        ledger["summary"][
            "accepted_candidate_artifact_root_contract_violations_by_source"
        ]
        == {}
    )
    assert (
        ledger["summary"]["accepted_candidate_request_sequence_contract_valid"]
        is True
    )
    assert ledger["summary"]["accepted_candidate_request_sequence_source_ids"] == [
        "inline_payload_0"
    ] == wing_readonly_preflight_accepted_candidate_request_sequence_source_ids(
        ["inline_payload_0"],
        ledger["summary"]["candidate_reviews"],
    )
    assert (
        ledger["summary"][
            "accepted_candidate_request_sequence_contract_violations_by_source"
        ]
        == wing_readonly_preflight_accepted_candidate_request_sequence_violations_by_source(
            ["inline_payload_0"],
            ledger["summary"]["candidate_reviews"],
        )
    )
    assert ledger["summary"]["readiness_matrix_update_allowed_by_decision"] is False
    assert ledger["summary"]["readiness_closure_allowed_by_decision"] is False
    assert ledger["summary"]["live_authority_granted"] is False


def test_review_decision_ledger_requires_record_prestart_proof_binding():
    record = _accepted_record()
    record.pop("artifact_manifest_verifier_passed_before_websocket")
    record["backend_port_prestart_summary_sha256"] = "4" * 64

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[record],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    violations = set(ledger["records"][0]["violations"])
    assert (
        "review_decision_artifact_manifest_verifier_passed_before_websocket_missing"
        in violations
    )
    assert "review_decision_backend_port_prestart_summary_sha256_mismatch" in violations
    assert ledger["records"][0]["operator_accepted_readonly_preflight_candidate"] is False
    assert ledger["records"][0]["live_authority_granted"] is False


def test_review_decision_ledger_blocks_accepted_candidate_without_request_sequence_trace():
    evidence = _evidence_with_candidate()
    evidence["artifacts"][0]["plan_request_sequence"] = []
    evidence["artifacts"][0]["request_sequence_matches_plan"] = False

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=evidence,
        decision_records=[_accepted_record()],
    )

    assert ledger["status"] == "operator_review_decision_records_available"
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    assert (
        ledger["summary"]["accepted_candidate_request_sequence_contract_valid"]
        is False
    )
    assert ledger["summary"]["accepted_candidate_request_sequence_source_ids"] == []
    assert ledger["summary"][
        "accepted_candidate_request_sequence_contract_violations_by_source"
    ] == {
        "inline_payload_0": [
            "plan_request_sequence_missing",
            "request_sequence_matches_plan_not_true",
        ]
    }
    assert (
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_REQUEST_SEQUENCE_BLOCKER
        in ledger["summary"]["blocked_by"]
    )
    assert ledger["summary"]["readiness_matrix_update_allowed_by_decision"] is False
    assert ledger["summary"]["live_authority_granted"] is False
    assert ledger["summary"]["full_autonomous_ready"] is False
    record = ledger["records"][0]
    assert record["operator_accepted_readonly_preflight_candidate"] is True
    assert record["readiness_matrix_update_allowed_by_decision"] is False
    assert record["live_authority_granted"] is False


def test_decision_review_report_surfaces_request_sequence_contract(monkeypatch):
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[_accepted_record()],
    )
    ledger["summary"]["manual_readiness_update_candidate_available"] = False
    ledger["summary"]["accepted_candidate_request_sequence_contract_valid"] = False
    ledger["summary"]["accepted_candidate_request_sequence_source_ids"] = []
    ledger["summary"][
        "accepted_candidate_request_sequence_contract_violations_by_source"
    ] = {
        "inline_payload_0": [
            "plan_request_sequence_missing",
            "request_sequence_matches_plan_not_true",
        ]
    }

    monkeypatch.setattr(
        decision_review_tool,
        "build_wing_readonly_preflight_evidence",
        lambda **kwargs: {"status": "readonly_preflight_candidate_available"},
    )
    monkeypatch.setattr(
        decision_review_tool,
        "build_wing_readonly_preflight_review_decision_ledger",
        lambda **kwargs: ledger,
    )

    report = decision_review_tool.build_decision_review_report()
    markdown = decision_review_tool.render_markdown(report)

    assert report["status"] == "blocked"
    assert report["manual_readiness_update_candidate_available"] is False
    assert report["accepted_candidate_request_sequence_contract_valid"] is False
    assert report["accepted_candidate_request_sequence_source_ids"] == []
    assert report[
        "accepted_candidate_request_sequence_contract_violations_by_source"
    ] == {
        "inline_payload_0": [
            "plan_request_sequence_missing",
            "request_sequence_matches_plan_not_true",
        ]
    }
    assert "valid accepted-candidate request-sequence trace" in report[
        "next_required_action"
    ]
    assert (
        "accepted_candidate_request_sequence_contract_valid: `false`"
        in markdown
    )
    assert "plan_request_sequence_missing" in markdown
    assert (
        decision_review_tool.WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_REQUEST_SEQUENCE_BLOCKER
        in markdown
    )


def test_decision_review_report_blocks_without_contact_trace(monkeypatch):
    evidence = _evidence_with_candidate()
    evidence["artifacts"][0]["contacts_backend"] = False
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=evidence,
        decision_records=[_accepted_record()],
    )

    monkeypatch.setattr(
        decision_review_tool,
        "build_wing_readonly_preflight_evidence",
        lambda **kwargs: {"status": "readonly_preflight_candidate_available"},
    )
    monkeypatch.setattr(
        decision_review_tool,
        "build_wing_readonly_preflight_review_decision_ledger",
        lambda **kwargs: ledger,
    )

    report = decision_review_tool.build_decision_review_report()
    markdown = decision_review_tool.render_markdown(report)

    assert ledger["summary"]["manual_readiness_update_candidate_available"] is True
    assert report["status"] == "blocked"
    assert report["manual_readiness_update_candidate_available"] is False
    assert report["accepted_candidate_contact_trace_complete"] is False
    assert report["accepted_candidate_contact_trace_source_ids"] == [
        "inline_payload_0"
    ]
    assert report["accepted_candidate_contacts_backend_by_source"] == {
        "inline_payload_0": False
    }
    assert report["accepted_candidate_queries_console_by_source"] == {
        "inline_payload_0": True
    }
    assert report["accepted_candidate_contact_trace_missing_source_ids"] == [
        "inline_payload_0"
    ]
    assert "valid accepted-candidate contact trace" in report[
        "next_required_action"
    ]
    assert "accepted_candidate_contact_trace_complete: `false`" in markdown
    assert '"inline_payload_0": false' in markdown
    assert (
        decision_review_tool.WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_CONTACT_TRACE_BLOCKER
        in markdown
    )


def test_decision_review_report_blocks_without_expected_artifact_trace(
    monkeypatch,
):
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[_accepted_record()],
    )
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is True
    candidate_review = ledger["summary"]["candidate_reviews"][0]
    candidate_review["expected_artifacts"] = []
    candidate_review["expected_summary_artifact"] = ""
    candidate_review["summary_path_matches_expected_artifact"] = False

    monkeypatch.setattr(
        decision_review_tool,
        "build_wing_readonly_preflight_evidence",
        lambda **kwargs: {"status": "readonly_preflight_candidate_available"},
    )
    monkeypatch.setattr(
        decision_review_tool,
        "build_wing_readonly_preflight_review_decision_ledger",
        lambda **kwargs: ledger,
    )

    report = decision_review_tool.build_decision_review_report()
    markdown = decision_review_tool.render_markdown(report)

    assert report["status"] == "blocked"
    assert report["manual_readiness_update_candidate_available"] is False
    assert report["accepted_candidate_contact_trace_complete"] is True
    assert report["accepted_candidate_expected_artifact_trace_complete"] is False
    assert report["accepted_candidate_expected_artifact_trace_source_ids"] == [
        "inline_payload_0"
    ]
    assert report[
        "accepted_candidate_expected_artifact_trace_missing_source_ids"
    ] == ["inline_payload_0"]
    assert report["accepted_candidate_expected_artifacts_by_source"] == {
        "inline_payload_0": []
    }
    assert report[
        "accepted_candidate_summary_path_matches_expected_artifact_by_source"
    ] == {"inline_payload_0": False}
    assert "valid accepted-candidate expected-artifact trace" in report[
        "next_required_action"
    ]
    assert "expected-artifact-bound accepted source id" in report[
        "next_required_action"
    ]
    assert (
        "accepted_candidate_expected_artifact_trace_complete: `false`"
        in markdown
    )
    assert "expected-artifact-bound accepted source id" in markdown
    assert (
        decision_review_tool.WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_EXPECTED_ARTIFACT_TRACE_BLOCKER
        in markdown
    )


def test_review_decision_ledger_blocks_accepted_candidate_without_prestart_contract():
    evidence = _evidence_with_candidate()
    evidence["artifacts"][0]["backend_port_prestart_check_required"] = False

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=evidence,
        decision_records=[_accepted_record(backend_port_prestart_check_required=False)],
    )

    assert ledger["status"] == "operator_review_decision_records_available"
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    assert ledger["summary"]["accepted_candidate_prestart_contract_valid"] is False
    assert ledger["summary"]["accepted_candidate_prestart_source_ids"] == []
    assert ledger["summary"][
        "accepted_candidate_prestart_contract_violations_by_source"
    ] == {"inline_payload_0": ["backend_port_prestart_check_required_not_true"]}
    assert (
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER
        in ledger["summary"]["blocked_by"]
    )
    assert ledger["summary"]["readiness_matrix_update_allowed_by_decision"] is False
    assert ledger["summary"]["live_authority_granted"] is False


def test_review_decision_ledger_blocks_accepted_candidate_with_artifact_root_violation():
    evidence = _evidence_with_candidate()
    evidence["artifacts"][0]["artifact_root"] = "artifacts/wing_readonly_preflight/.."

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=evidence,
        decision_records=[
            _accepted_record(artifact_root="artifacts/wing_readonly_preflight/.."),
        ],
    )

    assert ledger["status"] == "operator_review_decision_records_available"
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    assert ledger["summary"]["accepted_candidate_prestart_contract_valid"] is True
    assert (
        ledger["summary"]["accepted_candidate_artifact_root_contract_valid"]
        is False
    )
    assert ledger["summary"]["accepted_candidate_artifact_roots"] == [
        "artifacts/wing_readonly_preflight/.."
    ]
    assert ledger["summary"][
        "accepted_candidate_artifact_root_contract_violations_by_source"
    ] == {"inline_payload_0": ["artifact_root_contains_parent_traversal"]}
    assert (
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_ARTIFACT_ROOT_BLOCKER
        in ledger["summary"]["blocked_by"]
    )
    assert ledger["summary"]["readiness_matrix_update_allowed_by_decision"] is False
    assert ledger["summary"]["live_authority_granted"] is False


def test_review_decision_ledger_blocks_accepted_candidate_without_prestart_summary_binding():
    evidence = _evidence_with_candidate()
    evidence["artifacts"][0]["backend_port_prestart_summary_sha256"] = None

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=evidence,
        decision_records=[_accepted_record(backend_port_prestart_summary_sha256=None)],
    )

    assert ledger["status"] == "operator_review_decision_records_available"
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    assert ledger["summary"]["accepted_candidate_prestart_contract_valid"] is False
    assert ledger["summary"]["accepted_candidate_prestart_source_ids"] == []
    assert ledger["summary"][
        "accepted_candidate_prestart_contract_violations_by_source"
    ] == {"inline_payload_0": ["backend_port_prestart_summary_sha256_missing"]}
    assert (
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER
        in ledger["summary"]["blocked_by"]
    )


def test_review_decision_ledger_blocks_accepted_candidate_with_malformed_prestart_digest():
    evidence = _evidence_with_candidate()
    evidence["artifacts"][0]["backend_port_prestart_summary_sha256"] = "not-a-sha256"

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=evidence,
        decision_records=[
            _accepted_record(backend_port_prestart_summary_sha256="not-a-sha256")
        ],
    )

    assert ledger["status"] == "operator_review_decision_records_available"
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    assert ledger["summary"]["accepted_candidate_prestart_contract_valid"] is False
    assert ledger["summary"]["accepted_candidate_prestart_source_ids"] == []
    assert ledger["summary"][
        "accepted_candidate_prestart_contract_violations_by_source"
    ] == {"inline_payload_0": ["backend_port_prestart_summary_sha256_invalid"]}
    assert (
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER
        in ledger["summary"]["blocked_by"]
    )


def test_review_decision_ledger_blocks_accepted_candidate_without_required_false_prestart_detail():
    evidence = _evidence_with_candidate()
    evidence["artifacts"][0]["backend_port_prestart_gate_live_authority_granted"] = True

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=evidence,
        decision_records=[
            _accepted_record(backend_port_prestart_gate_live_authority_granted=True)
        ],
    )

    assert ledger["status"] == "operator_review_decision_records_available"
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    assert ledger["summary"]["accepted_candidate_prestart_contract_valid"] is False
    assert ledger["summary"]["accepted_candidate_prestart_source_ids"] == []
    assert ledger["summary"]["accepted_candidate_verifier_ws_proof_valid"] is True
    assert ledger["summary"]["accepted_candidate_verifier_ws_proof_source_ids"] == [
        "inline_payload_0"
    ]
    assert ledger["summary"][
        "accepted_candidate_prestart_contract_violations_by_source"
    ] == {
        "inline_payload_0": [
            "backend_port_prestart_gate_live_authority_granted_not_false"
        ]
    }
    assert (
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER
        in ledger["summary"]["blocked_by"]
    )


def test_review_decision_ledger_blocks_accepted_candidate_without_verifier_websocket_proof():
    evidence = _evidence_with_candidate()
    evidence["artifacts"][0]["artifact_manifest_verifier_websocket_opened_after_pass"] = False

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=evidence,
        decision_records=[
            _accepted_record(
                artifact_manifest_verifier_websocket_opened_after_pass=False
            )
        ],
    )

    assert ledger["status"] == "operator_review_decision_records_available"
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    assert ledger["summary"]["accepted_candidate_prestart_contract_valid"] is False
    assert ledger["summary"]["accepted_candidate_prestart_source_ids"] == []
    assert ledger["summary"][
        "accepted_candidate_prestart_contract_violations_by_source"
    ] == {
        "inline_payload_0": [
            "artifact_manifest_verifier_websocket_opened_after_pass_not_true"
        ]
    }
    assert ledger["summary"][
        "accepted_candidate_verifier_ws_proof_violations_by_source"
    ] == {
        "inline_payload_0": [
            "artifact_manifest_verifier_websocket_opened_after_pass_not_true"
        ]
    }
    assert (
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER
        in ledger["summary"]["blocked_by"]
    )
    assert (
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_VERIFIER_WS_PROOF_BLOCKER
        in ledger["summary"]["blocked_by"]
    )


def test_review_decision_ledger_blocks_accepted_candidate_without_expected_artifact_binding():
    evidence = _evidence_with_candidate()
    evidence["artifacts"][0]["summary_path_matches_expected_artifact"] = False

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=evidence,
        decision_records=[_accepted_record()],
    )

    assert ledger["status"] == "operator_review_decision_records_available"
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    assert ledger["summary"]["accepted_candidate_prestart_contract_valid"] is False
    assert ledger["summary"]["accepted_candidate_prestart_source_ids"] == []
    assert ledger["summary"][
        "accepted_candidate_prestart_contract_violations_by_source"
    ] == {"inline_payload_0": ["summary_path_not_bound_to_expected_artifact"]}
    assert (
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER
        in ledger["summary"]["blocked_by"]
    )


def test_review_decision_ledger_blocks_accepted_candidate_with_mismatched_expected_artifact_paths():
    evidence = _evidence_with_candidate()
    evidence["artifacts"][0]["expected_summary_artifact"] = (
        "artifacts/wing_readonly_preflight/other/summary.json"
    )

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=evidence,
        decision_records=[_accepted_record()],
    )

    assert ledger["status"] == "operator_review_decision_records_available"
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    assert ledger["summary"]["accepted_candidate_prestart_contract_valid"] is False
    assert ledger["summary"]["accepted_candidate_prestart_source_ids"] == []
    assert ledger["summary"][
        "accepted_candidate_prestart_contract_violations_by_source"
    ] == {"inline_payload_0": ["expected_summary_artifact_mismatch"]}
    assert (
        WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER
        in ledger["summary"]["blocked_by"]
    )

    evidence = _evidence_with_candidate()
    evidence["artifacts"][0]["expected_events_artifact"] = (
        "artifacts/wing_readonly_preflight/other/events.jsonl"
    )

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=evidence,
        decision_records=[_accepted_record()],
    )

    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    assert ledger["summary"]["accepted_candidate_prestart_contract_valid"] is False
    assert ledger["summary"][
        "accepted_candidate_prestart_contract_violations_by_source"
    ] == {"inline_payload_0": ["expected_events_artifact_mismatch"]}


def test_review_decision_ledger_loads_decision_record_from_json_file(tmp_path):
    decision_path = tmp_path / "review-decision.json"
    decision_path.write_text(
        json.dumps(_accepted_record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_paths=(decision_path,),
    )

    assert ledger["status"] == "operator_review_decision_records_available"
    assert ledger["summary"]["decision_record_count"] == 1
    assert ledger["summary"]["accepted_record_count"] == 1
    assert ledger["records"][0]["review_id"] == "review-wing-readonly-1"


def test_review_decision_ledger_records_rejected_candidate_without_authority():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[_accepted_record(decision="rejected", review_id="review-reject")],
    )

    assert ledger["status"] == "operator_review_decision_records_available"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["rejected_record_count"] == 1
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    assert ledger["records"][0]["operator_rejected_readonly_preflight_candidate"] is True


def test_review_decision_ledger_rejects_duplicate_conflicting_candidate_decisions():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[
            _accepted_record(review_id="review-accept"),
            _accepted_record(
                review_id="review-reject",
                decision="rejected",
                reviewed_at="2026-06-07T01:10:00Z",
            ),
        ],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["rejected_record_count"] == 0
    assert ledger["summary"]["invalid_record_count"] == 2
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    assert ledger["summary"]["duplicate_source_ids"] == ["inline_payload_0"]
    assert ledger["summary"]["conflicting_source_ids"] == ["inline_payload_0"]
    assert ledger["summary"]["invalid_record_ids"] == ["review-accept", "review-reject"]
    for record in ledger["records"]:
        assert record["valid"] is False
        assert "review_decision_duplicate_artifact_source" in record["violations"]
        assert (
            "review_decision_conflicting_artifact_source_decisions"
            in record["violations"]
        )
        assert record["operator_accepted_readonly_preflight_candidate"] is False
        assert record["operator_rejected_readonly_preflight_candidate"] is False


def test_review_decision_ledger_rejects_duplicate_accepted_candidate_decisions():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[
            _accepted_record(review_id="review-accept-a"),
            _accepted_record(
                review_id="review-accept-b",
                reviewed_at="2026-06-07T01:10:00Z",
            ),
        ],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["rejected_record_count"] == 0
    assert ledger["summary"]["invalid_record_count"] == 2
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    assert ledger["summary"]["duplicate_source_ids"] == ["inline_payload_0"]
    assert ledger["summary"]["conflicting_source_ids"] == []
    assert ledger["summary"]["invalid_record_ids"] == [
        "review-accept-a",
        "review-accept-b",
    ]
    for record in ledger["records"]:
        assert record["valid"] is False
        assert "review_decision_duplicate_artifact_source" in record["violations"]
        assert (
            "review_decision_conflicting_artifact_source_decisions"
            not in record["violations"]
        )
        assert record["operator_accepted_readonly_preflight_candidate"] is False
        assert record["operator_rejected_readonly_preflight_candidate"] is False


def test_review_decision_ledger_rejects_unsafe_or_unmatched_decision():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[
            _accepted_record(
                run_id="wrong-run",
                checklist_ids=["confirm_artifact_paths"],
                live_authority_granted=True,
            )
        ],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["invalid_record_count"] == 1
    assert ledger["summary"]["invalid_record_ids"] == ["review-wing-readonly-1"]
    violations = set(ledger["records"][0]["violations"])
    assert "review_decision_run_id_mismatch" in violations
    assert "review_decision_checklist_ids_mismatch" in violations
    assert "review_decision_unsafe_authority_claim" in violations
    assert ledger["records"][0]["unsafe_authority_flags"] == ["live_authority_granted"]


def test_review_decision_ledger_rejects_old_checklist_without_event_run_id_binding():
    old_checklist = [
        item
        for item in REQUIRED_REVIEW_CHECKLIST_IDS
        if item != "confirm_event_log_run_id_binding"
    ]

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[_accepted_record(checklist_ids=old_checklist)],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["invalid_record_count"] == 1
    assert "review_decision_checklist_ids_mismatch" in (
        ledger["records"][0]["violations"]
    )


def test_review_decision_ledger_rejects_record_without_event_run_id_verdict():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[_accepted_record(event_log_run_id_matches_summary=False)],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["invalid_record_count"] == 1
    assert "review_decision_event_log_run_id_not_confirmed" in (
        ledger["records"][0]["violations"]
    )
    assert ledger["records"][0]["event_log_run_id_matches_summary"] is False


def test_review_decision_ledger_rejects_record_without_request_id_correlation_verdict():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[
            _accepted_record(
                request_id_policy="",
                request_id_correlation_required=False,
            )
        ],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["invalid_record_count"] == 1
    violations = set(ledger["records"][0]["violations"])
    assert "review_decision_request_id_policy_mismatch" in violations
    assert "review_decision_request_id_correlation_not_confirmed" in violations
    assert ledger["records"][0]["request_id_policy"] == ""
    assert ledger["records"][0]["request_id_correlation_required"] is False


def test_review_decision_ledger_rejects_stale_scope_fields():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[
            _accepted_record(
                channels=[4],
                value_types=["fader"],
                operator_confirmed_channels=[4],
                operator_confirmed_session_ready=False,
                readback_count=1,
                physical_readback_count=1,
            )
        ],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["invalid_record_count"] == 1
    violations = set(ledger["records"][0]["violations"])
    assert "review_decision_channels_mismatch" in violations
    assert "review_decision_value_types_mismatch" in violations
    assert "review_decision_operator_confirmed_channels_mismatch" in violations
    assert "review_decision_operator_session_not_confirmed" in violations
    assert "review_decision_readback_count_mismatch" in violations
    assert "review_decision_physical_readback_count_mismatch" in violations


def test_review_decision_ledger_rejects_coerced_channel_scope_fields():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[
            _accepted_record(
                channels=["3"],
                operator_confirmed_channels=[3.0],
            )
        ],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["invalid_record_count"] == 1
    violations = set(ledger["records"][0]["violations"])
    assert "review_decision_channels_mismatch" in violations
    assert "review_decision_operator_confirmed_channels_mismatch" in violations
    assert ledger["records"][0]["channels"] == []
    assert ledger["records"][0]["operator_confirmed_channels"] == []


def test_review_decision_ledger_rejects_stale_artifact_hashes():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[
            _accepted_record(
                summary_sha256="3" * 64,
                events_sha256="4" * 64,
            )
        ],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["invalid_record_count"] == 1
    violations = set(ledger["records"][0]["violations"])
    assert "review_decision_summary_sha256_mismatch" in violations
    assert "review_decision_events_sha256_mismatch" in violations


def test_review_decision_ledger_rejects_missing_or_stale_artifact_root():
    missing_root = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[_accepted_record(artifact_root="")],
    )
    stale_root = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[_accepted_record(artifact_root="artifacts/stale_preflight")],
    )

    assert missing_root["status"] == "operator_review_decision_replacement_required"
    assert missing_root["summary"]["invalid_record_count"] == 1
    assert (
        "review_decision_artifact_root_missing"
        in missing_root["records"][0]["violations"]
    )
    assert stale_root["status"] == "operator_review_decision_replacement_required"
    assert stale_root["summary"]["invalid_record_count"] == 1
    assert (
        "review_decision_artifact_root_mismatch"
        in stale_root["records"][0]["violations"]
    )


def test_review_decision_ledger_rejects_missing_review_id():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[_accepted_record(review_id="")],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["invalid_record_count"] == 1
    assert "review_decision_review_id_missing" in ledger["records"][0]["violations"]
    assert (
        ledger["records"][0]["operator_accepted_readonly_preflight_candidate"]
        is False
    )


def test_review_decision_ledger_rejects_unedited_template_status():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[
            _accepted_record(
                template_status=WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS,
            )
        ],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["invalid_record_count"] == 1
    assert (
        "review_decision_template_status_unedited"
        in ledger["records"][0]["violations"]
    )
    assert ledger["records"][0]["template_status"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS
    )
    assert (
        ledger["records"][0]["operator_accepted_readonly_preflight_candidate"]
        is False
    )


def test_review_decision_ledger_rejects_malformed_artifact_hashes():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[
            _accepted_record(
                summary_sha256="not-a-sha256",
                events_sha256="also-not-a-sha256",
            )
        ],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["invalid_record_count"] == 1
    violations = set(ledger["records"][0]["violations"])
    assert "review_decision_summary_sha256_invalid" in violations
    assert "review_decision_events_sha256_invalid" in violations


def test_review_decision_ledger_rejects_pre_capture_review_decision():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[
            _accepted_record(reviewed_at="2026-06-07T00:59:59Z"),
        ],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["invalid_record_count"] == 1
    violations = set(ledger["records"][0]["violations"])
    assert "review_decision_reviewed_at_before_artifact_captured_at" in violations


def test_review_decision_ledger_rejects_stale_artifact_capture_timestamp():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[
            _accepted_record(artifact_captured_at="2026-06-07T00:59:59Z"),
        ],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["invalid_record_count"] == 1
    violations = set(ledger["records"][0]["violations"])
    assert "review_decision_artifact_captured_at_mismatch" in violations


def test_review_decision_ledger_rejects_non_utc_timestamp():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[
            _accepted_record(reviewed_at="2026-06-07 04:05:00"),
        ],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["records"][0]["valid"] is False
    assert "review_decision_reviewed_at_invalid" in ledger["records"][0]["violations"]


def test_review_decision_ledger_rejects_invalid_candidate_forbidden_request_types():
    invalid_summary = _valid_summary()
    invalid_summary["forbidden_request_types_sent"] = ["set_gain"]

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=build_wing_readonly_preflight_evidence(
            artifact_payloads=(invalid_summary,),
            artifact_roots=(),
        ),
        decision_records=[_accepted_record()],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["invalid_record_count"] == 1
    assert ledger["summary"]["accepted_record_count"] == 0
    assert "review_decision_candidate_missing" in ledger["records"][0]["violations"]


def test_review_decision_ledger_rejects_records_without_artifact_source_id():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate(),
        decision_records=[_accepted_record(artifact_source_id="")],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["invalid_record_count"] == 1
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["records"][0]["valid"] is False
    assert "review_decision_candidate_missing" in ledger["records"][0]["violations"]
    assert "review_decision_artifact_source_missing" in ledger["records"][0]["violations"]


def test_review_decision_ledger_is_transport_free():
    source = Path("backend/wing_readonly_preflight_review_decision.py").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
        "set_fader(",
        "set_gain(",
        "apply_operator_proposal(",
    ]
    for token in forbidden:
        assert token not in source
