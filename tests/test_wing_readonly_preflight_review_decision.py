import json
from pathlib import Path

from backend.wing_readonly_preflight_evidence import build_wing_readonly_preflight_evidence
from backend.wing_readonly_preflight_review_decision import (
    REQUIRED_REVIEW_CHECKLIST_IDS,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_METHOD_ID,
    build_wing_readonly_preflight_review_decision_ledger,
)


def _valid_summary() -> dict:
    return {
        "run_id": "20260606T120000Z_wing_readonly_preflight",
        "status": "passed",
        "reason": "wing_readonly_preflight_passed",
        "captured_at": "2026-06-07T01:00:00Z",
        "proof_artifact_schema": "wing_readonly_preflight.v1",
        "proof_runner": "wing_readonly_preflight_runner",
        "connection_mode": "wing",
        "channels": [3],
        "operator_channel_confirmation_required": True,
        "operator_confirmed_channels": [3],
        "operator_session_confirmation_required": True,
        "operator_confirmed_session_ready": True,
        "value_types": ["fader", "gain"],
        "require_physical_readback": True,
        "request_id_policy": "required_for_every_query_mixer_parameter_response",
        "expected_plan_path": "artifacts/wing_readonly_preflight_plan/latest_plan.json",
        "expected_plan_sha256": "0" * 64,
        "expected_plan_match": True,
        "expected_plan_violations": [],
        "expected_plan_required_for_real_run": True,
        "physical_preflight_requires_existing_backend_connection": True,
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
        "summary_path": "artifacts/wing_readonly_preflight/run/summary.json",
        "summary_sha256": "1" * 64,
        "events_path": "artifacts/wing_readonly_preflight/run/events.jsonl",
        "events_sha256": "2" * 64,
    }


def _evidence_with_candidate() -> dict:
    return build_wing_readonly_preflight_evidence(
        artifact_payloads=(_valid_summary(),),
        artifact_roots=(),
    )


def _accepted_record(**overrides) -> dict:
    record = {
        "review_id": "review-wing-readonly-1",
        "artifact_source_id": "inline_payload_0",
        "run_id": "20260606T120000Z_wing_readonly_preflight",
        "artifact_captured_at": "2026-06-07T01:00:00Z",
        "summary_path": "artifacts/wing_readonly_preflight/run/summary.json",
        "summary_sha256": "1" * 64,
        "events_path": "artifacts/wing_readonly_preflight/run/events.jsonl",
        "events_sha256": "2" * 64,
        "decision": "accepted",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-07T01:05:00Z",
        "checklist_ids": list(REQUIRED_REVIEW_CHECKLIST_IDS),
    }
    record.update(overrides)
    return record


def test_review_decision_ledger_requires_preflight_candidate_first():
    ledger = build_wing_readonly_preflight_review_decision_ledger()

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
    assert ledger["summary"]["readiness_matrix_update_allowed_by_decision"] is False
    assert ledger["summary"]["live_authority_granted"] is False


def test_review_decision_ledger_tracks_unreviewed_preflight_candidate():
    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=_evidence_with_candidate()
    )

    assert ledger["status"] == "operator_review_required"
    assert ledger["summary"]["review_candidate_count"] == 1
    assert ledger["summary"]["unreviewed_candidate_count"] == 1
    assert ledger["summary"]["unreviewed_candidate_source_ids"] == ["inline_payload_0"]
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
    assert ledger["summary"]["readiness_matrix_update_allowed_by_decision"] is False
    assert ledger["summary"]["readiness_closure_allowed_by_decision"] is False
    assert ledger["summary"]["live_authority_granted"] is False
    assert ledger["summary"]["full_autonomous_ready"] is False
    record = ledger["records"][0]
    assert record["operator_accepted_readonly_preflight_candidate"] is True
    assert record["readiness_matrix_update_allowed_by_decision"] is False
    assert record["live_authority_granted"] is False


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
