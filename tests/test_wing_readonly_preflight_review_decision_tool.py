import hashlib
import json
from pathlib import Path

from backend.wing_readonly_preflight_evidence import (
    EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND,
    EXPECTED_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    EXPECTED_PLAN_ALLOWED_REQUEST_TYPES,
    EXPECTED_PLAN_FAIL_CLOSED_CONDITIONS,
    EXPECTED_PLAN_FORBIDDEN_REQUEST_TYPES,
    EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE_SUMMARY,
    EXPECTED_SELECTED_MVP_BLOCKER,
    EXPECTED_SELECTED_MVP_BLOCKER_STATUS,
)
from backend.wing_readonly_preflight_review_decision import REQUIRED_REVIEW_CHECKLIST_IDS
from backend.wing_readonly_preflight_contract import (
    WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER,
    WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_DECISION_STATUS,
    expected_supervised_prestart_gate,
)
from tools import review_wing_readonly_preflight_decisions as decision_tool


def test_decision_review_ready_status_matches_supervised_gate_contract():
    assert decision_tool.ACCEPTED_DECISION_READY_STATUS == (
        WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_DECISION_STATUS
    )
    assert (
        expected_supervised_prestart_gate()["requires_operator_decision_status"]
        == decision_tool.ACCEPTED_DECISION_READY_STATUS
    )
    assert (
        decision_tool.DECISION_REVIEW_EXIT_CODES[
            decision_tool.ACCEPTED_DECISION_READY_STATUS
        ]
        == 0
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
        "value_types": ["fader", "gain", "pan"],
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
            EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND
        ),
        "backend_port_prestart_check_failure_detail_fields": list(
            EXPECTED_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
        ),
        "backend_port_prestart_summary_required": True,
        "backend_port_prestart_summary_path": "artifacts/no_wing_release_check/run/summary.json",
        "backend_port_prestart_summary_sha256": "2" * 64,
        "backend_port_prestart_summary_type": "no_wing_release_check",
        "backend_port_prestart_summary_status": "passed",
        "backend_port_prestart_summary_connection_policy": (
            "no_hardware_no_network_no_live_writes"
        ),
        "backend_port_prestart_summary_live_write": False,
        "backend_port_prestart_summary_contacts_backend": False,
        "backend_port_prestart_summary_queries_console": False,
        "backend_port_prestart_summary_executes_tests": False,
        "backend_port_prestart_check_status": "passed",
        "read_only": True,
        "live_write": False,
        "write_request_sent": False,
        "proposal_request_sent": False,
        "sent_request_types": [
            "get_operator_mode",
            "get_state",
            "query_mixer_parameter",
            "query_mixer_parameter",
            "query_mixer_parameter",
        ],
        "forbidden_request_types_sent": [],
        "readback_count": 3,
        "physical_readback_count": 3,
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
            {
                "type": "mixer_parameter_value",
                "request_id": "wing-readonly-preflight:ch3:pan",
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "channel": 3,
                "value_type": "pan",
                "address": "/ch/3/pan",
                "value": 0.0,
            },
        ],
    }


def _event(seq: int, event_type: str, payload: dict) -> dict:
    return {
        "seq": seq,
        "timestamp": "2026-06-06T12:00:00.000Z",
        "event_type": event_type,
        "payload": payload,
    }


def _valid_events(summary: dict) -> list[dict]:
    events = [
        _event(1, "send", {"type": "get_operator_mode"}),
        _event(2, "recv", {"type": "operator_mode_status", "mode": "assist", "connection_mode": "wing"}),
        _event(3, "send", {"type": "get_state"}),
        _event(4, "recv", {"type": "state_update", "mode": "wing", "state": {"/ch/3/fdr": -12.0}}),
    ]
    seq = 5
    for row in summary["readbacks"]:
        events.append(
            _event(
                seq,
                "send",
                {
                    "type": "query_mixer_parameter",
                    "request_id": row["request_id"],
                    "channel": row["channel"],
                    "value_type": row["value_type"],
                    "address": row["address"],
                },
            )
        )
        events.append(_event(seq + 1, "recv", row))
        seq += 2
    return events


def _expected_plan_request_sequence(summary: dict) -> list[dict]:
    return [
        {"type": "get_operator_mode"},
        {"type": "get_state"},
        *[
            {
                "type": "query_mixer_parameter",
                "request_id": row["request_id"],
                "channel": row["channel"],
                "value_type": row["value_type"],
                "address": row["address"],
            }
            for row in summary["readbacks"]
        ],
    ]


def _valid_expected_plan(summary: dict) -> dict:
    return {
        "type": "wing_readonly_preflight_offline_plan",
        "status": "ready_for_operator_review",
        "proof_artifact_schema": summary["proof_artifact_schema"],
        "proof_runner": summary["proof_runner"],
        "selected_mvp_blocker": EXPECTED_SELECTED_MVP_BLOCKER,
        "selected_mvp_blocker_status": EXPECTED_SELECTED_MVP_BLOCKER_STATUS,
        "plan_output": summary["expected_plan_path"],
        "channels": summary["channels"],
        "operator_channel_confirmation_required": summary[
            "operator_channel_confirmation_required"
        ],
        "operator_confirmed_channels": summary["operator_confirmed_channels"],
        "operator_session_confirmation_required": summary[
            "operator_session_confirmation_required"
        ],
        "operator_confirmed_session_ready": summary[
            "operator_confirmed_session_ready"
        ],
        "value_types": summary["value_types"],
        "require_physical_readback": summary["require_physical_readback"],
        "read_only": summary["read_only"],
        "live_write": summary["live_write"],
        "expected_plan_required_for_real_run": summary[
            "expected_plan_required_for_real_run"
        ],
        "physical_preflight_requires_existing_backend_connection": summary[
            "physical_preflight_requires_existing_backend_connection"
        ],
        "backend_port_prestart_check_required": True,
        "backend_port_prestart_check_must_pass_before_starting_backend": True,
        "backend_port_prestart_check_command": list(
            EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND
        ),
        "backend_port_prestart_check_failure_detail_fields": list(
            EXPECTED_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
        ),
        "backend_port_prestart_summary_required": summary[
            "backend_port_prestart_summary_required"
        ],
        "backend_port_prestart_summary_path": summary[
            "backend_port_prestart_summary_path"
        ],
        "backend_port_prestart_summary_sha256": summary[
            "backend_port_prestart_summary_sha256"
        ],
        "backend_port_prestart_summary_type": summary[
            "backend_port_prestart_summary_type"
        ],
        "backend_port_prestart_summary_status": summary[
            "backend_port_prestart_summary_status"
        ],
        "backend_port_prestart_summary_connection_policy": summary[
            "backend_port_prestart_summary_connection_policy"
        ],
        "backend_port_prestart_summary_live_write": summary[
            "backend_port_prestart_summary_live_write"
        ],
        "backend_port_prestart_summary_contacts_backend": summary[
            "backend_port_prestart_summary_contacts_backend"
        ],
        "backend_port_prestart_summary_queries_console": summary[
            "backend_port_prestart_summary_queries_console"
        ],
        "backend_port_prestart_summary_executes_tests": summary[
            "backend_port_prestart_summary_executes_tests"
        ],
        "backend_port_prestart_check_status": summary[
            "backend_port_prestart_check_status"
        ],
        "operator_required": True,
        "manual_review_required_before_readiness_update": True,
        "contacts_backend": False,
        "queries_console": False,
        "execution_allowed_by_plan": False,
        "live_authority_granted": False,
        "allowed_request_types": list(EXPECTED_PLAN_ALLOWED_REQUEST_TYPES),
        "forbidden_request_types": list(EXPECTED_PLAN_FORBIDDEN_REQUEST_TYPES),
        "request_id_policy": summary["request_id_policy"],
        "expected_response_types_by_request_type": summary[
            "expected_response_types_by_request_type"
        ],
        "request_sequence": _expected_plan_request_sequence(summary),
        "fail_closed_conditions": list(EXPECTED_PLAN_FAIL_CLOSED_CONDITIONS),
    }


def _write_artifact(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True)
    summary = _valid_summary()
    summary_path = run_dir / "summary.json"
    events_path = run_dir / "events.jsonl"
    plan_path = run_dir / "expected_plan.json"
    prestart_path = run_dir / "no_wing_prestart_summary.json"
    summary["summary_path"] = str(summary_path)
    summary["events_path"] = str(events_path)
    summary["expected_plan_path"] = str(plan_path)
    summary["backend_port_prestart_summary_path"] = str(prestart_path)
    prestart_payload = {
        "type": "no_wing_release_check",
        "status": "passed",
        "connection_policy": "no_hardware_no_network_no_live_writes",
        "live_write": False,
        "contacts_backend": False,
        "queries_console": False,
        "executes_tests": False,
        "results": [
            {
                "name": "backend_port_free",
                "status": "passed",
                "detail": {
                    "safe_to_stop_listener_by_gate": False,
                    "live_authority_granted": False,
                },
            }
        ],
    }
    prestart_path.write_text(
        json.dumps(prestart_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["backend_port_prestart_summary_sha256"] = _sha256_file(prestart_path)
    plan_bytes = json.dumps(_valid_expected_plan(summary), sort_keys=True).encode("utf-8")
    plan_path.write_bytes(plan_bytes)
    summary["expected_plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    events_path.write_text(
        "".join(json.dumps(row) + "\n" for row in _valid_events(summary)),
        encoding="utf-8",
    )
    return summary_path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision_record(summary_path: Path, **overrides) -> dict:
    events_path = summary_path.parent / "events.jsonl"
    record = {
        "review_id": "review-wing-readonly-1",
        "artifact_source_id": str(summary_path),
        "run_id": "20260606T120000Z_wing_readonly_preflight",
        "artifact_captured_at": "2026-06-07T01:00:00Z",
        "summary_path": str(summary_path),
        "summary_sha256": _sha256_file(summary_path),
        "events_path": str(events_path),
        "events_sha256": _sha256_file(events_path),
        "decision": "accepted",
        "reviewed_by": "operator",
        "reviewed_at": "2026-06-07T01:05:00Z",
        "checklist_ids": list(REQUIRED_REVIEW_CHECKLIST_IDS),
    }
    record.update(overrides)
    return record


def _write_decision(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_decision_review_report_accepts_operator_decision_without_live_authority(tmp_path: Path):
    summary_path = _write_artifact(tmp_path / "preflight" / "run")
    decision_path = _write_decision(
        tmp_path / "decisions" / "accepted.json",
        _decision_record(summary_path),
    )

    report = decision_tool.build_decision_review_report(
        artifact_roots=(tmp_path / "preflight",),
        decision_paths=(decision_path,),
    )

    assert report["type"] == "wing_readonly_preflight_review_decision_report"
    assert report["status"] == decision_tool.ACCEPTED_DECISION_READY_STATUS
    assert report["ledger_status"] == "operator_review_decision_records_available"
    assert report["read_only"] is True
    assert report["live_write"] is False
    assert report["queries_console"] is False
    assert report["executes_tests"] is False
    assert report["execution_allowed_by_decision"] is False
    assert report["readiness_matrix_update_allowed_by_decision"] is False
    assert report["live_authority_granted"] is False
    assert report["full_autonomous_ready"] is False
    assert report["accepted_record_count"] == 1
    assert report["manual_readiness_update_candidate_available"] is True
    assert report["accepted_source_ids"] == [str(summary_path)]
    assert report["accepted_candidate_prestart_contract_valid"] is True
    assert report["accepted_candidate_prestart_source_ids"] == [str(summary_path)]
    assert report["candidate_reviews"][0]["backend_port_prestart_check_required"] is True
    assert (
        report["candidate_reviews"][0][
            "backend_port_prestart_check_must_pass_before_starting_backend"
        ]
        is True
    )
    assert decision_tool.exit_code_for_report(report) == 0
    markdown = decision_tool.render_markdown(report)
    assert decision_tool.ACCEPTED_DECISION_READY_STATUS in markdown
    assert "accepted_candidate_prestart_contract_valid: `true`" in markdown
    assert "live_authority_granted: `false`" in markdown
    assert "Candidate Prestart Contract" in markdown
    assert "backend_port_prestart_check_command" in markdown
    assert "backend_port_prestart_check_failure_detail_fields" in markdown
    assert "listener_processes" in markdown
    assert "listener_confirmation_checklist" in markdown
    assert str(summary_path) in markdown
    assert "does not contact WING" in markdown


def test_decision_review_report_blocks_accepted_decision_without_prestart_trace(
    monkeypatch,
):
    def fake_ledger(**_kwargs):
        return {
            "status": "operator_review_decision_records_available",
            "summary": {
                "preflight_evidence_status": "readonly_preflight_candidate_available",
                "review_candidate_count": 1,
                "decision_record_count": 1,
                "accepted_record_count": 1,
                "rejected_record_count": 0,
                "invalid_record_count": 0,
                "unreviewed_candidate_count": 0,
                "accepted_source_ids": ["source-a"],
                "accepted_candidate_prestart_contract_valid": False,
                "accepted_candidate_prestart_source_ids": [],
                "rejected_source_ids": [],
                "invalid_record_ids": [],
                "duplicate_source_ids": [],
                "conflicting_source_ids": [],
                "unreviewed_candidate_source_ids": [],
                "candidate_reviews": [
                    {
                        "source_id": "source-a",
                        "status": "valid_readonly_preflight_artifact",
                        "run_id": "run-a",
                        "summary_path": "summary.json",
                        "backend_port_prestart_check_required": False,
                        "backend_port_prestart_check_must_pass_before_starting_backend": True,
                        "backend_port_prestart_check_command": [
                            "PYTHONPATH=backend",
                            "python3",
                            "tools/no_wing_release_check.py",
                            "--quick",
                        ],
                        "backend_port_prestart_check_failure_detail_fields": [
                            "listener_processes",
                            "listener_confirmation_checklist",
                            "preflight_blocked_until_listener_confirmed",
                            "intended_backend_confirmation_required",
                            "safe_to_stop_listener_by_gate",
                            "fail_closed_reason",
                            "operator_next_step",
                        ],
                    }
                ],
                "required_checklist_ids": list(REQUIRED_REVIEW_CHECKLIST_IDS),
                "manual_readiness_update_candidate_available": False,
                "blocked_by": [WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER],
                "next_global_step": "Manual readiness matrix update remains separate.",
            },
            "safety": {
                "read_only": True,
                "live_write": False,
                "mutates_console_state": False,
                "queries_console": False,
                "executes_tests": False,
                "operator_required": True,
                "execution_allowed_by_decision": False,
                "readiness_matrix_update_allowed_by_decision": False,
                "readiness_closure_allowed_by_decision": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
            },
        }

    monkeypatch.setattr(
        decision_tool,
        "build_wing_readonly_preflight_review_decision_ledger",
        fake_ledger,
    )

    report = decision_tool.build_decision_review_report(artifact_roots=())

    assert report["status"] == "blocked"
    assert report["accepted_record_count"] == 1
    assert report["manual_readiness_update_candidate_available"] is False
    assert report["accepted_candidate_prestart_contract_valid"] is False
    assert report["accepted_candidate_prestart_source_ids"] == []
    assert WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER in report[
        "blocked_by"
    ]
    assert "accepted-candidate prestart trace" in report["next_required_action"]
    assert decision_tool.exit_code_for_report(report) == 3
    markdown = decision_tool.render_markdown(report)
    assert "Status: `blocked`" in markdown
    assert "accepted_candidate_prestart_contract_valid: `false`" in markdown
    assert WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER in markdown
    assert "- none" not in markdown


def test_decision_review_report_rejects_pending_template(tmp_path: Path):
    summary_path = _write_artifact(tmp_path / "preflight" / "run")
    template_path = _write_decision(
        tmp_path / "decisions" / "pending_template.json",
        _decision_record(
            summary_path,
            decision="pending_operator_decision",
            reviewed_by="",
            reviewed_at="",
        ),
    )

    report = decision_tool.build_decision_review_report(
        artifact_roots=(tmp_path / "preflight",),
        decision_paths=(template_path,),
    )

    assert report["status"] == "blocked"
    assert report["invalid_record_count"] == 1
    assert report["manual_readiness_update_candidate_available"] is False
    assert decision_tool.exit_code_for_report(report) == 3
    record = report["ledger"]["records"][0]
    assert "review_decision_unknown" in record["violations"]
    assert "review_decision_reviewer_missing" in record["violations"]
    assert "review_decision_reviewed_at_missing" in record["violations"]


def test_decision_review_report_treats_rejection_as_nonready(tmp_path: Path):
    summary_path = _write_artifact(tmp_path / "preflight" / "run")
    decision_path = _write_decision(
        tmp_path / "decisions" / "rejected.json",
        _decision_record(summary_path, decision="rejected"),
    )

    report = decision_tool.build_decision_review_report(
        artifact_roots=(tmp_path / "preflight",),
        decision_paths=(decision_path,),
    )

    assert report["status"] == "candidate_rejected"
    assert report["accepted_record_count"] == 0
    assert report["rejected_record_count"] == 1
    assert report["manual_readiness_update_candidate_available"] is False
    assert decision_tool.exit_code_for_report(report) == 4


def test_decision_review_report_blocks_conflicting_duplicate_decisions(tmp_path: Path):
    summary_path = _write_artifact(tmp_path / "preflight" / "run")
    accepted_path = _write_decision(
        tmp_path / "decisions" / "accepted.json",
        _decision_record(summary_path, review_id="review-accept"),
    )
    rejected_path = _write_decision(
        tmp_path / "decisions" / "rejected.json",
        _decision_record(
            summary_path,
            review_id="review-reject",
            decision="rejected",
            reviewed_at="2026-06-07T01:10:00Z",
        ),
    )

    report = decision_tool.build_decision_review_report(
        artifact_roots=(tmp_path / "preflight",),
        decision_paths=(accepted_path, rejected_path),
    )

    assert report["status"] == "blocked"
    assert report["accepted_record_count"] == 0
    assert report["rejected_record_count"] == 0
    assert report["invalid_record_count"] == 2
    assert report["manual_readiness_update_candidate_available"] is False
    assert report["duplicate_source_ids"] == [str(summary_path)]
    assert report["conflicting_source_ids"] == [str(summary_path)]
    assert decision_tool.exit_code_for_report(report) == 3
    markdown = decision_tool.render_markdown(report)
    assert f'duplicate_source_ids: `["{summary_path}"]`' in markdown
    assert f'conflicting_source_ids: `["{summary_path}"]`' in markdown


def test_decision_review_report_blocks_duplicate_accepted_decisions(tmp_path: Path):
    summary_path = _write_artifact(tmp_path / "preflight" / "run")
    accepted_a_path = _write_decision(
        tmp_path / "decisions" / "accepted-a.json",
        _decision_record(summary_path, review_id="review-accept-a"),
    )
    accepted_b_path = _write_decision(
        tmp_path / "decisions" / "accepted-b.json",
        _decision_record(
            summary_path,
            review_id="review-accept-b",
            reviewed_at="2026-06-07T01:10:00Z",
        ),
    )

    report = decision_tool.build_decision_review_report(
        artifact_roots=(tmp_path / "preflight",),
        decision_paths=(accepted_a_path, accepted_b_path),
    )

    assert report["status"] == "blocked"
    assert report["accepted_record_count"] == 0
    assert report["rejected_record_count"] == 0
    assert report["invalid_record_count"] == 2
    assert report["manual_readiness_update_candidate_available"] is False
    assert report["duplicate_source_ids"] == [str(summary_path)]
    assert report["conflicting_source_ids"] == []
    assert decision_tool.exit_code_for_report(report) == 3
    markdown = decision_tool.render_markdown(report)
    assert f'duplicate_source_ids: `["{summary_path}"]`' in markdown
    assert "conflicting_source_ids: `[]`" in markdown


def test_decision_review_report_requires_preflight_candidate_first(tmp_path: Path):
    report = decision_tool.build_decision_review_report(
        artifact_roots=(tmp_path / "missing",),
        decision_roots=(tmp_path / "decisions",),
    )

    assert report["status"] == "blocked"
    assert report["review_candidate_count"] == 0
    assert decision_tool.exit_code_for_report(report) == 2


def test_decision_review_tool_main_writes_output(tmp_path: Path):
    summary_path = _write_artifact(tmp_path / "preflight" / "run")
    decision_path = _write_decision(
        tmp_path / "decisions" / "accepted.json",
        _decision_record(summary_path),
    )
    output = tmp_path / "decision-review.md"

    exit_code = decision_tool.main(
        [
            "--artifact-root",
            str(tmp_path / "preflight"),
            "--decision-path",
            str(decision_path),
            "--output",
            str(output),
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert "WING Read-Only Preflight Decision Review" in output.read_text(encoding="utf-8")


def test_decision_review_tool_is_transport_free():
    source = Path("tools/review_wing_readonly_preflight_decisions.py").read_text(
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
