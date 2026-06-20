import hashlib
import json
from pathlib import Path

import pytest

from backend.wing_readonly_preflight_evidence import (
    EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND,
    EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT,
    EXPECTED_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS,
    EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS,
    EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS,
    EXPECTED_PLAN_ALLOWED_REQUEST_TYPES,
    EXPECTED_PLAN_FAIL_CLOSED_CONDITIONS,
    EXPECTED_PLAN_FORBIDDEN_REQUEST_TYPES,
    EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE_SUMMARY,
    EXPECTED_SELECTED_MVP_BLOCKER,
    EXPECTED_SELECTED_MVP_BLOCKER_STATUS,
)
from backend.wing_readonly_preflight_contract import (
    WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
    WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_PRESTART_PROOF_FIELDS,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS,
    WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS,
    WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME,
)
from backend.wing_readonly_preflight_review_decision import (
    REQUIRED_REVIEW_CHECKLIST_IDS,
    build_wing_readonly_preflight_review_decision_ledger,
)
from tools import review_wing_readonly_preflight_artifacts as review_tool


def test_manual_review_checklist_ids_match_shared_contract():
    checklist_ids = tuple(row["id"] for row in review_tool.manual_review_checklist())

    assert checklist_ids == WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS
    assert (
        REQUIRED_REVIEW_CHECKLIST_IDS
        == WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS
    )


def test_review_decision_template_required_fields_match_shared_contract():
    assert review_tool.REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
    )
    assert review_tool.review_decision_template_required_fields() == list(
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
    )
    assert (
        review_tool.REVIEW_DECISION_TEMPLATE_STATUS
        == WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS
    )
    assert (
        review_tool.review_decision_template_status()
        == WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS
    )
    assert review_tool.review_decision_template_required_edits() == list(
        review_tool.REVIEW_DECISION_TEMPLATE_REQUIRED_EDITS
    )
    assert (
        "remove template_status or replace the generated pending marker before ledger acceptance"
        in review_tool.review_decision_template_required_edits()
    )
    assert review_tool.DEFAULT_DECISION_TEMPLATE_OUTPUT == (
        Path(__file__).resolve().parents[1]
        / WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT
    )


def _valid_summary() -> dict:
    run_id = "20260606T120000Z_wing_readonly_preflight"
    return {
        "run_id": run_id,
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
        "backend_port_prestart_check_command_text": (
            EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
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
            for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
        },
        **{
            field: False
            for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
        },
        **{
            field: True
            for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS
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
            "query_mixer_parameter",
        ],
        "contacts_backend": True,
        "queries_console": True,
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


def _event(
    seq: int,
    event_type: str,
    payload: dict,
    *,
    run_id: str = "20260606T120000Z_wing_readonly_preflight",
) -> dict:
    return {
        "run_id": run_id,
        "seq": seq,
        "timestamp": "2026-06-06T12:00:00.000Z",
        "event_type": event_type,
        "payload": payload,
    }


def _valid_events(summary: dict) -> list[dict]:
    events = [
        _event(1, "send", {"type": "get_operator_mode"}, run_id=summary["run_id"]),
        _event(
            2,
            "recv",
            {"type": "operator_mode_status", "mode": "assist", "connection_mode": "wing"},
            run_id=summary["run_id"],
        ),
        _event(3, "send", {"type": "get_state"}, run_id=summary["run_id"]),
        _event(
            4,
            "recv",
            {"type": "state_update", "mode": "wing", "state": {"/ch/3/fdr": -12.0}},
            run_id=summary["run_id"],
        ),
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
                run_id=summary["run_id"],
            )
        )
        events.append(_event(seq + 1, "recv", row, run_id=summary["run_id"]))
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


def _artifact_root_from_summary(summary: dict) -> str:
    return str(Path(summary["summary_path"]).parent.parent)


def _expected_artifacts_for_summary(summary: dict) -> list[str]:
    artifact_root = _artifact_root_from_summary(summary)
    return [
        f"{artifact_root}/<run_id>/{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}",
        f"{artifact_root}/<run_id>/{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}",
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
        "artifact_root": _artifact_root_from_summary(summary),
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
        "backend_port_prestart_check_command_text": (
            EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
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
        "backend_port_prestart_summary_read_only": summary[
            "backend_port_prestart_summary_read_only"
        ],
        "backend_port_prestart_summary_live_write": summary[
            "backend_port_prestart_summary_live_write"
        ],
        "backend_port_prestart_summary_contacts_backend": summary[
            "backend_port_prestart_summary_contacts_backend"
        ],
        "backend_port_prestart_summary_contacts_wing": summary[
            "backend_port_prestart_summary_contacts_wing"
        ],
        "backend_port_prestart_summary_mutates_console_state": summary[
            "backend_port_prestart_summary_mutates_console_state"
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
        "artifact_manifest_verifier_must_pass_before_websocket": summary[
            "artifact_manifest_verifier_must_pass_before_websocket"
        ],
        "artifact_manifest_verifier_passed_before_websocket": summary[
            "artifact_manifest_verifier_passed_before_websocket"
        ],
        "artifact_manifest_verifier_websocket_opened_after_pass": False,
        **{
            field: summary[field]
            for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
        },
        **{
            field: summary[field]
            for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
        },
        **{
            field: summary[field]
            for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS
        },
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
        "expected_artifacts": _expected_artifacts_for_summary(summary),
        "fail_closed_conditions": list(EXPECTED_PLAN_FAIL_CLOSED_CONDITIONS),
    }


def _write_artifact(
    run_dir: Path,
    *,
    summary: dict | None = None,
    tamper_events: bool = False,
) -> Path:
    summary = summary or _valid_summary()
    requested_run_dir_name = run_dir.name
    if requested_run_dir_name not in {"run", summary["run_id"]}:
        second = (sum(ord(char) for char in requested_run_dir_name) % 59) + 1
        summary["run_id"] = f"20260606T1200{second:02d}Z_wing_readonly_preflight"
    if WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT not in run_dir.as_posix():
        run_dir = run_dir.parent / WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT
    run_dir = run_dir if run_dir.name == summary["run_id"] else run_dir / summary["run_id"]
    run_dir.mkdir(parents=True)
    summary_path = run_dir / WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME
    events_path = run_dir / WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME
    plan_path = run_dir / "expected_plan.json"
    prestart_path = run_dir / "no_wing_prestart_summary.json"
    summary["artifact_root"] = str(run_dir.parent)
    summary["summary_path"] = str(summary_path)
    summary["events_path"] = str(events_path)
    summary["expected_plan_path"] = str(plan_path)
    summary["backend_port_prestart_summary_path"] = str(prestart_path)
    prestart_payload = {
        "type": "no_wing_release_check",
        "status": "passed",
        "connection_policy": "no_hardware_no_network_no_live_writes",
        "read_only": True,
        "live_write": False,
        "contacts_backend": False,
        "contacts_wing": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "results": [
            {
                "name": "backend_port_free",
                "status": "passed",
                "detail": {
                    "backend_port_prestart_check_failure_detail_fields": list(
                        EXPECTED_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
                    ),
                    **{
                        field: False
                        for field in (
                            EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
                        )
                    },
                    **{
                        field: False
                        for field in (
                            EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
                        )
                    },
                    **{
                        field: True
                        for field in (
                            EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS
                        )
                    },
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
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    events = _valid_events(summary)
    if tamper_events:
        events.insert(2, _event(99, "send", {"type": "set_gain", "channel": 3, "value": 1.0}))
    events_path.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    return summary_path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_review_report_accepts_clean_artifact_without_live_authority(tmp_path: Path):
    summary_path = _write_artifact(tmp_path / "preflight" / "run")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_artifacts = _expected_artifacts_for_summary(summary)

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))

    assert report["status"] == "candidate_ready_for_manual_review"
    assert report["readiness_update_candidate_available"] is True
    assert report["execution_allowed_by_review"] is False
    assert report["live_authority_granted"] is False
    assert report["full_autonomous_ready"] is False
    assert report["source_authority_denied"] is True
    assert report["source_authority_flags"] == {
        "readiness_closure_allowed_by_evidence": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }
    assert report["manual_review_required_before_readiness_update"] is True
    assert report["manual_review_decision_required"] is True
    assert report["request_id_policy"] == "required_for_every_query_mixer_parameter_response"
    assert report["request_id_correlation_required"] is True
    assert report["review_decision_template_available"] is True
    assert report["review_decision_template_required_fields"] == list(
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
    )
    assert report["review_decision_template_required_edits"] == list(
        review_tool.REVIEW_DECISION_TEMPLATE_REQUIRED_EDITS
    )
    assert (
        "remove template_status or replace the generated pending marker before ledger acceptance"
        in report["review_decision_template_required_edits"]
    )
    assert [row["id"] for row in report["manual_review_checklist"]] == [
        "confirm_artifact_paths",
        "confirm_operator_channel_scope",
        "confirm_operator_session_ready",
        "confirm_physical_readbacks",
        "confirm_event_log_readonly_sequence",
        "confirm_event_log_run_id_binding",
        "confirm_no_live_authority",
    ]
    assert report["artifact_sources"] == [str(summary_path)]
    assert report["artifact_reviews"] == [
        {
            "source_id": str(summary_path),
            "status": "valid_readonly_preflight_artifact",
            "run_id": "20260606T120000Z_wing_readonly_preflight",
            "captured_at": "2026-06-07T01:00:00Z",
            "artifact_root": str(summary_path.parent.parent),
            "summary_path": str(summary_path),
            "summary_sha256": _sha256_file(summary_path),
            "events_path": str(summary_path.parent / "events.jsonl"),
            "events_sha256": _sha256_file(summary_path.parent / "events.jsonl"),
            "event_log_run_id_matches_summary": True,
            "expected_artifacts": expected_artifacts,
            "expected_summary_artifact": str(summary_path),
            "expected_events_artifact": str(summary_path.parent / "events.jsonl"),
            "summary_path_matches_expected_artifact": True,
            "events_path_matches_expected_artifact": True,
            "contacts_backend": True,
            "queries_console": True,
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
            "channels": [3],
            "operator_confirmed_channels": [3],
            "operator_session_confirmation_required": True,
            "operator_confirmed_session_ready": True,
            "backend_port_prestart_check_required": True,
            "backend_port_prestart_check_must_pass_before_starting_backend": True,
            "backend_port_prestart_check_command": list(
                EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND
            ),
            "backend_port_prestart_check_command_text": (
                EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
            ),
            "backend_port_prestart_check_failure_detail_fields": list(
                EXPECTED_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
            ),
            "backend_port_prestart_summary_required": True,
            "backend_port_prestart_summary_path": str(
                summary_path.parent / "no_wing_prestart_summary.json"
            ),
            "backend_port_prestart_summary_sha256": _sha256_file(
                summary_path.parent / "no_wing_prestart_summary.json"
            ),
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
                for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS
            },
            **{
                field: False
                for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS
            },
            **{
                field: True
                for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS
            },
            "operator_next_step": None,
            "value_types": ["fader", "gain", "pan"],
            "readback_count": 3,
            "physical_readback_count": 3,
            "violations": [],
        }
        ]
    assert review_tool.exit_code_for_report(report) == 0
    markdown = review_tool.render_markdown(report)
    assert "candidate_ready_for_manual_review" in markdown
    assert "captured_at: `2026-06-07T01:00:00Z`" in markdown
    assert f"artifact_root: `{summary_path.parent.parent}`" in markdown
    assert f"summary_path: `{summary_path}`" in markdown
    assert f"summary_sha256: `{_sha256_file(summary_path)}`" in markdown
    assert f"events_path: `{summary_path.parent / 'events.jsonl'}`" in markdown
    assert f"events_sha256: `{_sha256_file(summary_path.parent / 'events.jsonl')}`" in markdown
    assert "event_log_run_id_matches_summary: `true`" in markdown
    assert f"expected_artifacts: `{json.dumps(expected_artifacts)}`" in markdown
    assert f"expected_summary_artifact: `{summary_path}`" in markdown
    assert f"expected_events_artifact: `{summary_path.parent / 'events.jsonl'}`" in markdown
    assert "summary_path_matches_expected_artifact: `true`" in markdown
    assert "events_path_matches_expected_artifact: `true`" in markdown
    assert "contacts_backend: `true`" in markdown
    assert "queries_console: `true`" in markdown
    assert "write_request_sent: `false`" in markdown
    assert "proposal_request_sent: `false`" in markdown
    assert (
        'sent_request_types: `["get_operator_mode", "get_state", "query_mixer_parameter", "query_mixer_parameter", "query_mixer_parameter"]`'
        in markdown
    )
    assert "forbidden_request_types_sent: `[]`" in markdown
    assert "operator_confirmed_channels: `[3]`" in markdown
    assert "operator_session_confirmation_required: `true`" in markdown
    assert "operator_confirmed_session_ready: `true`" in markdown
    assert "backend_port_prestart_check_required: `true`" in markdown
    assert "backend_port_prestart_check_must_pass_before_starting_backend: `true`" in markdown
    assert "backend_port_prestart_check_command" in markdown
    assert "backend_port_prestart_check_command_text" in markdown
    assert EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT in markdown
    assert "backend_port_prestart_check_failure_detail_fields" in markdown
    assert "backend_port_prestart_summary_required: `true`" in markdown
    assert "backend_port_prestart_summary_type: `no_wing_release_check`" in markdown
    assert "backend_port_prestart_summary_status: `passed`" in markdown
    assert "backend_port_prestart_summary_read_only: `true`" in markdown
    assert "backend_port_prestart_summary_contacts_backend: `false`" in markdown
    assert "backend_port_prestart_summary_contacts_wing: `false`" in markdown
    assert (
        "backend_port_prestart_summary_mutates_console_state: `false`"
        in markdown
    )
    assert "backend_port_prestart_summary_queries_console: `false`" in markdown
    assert "backend_port_prestart_summary_executes_tests: `false`" in markdown
    assert "backend_port_prestart_check_status: `passed`" in markdown
    assert (
        "artifact_manifest_verifier_must_pass_before_websocket: `true`"
        in markdown
    )
    assert "artifact_manifest_verifier_passed_before_websocket: `true`" in markdown
    assert (
        "artifact_manifest_verifier_websocket_opened_after_pass: `true`"
        in markdown
    )
    for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS:
        assert f"{field}: `false`" in markdown
    for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS:
        assert f"{field}: `false`" in markdown
    for field in EXPECTED_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS:
        assert f"{field}: `true`" in markdown
    assert "listener_processes" in markdown
    assert "listener_confirmation_checklist" in markdown
    assert "operator_next_step: `None`" in markdown
    assert "Manual Review Gate" in markdown
    assert "confirm_operator_session_ready" in markdown
    assert "confirm_event_log_readonly_sequence" in markdown
    assert "confirm_event_log_run_id_binding" in markdown
    assert "manual_review_decision_required: `true`" in markdown
    assert "source_authority_denied: `true`" in markdown
    assert (
        'source_authority_flags: `{"full_autonomous_ready": false, "live_authority_granted": false, "readiness_closure_allowed_by_evidence": false}`'
        in markdown
    )
    assert "request_id_policy: `required_for_every_query_mixer_parameter_response`" in markdown
    assert "request_id_correlation_required: `true`" in markdown
    assert "review_decision_template_available: `true`" in markdown
    assert "review_decision_template_required_edits" in markdown
    assert "Review Decision Template" in markdown
    assert "required_edits" in markdown
    assert (
        "remove template_status or replace the generated pending marker before ledger acceptance"
        in markdown
    )
    assert "does not contact WING" in markdown


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("readiness_closure_allowed_by_evidence", None),
        ("live_authority_granted", "false"),
        ("full_autonomous_ready", 0),
    ],
)
def test_review_report_blocks_candidate_when_source_authority_flag_is_not_false(
    monkeypatch,
    field: str,
    value,
):
    summary = {
        "artifact_count": 1,
        "valid_candidate_count": 1,
        "invalid_candidate_count": 0,
        "readiness_matrix_update_allowed": True,
        "readiness_closure_allowed_by_evidence": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "latest_valid_candidate_run_id": "20260606T120000Z_wing_readonly_preflight",
        "blocked_by": [],
        "violations": [],
    }
    summary[field] = value
    evidence = {
        "status": "readonly_preflight_candidate_available",
        "summary": summary,
        "safety": {
            "read_only": True,
            "live_write": False,
            "queries_console": False,
        },
        "artifacts": [],
    }
    monkeypatch.setattr(
        review_tool,
        "build_wing_readonly_preflight_evidence",
        lambda **_: evidence,
    )

    report = review_tool.build_review_report()

    assert report["status"] == "blocked"
    assert report["readiness_update_candidate_available"] is False
    assert report["manual_review_decision_required"] is False
    assert report["review_decision_template_available"] is False
    assert report["source_authority_denied"] is False
    assert report["source_authority_flags"][field] == value
    assert report["live_authority_granted"] is False
    assert report["full_autonomous_ready"] is False
    assert review_tool.exit_code_for_report(report) == 3


def test_review_decision_template_links_clean_candidate_but_requires_operator_edit(tmp_path: Path):
    summary_path = _write_artifact(tmp_path / "preflight" / "run")
    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))

    template = review_tool.build_review_decision_template(
        report,
        generated_at="2026-06-07T02:00:00Z",
    )

    assert template["template_status"] == (
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS
    )
    assert template["template_generated_at"] == "2026-06-07T02:00:00Z"
    assert template["review_id"] == "review-wing-readonly-20260606T120000Z_wing_readonly_preflight"
    assert template["artifact_source_id"] == str(summary_path)
    assert template["run_id"] == "20260606T120000Z_wing_readonly_preflight"
    assert template["artifact_captured_at"] == "2026-06-07T01:00:00Z"
    assert template["artifact_root"] == str(summary_path.parent.parent)
    assert template["summary_path"] == str(summary_path)
    assert template["summary_sha256"] == _sha256_file(summary_path)
    assert template["events_path"] == str(summary_path.parent / "events.jsonl")
    assert template["events_sha256"] == _sha256_file(summary_path.parent / "events.jsonl")
    assert template["channels"] == [3]
    assert template["value_types"] == ["fader", "gain", "pan"]
    assert template["operator_confirmed_channels"] == [3]
    assert template["operator_confirmed_session_ready"] is True
    assert template["readback_count"] == 3
    assert template["physical_readback_count"] == 3
    assert (
        template["request_id_policy"]
        == "required_for_every_query_mixer_parameter_response"
    )
    assert template["request_id_correlation_required"] is True
    assert template["event_log_run_id_matches_summary"] is True
    for field in WING_READONLY_PREFLIGHT_REVIEW_DECISION_PRESTART_PROOF_FIELDS:
        assert template[field] == report["artifact_reviews"][0][field]
    assert template["decision"] == "pending_operator_decision"
    assert template["reviewed_by"] == ""
    assert template["reviewed_at"] == ""
    assert template["checklist_ids"] == list(REQUIRED_REVIEW_CHECKLIST_IDS)
    assert template["live_authority_granted"] is False
    assert template["readiness_matrix_update_allowed_by_decision"] is False

    ledger = build_wing_readonly_preflight_review_decision_ledger(
        preflight_evidence=report["evidence"],
        decision_records=[template],
    )

    assert ledger["status"] == "operator_review_decision_replacement_required"
    assert ledger["summary"]["accepted_record_count"] == 0
    assert ledger["summary"]["manual_readiness_update_candidate_available"] is False
    violations = set(ledger["records"][0]["violations"])
    assert "review_decision_unknown" in violations
    assert "review_decision_reviewer_missing" in violations
    assert "review_decision_reviewed_at_missing" in violations
    assert "review_decision_template_status_unedited" in violations


def test_review_decision_template_rejects_mutated_prestart_proof_binding(tmp_path: Path):
    _write_artifact(tmp_path / "preflight" / "run")
    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0]["backend_port_prestart_summary_sha256"] = ""

    with pytest.raises(
        ValueError,
        match="decision_template_candidate_prestart_proof_incomplete",
    ):
        review_tool.build_review_decision_template(report)

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0][
        "artifact_manifest_verifier_passed_before_websocket"
    ] = False

    with pytest.raises(
        ValueError,
        match="decision_template_candidate_prestart_proof_incomplete",
    ):
        review_tool.build_review_decision_template(report)


def test_review_decision_template_rejects_coerced_readback_counts(tmp_path: Path):
    _write_artifact(tmp_path / "preflight" / "run")
    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0]["readback_count"] = True

    with pytest.raises(ValueError, match="decision_template_candidate_identity_incomplete"):
        review_tool.build_review_decision_template(report)

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0]["physical_readback_count"] = "3"

    with pytest.raises(ValueError, match="decision_template_candidate_identity_incomplete"):
        review_tool.build_review_decision_template(report)


def test_review_decision_template_rejects_coerced_channel_scope(tmp_path: Path):
    _write_artifact(tmp_path / "preflight" / "run")
    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0]["channels"] = ["3"]
    report["artifact_reviews"][0]["operator_confirmed_channels"] = ["3"]

    with pytest.raises(ValueError, match="decision_template_candidate_identity_incomplete"):
        review_tool.build_review_decision_template(report)

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0]["operator_confirmed_channels"] = [3.0]

    with pytest.raises(ValueError, match="decision_template_candidate_identity_incomplete"):
        review_tool.build_review_decision_template(report)


def test_review_decision_template_rejects_invalid_value_types(tmp_path: Path):
    _write_artifact(tmp_path / "preflight" / "run")
    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0]["value_types"] = ["fader", 3]

    with pytest.raises(ValueError, match="decision_template_candidate_identity_incomplete"):
        review_tool.build_review_decision_template(report)

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0]["value_types"] = ["fader", "eq"]

    with pytest.raises(ValueError, match="decision_template_candidate_identity_incomplete"):
        review_tool.build_review_decision_template(report)

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0]["value_types"] = ["fader", "fader"]

    with pytest.raises(ValueError, match="decision_template_candidate_identity_incomplete"):
        review_tool.build_review_decision_template(report)


def test_review_decision_template_rejects_malformed_artifact_hashes(tmp_path: Path):
    _write_artifact(tmp_path / "preflight" / "run")
    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0]["summary_sha256"] = "not-a-sha256"

    with pytest.raises(ValueError, match="decision_template_candidate_identity_incomplete"):
        review_tool.build_review_decision_template(report)

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0]["events_sha256"] = "also-not-a-sha256"

    with pytest.raises(ValueError, match="decision_template_candidate_identity_incomplete"):
        review_tool.build_review_decision_template(report)


def test_review_decision_template_rejects_unbound_expected_artifact_paths(tmp_path: Path):
    _write_artifact(tmp_path / "preflight" / "run")
    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0]["summary_path_matches_expected_artifact"] = False

    with pytest.raises(ValueError, match="decision_template_candidate_identity_incomplete"):
        review_tool.build_review_decision_template(report)

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0]["events_path_matches_expected_artifact"] = False

    with pytest.raises(ValueError, match="decision_template_candidate_identity_incomplete"):
        review_tool.build_review_decision_template(report)

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0]["expected_summary_artifact"] = str(
        tmp_path / "other" / "summary.json"
    )

    with pytest.raises(ValueError, match="decision_template_candidate_identity_incomplete"):
        review_tool.build_review_decision_template(report)

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))
    report["artifact_reviews"][0]["expected_events_artifact"] = str(
        tmp_path / "other" / "events.jsonl"
    )

    with pytest.raises(ValueError, match="decision_template_candidate_identity_incomplete"):
        review_tool.build_review_decision_template(report)


def test_review_decision_template_requires_explicit_source_for_multiple_candidates(tmp_path: Path):
    first_summary = _write_artifact(tmp_path / "preflight" / "run1")
    second_summary = _write_artifact(tmp_path / "preflight" / "run2")
    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))

    try:
        review_tool.build_review_decision_template(report)
    except ValueError as exc:
        assert str(exc) == "decision_template_requires_single_candidate_or_source_id"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected explicit source id requirement")

    template = review_tool.build_review_decision_template(
        report,
        artifact_source_id=str(second_summary),
        generated_at="2026-06-07T02:00:00Z",
    )

    assert template["artifact_source_id"] == str(second_summary)
    assert template["summary_path"] == str(second_summary)
    assert template["artifact_source_id"] != str(first_summary)


def test_review_report_blocks_missing_artifacts(tmp_path: Path):
    report = review_tool.build_review_report(artifact_roots=(tmp_path / "missing",))

    assert report["status"] == "blocked"
    assert report["artifact_count"] == 0
    assert report["manual_review_required_before_readiness_update"] is True
    assert report["manual_review_decision_required"] is False
    assert report["review_decision_template_available"] is False
    assert "operator_must_run_wing_readonly_preflight_runner" in report["blocked_by"]
    assert review_tool.exit_code_for_report(report) == 2


def test_review_report_blocks_tampered_event_log(tmp_path: Path):
    _write_artifact(tmp_path / "preflight" / "run", tamper_events=True)

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))

    assert report["status"] == "blocked"
    assert report["artifact_count"] == 1
    assert report["invalid_candidate_count"] == 1
    assert report["artifact_reviews"][0]["operator_confirmed_channels"] == [3]
    assert report["artifact_reviews"][0]["operator_confirmed_session_ready"] is True
    assert report["artifact_reviews"][0]["forbidden_request_types_sent"] == ["set_gain"]
    assert report["artifact_reviews"][0]["write_request_sent"] is False
    assert report["artifact_reviews"][0]["proposal_request_sent"] is False
    assert "event_log_forbidden_request_type_sent" in report["violations"]
    assert "forbidden_request_types_sent_not_empty" in report["violations"]
    markdown = review_tool.render_markdown(report)
    assert 'forbidden_request_types_sent: `["set_gain"]`' in markdown
    assert review_tool.exit_code_for_report(report) == 3


def test_review_report_surfaces_blocked_runner_operator_next_step(tmp_path: Path):
    summary = _valid_summary()
    summary.update(
        {
            "status": "blocked",
            "reason": "expected_plan_mismatch",
            "expected_plan_match": False,
            "expected_plan_violations": [
                "expected_plan_backend_port_prestart_check_failure_detail_fields_mismatch",
            ],
            "contacts_backend": True,
            "queries_console": True,
            "operator_next_step": (
                "Regenerate the offline expected plan with the current runtime args "
                "and prestart contract before opening the backend WebSocket."
            ),
        }
    )
    _write_artifact(tmp_path / "preflight" / "run", summary=summary)

    report = review_tool.build_review_report(artifact_roots=(tmp_path / "preflight",))

    assert report["status"] == "blocked"
    assert report["readiness_update_candidate_available"] is False
    assert report["artifact_reviews"][0]["operator_next_step"] == (
        "Regenerate the offline expected plan with the current runtime args "
        "and prestart contract before opening the backend WebSocket."
    )
    assert report["artifact_reviews"][0]["contacts_backend"] is True
    assert report["artifact_reviews"][0]["queries_console"] is True
    assert report["artifact_operator_next_steps"] == [
        "Regenerate the offline expected plan with the current runtime args "
        "and prestart contract before opening the backend WebSocket."
    ]
    assert report["next_required_action"] == (
        "Regenerate the offline expected plan with the current runtime args "
        "and prestart contract before opening the backend WebSocket."
    )
    assert "status_not_passed" in report["violations"]
    markdown = review_tool.render_markdown(report)
    assert "operator_next_step" in markdown
    assert "artifact_operator_next_steps" in markdown
    assert "contacts_backend: `true`" in markdown
    assert "queries_console: `true`" in markdown
    assert "before opening the backend WebSocket" in markdown
    assert "live_authority_granted: `false`" in markdown


def test_review_tool_main_writes_output_and_returns_gate_status(tmp_path: Path):
    _write_artifact(tmp_path / "preflight" / "run")
    output = tmp_path / "review.md"
    template_output = tmp_path / "decision-template.json"

    exit_code = review_tool.main(
        [
            "--artifact-root",
            str(tmp_path / "preflight"),
            "--output",
            str(output),
            "--decision-template-output",
            str(template_output),
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert "WING Read-Only Preflight Artifact Review" in output.read_text(encoding="utf-8")
    template = json.loads(template_output.read_text(encoding="utf-8"))
    assert template["decision"] == "pending_operator_decision"
    assert template["reviewed_by"] == ""


def test_review_tool_is_transport_free():
    source = Path("tools/review_wing_readonly_preflight_artifacts.py").read_text(encoding="utf-8")

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
