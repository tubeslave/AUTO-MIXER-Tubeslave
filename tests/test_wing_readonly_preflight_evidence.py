import copy
import hashlib
import json
from pathlib import Path

from backend.wing_readonly_preflight_evidence import (
    EXPECTED_BACKEND_PORT_PRESTART_CHECK_COMMAND,
    EXPECTED_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    DEFAULT_WING_READONLY_PREFLIGHT_ARTIFACT_ROOTS,
    EXPECTED_PLAN_ALLOWED_REQUEST_TYPES,
    EXPECTED_PLAN_FAIL_CLOSED_CONDITIONS,
    EXPECTED_PLAN_FORBIDDEN_REQUEST_TYPES,
    EXPECTED_SELECTED_MVP_BLOCKER,
    EXPECTED_SELECTED_MVP_BLOCKER_STATUS,
    WING_READONLY_PREFLIGHT_EVIDENCE_METHOD_ID,
    build_wing_readonly_preflight_evidence,
    validate_wing_readonly_preflight_artifact,
)
from backend.processing_registry import (
    build_processing_registry_snapshot,
    get_pipeline_profile,
    get_processing_method,
)
from backend.project_goal_capability_matrix import get_project_goal_capability


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
        "events_path": "artifacts/wing_readonly_preflight/run/events.jsonl",
        "summary_path": "artifacts/wing_readonly_preflight/run/summary.json",
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
        "request_sequence": _expected_plan_request_sequence(summary),
        "fail_closed_conditions": list(EXPECTED_PLAN_FAIL_CLOSED_CONDITIONS),
    }


def _write_expected_plan(summary_path: Path, plan: dict) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    plan_path = Path(summary["expected_plan_path"])
    plan_bytes = json.dumps(plan, sort_keys=True).encode("utf-8")
    plan_path.write_bytes(plan_bytes)
    summary["expected_plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def _write_prestart_summary(path: Path, *, status: str = "passed") -> None:
    payload = {
        "type": "no_wing_release_check",
        "status": status,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_file_artifact(artifact_dir: Path, summary: dict | None = None, events: list[dict] | None = None) -> Path:
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "summary.json"
    events_path = artifact_dir / "events.jsonl"
    plan_path = artifact_dir / "expected_plan.json"
    prestart_path = artifact_dir / "no_wing_prestart_summary.json"
    payload = copy.deepcopy(summary or _valid_summary())
    payload["events_path"] = str(events_path)
    payload["summary_path"] = str(artifact_path)
    payload["expected_plan_path"] = str(plan_path)
    payload["backend_port_prestart_summary_path"] = str(prestart_path)
    _write_prestart_summary(prestart_path)
    payload["backend_port_prestart_summary_sha256"] = hashlib.sha256(
        prestart_path.read_bytes()
    ).hexdigest()
    plan_bytes = json.dumps(_valid_expected_plan(payload), sort_keys=True).encode("utf-8")
    plan_path.write_bytes(plan_bytes)
    payload["expected_plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    event_rows = events if events is not None else _valid_events(payload)
    events_path.write_text(
        "".join(json.dumps(row) + "\n" for row in event_rows),
        encoding="utf-8",
    )
    return artifact_path


def test_wing_readonly_preflight_evidence_reports_missing_without_authority():
    evidence = build_wing_readonly_preflight_evidence(artifact_roots=("does-not-exist/preflight",))

    assert evidence["type"] == "wing_readonly_preflight_evidence"
    assert evidence["method_id"] == WING_READONLY_PREFLIGHT_EVIDENCE_METHOD_ID
    assert evidence["status"] == "operator_preflight_required"
    assert evidence["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "operator_required": True,
        "execution_allowed_by_evidence": False,
        "readiness_closure_allowed_by_evidence": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }
    assert evidence["summary"]["artifact_search_roots"] == ["does-not-exist/preflight"]
    assert evidence["summary"]["artifact_count"] == 0
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["accepted_as_readonly_preflight_candidate"] is False
    assert evidence["summary"]["readiness_matrix_update_allowed"] is False
    assert evidence["summary"]["readiness_closure_allowed_by_evidence"] is False
    assert evidence["summary"]["live_authority_granted"] is False
    assert evidence["summary"]["blocked_by"] == [
        "operator_must_run_wing_readonly_preflight_runner"
    ]
    assert {row["status"] for row in evidence["mvp_blocker_matrix"]} == {
        "operator_preflight_required"
    }


def test_valid_physical_preflight_candidate_updates_matrix_but_keeps_manual_review_gate():
    evidence = build_wing_readonly_preflight_evidence(
        artifact_payloads=(_valid_summary(),),
        artifact_roots=(),
    )

    assert evidence["status"] == "readonly_preflight_candidate_available"
    assert evidence["summary"]["artifact_count"] == 1
    assert evidence["summary"]["valid_candidate_count"] == 1
    assert evidence["summary"]["invalid_candidate_count"] == 0
    assert evidence["summary"]["accepted_as_readonly_preflight_candidate"] is True
    assert evidence["summary"]["operator_review_required"] is True
    assert evidence["summary"]["mvp_blocker_candidate_count"] == 4
    assert evidence["summary"]["mvp_blockers_closed_by_evidence"] == []
    assert evidence["summary"]["readiness_matrix_update_allowed"] is True
    assert evidence["summary"]["readiness_closure_allowed_by_evidence"] is False
    assert evidence["summary"]["live_authority_granted"] is False
    assert evidence["summary"]["latest_valid_candidate_run_id"] == "20260606T120000Z_wing_readonly_preflight"
    assert evidence["summary"]["latest_valid_candidate_captured_at"] == "2026-06-07T01:00:00Z"
    assert evidence["summary"]["blocked_by"] == [
        "operator_must_review_readonly_preflight_candidate_before_readiness_update"
    ]
    assert {row["id"] for row in evidence["mvp_blocker_matrix"]} == {
        "wing_readonly_connection_proof",
        "physical_console_readback_proof",
        "operator_visible_evidence_report",
        "readiness_matrix_update_based_on_real_proof",
    }
    assert {row["status"] for row in evidence["mvp_blocker_matrix"]} == {
        "candidate_available"
    }
    assert all(row["closed"] is False for row in evidence["mvp_blocker_matrix"])
    assert evidence["artifacts"][0]["accepted_as_readonly_preflight_candidate"] is True
    assert evidence["artifacts"][0]["captured_at"] == "2026-06-07T01:00:00Z"
    assert evidence["artifacts"][0]["summary_path"] == "artifacts/wing_readonly_preflight/run/summary.json"
    assert evidence["artifacts"][0]["events_path"] == "artifacts/wing_readonly_preflight/run/events.jsonl"
    assert evidence["artifacts"][0]["physical_readback_count"] == 3
    assert evidence["artifacts"][0]["channels"] == [3]
    assert evidence["artifacts"][0]["operator_confirmed_channels"] == [3]
    assert evidence["artifacts"][0]["operator_session_confirmation_required"] is True
    assert evidence["artifacts"][0]["operator_confirmed_session_ready"] is True
    assert evidence["artifacts"][0]["backend_port_prestart_summary_required"] is True
    assert (
        evidence["artifacts"][0]["backend_port_prestart_summary_type"]
        == "no_wing_release_check"
    )
    assert evidence["artifacts"][0]["backend_port_prestart_summary_status"] == "passed"
    assert evidence["artifacts"][0]["backend_port_prestart_check_status"] == "passed"
    assert evidence["artifacts"][0]["violations"] == []


def test_preflight_evidence_rejects_non_physical_or_forbidden_request_artifact():
    artifact = copy.deepcopy(_valid_summary())
    artifact["require_physical_readback"] = False
    artifact["physical_readback_count"] = 1
    artifact["sent_request_types"].append("set_gain")
    artifact["forbidden_request_types_sent"] = ["set_gain"]
    artifact["readbacks"][1]["status"] = "verified_from_cache"
    artifact["readbacks"][1]["source"] = "mixer_state_cache"
    artifact["apply_result"] = {"status": "applied"}

    validation = validate_wing_readonly_preflight_artifact(artifact)
    evidence = build_wing_readonly_preflight_evidence(
        artifact_payloads=(artifact,),
        artifact_roots=(),
    )

    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert evidence["summary"]["readiness_matrix_update_allowed"] is False
    assert evidence["summary"]["live_authority_granted"] is False
    assert "require_physical_readback_not_true" in evidence["summary"]["violations"]
    assert "forbidden_request_types_sent_not_empty" in evidence["summary"]["violations"]
    assert "unexpected_sent_request_types" in evidence["summary"]["violations"]
    assert "not_all_readbacks_physical" in evidence["summary"]["violations"]
    assert "readback_1_not_physical" in evidence["summary"]["violations"]
    assert "mutation_evidence_field_present:apply_result" in evidence["summary"]["violations"]


def test_preflight_evidence_rejects_missing_operator_channel_confirmation():
    artifact = copy.deepcopy(_valid_summary())
    artifact.pop("operator_channel_confirmation_required")
    artifact.pop("operator_confirmed_channels")

    validation = validate_wing_readonly_preflight_artifact(artifact)

    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "operator_channel_confirmation_required_not_true" in validation["violations"]
    assert "operator_confirmed_channels_missing" in validation["violations"]
    assert "operator_confirmed_channels_mismatch" in validation["violations"]


def test_preflight_evidence_rejects_missing_operator_session_confirmation():
    artifact = copy.deepcopy(_valid_summary())
    artifact.pop("operator_session_confirmation_required")
    artifact.pop("operator_confirmed_session_ready")

    validation = validate_wing_readonly_preflight_artifact(artifact)

    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "operator_session_confirmation_required_not_true" in validation["violations"]
    assert "operator_confirmed_session_ready_not_true" in validation["violations"]


def test_preflight_evidence_rejects_missing_expected_plan_link():
    artifact = copy.deepcopy(_valid_summary())
    artifact.pop("expected_plan_path")
    artifact.pop("expected_plan_sha256")
    artifact.pop("expected_plan_match")
    artifact.pop("expected_plan_violations")
    artifact.pop("expected_plan_required_for_real_run")

    validation = validate_wing_readonly_preflight_artifact(artifact)

    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "expected_plan_path_missing" in validation["violations"]
    assert "expected_plan_sha256_missing" in validation["violations"]
    assert "expected_plan_match_not_true" in validation["violations"]
    assert "expected_plan_violations_not_empty" in validation["violations"]
    assert (
        "expected_plan_required_for_real_run_not_true" in validation["violations"]
    )


def test_preflight_evidence_rejects_missing_existing_backend_connection_boundary():
    artifact = copy.deepcopy(_valid_summary())
    artifact.pop("physical_preflight_requires_existing_backend_connection")

    validation = validate_wing_readonly_preflight_artifact(artifact)

    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert (
        "physical_preflight_requires_existing_backend_connection_not_true"
        in validation["violations"]
    )


def test_preflight_evidence_rejects_operator_channel_confirmation_mismatch():
    artifact = copy.deepcopy(_valid_summary())
    artifact["operator_confirmed_channels"] = [4]

    validation = validate_wing_readonly_preflight_artifact(artifact)

    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "operator_confirmed_channels_mismatch" in validation["violations"]


def test_preflight_evidence_rejects_readbacks_for_unconfirmed_channel():
    artifact = copy.deepcopy(_valid_summary())
    artifact["readbacks"][0]["channel"] = 4
    artifact["readbacks"][0]["address"] = "/ch/4/fdr"

    validation = validate_wing_readonly_preflight_artifact(artifact)
    evidence = build_wing_readonly_preflight_evidence(
        artifact_payloads=(artifact,),
        artifact_roots=(),
    )

    expected_violation = "readback_targets_do_not_match_declared_channels_and_value_types"
    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert expected_violation in validation["violations"]
    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert expected_violation in evidence["summary"]["violations"]


def test_preflight_evidence_rejects_readback_address_mismatch():
    artifact = copy.deepcopy(_valid_summary())
    artifact["readbacks"][0]["address"] = "/ch/4/fdr"

    validation = validate_wing_readonly_preflight_artifact(artifact)

    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "readback_0_address_mismatch" in validation["violations"]


def test_preflight_evidence_rejects_missing_or_non_numeric_readback_value():
    artifact = copy.deepcopy(_valid_summary())
    artifact["readbacks"][0].pop("value")
    artifact["readbacks"][1]["value"] = "not-a-number"

    validation = validate_wing_readonly_preflight_artifact(artifact)

    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "readback_0_value_missing_or_non_numeric" in validation["violations"]
    assert "readback_1_value_missing_or_non_numeric" in validation["violations"]


def test_wing_readonly_preflight_evidence_discovers_summary_json_files(tmp_path):
    artifact_dir = tmp_path / "preflight" / "run"
    artifact_path = _write_file_artifact(artifact_dir)
    broken_dir = tmp_path / "preflight" / "broken"
    broken_dir.mkdir()
    (broken_dir / "summary.json").write_text("{", encoding="utf-8")

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["artifact_search_roots"] == [str(tmp_path / "preflight")]
    assert evidence["summary"]["artifact_count"] == 2
    assert evidence["summary"]["valid_candidate_count"] == 1
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert evidence["summary"]["accepted_as_readonly_preflight_candidate"] is False
    assert evidence["summary"]["mvp_blocker_candidate_count"] == 0
    assert evidence["summary"]["readiness_matrix_update_allowed"] is False
    assert evidence["summary"]["latest_valid_candidate_run_id"] == "20260606T120000Z_wing_readonly_preflight"
    assert evidence["summary"]["blocked_by"] == [
        "operator_must_replace_invalid_readonly_preflight_artifact"
    ]
    assert {row["status"] for row in evidence["mvp_blocker_matrix"]} == {
        "artifact_replacement_required"
    }
    assert any(row["source_id"] == str(artifact_path) for row in evidence["artifacts"])
    assert any(
        "artifact_load_error:JSONDecodeError" in row["violations"]
        for row in evidence["artifacts"]
    )


def test_file_preflight_artifact_requires_matching_event_log(tmp_path):
    artifact_dir = tmp_path / "preflight" / "missing-events"
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "summary.json"
    summary = _valid_summary()
    summary["events_path"] = str(artifact_dir / "events.jsonl")
    artifact_path.write_text(json.dumps(summary), encoding="utf-8")

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert "event_log_missing" in evidence["summary"]["violations"]
    assert "event_log_missing_or_invalid" in evidence["summary"]["violations"]


def test_file_preflight_artifact_requires_existing_expected_plan_file(tmp_path):
    artifact_dir = tmp_path / "preflight" / "missing-plan"
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "summary.json"
    events_path = artifact_dir / "events.jsonl"
    summary = _valid_summary()
    summary["summary_path"] = str(artifact_path)
    summary["events_path"] = str(events_path)
    summary["expected_plan_path"] = str(artifact_dir / "missing_plan.json")
    artifact_path.write_text(json.dumps(summary), encoding="utf-8")
    events_path.write_text(
        "".join(json.dumps(row) + "\n" for row in _valid_events(summary)),
        encoding="utf-8",
    )

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert "expected_plan_file_missing" in evidence["summary"]["violations"]


def test_file_preflight_artifact_requires_matching_expected_plan_sha(tmp_path):
    artifact_dir = tmp_path / "preflight" / "tampered-plan"
    _write_file_artifact(artifact_dir)
    summary_path = artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    Path(summary["expected_plan_path"]).write_text("tampered", encoding="utf-8")

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert "expected_plan_sha256_mismatch" in evidence["summary"]["violations"]


def test_file_preflight_artifact_requires_expected_plan_metadata_to_match_summary(tmp_path):
    artifact_path = _write_file_artifact(tmp_path / "preflight" / "plan-mismatch")
    summary = json.loads(artifact_path.read_text(encoding="utf-8"))
    plan = _valid_expected_plan(summary)
    plan["channels"] = [4]
    _write_expected_plan(artifact_path, plan)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert "expected_plan_channels_mismatch" in evidence["summary"]["violations"]


def test_file_preflight_artifact_requires_expected_plan_policy_metadata(tmp_path):
    artifact_path = _write_file_artifact(tmp_path / "preflight" / "plan-policy-mismatch")
    summary = json.loads(artifact_path.read_text(encoding="utf-8"))
    plan = _valid_expected_plan(summary)
    plan["selected_mvp_blocker"] = "Operator proposal queue import"
    plan["selected_mvp_blocker_status"] = "closed"
    plan["operator_required"] = False
    plan["plan_output"] = "artifacts/other_plan.json"
    plan["backend_port_prestart_check_required"] = False
    plan["backend_port_prestart_check_must_pass_before_starting_backend"] = False
    plan["backend_port_prestart_check_command"] = ["python3", "other_check.py"]
    plan["backend_port_prestart_check_failure_detail_fields"] = ["other_field"]
    plan["allowed_request_types"] = ["get_operator_mode", "set_gain"]
    plan["forbidden_request_types"] = ["set_fader"]
    plan["fail_closed_conditions"] = ["backend_not_in_wing_mode"]
    _write_expected_plan(artifact_path, plan)

    evidence = build_wing_readonly_preflight_evidence(
        artifact_roots=(tmp_path / "preflight",)
    )

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    violations = evidence["summary"]["violations"]
    assert "expected_plan_selected_mvp_blocker_mismatch" in violations
    assert "expected_plan_selected_mvp_blocker_status_mismatch" in violations
    assert "expected_plan_operator_required_not_true" in violations
    assert "expected_plan_plan_output_mismatch" in violations
    assert (
        "expected_plan_backend_port_prestart_check_required_not_true"
        in violations
    )
    assert (
        "expected_plan_backend_port_prestart_check_must_pass_before_starting_backend_not_true"
        in violations
    )
    assert (
        "expected_plan_backend_port_prestart_check_command_mismatch"
        in violations
    )
    assert (
        "expected_plan_backend_port_prestart_check_failure_detail_fields_mismatch"
        in violations
    )
    assert "expected_plan_allowed_request_types_mismatch" in violations
    assert "expected_plan_forbidden_request_types_mismatch" in violations
    assert "expected_plan_forbidden_request_type_missing:set_gain" in violations
    assert "expected_plan_fail_closed_conditions_mismatch" in violations
    assert (
        "expected_plan_fail_closed_condition_missing:physical_readback_missing"
        in violations
    )
    assert (
        "expected_plan_fail_closed_condition_missing:stale_backend_port_prestart_detail_contract"
        in violations
    )
    assert (
        "expected_plan_fail_closed_condition_missing:manual_review_decision_missing_or_invalid"
        in violations
    )


def test_file_preflight_artifact_requires_summary_prestart_contract(tmp_path):
    artifact_path = _write_file_artifact(tmp_path / "preflight" / "summary-prestart-mismatch")
    summary = json.loads(artifact_path.read_text(encoding="utf-8"))
    summary["backend_port_prestart_check_required"] = False
    summary["backend_port_prestart_check_must_pass_before_starting_backend"] = False
    summary["backend_port_prestart_check_command"] = ["python3", "other_check.py"]
    summary["backend_port_prestart_check_failure_detail_fields"] = ["other_field"]
    artifact_path.write_text(json.dumps(summary), encoding="utf-8")

    evidence = build_wing_readonly_preflight_evidence(
        artifact_roots=(tmp_path / "preflight",)
    )

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    violations = evidence["summary"]["violations"]
    assert "backend_port_prestart_check_required_not_true" in violations
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend_not_true"
        in violations
    )
    assert "backend_port_prestart_check_command_mismatch" in violations
    assert "backend_port_prestart_check_failure_detail_fields_mismatch" in violations


def test_file_preflight_artifact_requires_prestart_summary_file(tmp_path):
    artifact_path = _write_file_artifact(tmp_path / "preflight" / "missing-prestart")
    summary = json.loads(artifact_path.read_text(encoding="utf-8"))
    Path(summary["backend_port_prestart_summary_path"]).unlink()
    artifact_path.write_text(json.dumps(summary), encoding="utf-8")

    evidence = build_wing_readonly_preflight_evidence(
        artifact_paths=(artifact_path,),
        artifact_roots=(),
    )

    assert "prestart_summary_file_missing" in evidence["summary"]["violations"]
    assert evidence["artifacts"][0]["accepted_as_readonly_preflight_candidate"] is False


def test_file_preflight_artifact_requires_matching_prestart_summary_sha(tmp_path):
    artifact_path = _write_file_artifact(tmp_path / "preflight" / "prestart-sha")
    summary = json.loads(artifact_path.read_text(encoding="utf-8"))
    Path(summary["backend_port_prestart_summary_path"]).write_text(
        json.dumps({"type": "no_wing_release_check", "status": "passed"}),
        encoding="utf-8",
    )
    artifact_path.write_text(json.dumps(summary), encoding="utf-8")

    evidence = build_wing_readonly_preflight_evidence(
        artifact_paths=(artifact_path,),
        artifact_roots=(),
    )

    assert "prestart_summary_sha256_mismatch" in evidence["summary"]["violations"]
    assert evidence["artifacts"][0]["accepted_as_readonly_preflight_candidate"] is False


def test_file_preflight_artifact_rejects_prestart_summary_contact_boundary(tmp_path):
    artifact_path = _write_file_artifact(tmp_path / "preflight" / "prestart-contact")
    summary = json.loads(artifact_path.read_text(encoding="utf-8"))
    prestart_path = Path(summary["backend_port_prestart_summary_path"])
    prestart = json.loads(prestart_path.read_text(encoding="utf-8"))
    prestart["contacts_backend"] = True
    prestart_path.write_text(json.dumps(prestart), encoding="utf-8")
    summary["backend_port_prestart_summary_sha256"] = hashlib.sha256(
        prestart_path.read_bytes()
    ).hexdigest()
    artifact_path.write_text(json.dumps(summary), encoding="utf-8")

    evidence = build_wing_readonly_preflight_evidence(
        artifact_paths=(artifact_path,),
        artifact_roots=(),
    )

    assert evidence["artifacts"][0]["accepted_as_readonly_preflight_candidate"] is False
    assert "prestart_summary_contacts_backend_not_false" in evidence["summary"][
        "violations"
    ]
    assert "prestart_summary_recorded_contacts_backend_mismatch" in evidence["summary"][
        "violations"
    ]


def test_file_preflight_artifact_requires_expected_plan_request_sequence_to_match_summary(tmp_path):
    artifact_path = _write_file_artifact(tmp_path / "preflight" / "request-sequence-mismatch")
    summary = json.loads(artifact_path.read_text(encoding="utf-8"))
    plan = _valid_expected_plan(summary)
    plan["request_sequence"][2]["channel"] = 4
    _write_expected_plan(artifact_path, plan)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert "expected_plan_request_sequence_mismatch" in evidence["summary"]["violations"]


def test_file_preflight_artifact_rejects_tampered_event_log_send_types(tmp_path):
    summary = _valid_summary()
    events = _valid_events(summary)
    events.insert(
        2,
        _event(
            99,
            "send",
            {
                "type": "set_gain",
                "channel": 3,
                "value_type": "gain",
                "value": 1.0,
            },
        ),
    )
    _write_file_artifact(tmp_path / "preflight" / "tampered", summary=summary, events=events)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert "event_log_sent_request_types_mismatch" in evidence["summary"]["violations"]
    assert "event_log_unexpected_sent_request_types" in evidence["summary"]["violations"]
    assert "event_log_forbidden_request_type_sent" in evidence["summary"]["violations"]
    assert "forbidden_request_types_sent_not_empty" in evidence["summary"]["violations"]


def test_file_preflight_evidence_rejects_tampered_event_log_connect_mixer_send(tmp_path):
    summary = _valid_summary()
    events = _valid_events(summary)
    events.insert(
        2,
        _event(
            99,
            "send",
            {
                "type": "connect_mixer",
                "channel": 3,
                "value_type": "connection",
                "value": "mixer",
            },
        ),
    )
    _write_file_artifact(tmp_path / "preflight" / "connect-mixer", summary=summary, events=events)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert "event_log_forbidden_request_type_sent" in evidence["summary"]["violations"]
    assert "event_log_unexpected_sent_request_types" in evidence["summary"]["violations"]
    assert "forbidden_request_types_sent_not_empty" in evidence["summary"]["violations"]


def test_file_preflight_artifact_rejects_event_payload_mutation_evidence(tmp_path):
    summary = _valid_summary()
    events = _valid_events(summary)
    events[1]["payload"] = {
        **events[1]["payload"],
        "apply_result": {"status": "applied"},
    }
    _write_file_artifact(tmp_path / "preflight" / "payload-mutation", summary=summary, events=events)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert (
        "event_log_mutation_evidence_field_present:1:apply_result"
        in evidence["summary"]["violations"]
    )


def test_file_preflight_artifact_rejects_nested_event_payload_live_authority(tmp_path):
    summary = _valid_summary()
    events = _valid_events(summary)
    events[3]["payload"] = {
        **events[3]["payload"],
        "state": {
            **events[3]["payload"]["state"],
            "safety": {"live_authority_granted": True},
        },
    }
    _write_file_artifact(tmp_path / "preflight" / "payload-authority", summary=summary, events=events)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert (
        "event_log_unsafe_authority_field_true:3:live_authority_granted"
        in evidence["summary"]["violations"]
    )


def test_file_preflight_artifact_rejects_non_wing_operator_mode_event(tmp_path):
    summary = _valid_summary()
    events = _valid_events(summary)
    events[1]["payload"] = {
        **events[1]["payload"],
        "connection_mode": "offline",
    }
    _write_file_artifact(
        tmp_path / "preflight" / "operator-offline",
        summary=summary,
        events=events,
    )

    evidence = build_wing_readonly_preflight_evidence(
        artifact_roots=(tmp_path / "preflight",)
    )

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert (
        "event_log_operator_mode_connection_mode_not_wing"
        in evidence["summary"]["violations"]
    )


def test_file_preflight_artifact_rejects_non_wing_state_update_event(tmp_path):
    summary = _valid_summary()
    events = _valid_events(summary)
    events[3]["payload"] = {
        **events[3]["payload"],
        "mode": "offline",
    }
    _write_file_artifact(
        tmp_path / "preflight" / "state-offline",
        summary=summary,
        events=events,
    )

    evidence = build_wing_readonly_preflight_evidence(
        artifact_roots=(tmp_path / "preflight",)
    )

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert "event_log_state_update_mode_not_wing" in evidence["summary"]["violations"]


def test_file_preflight_artifact_rejects_wrong_event_request_order(tmp_path):
    summary = _valid_summary()
    events = _valid_events(summary)
    events[0], events[2] = events[2], events[0]
    _write_file_artifact(tmp_path / "preflight" / "wrong-order", summary=summary, events=events)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert "event_log_request_sequence_invalid" in evidence["summary"]["violations"]


def test_file_preflight_artifact_rejects_grouped_send_recv_events(tmp_path):
    summary = _valid_summary()
    events = _valid_events(summary)
    grouped = [
        events[0],
        events[2],
        events[1],
        events[3],
        *events[4:],
    ]
    _write_file_artifact(tmp_path / "preflight" / "grouped-events", summary=summary, events=grouped)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert "event_log_send_recv_sequence_invalid" in evidence["summary"]["violations"]
    assert "event_log_response_type_mismatch:0" in evidence["summary"]["violations"]


def test_file_preflight_artifact_rejects_send_error_event(tmp_path):
    summary = _valid_summary()
    _write_file_artifact(
        tmp_path / "preflight" / "send-error",
        summary=summary,
        events=[
            _event(
                1,
                "send_error",
                {
                    "request_type": "get_operator_mode",
                    "error_type": "RuntimeError",
                    "error": "send failed",
                },
            ),
        ],
    )

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert "event_log_send_error_present" in evidence["summary"]["violations"]
    assert "event_log_missing_or_invalid" in evidence["summary"]["violations"]
    assert "event_log_send_recv_sequence_invalid" in evidence["summary"]["violations"]


def test_file_preflight_artifact_rejects_non_contiguous_event_sequence(tmp_path):
    summary = _valid_summary()
    events = _valid_events(summary)
    events[3]["seq"] = 99
    _write_file_artifact(tmp_path / "preflight" / "bad-seq", summary=summary, events=events)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert "event_log_seq_sequence_invalid" in evidence["summary"]["violations"]


def test_file_preflight_artifact_rejects_invalid_event_timestamp(tmp_path):
    summary = _valid_summary()
    events = _valid_events(summary)
    events[2]["timestamp"] = "2026-06-06 12:00:00"
    _write_file_artifact(tmp_path / "preflight" / "bad-timestamp", summary=summary, events=events)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert "event_log_timestamp_invalid:2" in evidence["summary"]["violations"]


def test_file_preflight_artifact_rejects_non_monotonic_event_timestamps(tmp_path):
    summary = _valid_summary()
    events = _valid_events(summary)
    events[3]["timestamp"] = "2026-06-06T11:59:59.999Z"
    _write_file_artifact(tmp_path / "preflight" / "backward-timestamp", summary=summary, events=events)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert "event_log_timestamp_sequence_invalid" in evidence["summary"]["violations"]


def test_file_preflight_artifact_rejects_event_timestamp_after_capture(tmp_path):
    summary = _valid_summary()
    events = _valid_events(summary)
    events[-1]["timestamp"] = "2026-06-07T01:00:00.001Z"
    _write_file_artifact(tmp_path / "preflight" / "future-event", summary=summary, events=events)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert "event_log_timestamp_after_captured_at:9" in evidence["summary"]["violations"]


def test_file_preflight_artifact_rejects_query_readback_identity_mismatch(tmp_path):
    summary = _valid_summary()
    events = _valid_events(summary)
    events[5]["payload"] = {
        **events[5]["payload"],
        "channel": 4,
        "address": "/ch/4/fdr",
    }
    _write_file_artifact(tmp_path / "preflight" / "identity-mismatch", summary=summary, events=events)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert "event_log_readbacks_mismatch" in evidence["summary"]["violations"]
    assert "event_log_query_readback_identity_mismatch:0" in evidence["summary"]["violations"]


def test_file_preflight_artifact_rejects_summary_readback_wrong_address(tmp_path):
    summary = _valid_summary()
    summary["readbacks"][0]["address"] = "/ch/3/in/set/trim"
    events = _valid_events(summary)
    _write_file_artifact(tmp_path / "preflight" / "wrong-readback-address", summary=summary, events=events)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert "readback_0_address_mismatch" in evidence["summary"]["violations"]


def test_file_preflight_artifact_rejects_query_readback_request_id_mismatch(tmp_path):
    summary = _valid_summary()
    events = _valid_events(summary)
    events[5]["payload"] = {
        **events[5]["payload"],
        "request_id": "stale:ch3:fader",
    }
    _write_file_artifact(tmp_path / "preflight" / "request-id-mismatch", summary=summary, events=events)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert "event_log_readbacks_mismatch" in evidence["summary"]["violations"]
    assert "event_log_query_readback_request_id_mismatch:0" in evidence["summary"]["violations"]
    assert "event_log_query_readback_identity_mismatch:0" in evidence["summary"]["violations"]


def test_file_preflight_artifact_rejects_query_outside_confirmed_channels(tmp_path):
    summary = _valid_summary()
    summary["readbacks"][0] = {
        **summary["readbacks"][0],
        "channel": 4,
        "address": "/ch/4/fdr",
    }
    events = _valid_events(summary)
    _write_file_artifact(tmp_path / "preflight" / "unconfirmed-channel", summary=summary, events=events)

    evidence = build_wing_readonly_preflight_evidence(artifact_roots=(tmp_path / "preflight",))

    assert evidence["status"] == "readonly_preflight_artifact_replacement_required"
    assert evidence["summary"]["valid_candidate_count"] == 0
    assert evidence["summary"]["invalid_candidate_count"] == 1
    assert (
        "readback_targets_do_not_match_declared_channels_and_value_types"
        in evidence["summary"]["violations"]
    )


def test_wing_readonly_preflight_evidence_uses_repo_default_roots():
    evidence = build_wing_readonly_preflight_evidence()

    assert evidence["summary"]["artifact_search_roots"] == list(
        DEFAULT_WING_READONLY_PREFLIGHT_ARTIFACT_ROOTS
    )
    assert evidence["safety"]["queries_console"] is False
    assert evidence["safety"]["executes_tests"] is False


def test_wing_readonly_preflight_evidence_is_transport_free():
    source = Path("backend/wing_readonly_preflight_evidence.py").read_text(encoding="utf-8")

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
        "set_fader(",
        "set_gain(",
        "set_eq(",
        "set_compressor(",
        "apply_operator_proposal(",
    ]
    for token in forbidden:
        assert token not in source


def test_wing_readonly_preflight_evidence_is_registry_shadow_only_not_executor():
    method = get_processing_method(WING_READONLY_PREFLIGHT_EVIDENCE_METHOD_ID)

    assert method is not None
    assert method.status == "shadow_only"
    assert method.domain == "evaluation"
    assert method.can_analyze is True
    assert method.can_write_live is False
    assert method.live_write_kinds == ()
    assert set(method.pipeline_profiles) == {"shadow_quality", "lab_dsp"}

    main = get_pipeline_profile("main_live_assist")
    assert main is not None
    assert WING_READONLY_PREFLIGHT_EVIDENCE_METHOD_ID not in main.executor_method_ids

    snapshot = build_processing_registry_snapshot(include_methods=False)
    evidence = snapshot["wing_readonly_preflight_evidence"]
    assert evidence["type"] == "wing_readonly_preflight_evidence"
    assert evidence["safety"]["live_write"] is False
    assert evidence["safety"]["readiness_closure_allowed_by_evidence"] is False
    assert evidence["summary"]["live_authority_granted"] is False


def test_wing_readonly_preflight_evidence_is_console_read_goal_support_path():
    row = get_project_goal_capability("console_read_parameters")

    assert row is not None
    assert WING_READONLY_PREFLIGHT_EVIDENCE_METHOD_ID in row.method_ids
    assert "backend/wing_readonly_preflight_evidence.py" in row.support_paths
    assert row.current_live_write_kinds == ()
