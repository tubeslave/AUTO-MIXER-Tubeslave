import copy
import json
from pathlib import Path

from backend.wing_readonly_preflight_evidence import (
    DEFAULT_WING_READONLY_PREFLIGHT_ARTIFACT_ROOTS,
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
        "proof_artifact_schema": "wing_readonly_preflight.v1",
        "proof_runner": "wing_readonly_preflight_runner",
        "connection_mode": "wing",
        "channels": [3],
        "operator_channel_confirmation_required": True,
        "operator_confirmed_channels": [3],
        "value_types": ["fader", "gain"],
        "require_physical_readback": True,
        "request_id_policy": "required_for_every_query_mixer_parameter_response",
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


def _write_file_artifact(artifact_dir: Path, summary: dict | None = None, events: list[dict] | None = None) -> Path:
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "summary.json"
    events_path = artifact_dir / "events.jsonl"
    payload = copy.deepcopy(summary or _valid_summary())
    payload["events_path"] = str(events_path)
    payload["summary_path"] = str(artifact_path)
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
    assert evidence["artifacts"][0]["summary_path"] == "artifacts/wing_readonly_preflight/run/summary.json"
    assert evidence["artifacts"][0]["events_path"] == "artifacts/wing_readonly_preflight/run/events.jsonl"
    assert evidence["artifacts"][0]["physical_readback_count"] == 2
    assert evidence["artifacts"][0]["channels"] == [3]
    assert evidence["artifacts"][0]["operator_confirmed_channels"] == [3]
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
