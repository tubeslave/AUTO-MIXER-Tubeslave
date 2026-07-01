import json
from pathlib import Path

from backend.autonomy_promotion_readiness import (
    build_autonomy_promotion_readiness_snapshot,
)
from backend.wing_readonly_preflight_evidence import (
    EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE_SUMMARY,
    build_wing_readonly_preflight_evidence,
)
from backend.wing_readonly_preflight_contract import (
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_MATCH_FIELDS,
    WING_READONLY_PREFLIGHT_PRESTART_BACKEND_CHECK_REQUIRED_FALSE_PLAN_FIELDS,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_CONNECTION_POLICY,
    WING_READONLY_PREFLIGHT_PRESTART_SUMMARY_TYPE,
)
from backend.wing_readonly_preflight_review_decision import (
    build_wing_readonly_preflight_review_decision_ledger,
)
from tools import evaluate_supervised_proof_artifacts as report_tool

PRESTART_BACKEND_PORT_CHECK_COMMAND = list(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
)
PRESTART_BACKEND_PORT_CHECK_COMMAND_TEXT = (
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
)
PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS = list(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
)
GAIN_REQUIRED_EVIDENCE_COMPONENTS = [
    "physical_pre_read",
    "bounded_gain_down_apply",
    "apply_readback",
    "rollback_readback",
    "emergency_stop_block",
    "unsafe_gain_up_followup_blocked",
    "gain_up_operator_authored_opt_in",
]


def valid_wing_preflight_summary() -> dict:
    return {
        "run_id": "20260606T150000Z_wing_readonly_preflight",
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
        "backend_port_prestart_check_command": PRESTART_BACKEND_PORT_CHECK_COMMAND,
        "backend_port_prestart_check_command_text": (
            PRESTART_BACKEND_PORT_CHECK_COMMAND_TEXT
        ),
        "backend_port_prestart_check_failure_detail_fields": (
            PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS
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
        "events_path": (
            "artifacts/wing_readonly_preflight/"
            "20260606T150000Z_wing_readonly_preflight/events.jsonl"
        ),
        "summary_path": (
            "artifacts/wing_readonly_preflight/"
            "20260606T150000Z_wing_readonly_preflight/summary.json"
        ),
    }


def patch_report_tool_wing_preflight_candidate(monkeypatch) -> dict:
    evidence = build_wing_readonly_preflight_evidence(
        artifact_payloads=(valid_wing_preflight_summary(),),
        artifact_roots=(),
    )
    monkeypatch.setattr(
        report_tool,
        "build_autonomy_promotion_readiness_snapshot",
        lambda: build_autonomy_promotion_readiness_snapshot(
            wing_readonly_preflight_evidence_snapshot=evidence,
        ),
    )
    monkeypatch.setattr(
        report_tool,
        "build_wing_readonly_preflight_review_decision_ledger",
        lambda: build_wing_readonly_preflight_review_decision_ledger(
            preflight_evidence=evidence,
            decision_roots=(),
        ),
    )
    return evidence


def write_summary(run_dir: Path, payload: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def assert_prestart_backend_port_check(summary: dict) -> None:
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


def dry_run_summary(proof_id: str = "dry-gain-proof") -> dict:
    return {
        "run_id": proof_id,
        "status": "passed",
        "channel": 3,
        "delta_db": -0.25,
        "physical_wing": False,
        "write_policy": "fake_mixer_supervised_gate_only",
        "old_value": 1.0,
        "target_value": 0.75,
        "apply_result": {
            "apply_result": {
                "status": "applied",
                "channel": 3,
                "value_type": "gain",
                "old_value": 1.0,
                "effective_value": 0.75,
                "readback_status": "verified_from_physical_readback",
                "pre_read_status": "verified_from_physical_readback",
                "proof_status": "verified",
                "rollback_available": True,
                "rollback_value": 1.0,
            },
        },
        "rollback_result": {
            "rollback_status": "applied",
            "rollback_readback_status": "verified_from_physical_readback",
            "rollback_proof_status": "verified",
        },
        "cooldown_apply_result": {"apply_result": {"reason": "cooldown"}},
        "post_emergency_apply_result": {"apply_result": {"reason": "emergency_stop_active"}},
    }


def physical_summary(proof_id: str = "physical-fader-proof") -> dict:
    payload = dry_run_summary(proof_id)
    payload.update(
        {
            "channel": 4,
            "value_type": "fader",
            "physical_wing": True,
            "old_value": -12.0,
            "target_value": -12.25,
            "delta_db": -0.25,
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
            "rollback_readback_source": "physical_readback",
        }
    )
    payload["apply_result"]["apply_result"].update(
        {
            "channel": 4,
            "value_type": "fader",
            "old_value": -12.0,
            "effective_value": -12.25,
            "rollback_value": -12.0,
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
        }
    )
    return payload


def cache_status_physical_summary(proof_id: str = "cache-status-physical-proof") -> dict:
    payload = physical_summary(proof_id)
    payload["apply_result"]["apply_result"]["pre_read_status"] = "verified_from_cache"
    payload["apply_result"]["apply_result"]["readback_status"] = "verified_from_cache"
    payload["rollback_result"]["rollback_readback_status"] = "verified_from_cache"
    return payload


def legacy_execute_wing_summary(proof_id: str = "legacy-physical-gain-proof") -> dict:
    payload = dry_run_summary(proof_id)
    payload.pop("physical_wing")
    payload.update(
        {
            "run_id": "20260523T104641Z_wing_operator_proposal_ch4_gain",
            "execute": True,
            "write_policy": "apply_operator_proposal_only",
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
            "rollback_readback_source": "physical_readback",
            "operator_reason": "operator approved small gain-down physical proof",
        }
    )
    payload["apply_result"]["apply_result"].update(
        {
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
        }
    )
    return payload


def timeout_after_apply_summary(proof_id: str = "timeout-after-apply-proof") -> dict:
    payload = physical_summary(proof_id)
    payload.update(
        {
            "status": "manual_review_required",
            "reason": "connection_or_timeout_after_apply_request",
            "proof_artifact_schema": "wing_operator_proposal_validation.v2",
            "proof_runner": "wing_operator_proposal_validation_runner",
            "proof_mode": "physical",
            "execute": True,
            "fail_closed": False,
            "live_apply_request_sent": True,
            "rollback_request_sent": False,
        }
    )
    return payload


def timeout_before_apply_summary(proof_id: str = "timeout-before-apply-proof") -> dict:
    payload = physical_summary(proof_id)
    payload.update(
        {
            "status": "blocked",
            "reason": "connection_or_timeout_fail_closed",
            "proof_artifact_schema": "wing_operator_proposal_validation.v2",
            "proof_runner": "wing_operator_proposal_validation_runner",
            "proof_mode": "physical",
            "execute": True,
            "fail_closed": True,
            "live_apply_request_sent": False,
            "rollback_request_sent": False,
        }
    )
    return payload


def test_proof_artifact_report_tool_is_read_only_and_transport_free():
    source = Path("tools/evaluate_supervised_proof_artifacts.py").read_text(encoding="utf-8")

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "requested_change",
        "requested_changes",
    ]
    for token in forbidden:
        assert token not in source


def test_discover_summary_paths_recurses_and_only_returns_summary_json(tmp_path: Path):
    first = write_summary(tmp_path / "a", dry_run_summary("a-proof"))
    second = write_summary(tmp_path / "nested" / "b", dry_run_summary("b-proof"))
    (tmp_path / "nested" / "b" / "events.jsonl").write_text("{}", encoding="utf-8")

    paths = report_tool.discover_summary_paths([tmp_path])

    assert paths == sorted([first, second])


def test_report_distinguishes_dry_run_physical_ignored_and_invalid_summaries(tmp_path: Path):
    dry = write_summary(tmp_path / "dry", dry_run_summary())
    physical = write_summary(tmp_path / "physical", physical_summary())
    legacy = write_summary(tmp_path / "legacy", legacy_execute_wing_summary())
    ignored = write_summary(tmp_path / "offline_validation", {"type": "offline_validation", "status": "passed"})
    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    invalid = invalid_dir / "summary.json"
    invalid.write_text("{not-json", encoding="utf-8")

    report = report_tool.build_supervised_proof_artifact_report(
        [dry, physical, legacy, ignored, invalid],
        require_physical=True,
    )

    assert report["status"] == "blocked"
    assert report["safety"]["live_write"] is False
    assert report["summary"]["source_summary_count"] == 5
    assert report["summary"]["artifact_count"] == 3
    assert report["summary"]["ignored_summary_count"] == 1
    assert report["summary"]["invalid_summary_count"] == 1
    assert report["summary"]["status_counts"]["dry_run_ready"] == 1
    assert report["summary"]["status_counts"]["supervised_candidate"] == 1
    assert report["summary"]["status_counts"]["blocked"] == 1
    assert report["summary"]["proof_mode_counts"]["legacy_ambiguous"] == 1
    assert report["summary"]["promotion_allowed_proof_ids"] == ["physical-fader-proof"]
    assert report["summary"]["dry_run_ready_proof_ids"] == ["dry-gain-proof"]
    assert report["summary"]["blocked_proof_ids"] == ["20260523T104641Z_wing_operator_proposal_ch4_gain"]
    assert report["summary"]["autonomous_ready_proof_ids"] == []
    assert report["summary"]["required_evidence_components_by_value_type"]["gain"] == (
        GAIN_REQUIRED_EVIDENCE_COMPONENTS
    )
    assert report["ignored_summaries"] == [str(ignored)]
    assert report["invalid_summaries"] == [{"artifact_path": str(invalid), "reason": "json_load_error:JSONDecodeError"}]


def test_report_blocks_physical_summary_that_uses_cache_readback_status(tmp_path: Path):
    artifact = write_summary(tmp_path / "cache_status_physical", cache_status_physical_summary())

    report = report_tool.build_supervised_proof_artifact_report(
        [artifact],
        require_physical=True,
    )

    assert report["status"] == "ok"
    assert report["summary"]["status_counts"]["supervised_candidate"] == 0
    assert report["summary"]["status_counts"]["blocked"] == 1
    assert report["summary"]["promotion_allowed_proof_ids"] == []
    assert report["summary"]["blocked_proof_ids"] == ["cache-status-physical-proof"]
    row = report["rows"][0]
    assert "physical_pre_read_status_required" in row["blockers"]
    assert "physical_apply_readback_status_required" in row["blockers"]
    assert "physical_rollback_readback_status_required" in row["blockers"]
    assert row["evidence"]["pre_read_source"] == "physical_readback"
    assert row["evidence"]["readback_source"] == "physical_readback"
    assert row["evidence"]["rollback_readback_source"] == "physical_readback"


def test_exit_code_requires_minimum_physical_candidates(tmp_path: Path):
    dry = write_summary(tmp_path / "dry", dry_run_summary())
    report = report_tool.build_supervised_proof_artifact_report([dry], require_physical=True)

    assert report_tool.exit_code_for_report(report, min_promotion_candidates=0) == 0
    assert report_tool.exit_code_for_report(report, min_promotion_candidates=1) == 2


def test_main_writes_report_and_returns_success_for_physical_candidate(
    tmp_path: Path,
    monkeypatch,
):
    patch_report_tool_wing_preflight_candidate(monkeypatch)
    write_summary(tmp_path / "physical", physical_summary())
    write_summary(tmp_path / "legacy", legacy_execute_wing_summary())
    output_path = tmp_path / "report.json"

    exit_code = report_tool.main(
        [
            "--artifact-root",
            str(tmp_path),
            "--require-physical",
            "--min-promotion-candidates",
            "1",
            "--output",
            str(output_path),
            "--quiet",
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["report_path"] == str(output_path)
    assert report["summary"]["promotion_allowed_proof_ids"] == ["physical-fader-proof"]
    assert report["summary"]["autonomous_ready_proof_ids"] == []
    assert report["promotion_evidence"]["safety"]["live_widening_allowed"] is False
    assert report["promotion_evidence"]["summary"]["safety_row_review_candidate_ids"] == [
        "fader_supervised_evidence"
    ]
    assert report["proof_remediation_plan"]["safety"]["execution_allowed_by_plan"] is False
    assert report["proof_remediation_plan"]["status"] == "manual_remediation_required"
    assert report["proof_remediation_plan"]["summary"]["legacy_ambiguous_count"] == 1
    assert report["proof_remediation_plan"]["actions"][0]["action_type"] == "rerun_current_physical_schema"
    readiness = report["autonomy_promotion_readiness"]
    assert readiness["safety"]["live_write"] is False
    preflight = readiness["summary"]["wing_readonly_preflight_evidence"]
    assert preflight["evidence_status"] == "readonly_preflight_candidate_available"
    assert preflight["readiness_update_candidate_available"] is True
    assert (
        "operator_must_review_readonly_preflight_candidate_before_readiness_update"
        in preflight["blocked_by"]
    )
    decision = report["wing_readonly_preflight_review_decision_ledger"]
    assert decision["safety"]["live_write"] is False
    assert decision["summary"]["accepted_record_count"] == 0
    assert decision["summary"]["manual_readiness_update_candidate_available"] is False
    assert report["autonomy_evidence_overlay"]["safety"]["live_widening_allowed"] is False
    assert report["autonomy_evidence_overlay"]["summary"]["autonomous_ready_target_ids"] == []
    assert report["autonomy_evidence_overlay"]["summary"]["manual_safety_review_candidate_target_ids"] == [
        "fader_supervised_promotion"
    ]
    assert report["autonomy_evidence_overlay"]["summary"]["remediation_required_target_ids"] == [
        "gain_trim_supervised_promotion"
    ]
    assert report["autonomy_evidence_overlay"]["summary"]["no_current_evidence_target_ids"] == []
    assert [
        row["value_type"] for row in report["autonomy_evidence_overlay"]["rows"]
    ] == ["fader", "gain"]
    assert report["autonomy_mode_transition_gate"]["safety"]["automatic_mode_change_allowed"] is False
    assert report["autonomy_mode_transition_gate"]["summary"]["autonomous_apply_allowed_modes"] == []
    assert report["autonomy_mode_transition_gate"]["summary"]["full_auto_readiness_gap_bucket_counts"]["read_surface"] > 0
    assert report["autonomy_mode_transition_gate"]["summary"]["supervised_current_live_write_kinds"] == [
        "fader",
        "gain",
    ]
    assert report["autonomy_mode_transition_gate"]["rows"][2]["mode"] == "supervised"
    assert report["autonomy_mode_transition_gate"]["rows"][2]["status"] == "limited_available"
    assert report["autonomy_mode_transition_gate"]["rows"][3]["mode"] == "auto"
    assert report["autonomy_mode_transition_gate"]["rows"][3]["status"] == "blocked"
    gap_rollup = report["summary"]["autonomy_gap_rollup"]
    assert gap_rollup["read_only"] is True
    assert gap_rollup["live_write"] is False
    assert gap_rollup["live_widening_allowed"] is False
    assert gap_rollup["full_autonomous_ready"] is False
    assert gap_rollup["source"] == "autonomy_mode_transition_gate.rows[mode=auto]"
    assert gap_rollup["full_autonomous_blocked"] is True
    assert gap_rollup["full_auto_gap_bucket_counts"]["read_surface"] > 0
    assert gap_rollup["full_auto_gap_bucket_counts"]["write_surface"] > 0
    assert "scene_preset_supervised_promotion" in gap_rollup["full_auto_gap_target_ids_by_bucket"]["read_surface"]
    assert gap_rollup["evidence_requirement_ids_by_action_family"]["bus_group_matrix"] == [
        "bus_group_assignment_cache_freshness",
        "bus_group_fader_readback",
        "matrix_send_readback",
        "main_bus_true_peak_headroom",
        "bus_group_physical_fixture",
        "hardware_readback_and_rollback_proof",
    ]
    assert gap_rollup["evidence_requirement_ids_by_action_family"]["mute"] == [
        "current_mute_cache_freshness",
        "mute_owner_intent_evidence",
        "signal_return_or_idle_window_evidence",
        "manual_mute_authority_review",
        "mute_physical_fixture",
        "hardware_readback_rollback_and_emergency_proof",
    ]
    assert gap_rollup["evidence_requirement_ids_by_action_family"]["gain"] == (
        GAIN_REQUIRED_EVIDENCE_COMPONENTS
    )
    assert "Add bus/group/matrix safety review" in (
        gap_rollup["next_required_evidence_by_action_family"]["bus_group_matrix"][0]
    )
    assert "fresh mute-state intent" in (
        gap_rollup["next_required_evidence_by_action_family"]["mute"][0]
    )
    assert gap_rollup["supervised_live_apply_allowed"] is True
    assert gap_rollup["supervised_current_live_write_kinds"] == [
        "fader",
        "gain",
    ]
    assert gap_rollup["supervised_current_live_write_surfaces"] == [
        "preamp",
        "trim",
        "fader",
    ]
    assert "Full Autonomous remains blocked" in gap_rollup["claim"]
    assert report["next_wing_proof_plan"]["safety"]["execution_allowed_by_plan"] is False
    assert report["next_wing_proof_plan"]["status"] == "manual_review_ready"
    assert report["next_wing_proof_plan"]["summary"]["planned_value_types"] == []
    assert report["next_wing_proof_plan"]["summary"]["step_count"] == 0
    assert report["next_wing_proof_plan"]["summary"]["readonly_preflight_required"] is True
    assert_prestart_backend_port_check(report["next_wing_proof_plan"]["summary"])
    assert (
        report["next_wing_proof_plan"]["summary"]["readonly_preflight_candidate_available"]
        is True
    )
    assert report["next_wing_proof_plan"]["summary"]["readonly_preflight_step_count"] == 0
    assert (
        report["next_wing_proof_plan"]["summary"][
            "readonly_preflight_review_decision_accepted"
        ]
        is False
    )
    assert (
        report["next_wing_proof_plan"]["summary"][
            "readonly_preflight_review_decision_status"
        ]
        == decision["status"]
    )
    assert report["next_wing_proof_plan"]["readonly_preflight_steps"] == []
    assert (
        report["next_wing_proof_plan"]["summary"]["readonly_preflight_evidence_status"]
        == preflight["evidence_status"]
    )
    assert (
        report["next_wing_proof_plan"]["summary"]["readonly_preflight_blocked_by"]
        == preflight["blocked_by"]
    )
    assert (
        "accepted_readonly_preflight_review_decision_required_before_supervised_proof"
        in report["next_wing_proof_plan"]["summary"]["global_blockers"]
    )
    assert (
        "operator_must_record_readonly_preflight_review_decision"
        in report["next_wing_proof_plan"]["summary"]["global_blockers"]
    )


def test_timeout_artifacts_remain_blocked_through_report_and_promotion_evidence(
    tmp_path: Path,
    monkeypatch,
):
    patch_report_tool_wing_preflight_candidate(monkeypatch)
    write_summary(tmp_path / "timeout_after_apply", timeout_after_apply_summary())
    write_summary(tmp_path / "timeout_before_apply", timeout_before_apply_summary())
    output_path = tmp_path / "report.json"

    exit_code = report_tool.main(
        [
            "--artifact-root",
            str(tmp_path),
            "--require-physical",
            "--min-promotion-candidates",
            "1",
            "--output",
            str(output_path),
            "--quiet",
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["summary"]["status_counts"]["blocked"] == 2
    assert report["summary"]["promotion_allowed_proof_ids"] == []
    assert report["summary"]["blocked_proof_ids"] == [
        "timeout-after-apply-proof",
        "timeout-before-apply-proof",
    ]
    rows_by_id = {row["proof_id"]: row for row in report["rows"]}
    assert "transport_failure_after_apply_requires_manual_review" in rows_by_id[
        "timeout-after-apply-proof"
    ]["blockers"]
    assert "transport_failure_before_apply_fail_closed" in rows_by_id[
        "timeout-before-apply-proof"
    ]["blockers"]

    promotion = report["promotion_evidence"]
    assert promotion["summary"]["safety_row_review_candidate_ids"] == []
    fader_row = next(row for row in promotion["rows"] if row["value_type"] == "fader")
    assert fader_row["status"] == "evidence_blocked"
    assert fader_row["physical_candidate_count"] == 0
    assert fader_row["blocked_proof_ids"] == [
        "timeout-after-apply-proof",
        "timeout-before-apply-proof",
    ]
    assert "blocked_artifacts_for_value_type" in fader_row["blockers"]
    assert "physical_candidate_missing" in fader_row["blockers"]
    assert report["next_wing_proof_plan"]["status"] == "manual_review_ready"
    assert report["next_wing_proof_plan"]["summary"]["step_count"] == 0
    assert_prestart_backend_port_check(report["next_wing_proof_plan"]["summary"])
    assert (
        "accepted_readonly_preflight_review_decision_required_before_supervised_proof"
        in report["next_wing_proof_plan"]["summary"]["global_blockers"]
    )
    assert (
        "operator_must_record_readonly_preflight_review_decision"
        in report["next_wing_proof_plan"]["summary"]["global_blockers"]
    )
