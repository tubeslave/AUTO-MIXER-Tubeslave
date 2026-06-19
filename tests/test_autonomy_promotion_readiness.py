from pathlib import Path

import pytest

from backend.autonomy_promotion_readiness import (
    build_autonomy_promotion_readiness_snapshot,
    iter_promotion_readiness_rows,
)
from backend.evaluation_integration_bundle_contract import REQUIRED_INTEGRATION_CHECKS
from backend.fault_injection_recovery_contract import (
    REQUIRED_AUDIT_FIELDS as RECOVERY_REQUIRED_AUDIT_FIELDS,
    REQUIRED_EVALUATION_TYPES as RECOVERY_REQUIRED_EVALUATION_TYPES,
    REQUIRED_FAULT_SCENARIOS,
    REQUIRED_RECOVERY_OUTCOMES,
)
from backend.integration_policy_contract import (
    REQUIRED_INTEGRATION_CONTROLS as POLICY_REQUIRED_INTEGRATION_CONTROLS,
    REQUIRED_INTEGRATION_EVIDENCE as POLICY_REQUIRED_INTEGRATION_EVIDENCE,
)
from backend.no_live_transport_contract import (
    REQUIRED_TRANSPORT_CONTROLS,
    REQUIRED_TRANSPORT_EVIDENCE,
    REQUIRED_ZERO_RUNTIME_OBSERVATIONS,
)
from backend.offline_replay_evaluation_contract import (
    REQUIRED_EVALUATION_TYPES,
    REQUIRED_QUALITY_AXES,
    REQUIRED_SAFETY_CHECKS,
    REQUIRED_SCENARIOS,
)
from backend.operator_report_contract import (
    REQUIRED_AUDIT_FIELDS as OPERATOR_REPORT_REQUIRED_AUDIT_FIELDS,
    REQUIRED_AUTHORITY_DENIAL_FLAGS as OPERATOR_REPORT_REQUIRED_AUTHORITY_DENIAL_FLAGS,
    REQUIRED_REPORT_BINDINGS,
    REQUIRED_REPORT_SECTIONS,
)
from backend.quality_threshold_policy_contract import (
    REQUIRED_POLICY_CHECKS as THRESHOLD_REQUIRED_POLICY_CHECKS,
    REQUIRED_THRESHOLD_AXES,
)
from backend.recovery_policy_contract import (
    REQUIRED_RECOVERY_CONTROLS,
    REQUIRED_RECOVERY_EVIDENCE,
)
from backend.safety_kernel_trace_contract import (
    REQUIRED_AUDIT_FIELDS as SAFETY_KERNEL_REQUIRED_AUDIT_FIELDS,
    REQUIRED_GUARD_CHECKS as SAFETY_KERNEL_REQUIRED_GUARD_CHECKS,
    REQUIRED_TRACE_SCENARIOS as SAFETY_KERNEL_REQUIRED_TRACE_SCENARIOS,
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


def _valid_wing_preflight_summary() -> dict:
    return {
        "run_id": "20260606T150000Z_wing_readonly_preflight",
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


def _artifact_bound_preflight_candidate_row() -> dict:
    summary = _valid_wing_preflight_summary()
    return {
        **summary,
        "source_id": summary["summary_path"],
        "accepted_as_readonly_preflight_candidate": True,
        "expected_artifacts": [
            summary["summary_path"],
            summary["events_path"],
        ],
        "expected_summary_artifact": summary["summary_path"],
        "expected_events_artifact": summary["events_path"],
        "summary_path_matches_expected_artifact": True,
        "events_path_matches_expected_artifact": True,
    }


def _readiness_evidence_snapshot_with_candidate(candidate: dict) -> dict:
    return {
        "status": "readonly_preflight_candidate_available",
        "summary": {
            "artifact_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "readiness_matrix_update_allowed": True,
            "readiness_closure_allowed_by_evidence": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "latest_valid_candidate_run_id": candidate["run_id"],
            "blocked_by": [],
            "violations": [],
        },
        "artifacts": [candidate],
        "mvp_blocker_matrix": [
            {"id": "wing_readonly_connection_proof", "status": "candidate_available"},
            {"id": "physical_console_readback_proof", "status": "candidate_available"},
            {"id": "operator_visible_evidence_report", "status": "candidate_available"},
            {
                "id": "readiness_matrix_update_based_on_real_proof",
                "status": "candidate_available",
            },
        ],
    }


def _replay_scenario(name: str) -> dict:
    return {
        "scenario": name,
        "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
        "safety_checks": {key: True for key in REQUIRED_SAFETY_CHECKS},
        "quality_axes": {
            key: {"status": "passed", "artifact": f"{name}/{key}.json"}
            for key in REQUIRED_QUALITY_AXES
        },
    }


def _threshold_axis(axis: str) -> dict:
    return {
        "axis": axis,
        "metric_name": f"{axis}_score",
        "direction": "min",
        "candidate_value": 0.8,
        "evidence_artifact": f"thresholds/{axis}.json",
        "rationale": f"Replay promotion threshold for {axis}",
        "rollback_policy": f"Revert {axis} threshold on regression",
    }


def _quality_threshold_policy_manifest(**overrides) -> dict:
    manifest = {
        "policy_id": "autonomy-readiness-threshold-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
        "threshold_policy": {
            "status": "approved",
            "approved_by": "owner",
            "approved_decision_id": "adr:offline-replay-thresholds",
            "approved_at": "2026-06-19T17:32:45Z",
        },
        "thresholds": [_threshold_axis(axis) for axis in REQUIRED_THRESHOLD_AXES],
    }
    manifest.update(overrides)
    return manifest


def _offline_replay_manifest(**overrides) -> dict:
    manifest = {
        "manifest_id": "autonomy-readiness-offline-replay",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "quality_threshold_policy_manifest": _quality_threshold_policy_manifest(),
        "scenarios": [_replay_scenario(name) for name in REQUIRED_SCENARIOS],
    }
    manifest.update(overrides)
    return manifest


def _fault_case(name: str) -> dict:
    return {
        "fault_scenario": name,
        "evaluation_types": list(RECOVERY_REQUIRED_EVALUATION_TYPES),
        "recovery_outcomes": {key: True for key in REQUIRED_RECOVERY_OUTCOMES},
        "audit": {key: f"{name}/{key}" for key in RECOVERY_REQUIRED_AUDIT_FIELDS},
    }


def _recovery_policy_manifest(**overrides) -> dict:
    manifest = {
        "policy_id": "autonomy-readiness-recovery-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "recovery_controls": {key: True for key in REQUIRED_RECOVERY_CONTROLS},
        "recovery_evidence": {key: True for key in REQUIRED_RECOVERY_EVIDENCE},
        "recovery_policy": {
            "status": "approved",
            "approved_by": "owner",
            "approved_decision_id": "adr:fault-recovery-policy",
            "approved_at": "2026-06-19T18:00:10Z",
        },
    }
    manifest.update(overrides)
    return manifest


def _fault_recovery_manifest(**overrides) -> dict:
    manifest = {
        "manifest_id": "autonomy-readiness-fault-recovery",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "recovery_policy_manifest": _recovery_policy_manifest(),
        "fault_scenarios": [_fault_case(name) for name in REQUIRED_FAULT_SCENARIOS],
    }
    manifest.update(overrides)
    return manifest


def _integration_policy_manifest(**overrides) -> dict:
    manifest = {
        "policy_id": "autonomy-readiness-integration-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "integration_controls": {
            key: True for key in POLICY_REQUIRED_INTEGRATION_CONTROLS
        },
        "integration_evidence": {
            key: True for key in POLICY_REQUIRED_INTEGRATION_EVIDENCE
        },
        "integration_policy": {
            "status": "approved",
            "approved_by": "owner",
            "approved_decision_id": "adr:integration-policy",
            "approved_at": "2026-06-19T18:10:23Z",
        },
    }
    manifest.update(overrides)
    return manifest


def _safety_kernel_trace_record(scenario: str, **overrides) -> dict:
    decision = "bounded_allow" if scenario == "supervised_fader_gain_allowed" else "reject"
    record = {
        "scenario": scenario,
        "safety_decision": decision,
        "guard_checks": {key: True for key in SAFETY_KERNEL_REQUIRED_GUARD_CHECKS},
        "audit": {
            key: f"{scenario}/{key}" for key in SAFETY_KERNEL_REQUIRED_AUDIT_FIELDS
        }
        | {"safety_decision": decision},
    }
    record.update(overrides)
    return record


def _safety_kernel_trace_manifest(**overrides) -> dict:
    manifest = {
        "trace_id": "autonomy-readiness-safety-kernel-trace",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "trace_records": [
            _safety_kernel_trace_record(scenario)
            for scenario in SAFETY_KERNEL_REQUIRED_TRACE_SCENARIOS
        ],
    }
    manifest.update(overrides)
    return manifest


def _transport_evidence_item(name: str) -> dict:
    return {
        "status": "passed",
        "artifact": f"no-live-transport/{name}.json",
        "sha256": f"{name}-sha256",
    }


def _no_live_transport_manifest(**overrides) -> dict:
    manifest = {
        "transport_id": "autonomy-readiness-no-live-transport",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "transport_controls": {key: True for key in REQUIRED_TRANSPORT_CONTROLS},
        "transport_evidence": {
            key: _transport_evidence_item(key) for key in REQUIRED_TRANSPORT_EVIDENCE
        },
        "runtime_observations": {
            key: 0 for key in REQUIRED_ZERO_RUNTIME_OBSERVATIONS
        },
        "forbidden_transport_tokens": [],
    }
    manifest.update(overrides)
    return manifest


def _operator_report_binding(name: str) -> dict:
    return {
        "status": "passed",
        "artifact": f"operator-report/{name}.json",
        "sha256": f"{name}-sha256",
    }


def _operator_report_manifest(**overrides) -> dict:
    manifest = {
        "report_id": "autonomy-readiness-operator-report",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "report_sections": {key: True for key in REQUIRED_REPORT_SECTIONS},
        "report_bindings": {
            key: _operator_report_binding(key) for key in REQUIRED_REPORT_BINDINGS
        },
        "authority_denial": {
            key: False for key in OPERATOR_REPORT_REQUIRED_AUTHORITY_DENIAL_FLAGS
        },
        "visible_blockers": [
            "M001 waits for real WING read-only artifact",
            "operator review required",
        ],
        "next_operator_action": "Run WING read-only preflight with operator.",
        "audit": {
            key: f"operator-report/{key}"
            for key in OPERATOR_REPORT_REQUIRED_AUDIT_FIELDS
        },
    }
    manifest.update(overrides)
    return manifest


def _complete_integration_bundle(**overrides) -> dict:
    manifest = {
        "manifest_id": "autonomy-readiness-integration-bundle",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "integration_policy_manifest": _integration_policy_manifest(),
        "safety_kernel_trace_manifest": _safety_kernel_trace_manifest(),
        "no_live_transport_manifest": _no_live_transport_manifest(),
        "operator_report_manifest": _operator_report_manifest(),
        "integration_checks": {key: True for key in REQUIRED_INTEGRATION_CHECKS},
        "offline_replay_manifest": _offline_replay_manifest(),
        "fault_recovery_manifest": _fault_recovery_manifest(),
    }
    manifest.update(overrides)
    return manifest


def test_autonomy_promotion_readiness_module_is_read_only_and_transport_free():
    source = Path("backend/autonomy_promotion_readiness.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
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


def test_promotion_readiness_snapshot_is_fail_closed_for_full_autonomous_mode():
    snapshot = build_autonomy_promotion_readiness_snapshot()

    assert snapshot["type"] == "autonomy_promotion_readiness"
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
        "supervised_only": True,
    }
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert "mode_full_autonomous_blocked" in snapshot["summary"]["global_blockers"]
    assert snapshot["summary"]["target_count"] == len(snapshot["rows"])
    preflight = snapshot["summary"]["wing_readonly_preflight_evidence"]
    assert preflight["read_only"] is True
    assert preflight["live_write"] is False
    assert preflight["live_widening_allowed"] is False
    assert preflight["automatic_safety_update_allowed"] is False
    assert preflight["readiness_closure_allowed_by_evidence"] is False
    assert preflight["live_authority_granted"] is False
    assert preflight["full_autonomous_ready"] is False
    assert preflight["source_authority_denied"] is True
    assert preflight["source_authority_flags"] == {
        "readiness_closure_allowed_by_evidence": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }
    assert preflight["evidence_status"] == "operator_preflight_required"
    assert preflight["readiness_update_candidate_available"] is False
    assert preflight["prestart_backend_port_check_required"] is True
    assert preflight["prestart_backend_port_check_must_pass_before_starting_backend"] is True
    assert preflight["prestart_backend_port_check_command"] == [
        "PYTHONPATH=backend",
        "python3",
        "tools/no_wing_release_check.py",
        "--quick",
    ]
    assert (
        preflight["prestart_backend_port_check_command_text"]
        == WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
    )
    assert preflight["prestart_backend_port_check_failure_detail_fields"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert preflight["prestart_backend_port_summary_required"] is False
    assert preflight["prestart_backend_port_summary_status"] is None
    assert preflight["prestart_backend_port_summary_live_write"] is None
    assert preflight["prestart_backend_port_summary_contacts_backend"] is None
    assert preflight["prestart_backend_port_summary_queries_console"] is None
    assert preflight["prestart_backend_port_summary_executes_tests"] is None
    assert preflight["prestart_backend_port_check_status"] is None
    assert preflight["candidate_blocker_ids"] == []
    assert preflight["closed_blocker_ids"] == []
    assert "operator_must_run_wing_readonly_preflight_runner" in preflight["blocked_by"]


def test_promotion_readiness_requires_offline_integration_bundle_contract():
    snapshot = build_autonomy_promotion_readiness_snapshot()
    integration = snapshot["summary"]["offline_integration_bundle"]

    assert "offline_integration_bundle_contract_not_ready" in (
        snapshot["summary"]["global_blockers"]
    )
    assert integration["read_only"] is True
    assert integration["offline_only"] is False
    assert integration["live_write"] is False
    assert integration["hardware_evidence"] is False
    assert integration["readiness_closure_allowed"] is False
    assert integration["live_authority_granted"] is False
    assert integration["source_authority_denied"] is True
    assert integration["ready_for_integration_promotion"] is False
    assert integration["status"] == "blocked"
    assert "offline_only_required" in integration["blockers"]
    assert "operator_report_contract_not_ready" in integration["blockers"]
    assert "offline_replay" in integration["child_contract_statuses"]
    assert integration["child_contract_statuses"]["operator_report"] == "blocked"
    assert "live authority or readiness closure" in integration["claim"]
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert snapshot["safety"]["full_autonomous_ready"] is False


def test_promotion_readiness_accepts_ready_offline_bundle_without_authority():
    snapshot = build_autonomy_promotion_readiness_snapshot(
        offline_integration_bundle_manifest=_complete_integration_bundle()
    )
    integration = snapshot["summary"]["offline_integration_bundle"]

    assert "offline_integration_bundle_contract_not_ready" not in (
        snapshot["summary"]["global_blockers"]
    )
    assert integration["read_only"] is True
    assert integration["offline_only"] is True
    assert integration["live_write"] is False
    assert integration["hardware_evidence"] is False
    assert integration["readiness_closure_allowed"] is False
    assert integration["live_authority_granted"] is False
    assert integration["contacts_wing"] is False
    assert integration["contacts_backend"] is False
    assert integration["source_authority_denied"] is True
    assert integration["ready_for_integration_promotion"] is True
    assert integration["status"] == "ready_for_integration_evaluation_promotion"
    assert integration["blockers"] == []
    assert integration["missing_integration_checks"] == []
    assert integration["child_contract_statuses"]["operator_report"] == (
        "ready_for_operator_report_use"
    )
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
        "supervised_only": True,
    }


def test_promotion_readiness_rejects_live_integration_bundle_claims():
    snapshot = build_autonomy_promotion_readiness_snapshot(
        offline_integration_bundle_manifest=_complete_integration_bundle(
            live_write=True,
            contacts_wing=True,
            contacts_backend=True,
            hardware_evidence=True,
            readiness_closure_allowed=True,
            live_authority_granted=True,
        )
    )
    integration = snapshot["summary"]["offline_integration_bundle"]

    assert "offline_integration_bundle_contract_not_ready" in (
        snapshot["summary"]["global_blockers"]
    )
    assert integration["read_only"] is True
    assert integration["live_write"] is False
    assert integration["hardware_evidence"] is False
    assert integration["readiness_closure_allowed"] is False
    assert integration["live_authority_granted"] is False
    assert integration["source_live_write_claimed"] is True
    assert integration["source_hardware_evidence_claimed"] is True
    assert integration["contacts_wing"] is True
    assert integration["contacts_backend"] is True
    assert integration["source_authority_denied"] is False
    assert integration["ready_for_integration_promotion"] is False
    assert "integration_bundle_forbids_live_write" in integration["blockers"]
    assert "integration_bundle_evidence_cannot_grant_live_authority" in (
        integration["blockers"]
    )
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert snapshot["safety"]["full_autonomous_ready"] is False


def test_fader_is_supervised_only_but_not_autonomous_ready():
    rows = {row.id: row for row in iter_promotion_readiness_rows()}
    row = rows["fader_supervised_promotion"]

    assert row.status == "supervised_live_allowed"
    assert row.supervised_live_allowed is True
    assert row.autonomous_ready is False
    assert row.safety_row_id == "fader_supervised"
    assert row.safety_status == "hardware_proven_narrow"
    assert "mode_full_autonomous_blocked" in row.blockers
    assert "not_autonomous_ready" in row.blockers
    assert "real_wing_proof_partial" in row.blockers
    assert "emergency_stop_proof_partial" in row.blockers


def test_gain_declared_executor_surface_is_still_safety_proof_blocked():
    rows = {row.id: row for row in iter_promotion_readiness_rows()}
    row = rows["gain_trim_supervised_promotion"]

    assert row.supervised_live_allowed is False
    assert row.autonomous_ready is False
    assert row.safety_row_id == "gain_supervised"
    assert row.safety_status == "dry_run_ready"
    assert "safety_live_not_allowed" in row.blockers
    assert "real_wing_proof_missing" in row.blockers
    assert "capability_declares_live_kind_without_safety_proof" in row.blockers
    assert row.evidence_requirement_ids == (
        "physical_pre_read",
        "bounded_gain_down_apply",
        "apply_readback",
        "rollback_readback",
        "emergency_stop_block",
        "unsafe_gain_up_followup_blocked",
        "gain_up_operator_authored_opt_in",
    )


def test_broad_console_surfaces_are_blocked_by_missing_proof_or_manual_only_policy():
    rows = {row.id: row for row in iter_promotion_readiness_rows()}

    eq = rows["eq_supervised_promotion"]
    routing = rows["routing_supervised_promotion"]
    scene_preset = rows["scene_preset_supervised_promotion"]
    bus = rows["bus_group_matrix_supervised_promotion"]
    mute = rows["mute_supervised_promotion"]

    assert eq.supervised_live_allowed is False
    assert "safety_live_not_allowed" in eq.blockers
    assert "fake_mixer_proof_missing" in eq.blockers
    assert "read_surface_channel_eq_dynamics_gate_partial" in eq.blockers

    assert routing.status == "manual_only_blocked"
    assert routing.action_family == "routing"
    assert routing.capability_ids == ("console_routing_control",)
    assert routing.read_surface_ids == ("routing_and_sends",)
    assert routing.evidence_requirement_ids == (
        "current_route_cache_freshness",
        "route_diff_evidence",
        "source_identity_downstream_trust",
        "routing_physical_fixture",
        "hardware_readback_and_rollback_proof",
    )
    assert "safety_status_blocked" in routing.blockers
    assert "write_surface_routing_snapshot_scene_writes_quarantined" in routing.blockers

    assert scene_preset.status == "manual_only_blocked"
    assert scene_preset.supervised_live_allowed is False
    assert scene_preset.autonomous_ready is False
    assert scene_preset.action_family == "scene_preset"
    assert scene_preset.capability_ids == ("console_scene_preset_control",)
    assert scene_preset.read_surface_ids == ("snapshots_scenes", "user_presets")
    assert scene_preset.evidence_requirement_ids == (
        "current_scene_preset_cache_freshness",
        "snapshots_scenes_snapshot_cache_freshness",
        "user_presets_snapshot_cache_freshness",
        "read_only_scoped_diff_evidence",
        "read_only_rollback_evidence",
        "scene_preset_physical_fixture",
        "hardware_readback_and_rollback_proof",
    )
    assert "safety_status_blocked" in scene_preset.blockers
    assert "write_surface_routing_snapshot_scene_writes_quarantined" in scene_preset.blockers
    assert "read_surface_snapshots_scenes_partial" in scene_preset.blockers
    assert "read_surface_user_presets_partial" in scene_preset.blockers

    assert bus.status == "manual_only_blocked"
    assert bus.supervised_live_allowed is False
    assert bus.safety_row_id == "bus_group_matrix"
    assert bus.safety_status == "quarantined"
    assert bus.evidence_requirement_ids == (
        "bus_group_assignment_cache_freshness",
        "bus_group_fader_readback",
        "matrix_send_readback",
        "main_bus_true_peak_headroom",
        "bus_group_physical_fixture",
        "hardware_readback_and_rollback_proof",
    )
    assert "safety_coverage_row_missing" not in bus.blockers
    assert "safety_status_quarantined" in bus.blockers
    assert "real_wing_proof_missing" in bus.blockers
    assert "rollback_readback_missing" in bus.blockers
    assert "emergency_stop_proof_missing" in bus.blockers
    assert "write_surface_mixer_transport_clients_quarantined" in bus.blockers

    assert mute.status == "manual_only_blocked"
    assert mute.supervised_live_allowed is False
    assert mute.autonomous_ready is False
    assert mute.safety_row_id == "mute_supervised"
    assert mute.safety_status == "quarantined"
    assert mute.evidence_requirement_ids == (
        "current_mute_cache_freshness",
        "mute_owner_intent_evidence",
        "signal_return_or_idle_window_evidence",
        "manual_mute_authority_review",
        "mute_physical_fixture",
        "hardware_readback_rollback_and_emergency_proof",
    )
    assert "safety_coverage_row_missing" not in mute.blockers
    assert "safety_status_quarantined" in mute.blockers
    assert "read_surface_channel_identity_and_basic_controls_partial" in mute.blockers
    assert "write_surface_voice_ai_direct_commands_quarantined" in mute.blockers


def test_no_readiness_row_widens_live_write_authority():
    for row in iter_promotion_readiness_rows():
        assert row.autonomous_ready is False
        if row.supervised_live_allowed:
            assert row.action_family == "fader"
        else:
            assert row.status in {"proof_blocked", "read_coverage_blocked", "manual_only_blocked"}


def test_readiness_snapshot_serializes_evidence_requirement_ids():
    snapshot = build_autonomy_promotion_readiness_snapshot()
    rows = {row["id"]: row for row in snapshot["rows"]}
    bus = rows["bus_group_matrix_supervised_promotion"]
    mute = rows["mute_supervised_promotion"]
    scene_preset = rows["scene_preset_supervised_promotion"]

    assert bus["supervised_live_allowed"] is False
    assert bus["autonomous_ready"] is False
    assert bus["evidence_requirement_ids"] == [
        "bus_group_assignment_cache_freshness",
        "bus_group_fader_readback",
        "matrix_send_readback",
        "main_bus_true_peak_headroom",
        "bus_group_physical_fixture",
        "hardware_readback_and_rollback_proof",
    ]
    assert mute["supervised_live_allowed"] is False
    assert mute["autonomous_ready"] is False
    assert mute["evidence_requirement_ids"] == [
        "current_mute_cache_freshness",
        "mute_owner_intent_evidence",
        "signal_return_or_idle_window_evidence",
        "manual_mute_authority_review",
        "mute_physical_fixture",
        "hardware_readback_rollback_and_emergency_proof",
    ]
    assert scene_preset["supervised_live_allowed"] is False
    assert scene_preset["autonomous_ready"] is False
    assert scene_preset["evidence_requirement_ids"] == [
        "current_scene_preset_cache_freshness",
        "snapshots_scenes_snapshot_cache_freshness",
        "user_presets_snapshot_cache_freshness",
        "read_only_scoped_diff_evidence",
        "read_only_rollback_evidence",
        "scene_preset_physical_fixture",
        "hardware_readback_and_rollback_proof",
    ]


def test_readiness_snapshot_groups_autonomy_gaps_without_granting_authority():
    snapshot = build_autonomy_promotion_readiness_snapshot()
    gap_summary = snapshot["summary"]["promotion_gap_summary"]

    assert gap_summary["read_only"] is True
    assert gap_summary["live_write"] is False
    assert gap_summary["full_autonomous_ready"] is False
    assert gap_summary["gap_bucket_counts"]["autonomy_mode"] >= snapshot["summary"]["target_count"]
    assert gap_summary["gap_bucket_counts"]["proof"] > 0
    assert gap_summary["gap_bucket_counts"]["read_surface"] > 0
    assert gap_summary["gap_bucket_counts"]["write_surface"] > 0
    assert gap_summary["gap_bucket_counts"]["evidence_requirement"] >= 24

    blocked_by_bucket = gap_summary["blocked_target_ids_by_bucket"]
    assert "fader_supervised_promotion" in blocked_by_bucket["autonomy_mode"]
    assert "scene_preset_supervised_promotion" in blocked_by_bucket["read_surface"]
    assert "scene_preset_supervised_promotion" in blocked_by_bucket["write_surface"]
    assert "bus_group_matrix_supervised_promotion" in blocked_by_bucket["evidence_requirement"]
    assert "mute_supervised_promotion" in blocked_by_bucket["evidence_requirement"]
    assert "routing_supervised_promotion" in blocked_by_bucket["safety"]

    evidence_by_family = gap_summary["evidence_requirement_ids_by_action_family"]
    assert evidence_by_family["bus_group_matrix"] == [
        "bus_group_assignment_cache_freshness",
        "bus_group_fader_readback",
        "matrix_send_readback",
        "main_bus_true_peak_headroom",
        "bus_group_physical_fixture",
        "hardware_readback_and_rollback_proof",
    ]
    assert evidence_by_family["mute"] == [
        "current_mute_cache_freshness",
        "mute_owner_intent_evidence",
        "signal_return_or_idle_window_evidence",
        "manual_mute_authority_review",
        "mute_physical_fixture",
        "hardware_readback_rollback_and_emergency_proof",
    ]
    assert evidence_by_family["routing"] == [
        "current_route_cache_freshness",
        "route_diff_evidence",
        "source_identity_downstream_trust",
        "routing_physical_fixture",
        "hardware_readback_and_rollback_proof",
    ]
    assert "read_only_scoped_diff_evidence" in evidence_by_family["scene_preset"]

    next_by_family = gap_summary["next_required_evidence_by_action_family"]
    evidence_by_family = gap_summary["evidence_requirement_ids_by_action_family"]
    assert evidence_by_family["gain"] == [
        "physical_pre_read",
        "bounded_gain_down_apply",
        "apply_readback",
        "rollback_readback",
        "emergency_stop_block",
        "unsafe_gain_up_followup_blocked",
        "gain_up_operator_authored_opt_in",
    ]
    assert any("Keep scene apply blocked" in item for item in next_by_family["scene_preset"])


def test_readiness_snapshot_requires_replacement_for_unbound_wing_preflight_artifact():
    evidence = build_wing_readonly_preflight_evidence(
        artifact_payloads=(_valid_wing_preflight_summary(),),
        artifact_roots=(),
    )

    snapshot = build_autonomy_promotion_readiness_snapshot(
        wing_readonly_preflight_evidence_snapshot=evidence,
    )
    preflight = snapshot["summary"]["wing_readonly_preflight_evidence"]

    assert preflight["evidence_status"] == "readonly_preflight_artifact_replacement_required"
    assert preflight["artifact_count"] == 1
    assert preflight["valid_candidate_count"] == 0
    assert preflight["invalid_candidate_count"] == 1
    assert preflight["readiness_update_candidate_available"] is False
    assert preflight["prestart_backend_port_check_required"] is True
    assert preflight["prestart_backend_port_check_must_pass_before_starting_backend"] is True
    assert preflight["prestart_backend_port_check_command"] == [
        "PYTHONPATH=backend",
        "python3",
        "tools/no_wing_release_check.py",
        "--quick",
    ]
    assert (
        preflight["prestart_backend_port_check_command_text"]
        == WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND_TEXT
    )
    assert preflight["prestart_backend_port_check_failure_detail_fields"] == list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    assert preflight["prestart_backend_port_summary_required"] is False
    assert preflight["prestart_backend_port_summary_type"] is None
    assert preflight["prestart_backend_port_summary_type_matches_contract"] is False
    assert preflight["prestart_backend_port_summary_status"] is None
    assert preflight["prestart_backend_port_summary_connection_policy"] is None
    assert (
        preflight["prestart_backend_port_summary_connection_policy_matches_contract"]
        is False
    )
    assert preflight["prestart_backend_port_summary_contract_valid"] is False
    assert preflight["prestart_backend_port_summary_contract_violations"] == [
        "no_valid_readonly_preflight_candidate",
    ]
    assert preflight["violations"] == ["artifact_root_missing"]
    assert preflight["prestart_backend_port_summary_live_write"] is None
    assert preflight["prestart_backend_port_summary_contacts_backend"] is None
    assert preflight["prestart_backend_port_summary_queries_console"] is None
    assert preflight["prestart_backend_port_summary_executes_tests"] is None
    assert preflight["prestart_backend_port_check_status"] is None
    assert preflight["candidate_blocker_ids"] == []
    assert preflight["closed_blocker_ids"] == []
    assert preflight["latest_valid_candidate_run_id"] is None
    assert preflight["readiness_closure_allowed_by_evidence"] is False
    assert preflight["live_authority_granted"] is False
    assert preflight["full_autonomous_ready"] is False
    assert "manual readiness review only" in preflight["claim"]
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert "fader_supervised_promotion" in snapshot["summary"]["blocked_target_ids"]
    rows = {row["id"]: row for row in snapshot["rows"]}
    assert rows["fader_supervised_promotion"]["autonomous_ready"] is False
    assert rows["gain_trim_supervised_promotion"]["supervised_live_allowed"] is False


def test_readiness_snapshot_accepts_artifact_bound_prestart_summary_contract():
    snapshot = build_autonomy_promotion_readiness_snapshot(
        wing_readonly_preflight_evidence_snapshot=(
            _readiness_evidence_snapshot_with_candidate(
                _artifact_bound_preflight_candidate_row()
            )
        ),
    )
    preflight = snapshot["summary"]["wing_readonly_preflight_evidence"]

    assert preflight["readiness_update_candidate_available"] is True
    assert preflight["prestart_backend_port_summary_contract_valid"] is True
    assert preflight["prestart_backend_port_summary_contract_violations"] == []
    assert preflight["readiness_closure_allowed_by_evidence"] is False
    assert preflight["live_authority_granted"] is False
    assert preflight["full_autonomous_ready"] is False
    assert preflight["source_authority_denied"] is True
    assert preflight["source_authority_flags"] == {
        "readiness_closure_allowed_by_evidence": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("readiness_closure_allowed_by_evidence", None),
        ("live_authority_granted", "false"),
        ("full_autonomous_ready", 0),
    ],
)
def test_readiness_snapshot_blocks_candidate_when_source_authority_flag_is_not_false(
    field,
    value,
):
    evidence = _readiness_evidence_snapshot_with_candidate(
        _artifact_bound_preflight_candidate_row()
    )
    evidence["summary"][field] = value

    snapshot = build_autonomy_promotion_readiness_snapshot(
        wing_readonly_preflight_evidence_snapshot=evidence,
    )
    preflight = snapshot["summary"]["wing_readonly_preflight_evidence"]

    assert preflight["readiness_update_candidate_available"] is False
    assert preflight["source_authority_denied"] is False
    assert preflight["source_authority_flags"][field] == value
    assert preflight["readiness_closure_allowed_by_evidence"] is False
    assert preflight["live_authority_granted"] is False
    assert preflight["full_autonomous_ready"] is False


def test_readiness_snapshot_fails_closed_without_verifier_websocket_proof():
    candidate = _artifact_bound_preflight_candidate_row()
    candidate["artifact_manifest_verifier_websocket_opened_after_pass"] = False
    snapshot = build_autonomy_promotion_readiness_snapshot(
        wing_readonly_preflight_evidence_snapshot=(
            _readiness_evidence_snapshot_with_candidate(candidate)
        ),
    )
    preflight = snapshot["summary"]["wing_readonly_preflight_evidence"]

    assert preflight["readiness_update_candidate_available"] is False
    assert preflight["prestart_backend_port_summary_contract_valid"] is False
    assert preflight["prestart_backend_port_summary_contract_violations"] == [
        "artifact_manifest_verifier_websocket_opened_after_pass_not_true"
    ]
    assert preflight["readiness_closure_allowed_by_evidence"] is False
    assert preflight["live_authority_granted"] is False
    assert preflight["full_autonomous_ready"] is False


def test_readiness_snapshot_fails_closed_without_valid_prestart_summary_contract():
    evidence = build_wing_readonly_preflight_evidence(
        artifact_payloads=(),
        artifact_roots=(),
    )

    snapshot = build_autonomy_promotion_readiness_snapshot(
        wing_readonly_preflight_evidence_snapshot=evidence,
    )
    preflight = snapshot["summary"]["wing_readonly_preflight_evidence"]

    assert preflight["readiness_update_candidate_available"] is False
    assert preflight["prestart_backend_port_summary_contract_valid"] is False
    assert preflight["prestart_backend_port_summary_contract_violations"] == [
        "no_valid_readonly_preflight_candidate"
    ]
    assert preflight["readiness_closure_allowed_by_evidence"] is False
    assert preflight["live_authority_granted"] is False
    assert preflight["full_autonomous_ready"] is False
