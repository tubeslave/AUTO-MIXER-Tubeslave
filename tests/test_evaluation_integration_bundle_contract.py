from pathlib import Path

from backend.evaluation_integration_bundle_contract import (
    REQUIRED_INTEGRATION_CHECKS,
    evaluate_integration_bundle_contract,
)
from backend.fault_injection_recovery_contract import (
    REQUIRED_AUDIT_FIELDS,
    REQUIRED_EVALUATION_TYPES as RECOVERY_REQUIRED_EVALUATION_TYPES,
    REQUIRED_FAULT_SCENARIOS,
    REQUIRED_RECOVERY_OUTCOMES,
)
from backend.integration_policy_contract import (
    REQUIRED_INTEGRATION_CONTROLS as POLICY_REQUIRED_INTEGRATION_CONTROLS,
    REQUIRED_INTEGRATION_EVIDENCE as POLICY_REQUIRED_INTEGRATION_EVIDENCE,
)
from backend.offline_replay_evaluation_contract import (
    REQUIRED_EVALUATION_TYPES,
    REQUIRED_QUALITY_AXES,
    REQUIRED_SAFETY_CHECKS,
    REQUIRED_SCENARIOS,
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


def _scenario(name: str) -> dict:
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
        "policy_id": "offline-replay-threshold-policy",
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
        "manifest_id": "offline-replay-fixture",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "quality_threshold_policy_manifest": _quality_threshold_policy_manifest(),
        "scenarios": [_scenario(name) for name in REQUIRED_SCENARIOS],
    }
    manifest.update(overrides)
    return manifest


def _fault_case(name: str) -> dict:
    return {
        "fault_scenario": name,
        "evaluation_types": list(RECOVERY_REQUIRED_EVALUATION_TYPES),
        "recovery_outcomes": {key: True for key in REQUIRED_RECOVERY_OUTCOMES},
        "audit": {key: f"{name}-{key}" for key in REQUIRED_AUDIT_FIELDS},
    }


def _recovery_policy_manifest(**overrides) -> dict:
    manifest = {
        "policy_id": "fault-recovery-policy",
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
        "manifest_id": "fault-recovery-fixture",
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
        "policy_id": "integration-policy-fixture",
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
            "approved_decision_id": "adr:integration-evaluation-policy",
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
            key: f"{scenario}-{key}" for key in SAFETY_KERNEL_REQUIRED_AUDIT_FIELDS
        }
        | {"safety_decision": decision},
    }
    record.update(overrides)
    return record


def _safety_kernel_trace_manifest(**overrides) -> dict:
    manifest = {
        "trace_id": "safety-kernel-trace-fixture",
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


def _complete_bundle(**overrides) -> dict:
    bundle = {
        "manifest_id": "integration-bundle-fixture",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "integration_policy_manifest": _integration_policy_manifest(),
        "safety_kernel_trace_manifest": _safety_kernel_trace_manifest(),
        "integration_checks": {key: True for key in REQUIRED_INTEGRATION_CHECKS},
        "offline_replay_manifest": _offline_replay_manifest(),
        "fault_recovery_manifest": _fault_recovery_manifest(),
    }
    bundle.update(overrides)
    return bundle


def test_integration_bundle_blocks_incomplete_child_contracts_and_pending_policy():
    result = evaluate_integration_bundle_contract(
        _complete_bundle(
            integration_policy_manifest=_integration_policy_manifest(
                integration_policy={"status": "PENDING_APPROVAL"}
            ),
            integration_checks={"analysis_trace_bound": True},
            offline_replay_manifest=_offline_replay_manifest(
                quality_threshold_policy_manifest=_quality_threshold_policy_manifest(
                    threshold_policy={"status": "PENDING_APPROVAL"}
                ),
                scenarios=[_scenario("soundcheck")],
            ),
            fault_recovery_manifest=_fault_recovery_manifest(
                fault_scenarios=[_fault_case("transport_timeout")],
            ),
        )
    )

    assert result["status"] == "blocked"
    assert result["ready_for_integration_evaluation_promotion"] is False
    assert result["authority"]["readiness_closure_allowed"] is False
    assert result["authority"]["live_authority_granted"] is False
    assert "approved_integration_policy_required" in result["blockers"]
    assert "integration_policy_contract_not_ready" in result["blockers"]
    assert (
        "integration_policy:owner_approved_integration_policy_required"
        in result["blockers"]
    )
    assert "offline_replay_contract_not_ready" in result["blockers"]
    assert "fault_recovery_contract_not_ready" in result["blockers"]
    assert "offline_replay:approved_quality_threshold_policy_required" in result["blockers"]
    assert "offline_replay:quality_threshold_policy_contract_not_ready" in result[
        "blockers"
    ]
    assert (
        "offline_replay:quality_threshold_policy:"
        "owner_approved_quality_threshold_policy_required"
        in result["blockers"]
    )
    assert "offline_replay:missing_required_scenario:studio" in result["blockers"]
    assert "fault_recovery:missing_required_fault_scenario:connection_loss" in result[
        "blockers"
    ]
    assert "missing_integration_check:fault_recovery_result_bound" in result["blockers"]


def test_integration_bundle_accepts_complete_offline_bundle_without_live_authority():
    result = evaluate_integration_bundle_contract(_complete_bundle())

    assert result["status"] == "ready_for_integration_evaluation_promotion"
    assert result["ready_for_integration_evaluation_promotion"] is True
    assert result["offline_only"] is True
    assert result["hardware_evidence"] is False
    assert result["live_write"] is False
    assert result["authority"] == {
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "contacts_wing": False,
        "contacts_backend": False,
    }
    assert result["summary"]["offline_replay_ready"] is True
    assert result["summary"]["fault_recovery_ready"] is True
    assert result["summary"]["safety_kernel_trace_ready"] is True
    assert result["summary"]["integration_policy_approved"] is True
    assert result["summary"]["integration_policy_contract_status"] == (
        "ready_for_integration_policy_use"
    )
    assert (
        result["child_results"]["integration_policy"][
            "ready_for_integration_policy_use"
        ]
        is True
    )
    assert (
        result["child_results"]["safety_kernel_trace"][
            "ready_for_safety_kernel_trace_use"
        ]
        is True
    )
    assert result["summary"]["missing_integration_checks"] == []
    assert result["blockers"] == []


def test_integration_bundle_requires_bound_integration_policy_manifest():
    result = evaluate_integration_bundle_contract(
        _complete_bundle(
            integration_policy_manifest={},
            integration_policy={
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:legacy-integration-policy",
            },
        )
    )

    assert result["status"] == "blocked"
    assert result["ready_for_integration_evaluation_promotion"] is False
    assert "approved_integration_policy_required" in result["blockers"]
    assert "integration_policy_contract_not_ready" in result["blockers"]
    assert "legacy_integration_policy_without_contract_manifest" in result[
        "blockers"
    ]
    assert "integration_policy:offline_only_required" in result["blockers"]
    assert result["summary"]["legacy_integration_policy_status"] == "approved"
    assert result["summary"]["integration_policy_status"] == ""
    assert result["summary"]["integration_policy_approved"] is False
    assert result["child_results"]["integration_policy"]["status"] == "blocked"
    assert result["authority"]["live_authority_granted"] is False


def test_integration_bundle_requires_bound_safety_kernel_trace_manifest():
    result = evaluate_integration_bundle_contract(
        _complete_bundle(safety_kernel_trace_manifest={})
    )

    assert result["status"] == "blocked"
    assert result["ready_for_integration_evaluation_promotion"] is False
    assert "safety_kernel_trace_contract_not_ready" in result["blockers"]
    assert (
        "legacy_safety_kernel_trace_bound_without_contract_manifest"
        in result["blockers"]
    )
    assert "safety_kernel_trace:offline_only_required" in result["blockers"]
    assert "safety_kernel_trace:missing_required_trace_scenario:stale_state_rejected" in (
        result["blockers"]
    )
    assert result["summary"]["safety_kernel_trace_ready"] is False
    assert result["summary"]["safety_kernel_trace_contract_status"] == "blocked"
    assert result["child_results"]["safety_kernel_trace"]["status"] == "blocked"
    assert result["authority"]["live_authority_granted"] is False


def test_integration_bundle_rejects_live_or_hardware_claims():
    result = evaluate_integration_bundle_contract(
        _complete_bundle(
            live_write=True,
            contacts_wing=True,
            contacts_backend=True,
            hardware_evidence=True,
            readiness_closure_allowed=True,
            live_authority_granted=True,
        )
    )

    assert result["status"] == "blocked"
    assert result["ready_for_integration_evaluation_promotion"] is False
    assert set(result["blockers"]) >= {
        "integration_bundle_forbids_live_write",
        "integration_bundle_forbids_wing_contact",
        "integration_bundle_forbids_backend_contact",
        "integration_bundle_evidence_must_not_be_labeled_hardware",
        "integration_bundle_evidence_cannot_close_readiness",
        "integration_bundle_evidence_cannot_grant_live_authority",
    }
    assert result["authority"]["live_authority_granted"] is False
    assert result["authority"]["readiness_closure_allowed"] is False


def test_integration_bundle_rejects_missing_integration_checks():
    checks = {key: True for key in REQUIRED_INTEGRATION_CHECKS}
    checks["safety_kernel_trace_bound"] = {"status": "pending"}
    checks["no_live_transport"] = False

    result = evaluate_integration_bundle_contract(
        _complete_bundle(integration_checks=checks)
    )

    assert result["status"] == "blocked"
    assert "missing_integration_check:safety_kernel_trace_bound" in result["blockers"]
    assert "missing_integration_check:no_live_transport" in result["blockers"]


def test_integration_bundle_source_has_no_live_transport_imports():
    source = Path("backend/evaluation_integration_bundle_contract.py").read_text()
    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "set_fader(",
        "set_gain(",
        "set_eq(",
        "set_compressor(",
        "route_output(",
        "load_snap(",
        "save_snap(",
    ]

    assert not any(token in source for token in forbidden)
