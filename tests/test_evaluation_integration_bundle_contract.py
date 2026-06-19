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
from backend.offline_replay_evaluation_contract import (
    REQUIRED_EVALUATION_TYPES,
    REQUIRED_QUALITY_AXES,
    REQUIRED_SAFETY_CHECKS,
    REQUIRED_SCENARIOS,
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
        "threshold_policy": {
            "status": "approved",
            "approved_by": "owner",
            "approved_decision_id": "adr:offline-replay-thresholds",
        },
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
        "recovery_policy": {
            "status": "approved",
            "approved_by": "owner",
            "approved_decision_id": "adr:fault-recovery-policy",
        },
        "fault_scenarios": [_fault_case(name) for name in REQUIRED_FAULT_SCENARIOS],
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
        "integration_policy": {
            "status": "approved",
            "approved_by": "owner",
            "approved_decision_id": "adr:integration-evaluation-policy",
        },
        "integration_checks": {key: True for key in REQUIRED_INTEGRATION_CHECKS},
        "offline_replay_manifest": _offline_replay_manifest(),
        "fault_recovery_manifest": _fault_recovery_manifest(),
    }
    bundle.update(overrides)
    return bundle


def test_integration_bundle_blocks_incomplete_child_contracts_and_pending_policy():
    result = evaluate_integration_bundle_contract(
        _complete_bundle(
            integration_policy={"status": "PENDING_APPROVAL"},
            integration_checks={"analysis_trace_bound": True},
            offline_replay_manifest=_offline_replay_manifest(
                threshold_policy={"status": "PENDING_APPROVAL"},
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
    assert "offline_replay_contract_not_ready" in result["blockers"]
    assert "fault_recovery_contract_not_ready" in result["blockers"]
    assert "offline_replay:approved_quality_threshold_policy_required" in result["blockers"]
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
    assert result["summary"]["integration_policy_approved"] is True
    assert result["summary"]["missing_integration_checks"] == []
    assert result["blockers"] == []


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
