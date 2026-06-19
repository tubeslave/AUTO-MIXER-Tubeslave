from pathlib import Path

from backend.fault_injection_recovery_contract import (
    REQUIRED_AUDIT_FIELDS,
    REQUIRED_EVALUATION_TYPES,
    REQUIRED_FAULT_SCENARIOS,
    REQUIRED_RECOVERY_OUTCOMES,
    evaluate_fault_injection_recovery_contract,
)


def _fault_case(name: str) -> dict:
    return {
        "fault_scenario": name,
        "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
        "recovery_outcomes": {key: True for key in REQUIRED_RECOVERY_OUTCOMES},
        "audit": {key: f"{name}-{key}" for key in REQUIRED_AUDIT_FIELDS},
    }


def _complete_manifest(**overrides) -> dict:
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


def test_fault_recovery_contract_blocks_incomplete_or_pending_policy_evidence():
    result = evaluate_fault_injection_recovery_contract(
        _complete_manifest(
            recovery_policy={"status": "PENDING_APPROVAL"},
            fault_scenarios=[_fault_case("transport_timeout")],
        )
    )

    assert result["status"] == "blocked"
    assert result["ready_for_fault_recovery_promotion"] is False
    assert result["authority"]["readiness_closure_allowed"] is False
    assert result["authority"]["live_authority_granted"] is False
    assert "approved_recovery_policy_required" in result["blockers"]
    assert "missing_required_fault_scenario:connection_loss" in result["blockers"]
    assert "missing_required_fault_scenario:stale_state" in result["blockers"]


def test_fault_recovery_contract_accepts_complete_simulator_bundle_without_live_authority():
    result = evaluate_fault_injection_recovery_contract(_complete_manifest())

    assert result["status"] == "ready_for_fault_recovery_promotion"
    assert result["ready_for_fault_recovery_promotion"] is True
    assert result["offline_only"] is True
    assert result["hardware_evidence"] is False
    assert result["live_write"] is False
    assert result["authority"] == {
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "contacts_wing": False,
        "contacts_backend": False,
    }
    assert result["summary"]["covered_fault_scenarios"] == list(REQUIRED_FAULT_SCENARIOS)
    assert result["summary"]["recovery_policy_approved"] is True
    assert result["blockers"] == []


def test_fault_recovery_contract_rejects_live_or_hardware_claims():
    result = evaluate_fault_injection_recovery_contract(
        _complete_manifest(
            live_write=True,
            contacts_wing=True,
            contacts_backend=True,
            hardware_evidence=True,
            readiness_closure_allowed=True,
            live_authority_granted=True,
        )
    )

    assert result["status"] == "blocked"
    assert result["ready_for_fault_recovery_promotion"] is False
    assert set(result["blockers"]) >= {
        "fault_recovery_contract_forbids_live_write",
        "fault_recovery_contract_forbids_wing_contact",
        "fault_recovery_contract_forbids_backend_contact",
        "simulator_fault_evidence_must_not_be_labeled_hardware",
        "simulator_fault_evidence_cannot_close_readiness",
        "simulator_fault_evidence_cannot_grant_live_authority",
    }
    assert result["authority"]["live_authority_granted"] is False
    assert result["authority"]["readiness_closure_allowed"] is False


def test_fault_recovery_contract_rejects_missing_recovery_outcomes_and_audit_fields():
    bad_case = _fault_case("emergency_stop")
    bad_case["recovery_outcomes"]["emergency_stop_verified"] = False
    bad_case["audit"]["rollback_reference"] = ""

    result = evaluate_fault_injection_recovery_contract(
        _complete_manifest(
            fault_scenarios=[
                _fault_case(name)
                for name in REQUIRED_FAULT_SCENARIOS
                if name != "emergency_stop"
            ]
            + [bad_case],
        )
    )

    assert result["status"] == "blocked"
    assert (
        "emergency_stop:missing_recovery_outcome:emergency_stop_verified"
        in result["blockers"]
    )
    assert "emergency_stop:missing_audit_field:rollback_reference" in result["blockers"]


def test_fault_recovery_contract_source_has_no_live_transport_imports():
    source = Path("backend/fault_injection_recovery_contract.py").read_text()
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
