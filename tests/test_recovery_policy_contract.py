from pathlib import Path

from backend.recovery_policy_contract import (
    REQUIRED_RECOVERY_CONTROLS,
    REQUIRED_RECOVERY_EVIDENCE,
    evaluate_recovery_policy_contract,
)


def _complete_manifest(**overrides) -> dict:
    manifest = {
        "policy_id": "candidate-recovery-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "recovery_controls": {key: True for key in REQUIRED_RECOVERY_CONTROLS},
        "recovery_evidence": {key: True for key in REQUIRED_RECOVERY_EVIDENCE},
        "recovery_policy": {"status": "candidate"},
    }
    manifest.update(overrides)
    return manifest


def test_incomplete_candidate_blocks_review_and_recovery_policy_use():
    result = evaluate_recovery_policy_contract(
        {
            "policy_id": "incomplete-recovery-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "recovery_controls": {"fault_detection_policy": True},
            "recovery_evidence": {"fault_scenario_matrix_bound": True},
            "recovery_policy": {"status": "PENDING_APPROVAL"},
        }
    )

    assert result["status"] == "blocked"
    assert result["ready_for_owner_review"] is False
    assert result["ready_for_recovery_policy_use"] is False
    assert result["authority"]["live_authority_granted"] is False
    assert result["authority"]["readiness_closure_allowed"] is False
    assert "missing_recovery_control:fail_closed_policy" in result["blockers"]
    assert "missing_recovery_evidence:rollback_reference_bound" in result["blockers"]
    assert "owner_approved_recovery_policy_required" in result["blockers"]


def test_complete_candidate_is_review_ready_but_not_approved_for_policy_use():
    result = evaluate_recovery_policy_contract(_complete_manifest())

    assert result["status"] == "ready_for_owner_review"
    assert result["ready_for_owner_review"] is True
    assert result["ready_for_recovery_policy_use"] is False
    assert result["recovery_policy_approved"] is False
    assert result["review_blockers"] == []
    assert result["blockers"] == ["owner_approved_recovery_policy_required"]
    assert result["summary"]["missing_recovery_controls"] == []
    assert result["summary"]["missing_recovery_evidence"] == []
    assert result["hardware_evidence"] is False


def test_owner_approved_complete_policy_is_ready_for_use_only_offline():
    result = evaluate_recovery_policy_contract(
        _complete_manifest(
            policy_id="approved-recovery-policy",
            recovery_policy={
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:recovery-policy",
                "approved_at": "2026-06-19T17:51:02Z",
            },
        )
    )

    assert result["status"] == "ready_for_recovery_policy_use"
    assert result["ready_for_owner_review"] is True
    assert result["ready_for_recovery_policy_use"] is True
    assert result["recovery_policy_approved"] is True
    assert result["blockers"] == []
    assert result["live_write"] is False
    assert result["authority"]["live_authority_granted"] is False
    assert result["authority"]["readiness_closure_allowed"] is False


def test_live_hardware_and_authority_claims_are_rejected():
    result = evaluate_recovery_policy_contract(
        _complete_manifest(
            offline_only=False,
            live_write=True,
            contacts_wing=True,
            contacts_backend=True,
            hardware_evidence=True,
            readiness_closure_allowed=True,
            live_authority_granted=True,
            recovery_policy={
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:recovery-policy",
                "approved_at": "2026-06-19T17:51:02Z",
            },
        )
    )

    assert result["ready_for_owner_review"] is False
    assert result["ready_for_recovery_policy_use"] is False
    assert result["authority"]["live_authority_granted"] is False
    assert "offline_only_required" in result["blockers"]
    assert "recovery_policy_forbids_live_write" in result["blockers"]
    assert "recovery_policy_forbids_wing_contact" in result["blockers"]
    assert "recovery_policy_forbids_backend_contact" in result["blockers"]
    assert "recovery_policy_evidence_must_not_be_labeled_hardware" in result["blockers"]
    assert "recovery_policy_evidence_cannot_close_readiness" in result["blockers"]
    assert "recovery_policy_evidence_cannot_grant_live_authority" in result["blockers"]


def test_recovery_policy_contract_source_has_no_live_transport_imports():
    source = Path("backend/recovery_policy_contract.py").read_text(encoding="utf-8")
    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        "requests",
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
