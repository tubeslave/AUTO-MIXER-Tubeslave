from pathlib import Path

from backend.safety_kernel_trace_contract import (
    REQUIRED_AUDIT_FIELDS,
    REQUIRED_GUARD_CHECKS,
    REQUIRED_TRACE_SCENARIOS,
    evaluate_safety_kernel_trace_contract,
)


def _trace_record(scenario: str, **overrides) -> dict:
    decision = (
        "bounded_allow"
        if scenario == "supervised_fader_gain_allowed"
        else "reject"
    )
    record = {
        "scenario": scenario,
        "safety_decision": decision,
        "guard_checks": {key: True for key in REQUIRED_GUARD_CHECKS},
        "audit": {
            "decision_id": f"{scenario}-decision",
            "checked_at": "2026-06-19T18:39:18Z",
            "candidate_action": f"{scenario}-candidate",
            "safety_decision": decision,
            "reason": f"{scenario} exercised deterministic safety kernel",
            "policy_version": "safety-kernel-offline-v1",
            "audit_reference": f"artifacts/safety-kernel/{scenario}.json",
        },
    }
    record.update(overrides)
    return record


def _complete_manifest(**overrides) -> dict:
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
            _trace_record(scenario) for scenario in REQUIRED_TRACE_SCENARIOS
        ],
    }
    manifest.update(overrides)
    return manifest


def test_incomplete_trace_blocks_safety_kernel_trace_use():
    result = evaluate_safety_kernel_trace_contract(
        _complete_manifest(
            trace_records=[
                _trace_record(
                    "supervised_fader_gain_allowed",
                    guard_checks={"mode_permission_checked": True},
                )
            ]
        )
    )

    assert result["status"] == "blocked"
    assert result["ready_for_safety_kernel_trace_use"] is False
    assert result["authority"]["live_authority_granted"] is False
    assert result["authority"]["readiness_closure_allowed"] is False
    assert "missing_required_trace_scenario:unsupported_live_write_rejected" in result[
        "blockers"
    ]
    assert (
        "supervised_fader_gain_allowed:missing_guard_check:"
        "live_write_allowlist_checked"
        in result["blockers"]
    )


def test_complete_offline_trace_is_ready_without_live_authority():
    result = evaluate_safety_kernel_trace_contract(_complete_manifest())

    assert result["status"] == "ready_for_safety_kernel_trace_use"
    assert result["ready_for_safety_kernel_trace_use"] is True
    assert result["offline_only"] is True
    assert result["hardware_evidence"] is False
    assert result["live_write"] is False
    assert result["authority"] == {
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "contacts_wing": False,
        "contacts_backend": False,
    }
    assert result["summary"]["missing_trace_scenarios"] == []
    assert result["blockers"] == []


def test_live_hardware_and_authority_claims_are_rejected():
    result = evaluate_safety_kernel_trace_contract(
        _complete_manifest(
            offline_only=False,
            live_write=True,
            contacts_wing=True,
            contacts_backend=True,
            hardware_evidence=True,
            readiness_closure_allowed=True,
            live_authority_granted=True,
        )
    )

    assert result["ready_for_safety_kernel_trace_use"] is False
    assert set(result["blockers"]) >= {
        "offline_only_required",
        "safety_kernel_trace_forbids_live_write",
        "safety_kernel_trace_forbids_wing_contact",
        "safety_kernel_trace_forbids_backend_contact",
        "safety_kernel_trace_must_not_be_labeled_hardware",
        "safety_kernel_trace_cannot_close_readiness",
        "safety_kernel_trace_cannot_grant_live_authority",
    }
    assert result["authority"]["live_authority_granted"] is False
    assert result["authority"]["readiness_closure_allowed"] is False


def test_wrong_decision_and_missing_audit_fields_are_rejected():
    audit = {
        key: f"audit-{key}" for key in REQUIRED_AUDIT_FIELDS if key != "reason"
    }
    audit["safety_decision"] = "allow"
    result = evaluate_safety_kernel_trace_contract(
        _complete_manifest(
            trace_records=[
                _trace_record(
                    scenario,
                    safety_decision="allow",
                    audit=audit,
                )
                if scenario == "unsupported_live_write_rejected"
                else _trace_record(scenario)
                for scenario in REQUIRED_TRACE_SCENARIOS
            ]
        )
    )

    assert result["ready_for_safety_kernel_trace_use"] is False
    assert "unsupported_live_write_rejected:unexpected_safety_decision" in result[
        "blockers"
    ]
    assert "unsupported_live_write_rejected:missing_audit_field:reason" in result[
        "blockers"
    ]


def test_safety_kernel_trace_contract_source_has_no_live_transport_imports():
    source = Path("backend/safety_kernel_trace_contract.py").read_text(
        encoding="utf-8"
    )
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
