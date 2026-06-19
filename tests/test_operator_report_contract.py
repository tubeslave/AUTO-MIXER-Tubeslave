from pathlib import Path

from backend.operator_report_contract import (
    REQUIRED_AUDIT_FIELDS,
    REQUIRED_AUTHORITY_DENIAL_FLAGS,
    REQUIRED_REPORT_BINDINGS,
    REQUIRED_REPORT_SECTIONS,
    evaluate_operator_report_contract,
)


def _binding(name: str) -> dict:
    return {
        "status": "passed",
        "artifact": f"operator-report/{name}.json",
        "sha256": f"sha256:{name}",
    }


def _complete_report(**overrides) -> dict:
    report = {
        "report_id": "operator-report-fixture",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "report_sections": {key: True for key in REQUIRED_REPORT_SECTIONS},
        "report_bindings": {key: _binding(key) for key in REQUIRED_REPORT_BINDINGS},
        "authority_denial": {
            key: False for key in REQUIRED_AUTHORITY_DENIAL_FLAGS
        },
        "visible_blockers": [
            "M001 waits for real WING read-only artifact",
            "operator review required",
        ],
        "next_operator_action": (
            "Run the supervised WING read-only preflight and review artifacts."
        ),
        "audit": {
            key: f"operator-report-{key}" for key in REQUIRED_AUDIT_FIELDS
        },
    }
    report.update(overrides)
    return report


def test_operator_report_blocks_incomplete_report_evidence():
    result = evaluate_operator_report_contract(
        _complete_report(
            report_sections={"current_status": True},
            report_bindings={"integration_bundle_artifact": _binding("bundle")},
            authority_denial={"live_authority_granted": False},
            visible_blockers=[],
            next_operator_action="",
            audit={"report_id": "operator-report"},
        )
    )

    assert result["status"] == "blocked"
    assert result["ready_for_operator_report_use"] is False
    assert "missing_report_section:authority_denial" in result["blockers"]
    assert "missing_report_binding:offline_replay_result" in result["blockers"]
    assert "missing_authority_denial_flag:readiness_closure_allowed" in (
        result["blockers"]
    )
    assert "visible_blockers_required" in result["blockers"]
    assert "next_operator_action_required" in result["blockers"]
    assert "missing_audit_field:generated_at" in result["blockers"]
    assert result["authority"]["live_authority_granted"] is False


def test_operator_report_accepts_complete_offline_report_without_authority():
    result = evaluate_operator_report_contract(_complete_report())

    assert result["status"] == "ready_for_operator_report_use"
    assert result["ready_for_operator_report_use"] is True
    assert result["offline_only"] is True
    assert result["hardware_evidence"] is False
    assert result["live_write"] is False
    assert result["authority"] == {
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "contacts_wing": False,
        "contacts_backend": False,
    }
    assert result["summary"]["missing_report_sections"] == []
    assert result["summary"]["missing_report_bindings"] == []
    assert result["summary"]["missing_authority_denial_flags"] == []
    assert result["summary"]["non_denied_authority_flags"] == []
    assert result["summary"]["visible_blocker_count"] == 2
    assert result["summary"]["next_operator_action_present"] is True
    assert result["summary"]["missing_audit_fields"] == []
    assert result["blockers"] == []


def test_operator_report_rejects_live_or_hardware_claims():
    result = evaluate_operator_report_contract(
        _complete_report(
            offline_only=False,
            live_write=True,
            contacts_wing=True,
            contacts_backend=True,
            hardware_evidence=True,
            readiness_closure_allowed=True,
            live_authority_granted=True,
            authority_denial={
                key: False for key in REQUIRED_AUTHORITY_DENIAL_FLAGS
            }
            | {"live_authority_granted": True},
        )
    )

    assert result["status"] == "blocked"
    assert set(result["blockers"]) >= {
        "offline_only_required",
        "operator_report_forbids_live_write",
        "operator_report_forbids_wing_contact",
        "operator_report_forbids_backend_contact",
        "operator_report_must_not_be_labeled_hardware",
        "operator_report_cannot_close_readiness",
        "operator_report_cannot_grant_live_authority",
        "authority_not_denied:live_authority_granted",
    }
    assert result["authority"]["live_authority_granted"] is False
    assert result["authority"]["readiness_closure_allowed"] is False


def test_operator_report_requires_bound_artifacts_not_loose_status():
    bindings = {key: True for key in REQUIRED_REPORT_BINDINGS}

    result = evaluate_operator_report_contract(_complete_report(report_bindings=bindings))

    assert result["status"] == "blocked"
    assert "missing_report_binding:integration_bundle_artifact" in result["blockers"]
    assert "missing_report_binding:no_live_transport_result" in result["blockers"]
    assert result["authority"]["live_authority_granted"] is False


def test_operator_report_source_has_no_live_transport_imports():
    source = Path("backend/operator_report_contract.py").read_text()
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
