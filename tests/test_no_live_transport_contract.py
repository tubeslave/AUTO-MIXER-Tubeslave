from pathlib import Path

from backend.no_live_transport_contract import (
    REQUIRED_TRANSPORT_CONTROLS,
    REQUIRED_TRANSPORT_EVIDENCE,
    REQUIRED_ZERO_RUNTIME_OBSERVATIONS,
    evaluate_no_live_transport_contract,
)


def _evidence(name: str) -> dict:
    return {
        "status": "passed",
        "artifact": f"artifacts/no-live-transport/{name}.json",
        "sha256": f"{name}-sha256",
    }


def _complete_manifest(**overrides) -> dict:
    manifest = {
        "transport_id": "no-live-transport-fixture",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "transport_controls": {key: True for key in REQUIRED_TRANSPORT_CONTROLS},
        "transport_evidence": {
            key: _evidence(key) for key in REQUIRED_TRANSPORT_EVIDENCE
        },
        "runtime_observations": {
            key: False if key == "websocket_opened" else 0
            for key in REQUIRED_ZERO_RUNTIME_OBSERVATIONS
        },
        "forbidden_transport_tokens": [],
    }
    manifest.update(overrides)
    return manifest


def test_no_live_transport_contract_blocks_incomplete_evidence():
    result = evaluate_no_live_transport_contract(
        _complete_manifest(
            transport_controls={"offline_executor_only": True},
            transport_evidence={"source_audit_artifact": _evidence("source")},
            runtime_observations={"websocket_opened": False},
        )
    )

    assert result["status"] == "blocked"
    assert result["ready_for_no_live_transport_use"] is False
    assert "missing_transport_control:no_wing_client" in result["blockers"]
    assert "missing_transport_evidence:runtime_config_artifact" in result["blockers"]
    assert "missing_runtime_observation:osc_messages_sent" in result["blockers"]
    assert result["authority"]["live_authority_granted"] is False
    assert result["authority"]["readiness_closure_allowed"] is False


def test_no_live_transport_contract_accepts_complete_offline_evidence():
    result = evaluate_no_live_transport_contract(_complete_manifest())

    assert result["status"] == "ready_for_no_live_transport_use"
    assert result["ready_for_no_live_transport_use"] is True
    assert result["offline_only"] is True
    assert result["hardware_evidence"] is False
    assert result["live_write"] is False
    assert result["authority"] == {
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "contacts_wing": False,
        "contacts_backend": False,
    }
    assert result["summary"]["missing_transport_controls"] == []
    assert result["summary"]["missing_transport_evidence"] == []
    assert result["summary"]["nonzero_runtime_observations"] == []
    assert result["blockers"] == []


def test_no_live_transport_contract_rejects_live_or_hardware_claims():
    result = evaluate_no_live_transport_contract(
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

    assert result["status"] == "blocked"
    assert result["ready_for_no_live_transport_use"] is False
    assert set(result["blockers"]) >= {
        "offline_only_required",
        "no_live_transport_forbids_live_write",
        "no_live_transport_forbids_wing_contact",
        "no_live_transport_forbids_backend_contact",
        "no_live_transport_must_not_be_labeled_hardware",
        "no_live_transport_cannot_close_readiness",
        "no_live_transport_cannot_grant_live_authority",
    }
    assert result["authority"]["live_authority_granted"] is False
    assert result["authority"]["readiness_closure_allowed"] is False


def test_no_live_transport_contract_rejects_observed_transport():
    result = evaluate_no_live_transport_contract(
        _complete_manifest(
            runtime_observations={
                "websocket_opened": True,
                "osc_messages_sent": 1,
                "live_write_requests_sent": 1,
                "mixer_client_calls": 1,
                "network_connections_opened": 1,
            },
            forbidden_transport_tokens=["WingClient", "EnhancedOSCClient"],
        )
    )

    assert result["status"] == "blocked"
    assert "live_transport_observed:websocket_opened" in result["blockers"]
    assert "live_transport_observed:osc_messages_sent" in result["blockers"]
    assert "forbidden_transport_token:WingClient" in result["blockers"]
    assert "forbidden_transport_token:EnhancedOSCClient" in result["blockers"]
    assert result["summary"]["nonzero_runtime_observations"] == [
        "websocket_opened",
        "osc_messages_sent",
        "live_write_requests_sent",
        "mixer_client_calls",
        "network_connections_opened",
    ]


def test_no_live_transport_contract_source_has_no_live_transport_imports():
    source = Path("backend/no_live_transport_contract.py").read_text()
    forbidden = [
        "from backend.wing_client",
        "import socket",
        "import websockets",
        "EnhancedOSCClient",
        "WingClient",
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
