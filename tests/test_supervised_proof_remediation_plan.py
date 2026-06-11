from pathlib import Path

from backend.supervised_proof_remediation_plan import build_supervised_proof_remediation_plan


def proof_report(*rows, invalid_summaries=None):
    return {
        "type": "supervised_proof_artifact_report",
        "summary": {
            "invalid_summary_count": len(invalid_summaries or []),
        },
        "rows": list(rows),
        "invalid_summaries": list(invalid_summaries or []),
    }


def blocked_legacy_row(proof_id: str = "legacy-gain-proof") -> dict:
    return {
        "proof_id": proof_id,
        "value_type": "gain",
        "channel": 4,
        "proof_mode": "legacy_ambiguous",
        "status": "blocked",
        "artifact_path": "artifacts/wing_telemetry/legacy/summary.json",
        "blockers": [
            "legacy_physical_schema_missing",
            "physical_proof_required_for_promotion",
        ],
        "evidence": {
            "delta_db": -0.25,
        },
    }


def blocked_nonlegacy_row(proof_id: str = "blocked-fader-proof") -> dict:
    return {
        "proof_id": proof_id,
        "value_type": "fader",
        "channel": 2,
        "proof_mode": "physical",
        "status": "blocked",
        "artifact_path": "artifacts/wing_telemetry/blocked/summary.json",
        "blockers": ["rollback_readback_missing"],
    }


def test_proof_remediation_plan_module_is_transport_free():
    source = Path("backend/supervised_proof_remediation_plan.py").read_text(encoding="utf-8")

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
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


def test_legacy_blocked_artifact_gets_current_schema_rerun_action_without_execution_authority():
    plan = build_supervised_proof_remediation_plan(
        proof_report(blocked_legacy_row()),
        backend_url="ws://proof-backend",
        artifact_root="artifacts/proof",
    )

    assert plan["type"] == "supervised_proof_remediation_plan"
    assert plan["status"] == "manual_remediation_required"
    assert plan["safety"] == {
        "read_only": True,
        "live_write": False,
        "execution_allowed_by_plan": False,
        "live_widening_allowed": False,
        "automatic_safety_update_allowed": False,
        "full_autonomous_ready": False,
        "operator_required": True,
    }
    assert plan["summary"]["blocked_artifact_count"] == 1
    assert plan["summary"]["legacy_ambiguous_count"] == 1
    assert plan["summary"]["blocker_counts"]["legacy_physical_schema_missing"] == 1

    action = plan["actions"][0]
    assert action["action_type"] == "rerun_current_physical_schema"
    assert action["proof_id"] == "legacy-gain-proof"
    assert action["value_type"] == "gain"
    assert action["channel"] == 4
    assert action["delta_db"] == -0.25
    assert action["execution_allowed_by_plan"] is False
    assert action["live_execution_required"] is True
    assert action["operator_approval_required"] is True
    assert "--execute" in action["command"]
    assert "--operator-confirm-channel" in action["command"]
    assert (
        action["command"][action["command"].index("--operator-confirm-channel") + 1]
        == "4"
    )
    assert "--operator-confirm-value-type" in action["command"]
    assert (
        action["command"][action["command"].index("--operator-confirm-value-type") + 1]
        == "gain"
    )
    assert "--operator-confirm-delta-db" in action["command"]
    assert (
        action["command"][action["command"].index("--operator-confirm-delta-db") + 1]
        == "-0.25"
    )
    assert "--prove-blocked-apply" in action["command"]
    assert "--prove-emergency-stop" in action["command"]
    assert "ws://proof-backend" in action["command"]
    assert "artifacts/proof" in action["command"]


def test_nonlegacy_blocked_artifact_gets_manual_review_action():
    plan = build_supervised_proof_remediation_plan(proof_report(blocked_nonlegacy_row()))

    action = plan["actions"][0]
    assert action["action_type"] == "manual_blocker_review"
    assert action["command"] == []
    assert action["live_execution_required"] is False
    assert action["operator_approval_required"] is False
    assert action["blockers"] == ["rollback_readback_missing"]


def test_invalid_summaries_get_cleanup_actions_and_no_artifacts_status_is_explicit():
    invalid = [{"artifact_path": "artifacts/bad/summary.json", "reason": "json_load_error:JSONDecodeError"}]

    plan = build_supervised_proof_remediation_plan(proof_report(invalid_summaries=invalid))
    empty_plan = build_supervised_proof_remediation_plan(proof_report())

    assert plan["status"] == "manual_remediation_required"
    assert plan["summary"]["invalid_summary_count"] == 1
    assert plan["actions"][0]["action_type"] == "fix_or_remove_invalid_summary"
    assert plan["actions"][0]["command"] == []
    assert empty_plan["status"] == "no_artifacts"
    assert empty_plan["actions"] == []
