from pathlib import Path

from backend.autonomy_mode_transition_gate import build_autonomy_mode_transition_gate


def proof_context(*rows, auto_mode_unlocked: bool = False):
    context = {
        "type": "supervised_proof_artifact_report",
        "summary": {"invalid_summary_count": 0},
        "rows": list(rows),
        "invalid_summaries": [],
    }
    if auto_mode_unlocked:
        context["auto_mode_unlocked"] = True
    return context


def physical_candidate(proof_id: str, value_type: str) -> dict:
    return {
        "proof_id": proof_id,
        "value_type": value_type,
        "proof_mode": "physical",
        "status": "supervised_candidate",
        "promotion_allowed": True,
        "autonomous_ready": False,
        "blockers": [],
    }


def legacy_blocked(proof_id: str, value_type: str) -> dict:
    return {
        "proof_id": proof_id,
        "value_type": value_type,
        "channel": 4,
        "proof_mode": "legacy_ambiguous",
        "status": "blocked",
        "promotion_allowed": False,
        "autonomous_ready": False,
        "artifact_path": "artifacts/wing_telemetry/legacy/summary.json",
        "blockers": [
            "legacy_physical_schema_missing",
            "physical_proof_required_for_promotion",
        ],
        "evidence": {"delta_db": -0.25},
    }


def test_mode_transition_gate_module_is_transport_free():
    source = Path("backend/autonomy_mode_transition_gate.py").read_text(encoding="utf-8")

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


def test_gate_keeps_analysis_and_assist_available_while_auto_is_blocked():
    gate = build_autonomy_mode_transition_gate(proof_context())
    rows = {row["mode"]: row for row in gate["rows"]}

    assert gate["type"] == "autonomy_mode_transition_gate"
    assert gate["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "automatic_mode_change_allowed": False,
        "automatic_safety_update_allowed": False,
        "full_autonomous_ready": False,
    }
    assert rows["analyze"]["status"] == "available"
    assert rows["analyze"]["runtime_entry_allowed"] is True
    assert rows["assist"]["status"] == "available"
    assert rows["assist"]["proposal_creation_allowed"] is True
    assert rows["auto"]["status"] == "blocked"
    assert rows["auto"]["runtime_entry_allowed"] is False
    assert rows["auto"]["autonomous_apply_allowed"] is False
    assert rows["auto"]["readiness_gap_bucket_counts"]["read_surface"] > 0
    assert rows["auto"]["readiness_gap_bucket_counts"]["write_surface"] > 0
    assert "readiness_gap_read_surface" in rows["auto"]["blockers"]
    assert "readiness_gap_write_surface" in rows["auto"]["blockers"]
    assert "scene_preset_supervised_promotion" in rows["auto"]["readiness_gap_target_ids_by_bucket"]["read_surface"]
    assert gate["summary"]["autonomous_apply_allowed_modes"] == []
    assert gate["summary"]["full_auto_readiness_gap_bucket_counts"]["read_surface"] > 0
    assert gate["summary"]["full_auto_readiness_gap_bucket_counts"]["write_surface"] > 0
    assert (
        "scene_preset_supervised_promotion"
        in gate["summary"]["full_auto_readiness_gap_target_ids_by_bucket"]["read_surface"]
    )


def test_gate_reports_supervised_as_limited_and_blocks_expansion_from_proof_overlay():
    gate = build_autonomy_mode_transition_gate(
        proof_context(
            physical_candidate("physical-fader-proof", "fader"),
            legacy_blocked("legacy-gain-proof", "gain"),
        )
    )
    rows = {row["mode"]: row for row in gate["rows"]}
    supervised = rows["supervised"]

    assert supervised["status"] == "blocked"
    assert supervised["runtime_entry_allowed"] is False
    assert supervised["live_apply_allowed"] is False
    assert supervised["current_live_write_kinds"] == []
    assert supervised["current_live_write_surfaces"] == []
    assert supervised["mode_expansion_allowed"] is False
    assert supervised["promotion_target_ids"] == []
    assert supervised["evidence_statuses"] == []
    assert "proof_remediation_required_before_promotion" in supervised["blockers"]
    assert "supervised_mode_is_narrow_fader_gain_only" in supervised["blockers"]
    assert supervised["readiness_gap_bucket_counts"]["read_surface"] > 0
    assert "readiness_gap_read_surface" not in supervised["blockers"]
    assert "supervised" not in gate["summary"]["live_apply_allowed_modes"]
    assert gate["summary"]["supervised_current_live_write_kinds"] == []
    assert gate["summary"]["supervised_current_live_write_surfaces"] == []
    assert gate["summary"]["supervised_promotion_target_ids"] == []
    assert gate["summary"]["blocked_modes"] == ["supervised", "auto"]


def test_gate_blocks_supervised_when_static_policy_has_no_live_target_even_with_evidence():
    custom_context = proof_context(physical_candidate("physical-gain-proof", "gain"))
    custom_context["autonomy_promotion_readiness"] = {
        "type": "autonomy_promotion_readiness",
        "summary": {"supervised_live_allowed_target_ids": []},
        "rows": [],
    }

    gate = build_autonomy_mode_transition_gate(custom_context)
    rows = {row["mode"]: row for row in gate["rows"]}

    assert rows["supervised"]["status"] == "blocked"
    assert rows["supervised"]["runtime_entry_allowed"] is False
    assert rows["supervised"]["live_apply_allowed"] is False
    assert rows["supervised"]["readiness_gap_bucket_counts"] == {}
    assert rows["supervised"]["readiness_gap_target_ids_by_bucket"] == {}


def test_gate_reflects_auto_mode_unlock_in_auto_blockers_for_non_wing_connections():
    locked_gate = build_autonomy_mode_transition_gate(
        proof_context(),
        connection_mode="local",
        wing_boundary_active=False,
    )
    unlocked_gate = build_autonomy_mode_transition_gate(
        proof_context(auto_mode_unlocked=True),
        connection_mode="local",
        wing_boundary_active=False,
    )

    locked_auto = {row["mode"]: row for row in locked_gate["rows"]}["auto"]
    unlocked_auto = {row["mode"]: row for row in unlocked_gate["rows"]}["auto"]

    assert "auto_mode_requires_explicit_runtime_unlock" in locked_auto["blockers"]
    assert "auto_mode_requires_explicit_runtime_unlock" not in unlocked_auto["blockers"]
