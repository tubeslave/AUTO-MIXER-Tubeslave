from pathlib import Path

from backend.autonomy_promotion_readiness import (
    build_autonomy_promotion_readiness_snapshot,
    iter_promotion_readiness_rows,
)


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


def test_broad_console_surfaces_are_blocked_by_missing_proof_or_manual_only_policy():
    rows = {row.id: row for row in iter_promotion_readiness_rows()}

    eq = rows["eq_supervised_promotion"]
    routing = rows["routing_scene_preset_supervised_promotion"]
    bus = rows["bus_group_matrix_supervised_promotion"]

    assert eq.supervised_live_allowed is False
    assert "safety_live_not_allowed" in eq.blockers
    assert "fake_mixer_proof_missing" in eq.blockers
    assert "read_surface_channel_eq_dynamics_gate_partial" in eq.blockers

    assert routing.status == "manual_only_blocked"
    assert "safety_status_blocked" in routing.blockers
    assert "write_surface_routing_snapshot_scene_writes_quarantined" in routing.blockers
    assert "read_surface_user_presets_partial" in routing.blockers

    assert bus.supervised_live_allowed is False
    assert "safety_coverage_row_missing" in bus.blockers
    assert "write_surface_mixer_transport_clients_quarantined" in bus.blockers


def test_no_readiness_row_widens_live_write_authority():
    for row in iter_promotion_readiness_rows():
        assert row.autonomous_ready is False
        if row.supervised_live_allowed:
            assert row.action_family == "fader"
        else:
            assert row.status in {"proof_blocked", "read_coverage_blocked", "manual_only_blocked"}
