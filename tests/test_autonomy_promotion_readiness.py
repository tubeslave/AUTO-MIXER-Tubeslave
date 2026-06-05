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
    routing = rows["routing_supervised_promotion"]
    scene_preset = rows["scene_preset_supervised_promotion"]
    bus = rows["bus_group_matrix_supervised_promotion"]

    assert eq.supervised_live_allowed is False
    assert "safety_live_not_allowed" in eq.blockers
    assert "fake_mixer_proof_missing" in eq.blockers
    assert "read_surface_channel_eq_dynamics_gate_partial" in eq.blockers

    assert routing.status == "manual_only_blocked"
    assert routing.action_family == "routing"
    assert routing.capability_ids == ("console_routing_control",)
    assert routing.read_surface_ids == ("routing_and_sends",)
    assert routing.evidence_requirement_ids == (
        "current_route_cache_freshness",
        "route_diff_evidence",
        "source_identity_downstream_trust",
        "routing_physical_fixture",
        "hardware_readback_and_rollback_proof",
    )
    assert "safety_status_blocked" in routing.blockers
    assert "write_surface_routing_snapshot_scene_writes_quarantined" in routing.blockers

    assert scene_preset.status == "manual_only_blocked"
    assert scene_preset.supervised_live_allowed is False
    assert scene_preset.autonomous_ready is False
    assert scene_preset.action_family == "scene_preset"
    assert scene_preset.capability_ids == ("console_scene_preset_control",)
    assert scene_preset.read_surface_ids == ("snapshots_scenes", "user_presets")
    assert scene_preset.evidence_requirement_ids == (
        "current_scene_preset_cache_freshness",
        "snapshots_scenes_snapshot_cache_freshness",
        "user_presets_snapshot_cache_freshness",
        "read_only_scoped_diff_evidence",
        "read_only_rollback_evidence",
        "scene_preset_physical_fixture",
        "hardware_readback_and_rollback_proof",
    )
    assert "safety_status_blocked" in scene_preset.blockers
    assert "write_surface_routing_snapshot_scene_writes_quarantined" in scene_preset.blockers
    assert "read_surface_snapshots_scenes_partial" in scene_preset.blockers
    assert "read_surface_user_presets_partial" in scene_preset.blockers

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


def test_readiness_snapshot_serializes_evidence_requirement_ids():
    snapshot = build_autonomy_promotion_readiness_snapshot()
    rows = {row["id"]: row for row in snapshot["rows"]}
    scene_preset = rows["scene_preset_supervised_promotion"]

    assert scene_preset["supervised_live_allowed"] is False
    assert scene_preset["autonomous_ready"] is False
    assert scene_preset["evidence_requirement_ids"] == [
        "current_scene_preset_cache_freshness",
        "snapshots_scenes_snapshot_cache_freshness",
        "user_presets_snapshot_cache_freshness",
        "read_only_scoped_diff_evidence",
        "read_only_rollback_evidence",
        "scene_preset_physical_fixture",
        "hardware_readback_and_rollback_proof",
    ]


def test_readiness_snapshot_groups_autonomy_gaps_without_granting_authority():
    snapshot = build_autonomy_promotion_readiness_snapshot()
    gap_summary = snapshot["summary"]["promotion_gap_summary"]

    assert gap_summary["read_only"] is True
    assert gap_summary["live_write"] is False
    assert gap_summary["full_autonomous_ready"] is False
    assert gap_summary["gap_bucket_counts"]["autonomy_mode"] >= snapshot["summary"]["target_count"]
    assert gap_summary["gap_bucket_counts"]["proof"] > 0
    assert gap_summary["gap_bucket_counts"]["read_surface"] > 0
    assert gap_summary["gap_bucket_counts"]["write_surface"] > 0
    assert gap_summary["gap_bucket_counts"]["evidence_requirement"] >= 12

    blocked_by_bucket = gap_summary["blocked_target_ids_by_bucket"]
    assert "fader_supervised_promotion" in blocked_by_bucket["autonomy_mode"]
    assert "scene_preset_supervised_promotion" in blocked_by_bucket["read_surface"]
    assert "scene_preset_supervised_promotion" in blocked_by_bucket["write_surface"]
    assert "routing_supervised_promotion" in blocked_by_bucket["safety"]

    evidence_by_family = gap_summary["evidence_requirement_ids_by_action_family"]
    assert evidence_by_family["routing"] == [
        "current_route_cache_freshness",
        "route_diff_evidence",
        "source_identity_downstream_trust",
        "routing_physical_fixture",
        "hardware_readback_and_rollback_proof",
    ]
    assert "read_only_scoped_diff_evidence" in evidence_by_family["scene_preset"]

    next_by_family = gap_summary["next_required_evidence_by_action_family"]
    assert any("Keep scene apply blocked" in item for item in next_by_family["scene_preset"])
