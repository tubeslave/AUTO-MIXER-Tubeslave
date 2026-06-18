from pathlib import Path

from backend.processing_registry import get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def test_broad_goal_methods_keep_safety_markers_without_live_widening():
    expected_notes = {
        "cross_adaptive_eq": {
            "arrangement_role_priority_evidence",
            "source_identity_downstream_trust",
            "no_live_writes",
        },
        "fx_send_proposal": {
            "monitor_safety_guard",
            "return_level_guard",
            "feedback_guard",
            "no_live_writes",
        },
        "master_bus_optimization_proposal": {
            "pa_feed_safety_gate",
            "reference_feed_exclusion",
            "true_peak_guard",
            "no_live_writes",
        },
        "routing_group_read_map": {"read_only_map", "no_live_writes"},
        "scene_preset_read_inventory": {
            "read_only_inventory",
            "no_snapshot_loads",
            "no_live_writes",
        },
    }

    for method_id, notes in expected_notes.items():
        method = get_processing_method(method_id)

        assert method is not None
        assert method.status in {"proposal_only", "shadow_only"}
        assert method.can_write_live is False
        assert method.live_write_kinds == ()
        assert notes <= set(method.notes)


def test_goal_capability_matrix_keeps_broad_console_writes_proposal_only():
    expected_fragments = {
        "cross_channel_unmasking": (
            "arrangement context",
            "source identity",
            "rollback",
        ),
        "effect_sends": ("monitor safety", "return", "rollback"),
        "master_bus_optimization": (
            "pa-feed safety",
            "reference-feed exclusion",
            "rollback",
        ),
        "console_fx_control": ("monitor safety", "rollback"),
        "console_scene_preset_control": (
            "read-only scene/preset inventory",
            "hardware proof",
        ),
        "console_pan_control": ("mono/phase", "rollback"),
    }

    for row_id, fragments in expected_fragments.items():
        row = get_project_goal_capability(row_id)

        assert row is not None
        assert row.status == "proposal_only"
        assert row.autonomy_mode_ceiling == "assisted"
        assert row.current_live_write_kinds == ()
        text = f"{row.blocker} {row.next_step}".lower()
        for fragment in fragments:
            assert fragment in text


def test_correction_backlog_tracks_safety_gates_for_broad_methods():
    backlog = Path(".ai/project-layer/correction_methods_backlog.md").read_text(
        encoding="utf-8"
    )

    for marker in (
        "| Cross-channel unmasking | proposal_only |",
        "arrangement-priority aware",
        "| FX / Sends | proposal_only |",
        "monitor-safety guard",
        "| Routing / Groups | read_only_map |",
        "| Scenes / Presets | read_only_inventory |",
        "| Master / Output | proposal_only |",
        "PA-feed safety",
        "reference-feed exclusion",
    ):
        assert marker in backlog
