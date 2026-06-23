"""Tests for the live shared-chat mix planner."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from autofoh_safety import ChannelEQMove, ChannelFaderMove, MasterFaderMove
from live_shared_mix import (
    LiveSharedMixChannel,
    LiveSharedMixConfig,
    build_live_shared_mix_plan,
)


def _sine(freq_hz: float, amp: float, seconds: float = 2.0, sample_rate: int = 48000):
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / float(sample_rate)
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _assert_action_top_level_denial(action):
    assert action.can_apply_live is False
    assert action.can_import_to_queue is False
    assert action.queue_import_allowed is False
    assert action.live_apply_allowed is False
    assert action.read_only_or_dry_run_only is True
    assert action.live_authority_granted is False
    assert action.readiness_closure_allowed is False
    assert action.queue_action == "not_queueable_until_promoted"
    assert action.live_write is False
    assert action.queue_mutation is False
    assert action.mutates_mixer is False


def _assert_payload_top_level_denial(payload):
    assert payload["queue_action"] == "not_queueable_until_promoted"
    assert payload["can_import_to_queue"] is False
    assert payload["can_apply_live"] is False
    assert payload["queue_import_allowed"] is False
    assert payload["live_apply_allowed"] is False
    assert payload["read_only_or_dry_run_only"] is True
    assert payload["live_authority_granted"] is False
    assert payload["readiness_closure_allowed"] is False
    assert payload["live_write"] is False
    assert payload["queue_mutation"] is False
    assert payload["mutates_mixer"] is False


def _assert_correction_authority_denial(correction):
    assert correction["read_only_or_dry_run_only"] is True
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    boundary = correction["live_authority_boundary"]
    assert boundary["live_authority_granted"] is False
    assert boundary["full_autonomous_ready"] is False
    assert boundary["readiness_closure_allowed"] is False
    assert boundary["wing_proof_required"] is True
    assert boundary["hardware_proof_required"] is True
    assert boundary["live_apply_allowed"] is False
    assert boundary["queue_import_allowed"] is False


def _channel(channel_id, name, role, audio, *, fader_db=-6.0, auto=True, priority=0.8):
    return LiveSharedMixChannel(
        channel_id=channel_id,
        name=name,
        role=role,
        stems=(role.upper(),),
        priority=priority,
        audio=audio,
        sample_rate=48000,
        fader_db=fader_db,
        muted=False,
        auto_corrections_enabled=auto,
        raw_settings={
            "fader_db": fader_db,
            "muted": False,
            "input_routing": {"main_group": "MOD", "main_channel": channel_id},
            "main_send": {"on": 1, "level_db": 0.0, "pre": 0},
        },
    )


def test_low_end_anchor_plans_source_eq_not_master_eq():
    config = LiveSharedMixConfig(analysis_window_sec=1.0, max_actions_per_pass=8)
    channels = [
        _channel(1, "KICK", "kick", _sine(72.0, 0.9)),
        _channel(2, "BASS", "bass", _sine(88.0, 0.7)),
    ]

    plan = build_live_shared_mix_plan(channels, 48000, config=config)

    assert any(isinstance(action, ChannelEQMove) for action in plan.actions)
    assert all(not isinstance(action, MasterFaderMove) for action in plan.actions)
    assert "source/stem fixes before master processing" in plan.report["rules"]


def test_lead_masking_cuts_competing_music_channel():
    config = LiveSharedMixConfig(analysis_window_sec=1.0, max_actions_per_pass=8)
    channels = [
        _channel(1, "LEAD VOX", "lead_vocal", _sine(2200.0, 0.15)),
        _channel(2, "GTR", "guitars", _sine(2800.0, 0.9)),
    ]

    plan = build_live_shared_mix_plan(channels, 48000, config=config)

    guitar_eq = [
        action
        for action in plan.actions
        if isinstance(action, ChannelEQMove) and action.channel_id == 2
    ]
    assert guitar_eq
    assert guitar_eq[0].gain_db < 0.0
    _assert_action_top_level_denial(guitar_eq[0])
    correction = guitar_eq[0].operator_correction
    assert correction["objective_family"] == "eq"
    assert correction["future_wing_readback_contract"]["target_kind"] == "channel_eq_band"
    assert correction["bounded_delta"]["delta_db"] == -1.0
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_payload_top_level_denial(source_evidence)
    summary = source_evidence["live_shared_mix_safety_summary"]
    assert summary["summary_type"] == "live_shared_mix_review"
    assert summary["value_type"] == "eq"
    assert summary["target_kind"] == "channel_eq_band"
    assert summary["target"] == 2
    assert summary["bounded_gain_delta"] is True
    assert summary["feedback_safety"] == "required_before_apply"
    assert summary["true_peak_headroom"] == "required_before_apply"
    _assert_payload_top_level_denial(summary)
    assert summary["can_authorize_live"] is False
    _assert_correction_authority_denial(correction)
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "bounded_gain_delta"
    ] is True


def test_lead_layer_fader_support_has_operator_correction_contract():
    config = LiveSharedMixConfig(
        analysis_window_sec=1.0,
        max_actions_per_pass=8,
        mirror_eq_enabled=False,
    )
    channels = [
        _channel(1, "LEAD VOX", "lead_vocal", _sine(2200.0, 0.45)),
        _channel(2, "LEAD DBL", "lead_vocal", _sine(2200.0, 0.04)),
    ]

    plan = build_live_shared_mix_plan(channels, 48000, config=config)

    fader_moves = [
        action
        for action in plan.actions
        if isinstance(action, ChannelFaderMove) and action.channel_id == 2
    ]
    assert fader_moves
    move = fader_moves[0]
    _assert_action_top_level_denial(move)
    correction = move.operator_correction
    assert move.delta_db > 0.0
    assert correction["objective_family"] == "fader_balance"
    assert correction["future_wing_readback_contract"]["target_kind"] == (
        "shared_mix_channel_or_stem_balance"
    )
    assert correction["bounded_delta"]["delta_db"] == round(move.delta_db, 1)
    assert correction["bounded_delta"]["parameter_deltas"]["target_plane_lufs"] == -22.0
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_payload_top_level_denial(source_evidence)
    summary = source_evidence["live_shared_mix_safety_summary"]
    assert summary["summary_type"] == "live_shared_mix_review"
    assert summary["value_type"] == "fader_balance"
    assert summary["target_kind"] == "shared_mix_channel_or_stem_balance"
    assert summary["target"] == 2
    assert summary["ayaic_level_plane"]["target_lufs"] == -22.0
    assert summary["fader_ceiling"] == "target_clamped_at_0_dbfs"
    assert summary["bounded_fader_delta"] is True
    _assert_payload_top_level_denial(summary)
    assert summary["can_authorize_live"] is False
    _assert_correction_authority_denial(correction)
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "ayaic_level_plane"
    ]["target_lufs"] == -22.0
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "source_or_stem_contributor"
    ] == "lead_vocal"
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "bounded_fader_delta"
    ] is True
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "rollback_fader_readback"
    ] == "required_before_apply"


def test_master_reference_overload_plans_main_cut_only():
    config = LiveSharedMixConfig(
        analysis_window_sec=1.0,
        max_actions_per_pass=8,
        master_peak_ceiling_db=-3.0,
        master_max_cut_db=1.0,
    )
    channels = [
        _channel(1, "LEAD VOX", "lead_vocal", _sine(2200.0, 0.15), auto=False),
    ]
    master = np.column_stack([_sine(1000.0, 0.9), _sine(1000.0, 0.9)])

    plan = build_live_shared_mix_plan(
        channels,
        48000,
        config=config,
        master_audio=master,
        master_current_fader_db=-2.0,
    )

    master_actions = [action for action in plan.actions if isinstance(action, MasterFaderMove)]
    assert len(master_actions) == 1
    assert master_actions[0].target_db == -3.0
    _assert_action_top_level_denial(master_actions[0])
    correction = master_actions[0].operator_correction
    assert correction["objective_family"] == "master_output"
    assert correction["future_wing_readback_contract"]["target_kind"] == "main_bus_fader"
    assert correction["bounded_delta"]["delta_db"] == -1.0
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_payload_top_level_denial(source_evidence)
    summary = source_evidence["live_shared_mix_safety_summary"]
    _assert_payload_top_level_denial(summary)
    assert summary["can_authorize_live"] is False
    _assert_correction_authority_denial(correction)
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "pa_feed_safe_for_master_change"
    ] == "requires_operator_confirmation"
    assert plan.report["master"]["action"] == "master_fader_cut"


def test_mirror_eq_cuts_masker_and_boosts_masked_priority_source():
    config = LiveSharedMixConfig(
        analysis_window_sec=1.0,
        max_actions_per_pass=2,
        mirror_eq_enabled=True,
        mirror_eq_max_actions_per_pass=4,
    )
    channels = [
        _channel(1, "LEAD VOX", "lead_vocal", _sine(2500.0, 0.45), priority=1.0),
        _channel(2, "GTR", "guitars", _sine(2500.0, 0.45), priority=0.45),
    ]

    plan = build_live_shared_mix_plan(channels, 48000, config=config)

    mirror_actions = [
        action
        for action in plan.actions
        if isinstance(action, ChannelEQMove) and action.reason.startswith("Mirror EQ")
    ]
    guitar_cuts = [action for action in mirror_actions if action.channel_id == 2 and action.gain_db < 0.0]
    lead_boosts = [action for action in mirror_actions if action.channel_id == 1 and action.gain_db > 0.0]
    assert guitar_cuts
    assert lead_boosts
    assert guitar_cuts[0].q == 4.0
    assert lead_boosts[0].q == 2.0
    assert plan.report["mirror_eq"]["mode"] == "cross_adaptive_full_mirror_eq"
    assert "mirror EQ: cut masker" in " ".join(plan.report["rules"])


def test_mirror_eq_can_be_disabled():
    config = LiveSharedMixConfig(
        analysis_window_sec=1.0,
        max_actions_per_pass=12,
        mirror_eq_enabled=False,
    )
    channels = [
        _channel(1, "LEAD VOX", "lead_vocal", _sine(2500.0, 0.45), priority=1.0),
        _channel(2, "GTR", "guitars", _sine(2500.0, 0.45), priority=0.45),
    ]

    plan = build_live_shared_mix_plan(channels, 48000, config=config)

    assert plan.report["mirror_eq"] == {"enabled": False, "reason": "disabled"}
    assert all(
        not (
            isinstance(action, ChannelEQMove)
            and action.reason.startswith("Mirror EQ")
        )
        for action in plan.actions
    )
