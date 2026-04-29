"""Tests for the live shared-chat mix planner."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from autofoh_safety import ChannelEQMove, MasterFaderMove
from live_shared_mix import (
    LiveSharedMixChannel,
    LiveSharedMixConfig,
    build_live_shared_mix_plan,
)


def _sine(freq_hz: float, amp: float, seconds: float = 2.0, sample_rate: int = 48000):
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / float(sample_rate)
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


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
