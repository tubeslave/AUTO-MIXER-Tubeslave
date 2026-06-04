import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from phase_aware_eq_guard import (
    EQActionPolicy,
    GroupEnableMode,
    PhaseAwareEQGuardConfig,
    PhaseAwareEQGuardPlugin,
    PhaseChannelInput,
    analyze_phase_aware_eq_guard,
)


def _sine(freq_hz, *, sample_rate=48000, seconds=0.2, amp=0.35):
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
    return (amp * np.sin(2.0 * np.pi * float(freq_hz) * t)).astype(np.float32)


def test_low_frequency_phase_cancellation_blocks_aggressive_eq_boost():
    sample_rate = 48000
    kick_anchor = _sine(70.0, sample_rate=sample_rate)
    kick_outer = -kick_anchor.copy()

    result = analyze_phase_aware_eq_guard(
        [
            PhaseChannelInput(1, kick_anchor, group_id="DRUMS", weight=1.0, label="kick in"),
            PhaseChannelInput(2, kick_outer, group_id="DRUMS", weight=0.6, label="kick out"),
        ],
        PhaseAwareEQGuardConfig(
            sample_rate=sample_rate,
            group_id="DRUMS",
            low_freq_focus=True,
            low_freq_loss_threshold_db=1.0,
        ),
    )

    assert result.method_id == "automixer_phase_aware_eq_guard_v1"
    assert result.eq_action_policy == EQActionPolicy.PHASE_CANDIDATE
    assert result.low_freq_focus is True
    assert result.suggested_actions
    assert result.suggested_actions[0]["action_type"] == "phase_alignment_candidate"
    assert result.pair_measurements[0].polarity_candidate is True
    assert result.pair_measurements[0].low_freq_sum_loss_db >= 1.0
    assert "hold aggressive EQ boosts" in " ".join(result.warnings)


def test_high_bleed_lowers_phase_confidence_and_holds_eq():
    sample_rate = 48000
    rng = np.random.default_rng(123)
    sample_count = int(sample_rate * 0.2)
    anchor = rng.normal(0.0, 0.1, sample_count).astype(np.float32)
    unrelated_bleed = rng.normal(0.0, 0.1, sample_count).astype(np.float32)

    result = analyze_phase_aware_eq_guard(
        [
            PhaseChannelInput(1, anchor, group_id="VOCALS", weight=1.0),
            PhaseChannelInput(2, unrelated_bleed, group_id="VOCALS", weight=0.5, bleed_ratio=0.9),
        ],
        PhaseAwareEQGuardConfig(sample_rate=sample_rate, group_id="VOCALS"),
    )

    assert result.eq_action_policy == EQActionPolicy.PHASE_UNRELIABLE
    assert result.pair_measurements[0].candidate_bleed_ratio == 0.9
    assert result.pair_measurements[0].phase_confidence < 0.42
    assert "longer or cleaner soundcheck" in " ".join(result.warnings)


def test_phase_lock_keeps_high_weight_anchor_and_only_proposes_candidate_moves():
    sample_rate = 48000
    rng = np.random.default_rng(456)
    anchor = rng.normal(0.0, 0.2, int(sample_rate * 0.2)).astype(np.float32)
    anchor = np.convolve(anchor, [0.25, 0.5, 0.25], mode="same").astype(np.float32)
    delayed = np.zeros_like(anchor)
    delayed[96:] = anchor[:-96]  # 2ms at 48kHz

    result = analyze_phase_aware_eq_guard(
        [
            PhaseChannelInput(1, anchor, group_id="BASS", weight=3.0, gain_db=0.0, label="bass DI"),
            PhaseChannelInput(2, delayed, group_id="BASS", weight=0.4, gain_db=0.0, label="bass mic"),
        ],
        PhaseAwareEQGuardConfig(
            sample_rate=sample_rate,
            group_id="BASS",
            group_enable_mode=GroupEnableMode.GROUP_PHASE_LOCK,
            low_freq_focus=True,
            delay_candidate_threshold_ms=0.3,
        ),
    )

    assert result.anchor_channel == 1
    assert result.eq_action_policy == EQActionPolicy.PHASE_CANDIDATE
    assert result.suggested_actions
    assert result.suggested_actions[0]["anchor_channel"] == 1
    assert result.suggested_actions[0]["candidate_channel"] == 2
    assert "delay_candidate" in result.suggested_actions[0]["actions"]
    assert "phase-lock mode" in " ".join(result.warnings)


def test_internal_only_group_ignores_unrelated_channels_even_when_they_are_louder():
    sample_rate = 48000
    drum_a = _sine(180.0, sample_rate=sample_rate, amp=0.2)
    drum_b = -drum_a
    loud_guitar = _sine(440.0, sample_rate=sample_rate, amp=0.9)

    result = analyze_phase_aware_eq_guard(
        [
            PhaseChannelInput(1, drum_a, group_id="DRUMS", weight=1.0),
            PhaseChannelInput(2, drum_b, group_id="DRUMS", weight=0.8),
            PhaseChannelInput(9, loud_guitar, group_id="GTR", weight=10.0),
        ],
        PhaseAwareEQGuardConfig(
            sample_rate=sample_rate,
            group_id="DRUMS",
            group_enable_mode=GroupEnableMode.GROUP_INTERNAL_ONLY,
            low_freq_focus=True,
            low_freq_loss_threshold_db=1.0,
        ),
    )

    assert result.metadata["channel_count"] == 2
    assert result.anchor_channel in {1, 2}
    assert all(measurement.candidate_channel in {1, 2} for measurement in result.pair_measurements)
    assert result.eq_action_policy == EQActionPolicy.PHASE_CANDIDATE


def test_group_off_and_global_disabled_return_disabled_policy_without_actions():
    audio = _sine(100.0)

    for config in (
        PhaseAwareEQGuardConfig(global_enabled=False),
        PhaseAwareEQGuardConfig(group_enable_mode=GroupEnableMode.GROUP_OFF),
    ):
        result = analyze_phase_aware_eq_guard(
            [
                PhaseChannelInput(1, audio, group_id="DRUMS"),
                PhaseChannelInput(2, audio, group_id="DRUMS"),
            ],
            config,
        )

        assert result.eq_action_policy == EQActionPolicy.GROUP_DISABLED
        assert result.suggested_actions == ()
        assert result.metadata["live_write_allowed"] is False


def test_gain_weight_group_gain_and_latency_metadata_are_reported():
    sample_rate = 48000
    anchor = _sine(220.0, sample_rate=sample_rate, amp=0.25)
    layer = _sine(220.0, sample_rate=sample_rate, amp=0.12)

    result = analyze_phase_aware_eq_guard(
        [
            PhaseChannelInput(1, anchor, group_id="VOCALS", weight=2.0, gain_db=-3.0),
            PhaseChannelInput(2, layer, group_id="VOCALS", weight=0.5, gain_db=-6.0),
        ],
        PhaseAwareEQGuardConfig(
            sample_rate=sample_rate,
            group_id="VOCALS",
            lookahead_buffer_ms=5.0,
            report_latency=True,
        ),
    )

    assert result.anchor_channel == 1
    assert result.group_gain_db < 0.0
    assert result.logic_latency_ms == 5.0
    assert result.report_latency is True
    assert result.is_group_graph is True
    assert result.is_freq_graph is True
    assert result.metadata["pi_surface_coverage"]["group_gain"] is True
    assert result.metadata["pi_surface_coverage"]["weight"] is True


def test_plugin_wrapper_processes_and_updates_group_mode():
    sample_rate = 48000
    audio = _sine(100.0, sample_rate=sample_rate)
    plugin = PhaseAwareEQGuardPlugin(
        PhaseAwareEQGuardConfig(sample_rate=sample_rate, group_id="DRUMS", low_freq_focus=True)
    )

    first = plugin.process(
        [
            PhaseChannelInput(1, audio, group_id="DRUMS", weight=1.0),
            PhaseChannelInput(2, -audio, group_id="DRUMS", weight=0.8),
        ]
    )
    updated = plugin.update_config(group_enable_mode="GROUP_OFF")
    second = plugin.process(
        [
            PhaseChannelInput(1, audio, group_id="DRUMS"),
            PhaseChannelInput(2, -audio, group_id="DRUMS"),
        ]
    )

    assert first.eq_action_policy == EQActionPolicy.PHASE_CANDIDATE
    assert updated.group_enable_mode == GroupEnableMode.GROUP_OFF
    assert second.eq_action_policy == EQActionPolicy.GROUP_DISABLED
    assert plugin.last_result == second


def test_phase_aware_eq_guard_registry_entry_is_proposal_only():
    from processing_registry import get_processing_method

    method = get_processing_method("phase_aware_eq_guard")

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_analyze is True
    assert method.can_propose is True
    assert method.can_write_live is False
    assert "lab_dsp" in method.pipeline_profiles
    assert "shadow_quality" in method.pipeline_profiles


def test_phase_aware_eq_guard_module_is_transport_free():
    source = Path("backend/phase_aware_eq_guard.py").read_text()

    forbidden_fragments = [
        "WingClient",
        "mixer_client",
        "pythonosc",
        "python-osc",
        "set_eq",
        "set_delay",
        "set_polarity",
        "set_fader",
        "set_gain",
        ".send(",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in source
