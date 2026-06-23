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
from processing_proposal_adapters import build_proposal_only_method_contracts


def _sine(freq_hz, *, sample_rate=48000, seconds=0.2, amp=0.35):
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
    return (amp * np.sin(2.0 * np.pi * float(freq_hz) * t)).astype(np.float32)


def _fresh_state(field_count=6):
    return {
        "status": "fresh",
        "field_count": field_count,
        "fresh_field_count": field_count,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _assert_top_level_denial(payload):
    assert payload["queue_action"] == "not_queueable_until_promoted"
    assert payload["can_import_to_queue"] is False
    assert payload["can_apply_live"] is False
    assert payload["queue_import_allowed"] is False
    assert payload["live_apply_allowed"] is False
    assert payload["read_only_or_dry_run_only"] is True
    assert payload["live_authority_granted"] is False
    assert payload["readiness_closure_allowed"] is False
    assert payload["mutates_mixer"] is False


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
            current_state_cache_freshness=_fresh_state(),
        ),
    )

    assert result.method_id == "automixer_phase_aware_eq_guard_v1"
    assert result.eq_action_policy == EQActionPolicy.PHASE_CANDIDATE
    assert result.low_freq_focus is True
    assert result.suggested_actions
    action = result.suggested_actions[0]
    assert action["action_type"] == "phase_alignment_candidate"
    assert action["live_write_allowed"] is False
    _assert_top_level_denial(action)
    summary = action["phase_eq_guard_safety_summary"]
    assert summary == action["evidence"]["phase_eq_guard_safety_summary"]
    assert summary["summary_type"] == "phase_aware_eq_guard_review"
    assert summary["anchor_channel"] == 1
    assert summary["candidate_channel"] == 2
    assert summary["eq_hold_policy"] == "hold_aggressive_eq_until_phase_review"
    assert summary["current_phase_audio_context_cache_no_live_authority"] is True
    _assert_top_level_denial(summary)
    correction = action["operator_correction"]
    assert correction["objective_family"] == "phase_alignment"
    assert correction["bounded_delta"]["unit"] == "phase_guard_policy"
    assert correction["future_wing_readback_contract"]["target_kind"] == (
        "group_phase_before_eq_policy"
    )
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    assert source_evidence["phase_eq_guard_safety_summary"] == summary
    assert source_evidence["current_phase_audio_context_cache_no_live_authority"] is True
    assert source_evidence["readback_required_before_promotion"] is True
    assert source_evidence["rollback_required_before_promotion"] is True
    assert result.pair_measurements[0].polarity_candidate is True
    assert result.pair_measurements[0].low_freq_sum_loss_db >= 1.0
    assert "hold aggressive EQ boosts" in " ".join(result.warnings)
    assert result.metadata["current_state_cache_freshness_status"] == "fresh"
    assert result.metadata["current_state_cache_freshness"]["fresh_field_count"] == 6
    assert result.metadata["current_phase_audio_context_cache_no_live_authority"] is True
    assert result.metadata["live_write_allowed"] is False
    _assert_top_level_denial(result.metadata)


def test_stale_phase_context_blocks_phase_candidates_and_holds_eq():
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
            current_state_cache_freshness={
                "status": "stale",
                "field_count": 6,
                "fresh_field_count": 4,
                "stale_field_count": 2,
                "stale_fields": ["phase_audio_window", "group_phase_measurements"],
                "read_only": True,
                "live_write": False,
                "queries_console": False,
            },
        ),
    )

    assert result.eq_action_policy == EQActionPolicy.PHASE_UNRELIABLE
    assert result.suggested_actions == ()
    assert result.pair_measurements == ()
    assert "phase context cache is not fresh" in " ".join(result.warnings)
    assert "phase_aware_eq_context_cache_not_fresh" in result.metadata["blockers"]
    assert result.metadata["current_state_cache_freshness_status"] == "stale"
    assert result.metadata["current_state_cache_freshness"]["stale_fields"] == [
        "phase_audio_window",
        "group_phase_measurements",
    ]
    assert result.metadata["live_write_allowed"] is False
    _assert_top_level_denial(result.metadata)


def test_missing_phase_context_blocks_phase_candidates_and_holds_eq():
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

    assert result.eq_action_policy == EQActionPolicy.PHASE_UNRELIABLE
    assert result.suggested_actions == ()
    assert result.pair_measurements == ()
    assert "phase_aware_eq_context_cache_not_fresh" in result.metadata["blockers"]
    assert result.metadata["current_state_cache_freshness_status"] == "not_provided"
    assert result.metadata["current_state_cache_freshness"]["provided"] is False
    assert "phase context cache is not fresh" in " ".join(result.warnings)
    assert result.metadata["live_write_allowed"] is False
    _assert_top_level_denial(result.metadata)


def test_live_authority_phase_context_blocks_phase_candidates_and_holds_eq():
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
            current_state_cache_freshness={
                "status": "fresh",
                "field_count": 6,
                "fresh_field_count": 6,
                "read_only": False,
                "live_write": True,
                "queries_console": True,
            },
        ),
    )

    assert result.eq_action_policy == EQActionPolicy.PHASE_UNRELIABLE
    assert result.suggested_actions == ()
    assert result.pair_measurements == ()
    assert "phase context cache has live authority" in " ".join(result.warnings)
    assert "phase_aware_eq_context_cache_live_authority_present" in result.metadata[
        "blockers"
    ]
    assert result.metadata["current_state_cache_freshness_status"] == "fresh"
    assert result.metadata["current_phase_audio_context_cache_no_live_authority"] is False
    assert result.metadata["current_state_cache_freshness"]["live_write"] is True
    assert result.metadata["live_write_allowed"] is False
    _assert_top_level_denial(result.metadata)


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
        PhaseAwareEQGuardConfig(
            sample_rate=sample_rate,
            group_id="VOCALS",
            current_state_cache_freshness=_fresh_state(),
        ),
    )

    assert result.eq_action_policy == EQActionPolicy.PHASE_UNRELIABLE
    assert result.pair_measurements[0].candidate_bleed_ratio == 0.9
    assert result.pair_measurements[0].phase_confidence < 0.42
    assert "longer or cleaner soundcheck" in " ".join(result.warnings)
    assert result.metadata["live_write_allowed"] is False
    _assert_top_level_denial(result.metadata)


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
            current_state_cache_freshness=_fresh_state(),
        ),
    )

    assert result.anchor_channel == 1
    assert result.eq_action_policy == EQActionPolicy.PHASE_CANDIDATE
    assert result.suggested_actions
    assert result.suggested_actions[0]["anchor_channel"] == 1
    assert result.suggested_actions[0]["candidate_channel"] == 2
    assert "delay_candidate" in result.suggested_actions[0]["actions"]
    _assert_top_level_denial(result.suggested_actions[0])
    _assert_top_level_denial(
        result.suggested_actions[0]["phase_eq_guard_safety_summary"]
    )
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
            current_state_cache_freshness=_fresh_state(),
        ),
    )

    assert result.metadata["channel_count"] == 2
    _assert_top_level_denial(result.metadata)
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
        _assert_top_level_denial(result.metadata)


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
            current_state_cache_freshness=_fresh_state(),
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
    _assert_top_level_denial(result.metadata)


def test_plugin_wrapper_processes_and_updates_group_mode():
    sample_rate = 48000
    audio = _sine(100.0, sample_rate=sample_rate)
    plugin = PhaseAwareEQGuardPlugin(
        PhaseAwareEQGuardConfig(
            sample_rate=sample_rate,
            group_id="DRUMS",
            low_freq_focus=True,
            current_state_cache_freshness=_fresh_state(),
        )
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
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_analyze is True
    assert method.can_propose is True
    assert method.can_write_live is False
    assert "lab_dsp" in method.pipeline_profiles
    assert "shadow_quality" in method.pipeline_profiles
    assert contracts["phase_aware_eq_guard"]["proposal_contract"]["value_type"] == "phase_eq_guard"
    assert (
        "current_phase_audio_context_cache_freshness"
        in contracts["phase_aware_eq_guard"]["proposal_contract"]["safety_evidence_required"]
    )
    assert (
        "current_phase_audio_context_cache_no_live_authority"
        in contracts["phase_aware_eq_guard"]["proposal_contract"]["safety_evidence_required"]
    )


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
