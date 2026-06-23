import pytest

from auto_fader import (
    AutoFaderController,
    combine_lufs,
    normalize_level_plane_instrument,
)


class FakeIntegratedMeter:
    def __init__(self, lufs):
        self.lufs = lufs

    def get_integrated_lufs(self):
        return self.lufs

    def get_block_count(self):
        return 1


class FakeChannelState:
    def __init__(self, instrument_type, mixer_channel, integrated_lufs):
        self.instrument_type = instrument_type
        self.mixer_channel = mixer_channel
        self.integrated_lufs = integrated_lufs
        self.integrated_lufs_meter = FakeIntegratedMeter(integrated_lufs)
        self.locked = False
        self.spectral_centroid = 0.0
        self.band_energy = {}
        self.band_energy_max = {}
        self.current_fader = 0.0


class FakeMixerClient:
    is_connected = True

    def __init__(self, faders):
        self.faders = dict(faders)
        self.set_calls = []

    def get_channel_fader(self, channel):
        return self.faders.get(channel, 0.0)

    def set_channel_fader(self, channel, value):
        self.set_calls.append((channel, value))
        self.faders[channel] = value


def make_controller(settings=None, mixer=None):
    config = {
        "automation": {
            "auto_fader": {
                "level_plane_balance_enabled": True,
                "fader_ceiling_db": 0.0,
                "level_plane_max_boost_db": 4.0,
                **(settings or {}),
            }
        }
    }
    return AutoFaderController(mixer_client=mixer, config=config)


def test_level_plane_targets_keep_vocal_in_front_of_guitar():
    controller = make_controller()

    lead_target, lead_meta = controller._resolve_level_plane_target_lufs("leadVocal")
    guitar_target, guitar_meta = controller._resolve_level_plane_target_lufs("electricGuitar")

    assert normalize_level_plane_instrument("leadVocal") == "lead_vocal"
    assert lead_meta["method"] == "level_plane"
    assert guitar_meta["method"] == "level_plane"
    assert lead_target == pytest.approx(-22.0)
    assert guitar_target == pytest.approx(-25.0)
    assert lead_target > guitar_target


def test_level_plane_second_pass_trims_multi_channel_groups():
    controller = make_controller({"level_plane_group_trim_limit_db": 3.0})
    controller.auto_balance_pass = 2
    active_channels = {
        1: {"pre_fader": -20.0, "ideal_fader": -5.0, "level_plane_group": "harmonic"},
        2: {"pre_fader": -20.0, "ideal_fader": -5.0, "level_plane_group": "harmonic"},
    }

    controller._apply_level_plane_group_pass(active_channels)

    assert combine_lufs([-25.0, -25.0]) == pytest.approx(-21.9897, abs=0.001)
    assert active_channels[1]["level_plane_group_trim_db"] == pytest.approx(-3.0)
    assert active_channels[2]["level_plane_group_trim_db"] == pytest.approx(-3.0)
    assert active_channels[1]["ideal_fader"] == pytest.approx(-8.0)
    assert active_channels[2]["ideal_fader"] == pytest.approx(-8.0)


def test_auto_balance_respects_unity_ceiling_and_boost_limit():
    mixer = FakeMixerClient({1: -10.0})
    controller = make_controller(mixer=mixer)
    controller.channels = {
        1: FakeChannelState("leadVocal", mixer_channel=1, integrated_lufs=-50.0),
    }
    controller.auto_balance_pass = 1

    controller._compute_auto_balance()
    result = controller.auto_balance_result[1]

    assert result["correction"] == pytest.approx(4.0)
    assert result["target_lufs"] == pytest.approx(-22.0)
    assert result["balance_method"] == "level_plane"


def _assert_operator_denial(payload):
    assert payload["can_apply_live"] is False
    assert payload["can_import_to_queue"] is False
    assert payload["queue_import_allowed"] is False
    assert payload["live_apply_allowed"] is False
    assert payload["read_only_or_dry_run_only"] is True
    assert payload["live_authority_granted"] is False
    assert payload["readiness_closure_allowed"] is False
    assert payload["queue_action"] == "not_queueable_until_promoted"
    assert payload["live_write"] is False
    assert payload["queue_mutation"] is False
    assert payload["mutates_mixer"] is False


def _assert_top_level_denial(payload):
    assert payload["applied"] is False
    assert payload["blocked"] is True
    assert payload["calls"] == []
    _assert_operator_denial(payload)


def test_apply_auto_balance_is_proposal_only_and_clips_to_unity_ceiling():
    mixer = FakeMixerClient({1: -2.0})
    controller = make_controller(mixer=mixer)
    controller.channels = {
        1: FakeChannelState("leadVocal", mixer_channel=1, integrated_lufs=-30.0),
    }
    controller.auto_balance_result = {
        1: {
            "correction": 5.0,
            "target_lufs": -22.0,
            "integrated_lufs": -30.0,
            "balance_method": "level_plane",
            "instrument_key": "lead_vocal",
            "level_plane_group": "vocals",
            "locked": False,
        },
    }

    assert controller.apply_auto_balance() is False
    assert mixer.faders[1] == pytest.approx(-2.0)
    assert mixer.set_calls == []

    payload = controller.last_auto_balance_apply_result
    _assert_top_level_denial(payload)
    assert payload["reason"] == "auto_fader_v1_auto_balance_apply_quarantined"
    assert payload["blocked_calls"] == [
        {
            "cmd": "set_channel_fader",
            "channel": 1,
            "value_db": 0.0,
            "current_db": -2.0,
            "delta_db": 2.0,
            "semantic": "auto_balance_apply_quarantined",
            "audio_channel": 1,
        }
    ]
    proposal = payload["proposals"][0]
    _assert_operator_denial(proposal)
    _assert_operator_denial(proposal["evidence"])
    assert proposal["proposed_action"]["target_fader_db"] == 0.0
    assert proposal["evidence"]["ayaic_level_plane"]["target_lufs"] == -22.0
    assert proposal["evidence"]["rollback_fader_readback"]["value_db"] == -2.0
    correction = proposal["operator_correction"]
    assert correction["objective_family"] == "fader_balance"
    assert correction["future_wing_readback_contract"]["target_kind"] == (
        "channel_fader_balance"
    )
    assert correction["bounded_delta"]["old_value"] == -2.0
    assert correction["bounded_delta"]["proposed_value"] == 0.0
    assert correction["bounded_delta"]["delta_db"] == 2.0
    assert correction["bounded_delta"]["parameter_deltas"]["target_plane_lufs"] == -22.0
    assert "auto_fader_v1_direct_apply_quarantined" in correction["safety_blockers"]
    assert correction["read_only_or_dry_run_only"] is True
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_operator_denial(source_evidence)
    assert source_evidence["ayaic_level_plane"]["target_lufs"] == -22.0
    assert source_evidence["rollback_fader_readback"]["value_db"] == -2.0
    assert controller.get_status()["last_auto_balance_apply_result"] == payload


def test_realtime_fader_update_is_proposal_only_and_preserves_current_fader():
    mixer = FakeMixerClient({7: -10.0})
    controller = make_controller(mixer=mixer)
    state = FakeChannelState("leadVocal", mixer_channel=7, integrated_lufs=-28.0)
    state.current_fader = -10.0
    controller.channels = {1: state}

    payload = controller._realtime_fader_apply_denial(
        {7: -8.2},
        {
            1: {
                "true_peak": -6.0,
                "baseline_lufs": -28.0,
                "baseline_peak": -8.0,
                "pre_fader_lufs": -26.5,
                "current_fader": -10.0,
                "target_fader": -8.2,
                "correction": 1.8,
                "fader_range": 3.0,
                "instrument_type": "leadVocal",
                "bleed_ratio": 0.0,
                "bleed_source": None,
            }
        },
        reason="auto_fader_v1_realtime_fader_apply_quarantined",
    )
    controller.last_realtime_fader_apply_result = payload

    _assert_top_level_denial(payload)
    assert mixer.faders[7] == pytest.approx(-10.0)
    assert mixer.set_calls == []
    assert state.current_fader == pytest.approx(-10.0)
    assert payload["reason"] == "auto_fader_v1_realtime_fader_apply_quarantined"
    assert payload["realtime_fader_result_preserved"] is True
    assert payload["blocked_calls"] == [
        {
            "cmd": "set_channel_fader",
            "channel": 7,
            "value_db": -8.2,
            "current_db": -10.0,
            "delta_db": 1.8,
            "semantic": "realtime_fader_adjustment_quarantined",
            "audio_channel": 1,
        }
    ]
    proposal = payload["proposals"][0]
    _assert_operator_denial(proposal)
    _assert_operator_denial(proposal["evidence"])
    assert proposal["evidence"]["realtime_fader_evidence"]["baseline_lufs"] == -28.0
    assert proposal["evidence"]["true_peak_headroom"]["current_true_peak_dbtp"] == -6.0
    assert proposal["evidence"]["rollback_fader_readback"]["value_db"] == -10.0
    correction = proposal["operator_correction"]
    assert correction["objective_family"] == "fader_balance"
    assert correction["bounded_delta"]["old_value"] == -10.0
    assert correction["bounded_delta"]["proposed_value"] == -8.2
    assert correction["bounded_delta"]["delta_db"] == 1.8
    assert correction["future_wing_readback_contract"]["target_kind"] == (
        "channel_fader_balance"
    )
    assert "realtime_fader_apply_remains_proposal_only" in correction[
        "safety_blockers"
    ]
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_operator_denial(source_evidence)
    assert source_evidence["realtime_fader_evidence"]["baseline_lufs"] == -28.0
    assert source_evidence["rollback_fader_readback"]["value_db"] == -10.0
    assert controller.get_status()["last_realtime_fader_apply_result"] == payload


def test_apply_auto_balance_missing_result_is_complete_operator_denial():
    mixer = FakeMixerClient({1: -2.0})
    controller = make_controller(mixer=mixer)

    assert controller.apply_auto_balance() is False

    payload = controller.last_auto_balance_apply_result
    _assert_top_level_denial(payload)
    assert payload["reason"] == "missing_auto_balance_result"
    assert payload["blocked_calls"] == []
    assert mixer.set_calls == []


def test_disabling_above_unity_reclamps_existing_ceiling():
    controller = make_controller({
        "allow_fader_above_unity": True,
        "fader_ceiling_db": 6.0,
    })

    assert controller.fader_ceiling_db == pytest.approx(6.0)

    controller.update_settings(allow_fader_above_unity=False)

    assert controller.fader_ceiling_db == pytest.approx(0.0)


def test_handle_external_fader_change_updates_matching_channel_within_limits():
    controller = make_controller()
    state = FakeChannelState("leadVocal", mixer_channel=7, integrated_lufs=-20.0)
    state.current_fader = -12.0
    controller.channels = {1: state}
    controller.realtime_enabled = True

    controller.handle_external_fader_change(7, -8.5)

    assert state.current_fader == pytest.approx(-8.5)


def test_handle_external_fader_change_clamps_to_controller_limits():
    controller = make_controller({
        "allow_fader_above_unity": True,
        "fader_ceiling_db": 6.0,
        "fader_floor_db": -24.0,
    })
    state = FakeChannelState("drums", mixer_channel=4, integrated_lufs=-20.0)
    state.current_fader = -5.0
    controller.channels = {1: state}
    controller.realtime_enabled = True

    controller.handle_external_fader_change(4, 10.0)
    assert state.current_fader == pytest.approx(6.0)

    controller.handle_external_fader_change(4, -200.0)
    assert state.current_fader == pytest.approx(-24.0)


def test_handle_external_fader_change_ignores_invalid_values():
    controller = make_controller()
    state = FakeChannelState("leadVocal", mixer_channel=2, integrated_lufs=-20.0)
    state.current_fader = -10.0
    controller.channels = {1: state}
    controller.realtime_enabled = True

    controller.handle_external_fader_change(2, float("nan"))
    controller.handle_external_fader_change(2, float("inf"))
    controller.handle_external_fader_change(2, "bad-value")

    assert state.current_fader == pytest.approx(-10.0)


def test_handle_external_fader_change_is_noop_when_realtime_disabled():
    controller = make_controller()
    state = FakeChannelState("leadVocal", mixer_channel=3, integrated_lufs=-20.0)
    state.current_fader = -8.0
    controller.channels = {1: state}
    controller.realtime_enabled = False

    controller.handle_external_fader_change(3, -2.0)

    assert state.current_fader == pytest.approx(-8.0)
