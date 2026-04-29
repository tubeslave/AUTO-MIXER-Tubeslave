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

    def get_channel_fader(self, channel):
        return self.faders.get(channel, 0.0)

    def set_channel_fader(self, channel, value):
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


def test_apply_auto_balance_clips_to_unity_ceiling():
    mixer = FakeMixerClient({1: -2.0})
    controller = make_controller(mixer=mixer)
    controller.channels = {
        1: FakeChannelState("leadVocal", mixer_channel=1, integrated_lufs=-30.0),
    }
    controller.auto_balance_result = {
        1: {"correction": 5.0, "locked": False},
    }

    assert controller.apply_auto_balance() is True
    assert mixer.faders[1] == pytest.approx(0.0)


def test_disabling_above_unity_reclamps_existing_ceiling():
    controller = make_controller({
        "allow_fader_above_unity": True,
        "fader_ceiling_db": 6.0,
    })

    assert controller.fader_ceiling_db == pytest.approx(6.0)

    controller.update_settings(allow_fader_above_unity=False)

    assert controller.fader_ceiling_db == pytest.approx(0.0)
