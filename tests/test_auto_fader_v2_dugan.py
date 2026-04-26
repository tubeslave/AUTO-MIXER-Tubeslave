"""Tests for AutoFader V2 Dugan/NOM integration safeguards."""

from auto_fader_v2.balance.dugan_automixer import DuganAutomixer, DuganAutomixSettings
from auto_fader_v2.controller import AutoFaderControllerV2


class DummyMixer:
    def __init__(self):
        self.faders = {}

    def get_channel_fader(self, channel_id):
        return self.faders.get(channel_id, 0.0)


def test_dugan_integration_excludes_drum_channels():
    controller = AutoFaderControllerV2.__new__(AutoFaderControllerV2)
    controller.dugan_automixer = DuganAutomixer(
        DuganAutomixSettings(
            active_threshold_db=-60.0,
            auto_mix_depth_db=6.0,
            last_hold_enabled=True,
        )
    )
    controller.latest_dugan_targets = {}
    controller.instrument_types = {
        1: "kick",
        2: "snare",
        3: "leadVocal",
        4: "electric_guitar",
    }
    controller.dugan_excluded_instruments = {"kick", "snare"}
    controller._initial_fader_positions = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    controller.fader_ceiling_db = 0.0
    controller.fader_floor_db = -60.0
    controller.allow_fader_above_unity = False
    controller.mixer_client = DummyMixer()

    adjustments = controller._calculate_dugan_adjustments({
        1: -20.0,
        2: -20.0,
        3: -20.0,
        4: -20.0,
    })

    assert 1 not in adjustments
    assert 2 not in adjustments
    assert adjustments[3] < 0.0
    assert adjustments[4] < 0.0
