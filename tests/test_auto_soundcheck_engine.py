"""Tests for AutoSoundcheckEngine safety wiring."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from auto_soundcheck_engine import AutoSoundcheckEngine
from observation_mixer import ObservationMixerClient


class FakeMixer:
    is_connected = True
    state = {}
    callbacks = {}

    def set_channel_gain(self, channel, value):
        raise AssertionError("Observation mode must intercept writes")


def test_selected_channels_drive_engine_channel_iterator():
    engine = AutoSoundcheckEngine(
        selected_channels=[4, 2, 4],
        num_channels=8,
        auto_discover=False,
    )

    assert engine._iter_channels() == [2, 4]
    assert engine._configured_channel_count() == 2


def test_observation_mode_wraps_mixer_and_intercepts_writes():
    observed = []
    engine = AutoSoundcheckEngine(
        observe_only=True,
        auto_discover=False,
        on_observation=observed.append,
    )
    engine.mixer_client = FakeMixer()

    engine._activate_observation_mode()

    assert isinstance(engine.mixer_client, ObservationMixerClient)
    assert engine.mixer_client.set_channel_gain(1, 5.0) is True
    assert any("operation" in item for item in observed)
