"""Tests for backend/observation_mixer.py."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from observation_mixer import ObservationMixerClient


class FakeMixer:
    def __init__(self):
        self.is_connected = True
        self.state = {"/ch/1/in/set/trim": 1.0, "/ch/1/fdr": -5.0}
        self.sent_queries = []

    def send(self, address, *args):
        self.sent_queries.append((address, args))
        if not args:
            return {"query": address}
        raise AssertionError("Write should not reach the real mixer")

    def get_channel_gain(self, channel):
        return self.state.get(f"/ch/{channel}/in/set/trim", 0.0)

    def get_channel_fader(self, channel):
        return self.state.get(f"/ch/{channel}/fdr", -144.0)

    def get_eq_on(self, channel):
        return 0

    def get_eq_band_gain(self, channel, band):
        return 0.0

    def get_eq_band_frequency(self, channel, band):
        return 1000.0

    def get_compressor_on(self, channel):
        return 0

    def get_compressor_gain(self, channel):
        return 0.0

    def get_mute(self, channel):
        return False


def test_observation_mixer_intercepts_writes_and_updates_shadow_state():
    fake = FakeMixer()
    operations = []
    mixer = ObservationMixerClient(fake, on_command=operations.append)

    assert mixer.set_channel_gain(1, 6.0) is True
    assert mixer.set_channel_fader(1, -2.5) is True

    assert mixer.get_channel_gain(1) == 6.0
    assert mixer.get_channel_fader(1) == -2.5
    assert len(operations) == 2
    assert fake.sent_queries == []


def test_observation_mixer_forwards_queries_but_not_write_sends():
    fake = FakeMixer()
    mixer = ObservationMixerClient(fake)

    result = mixer.send("/ch/1/fdr")
    assert result == {"query": "/ch/1/fdr"}

    assert mixer.send("/ch/1/fdr", -3.0) is True
    assert fake.sent_queries == [("/ch/1/fdr", ())]
    assert mixer.state["/ch/1/fdr"] == -3.0


def test_observation_mixer_summary_groups_operations():
    fake = FakeMixer()
    mixer = ObservationMixerClient(fake)

    mixer.set_channel_gain(1, 3.0)
    mixer.set_eq_on(1, 1)
    mixer.send("/main/1/fdr", -1.0)

    summary = mixer.get_summary()
    assert summary["total_operations"] == 3
    assert summary["channels"]["1"]["count"] == 2
    assert summary["channels"]["global"]["count"] == 1
