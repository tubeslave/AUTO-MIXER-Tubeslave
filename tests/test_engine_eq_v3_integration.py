"""Интеграция EQ v3 в AutoSoundcheckEngine: прекомпьют + fallback на пресеты."""

import numpy as np

from auto_soundcheck_engine import AutoSoundcheckEngine

SR = 48000


class _FakeCapture:
    """Минимальный AudioCapture: отдаёт заготовленные буферы."""

    def __init__(self, buffers):
        self.buffers = buffers

    def get_buffer(self, channel, num_samples):
        x = self.buffers.get(channel)
        if x is None:
            return np.zeros(0, dtype=np.float32)
        return x[-num_samples:].astype(np.float32)


def _engine_with_audio(buffers, roles):
    engine = AutoSoundcheckEngine(
        selected_channels=list(buffers), num_channels=8, auto_discover=False,
    )
    engine.audio_capture = _FakeCapture(buffers)
    engine.sample_rate = SR
    for ch, role in roles.items():
        info = engine.channels.get(ch)
        if info is None:
            from auto_soundcheck_engine import ChannelInfo
            info = ChannelInfo(channel=ch)
            engine.channels[ch] = info
        info.has_signal = True
        info.auto_corrections_enabled = True
        info.source_role = role
    return engine


def _ring_noise(res_freq=445.0, n_sec=1.0):
    rng = np.random.default_rng(7)
    t = np.arange(int(n_sec * SR)) / SR
    return (0.1 * rng.normal(0, 1, len(t)) + 0.4 * np.sin(2 * np.pi * res_freq * t))


class TestEngineEQv3:
    def test_precompute_returns_proposals(self):
        engine = _engine_with_audio(
            {2: _ring_noise(), 3: _ring_noise(300.0)},
            {2: "guitar", 3: "snare"},
        )
        proposals = engine._compute_eq_v3_proposals()
        assert set(proposals) == {2, 3}
        assert proposals[2].to_dict()["method"] == "eq_v3_soundcheck"

    def test_no_capture_falls_back_empty(self):
        engine = AutoSoundcheckEngine(
            selected_channels=[2], num_channels=8, auto_discover=False,
        )
        engine.audio_capture = None
        assert engine._compute_eq_v3_proposals() == {}

    def test_short_buffers_skipped(self):
        engine = _engine_with_audio(
            {2: np.zeros(128)},
            {2: "guitar"},
        )
        assert engine._compute_eq_v3_proposals() == {}

    def test_channels_without_signal_excluded(self):
        engine = _engine_with_audio({2: _ring_noise()}, {2: "guitar"})
        engine.channels[2].has_signal = False
        assert engine._compute_eq_v3_proposals() == {}
