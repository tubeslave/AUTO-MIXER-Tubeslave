"""Inverse EQ Eight norm ↔ dB/Hz/Q (readback vs set mapping)."""

import math

from ableton_client import (
    _eq_norm_to_freq_hz,
    _eq_norm_to_gain_db,
    _eq_norm_to_q,
)


def test_gain_db_roundtrip():
    for db in (-15.0, -6.0, 0.0, 6.0, 12.0, 15.0):
        n = max(0.0, min(1.0, (db + 15.0) / 30.0))
        assert abs(_eq_norm_to_gain_db(n) - db) < 1e-6


def test_freq_hz_roundtrip_mid_range():
    for hz in (50.0, 200.0, 1000.0, 5000.0, 12000.0):
        n = math.log10(hz / 20.0) / math.log10(20000.0 / 20.0)
        out = _eq_norm_to_freq_hz(n)
        assert abs(out - hz) / hz < 0.002


def test_freq_clamp_edges():
    assert _eq_norm_to_freq_hz(0.0) == 20.0
    assert _eq_norm_to_freq_hz(1.0) == 20000.0


def test_q_roundtrip():
    for q in (0.1, 1.0, 9.0, 18.0):
        n = max(0.0, min(1.0, (q - 0.1) / 17.9))
        assert abs(_eq_norm_to_q(n) - q) < 1e-6
