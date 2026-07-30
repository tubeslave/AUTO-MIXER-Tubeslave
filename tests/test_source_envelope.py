"""Синтетические тесты конвертов источников (Ф8, DEC-0008)."""

import numpy as np
import pytest

from source_envelope import (
    CORE_MAX_CUT_DB,
    IdentityMask,
    activity_mask,
    build_envelope,
    compute_identity_mask,
    eq_cut_allowed,
    measure_crest_db,
)

SR = 48000


def tone(freq, seconds=2.0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def kick_like(seconds=2.0):
    """Транзиентный источник: удары 60 Гц с быстрым затуханием, 2 Гц."""
    n = int(seconds * SR)
    x = np.zeros(n)
    hit = int(0.12 * SR)
    envelope = np.exp(-np.arange(hit) / (0.02 * SR))
    t = np.arange(hit) / SR
    pulse = 0.9 * envelope * np.sin(2 * np.pi * 60.0 * t)
    for start in range(0, n - hit, SR // 2):
        x[start:start + hit] += pulse
    return x


class TestActivityMask:
    def test_continuous_tone_fully_active(self):
        act = activity_mask(tone(440), SR)
        assert act.mean() > 0.95

    def test_silence_inactive(self):
        act = activity_mask(np.zeros(SR * 2), SR)
        assert act.sum() == 0

    def test_sparse_hits_partially_active(self):
        act = activity_mask(kick_like(), SR)
        assert 0.05 < act.mean() < 0.7


class TestIdentityMask:
    def test_tone_core_at_tone_band(self):
        mask = compute_identity_mask(tone(1000), SR)
        assert mask.confidence > 0.9
        assert 1000.0 in mask.core_bands_hz
        assert 8000.0 not in mask.core_bands_hz

    def test_kick_core_in_low_bands(self):
        mask = compute_identity_mask(kick_like(), SR)
        assert mask.core_bands_hz, "ядро должно найтись"
        assert min(mask.core_bands_hz) <= 80.0

    def test_silent_channel_gets_no_trusted_mask(self):
        mask = compute_identity_mask(np.zeros(SR * 2) + 1e-7, SR)
        assert mask.confidence == 0.0 and mask.core_bands_hz == []

    def test_is_core_maps_to_nearest_band(self):
        mask = compute_identity_mask(tone(1000), SR)
        assert mask.is_core(980.0)      # ближайший центр 1000
        assert not mask.is_core(7900.0)


class TestCrest:
    def test_sine_crest_3db(self):
        assert measure_crest_db(tone(440)) == pytest.approx(3.01, abs=0.1)

    def test_transient_crest_high(self):
        act = activity_mask(kick_like(), SR)
        crest = measure_crest_db(kick_like(), act, SR)
        assert crest > 10.0


class TestEnvelope:
    def test_kick_envelope(self):
        env = build_envelope("kick", kick_like(), SR)
        assert env.crest_target_db == (14.0, 24.0)
        assert env.touch_zones_hz[0] == (50.0, 70.0)
        assert env.measured_crest_db > 10.0

    def test_unknown_instrument_defaults(self):
        env = build_envelope("theremin", tone(700), SR)
        assert env.crest_target_db == (8.0, 18.0)
        assert env.touch_zones_hz == ()

    def test_sustained_source_fails_transient_target(self):
        env = build_envelope("kick", tone(60), SR)  # гул вместо ударов
        assert not env.crest_ok


class TestCoreProtection:
    def test_cut_in_core_capped(self):
        mask = compute_identity_mask(tone(1000), SR)
        assert eq_cut_allowed(mask, 1000.0, -6.0) == CORE_MAX_CUT_DB

    def test_cut_outside_core_free(self):
        mask = compute_identity_mask(tone(1000), SR)
        assert eq_cut_allowed(mask, 8000.0, -6.0) == -6.0

    def test_boost_untouched(self):
        mask = compute_identity_mask(tone(1000), SR)
        assert eq_cut_allowed(mask, 1000.0, 2.0) == 2.0

    def test_low_confidence_mask_protects_everything(self):
        weak = IdentityMask(core_bands_hz=[1000.0], confidence=0.2)
        assert eq_cut_allowed(weak, 8000.0, -6.0) == CORE_MAX_CUT_DB
