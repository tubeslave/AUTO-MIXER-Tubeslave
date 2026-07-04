"""Dugan gain sharing: константный суммарный гейн, приоритеты, сглаживание."""

import numpy as np
import pytest

from dugan_gain_sharing import (
    DuganGainSharing,
    dugan_gains_db,
    is_speech_segment,
)


class TestFormula:
    def test_constant_total_gain(self):
        """Сумма мощностных долей = 1 (суммарный гейн системы 0 дБ)."""
        gains = dugan_gains_db({"a": -30.0, "b": -36.0, "c": -50.0})
        total_power = sum(10 ** (g / 10.0) for g in gains.values())
        assert total_power == pytest.approx(1.0)

    def test_all_gains_non_positive(self):
        gains = dugan_gains_db({"a": -20.0, "b": -20.0})
        assert all(g <= 0.0 for g in gains.values())

    def test_dominant_mic_gets_most_gain(self):
        gains = dugan_gains_db({"host": -20.0, "guest": -40.0})
        assert gains["host"] > gains["guest"]
        # доминирующий канал почти не приглушён
        assert gains["host"] > -0.5
        # слабый приглушён примерно на разницу уровней
        assert gains["guest"] == pytest.approx(gains["host"] - 20.0, abs=0.1)

    def test_equal_mics_share_equally(self):
        gains = dugan_gains_db({"a": -25.0, "b": -25.0})
        assert gains["a"] == pytest.approx(gains["b"])
        assert gains["a"] == pytest.approx(-3.01, abs=0.05)  # 10*log10(1/2)

    def test_weights_shift_share_without_breaking_sum(self):
        base = dugan_gains_db({"a": -25.0, "b": -25.0})
        weighted = dugan_gains_db({"a": -25.0, "b": -25.0}, {"a": 6.0})
        assert weighted["a"] > base["a"]
        assert weighted["b"] < base["b"]
        total_power = sum(10 ** (g / 10.0) for g in weighted.values())
        assert total_power == pytest.approx(1.0)

    def test_empty_input(self):
        assert dugan_gains_db({}) == {}


class TestLiveSmoothing:
    def _make(self, **kw):
        clock = {"t": 0.0}

        def now():
            return clock["t"]

        sharing = DuganGainSharing(time_provider=now, **kw)
        return sharing, clock

    def test_rate_limited(self):
        sharing, clock = self._make(max_rate_db_per_sec=12.0,
                                    min_update_interval_s=0.1)
        clock["t"] = 0.1
        out = sharing.update({"a": -20.0, "b": -50.0})
        # за 0.1 c гейн не может уйти дальше 1.2 дБ от нуля
        assert out["b"] >= -1.3

    def test_converges_to_formula(self):
        sharing, clock = self._make(max_rate_db_per_sec=24.0,
                                    min_update_interval_s=0.1)
        target = dugan_gains_db({"a": -20.0, "b": -40.0})
        out = {}
        for i in range(1, 40):
            clock["t"] = 0.12 * i
            out = sharing.update({"a": -20.0, "b": -40.0}) or out
        assert out["b"] == pytest.approx(max(target["b"], sharing.floor_db), abs=0.3)

    def test_throttle_interval_respected(self):
        sharing, clock = self._make(min_update_interval_s=0.12)
        clock["t"] = 0.2
        assert sharing.update({"a": -20.0}) != {}
        clock["t"] = 0.25  # 50 мс — чаще троттлера WING
        assert sharing.update({"a": -20.0}) == {}

    def test_floor_not_exceeded(self):
        sharing, clock = self._make(floor_db=-15.0, max_rate_db_per_sec=1000.0)
        clock["t"] = 1.0
        out = sharing.update({"a": -10.0, "b": -60.0})
        assert out["b"] >= -15.0


class TestSpeechGate:
    def test_speech_segment_detected(self):
        assert is_speech_segment(3, music_activity_score=0.1)

    def test_music_blocks(self):
        assert not is_speech_segment(3, music_activity_score=0.8)

    def test_single_mic_blocks(self):
        assert not is_speech_segment(1, music_activity_score=0.0)
