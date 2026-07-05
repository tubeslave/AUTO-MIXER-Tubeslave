"""Gain v3: gain-set, lock/зависимости/trim-компенсация, live-детекторы."""

import numpy as np
import pytest

from heuristics.gain_v3 import (
    CLASS_GAIN_PROFILE,
    ChannelGainStats,
    GainLock,
    GainSupervisor,
    GainSupervisorConfig,
    channel_gain_stats,
    dependency_shifts,
    derive_gain_set,
    detect_clip,
    quantize_toward_headroom,
    trim_compensation,
)

SR = 48000
RNG = np.random.default_rng(21)


def _bursty(peak_amp=0.1, n_sec=4.0):
    """Сигнал с тихими и громкими блоками (громкое место + паузы)."""
    t = np.arange(int(n_sec * SR)) / SR
    env = np.where((t % 2.0) < 1.0, 0.02, 1.0)
    return peak_amp * env * np.sin(2 * np.pi * 220.0 * t)


def _stats(peak_db=-24.0, crest=15.0, floor=-70.0, l5=None, present=True):
    return ChannelGainStats(
        sample_peak_db=peak_db, true_peak_est_db=peak_db + 3.0,
        rms_db=peak_db - crest, crest_db=crest,
        loud_l5_db=(l5 if l5 is not None else peak_db - 6.0),
        noise_floor_db=floor, signal_present=present,
    )


class TestGainSet:
    def test_delta_hits_class_target_with_true_peak_allowance(self):
        stats = _stats(peak_db=-30.0)
        res = derive_gain_set(stats, "lead_vocal")
        # true peak est −27; цель −13 → дельта 14, квантование вниз
        assert res.gain_delta_db == pytest.approx(14.0, abs=0.5)
        assert res.gain_delta_db <= (-13.0) - (-27.0)  # в сторону headroom

    def test_quantize_always_toward_headroom(self):
        assert quantize_toward_headroom(3.9, 0.5) == pytest.approx(3.5)
        assert quantize_toward_headroom(-3.1, 0.5) == pytest.approx(-3.5)

    def test_no_signal_returns_none(self):
        assert derive_gain_set(_stats(present=False), "kick") is None

    def test_pad_for_line_hot_source(self):
        stats = _stats(peak_db=-6.0, floor=-80.0)   # горячий DI
        res = derive_gain_set(stats, "keys")
        assert res.pad_engaged
        # pad −20 учтён ДО квантования: дельта сдвинута на +20
        assert res.gain_delta_db > 0.0

    def test_pad_when_needed_gain_below_preamp_min(self):
        stats = _stats(peak_db=-4.0, floor=-80.0)
        res = derive_gain_set(stats, "guitar", current_gain_db=0.0,
                              gain_min_db=-10.0)
        assert res.pad_engaged

    def test_noise_floor_flag_is_source_problem_not_gain(self):
        stats = _stats(peak_db=-30.0, floor=-58.0)
        res = derive_gain_set(stats, "lead_vocal")
        assert any("check_cable_or_source" in f for f in res.snr_flags)

    def test_clean_floor_no_flags(self):
        stats = _stats(peak_db=-16.0, floor=-85.0)
        res = derive_gain_set(stats, "guitar")
        assert res.snr_flags == []

    def test_stats_from_audio(self):
        stats = channel_gain_stats(_bursty(), SR)
        assert stats.signal_present
        assert stats.noise_floor_db < stats.loud_l5_db
        assert stats.true_peak_est_db == pytest.approx(
            stats.sample_peak_db + 3.0)

    def test_drums_more_conservative_than_vocal(self):
        assert (CLASS_GAIN_PROFILE["snare"]["target_peak_dbfs"]
                < CLASS_GAIN_PROFILE["lead_vocal"]["target_peak_dbfs"])


class TestLockAndCompensation:
    def test_dependency_graph_shifts(self):
        shifts = dependency_shifts(+3.0)
        assert shifts["gate_threshold_rederive_db"] == pytest.approx(3.0)
        assert shifts["comp_threshold_rederive_db"] == pytest.approx(3.0)
        assert shifts["monitor_send_level_shift_db"] == pytest.approx(3.0)

    def test_trim_compensation_law(self):
        """Закон Gain Tracking: trim += −Δgain; динамика уведомляется."""
        comp = trim_compensation(+4.0)
        assert comp["trim_delta_db"] == pytest.approx(-4.0)
        assert comp["dynamics_effective_threshold_shift_db"] == pytest.approx(4.0)

    def test_lock_serializable(self):
        import json
        lock = GainLock(channel=3, gain_db=24.0, pad_engaged=False,
                        stats=_stats())
        json.dumps(lock.to_dict())


class TestDetectors:
    def test_clip_needs_consecutive_fullscale(self):
        x = np.zeros(1000)
        x[500:503] = 1.0
        assert detect_clip(x)
        x2 = np.zeros(1000)
        x2[::100] = 1.0                       # одиночные — не клип
        assert not detect_clip(x2)

    def test_clean_signal_not_clip(self):
        assert not detect_clip(0.5 * np.sin(np.arange(4800) / 10.0))


class TestSupervisor:
    def _snap(self, **kw):
        return GainLock(channel=1, gain_db=20.0, pad_engaged=False,
                        stats=_stats(**kw))

    def test_default_is_alert_only_no_auto_actions(self):
        sup = GainSupervisor(self._snap())
        clipped = np.concatenate([np.ones(64), 0.5 * RNG.normal(0, 1, SR)])
        alerts = sup.step(clipped, SR, dt_s=1.0)
        assert any(a["name"] == "CLIP" for a in alerts)
        assert sup.trim_db == 0.0            # дефолт: только оповестить человека

    def test_auto_trim_down_on_sustained_clip_rate_limited(self):
        sup = GainSupervisor(self._snap(), GainSupervisorConfig(auto_trim_enabled=True))
        clipped = np.concatenate([np.ones(2000), 0.9 * np.sin(np.arange(SR) / 8.0)])
        for _ in range(2):
            sup.step(clipped, SR, dt_s=1.0)
        assert sup.trim_db < 0.0
        assert sup.trim_db >= -3.0 * 2       # быстрая полоса 3 дБ/с

    def test_trim_bounded_at_6db(self):
        sup = GainSupervisor(self._snap(), GainSupervisorConfig(auto_trim_enabled=True))
        clipped = np.ones(SR)
        for _ in range(20):
            sup.step(clipped, SR, dt_s=1.0)
        assert sup.trim_db >= -6.0 - 1e-9    # кламп ±6

    def test_too_quiet_alert_only_never_boost(self):
        """Правило проекта: trim-boost в live только по opt-in оператора."""
        sup = GainSupervisor(self._snap(l5=-12.0),
                             GainSupervisorConfig(auto_trim_enabled=True))
        # активный, но тихий канал: всплески −34 dBFS против snap L5 −12
        t = np.arange(4 * SR) / SR
        env = np.where((t % 2.0) < 1.0, 0.02, 0.0002)
        quiet = env * np.sin(2 * np.pi * 220.0 * t)
        alerts = []
        for _ in range(61):
            alerts = sup.step(quiet, SR, dt_s=1.0)
        assert any(a["name"] == "TOO_QUIET" for a in alerts)
        assert sup.trim_db == 0.0            # никакого авто-boost

    def test_crest_collapse_after_hold(self):
        sup = GainSupervisor(self._snap(crest=20.0))
        # активный сигнал с задавленным crest (~3 дБ): лимит выше по тракту
        t = np.arange(4 * SR) / SR
        env = np.where((t % 2.0) < 1.5, 1.0, 0.001)
        squashed = 0.5 * env * np.sin(2 * np.pi * 220.0 * t)
        alerts = []
        for _ in range(6):
            alerts = sup.step(squashed, SR, dt_s=1.0)
        assert any(a["name"] == "CREST_COLLAPSE" for a in alerts)

    def test_floor_jump_alert(self):
        sup = GainSupervisor(self._snap(floor=-80.0))
        noisy = 0.05 * RNG.normal(0, 1, SR)            # пол ~−26
        alerts = sup.step(noisy, SR, dt_s=1.0)
        assert any(a["name"] == "FLOOR_JUMP" for a in alerts)

    def test_revert_returns_trim_to_zero(self):
        sup = GainSupervisor(self._snap(), GainSupervisorConfig(auto_trim_enabled=True))
        for _ in range(3):
            sup.step(np.ones(SR), SR, dt_s=1.0)
        assert sup.trim_db < 0.0
        sup.revert_trim()
        assert sup.trim_db == 0.0
        assert any(e["name"] == "TRIM_REVERTED" for e in sup.events)
