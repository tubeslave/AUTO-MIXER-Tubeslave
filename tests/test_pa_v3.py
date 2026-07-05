"""PA v3: H1/когерентность, ESS, калибровка, усреднение, EQ, саб/топ, live."""

import numpy as np
import pytest

from heuristics.pa_v3 import (
    HFDriftLane,
    SPL_LIMITS,
    SPLSupervisor,
    SystemSnapshot,
    align_sub_top,
    apply_cal,
    bass_excess_indicator,
    ess_deconvolve,
    ess_sweep,
    limiter_gr_alert,
    limiter_threshold_policy,
    load_cal_file,
    solve_fill,
    solve_system_eq,
    spatial_average,
    target_curve_db,
    transfer_function_h1,
)

SR = 48000
RNG = np.random.default_rng(61)


class TestTransferFunction:
    def test_flat_system_unity_high_coherence(self):
        x = RNG.normal(0, 0.3, 8 * SR)
        tf = transfer_function_h1(x, 0.5 * x, SR)
        mid = (tf["freqs"] > 100) & (tf["freqs"] < 10000)
        assert np.median(tf["magnitude_db"][mid]) == pytest.approx(-6.0, abs=0.5)
        assert np.median(tf["coherence"][mid]) > 0.95

    def test_noise_kills_coherence(self):
        x = RNG.normal(0, 0.3, 8 * SR)
        y = RNG.normal(0, 0.3, 8 * SR)          # некоррелированный
        tf = transfer_function_h1(x, y, SR)
        mid = (tf["freqs"] > 100) & (tf["freqs"] < 10000)
        assert np.median(tf["coherence"][mid]) < 0.3


class TestESS:
    def test_deconvolution_recovers_delay(self):
        sweep = ess_sweep(20.0, 20000.0, 4.0, SR)
        delay_samples = 480                       # 10 мс
        recording = np.concatenate([np.zeros(delay_samples), 0.7 * sweep])
        result = ess_deconvolve(sweep, recording, SR)
        assert result["delay_s"] == pytest.approx(0.010, abs=0.0005)

    def test_sweep_spans_band(self):
        sweep = ess_sweep(20.0, 20000.0, 4.0, SR)
        spec = np.abs(np.fft.rfft(sweep))
        freqs = np.fft.rfftfreq(len(sweep), 1.0 / SR)
        assert spec[(freqs > 50) & (freqs < 15000)].min() > 0.0


class TestCalibration:
    CAL = ["* comment line", "; another", "20 -1.5", "1000, 0.0", "20000 3.0"]

    def test_load_skips_comments(self):
        freqs, dbs = load_cal_file(self.CAL)
        assert list(freqs) == [20.0, 1000.0, 20000.0]
        assert list(dbs) == [-1.5, 0.0, 3.0]

    def test_apply_interpolates_and_subtracts(self):
        freqs, dbs = load_cal_file(self.CAL)
        fft_freqs = np.array([20.0, 1000.0, 20000.0])
        corrected = apply_cal(np.zeros(3), fft_freqs, freqs, dbs)
        assert corrected[0] == pytest.approx(1.5)   # микрофон занижал −1.5
        assert corrected[2] == pytest.approx(-3.0)  # микрофон завышал +3


class TestSpatialAveraging:
    def test_coherence_weighted_and_consistency_mask(self):
        pos1 = np.array([0.0, 0.0, 10.0])
        pos2 = np.array([0.0, 0.5, -10.0])         # третий бин: рябь ±10
        coh = np.array([0.9, 0.9, 0.9])
        result = spatial_average([pos1, pos2], [coh, coh])
        assert result["consistent"][0] and result["consistent"][1]
        assert not result["consistent"][2]          # позиционная рябь


class TestSystemEQ:
    def test_prefer_cut_and_reliability_gate(self):
        freqs = np.linspace(40, 16000, 512)
        target = target_curve_db(freqs, "live_standard")
        measured = target.copy()
        peak_bin = 100
        measured[peak_bin - 3:peak_bin + 3] += 5.0   # устойчивый пик +5
        coherence = np.full_like(freqs, 0.9)
        consistent = np.ones_like(freqs, dtype=bool)
        result = solve_system_eq(freqs, measured, target, coherence, consistent)
        assert result["filters"], "фильтр не выдан"
        assert result["filters"][0]["gain_db"] < 0    # prefer cut
        assert result["filters"][0]["gain_db"] >= -6.0

    def test_unreliable_regions_not_eqd(self):
        freqs = np.linspace(40, 16000, 512)
        target = np.zeros_like(freqs)
        measured = target.copy()
        measured[200:210] += 8.0
        coherence = np.full_like(freqs, 0.3)          # некогерентно (HVAC/толпа)
        consistent = np.ones_like(freqs, dtype=bool)
        result = solve_system_eq(freqs, measured, target, coherence, consistent)
        assert result["filters"] == []                # «EQ тренда, а не ряби»

    def test_filter_budget(self):
        freqs = np.linspace(40, 16000, 512)
        target = np.zeros_like(freqs)
        measured = 4.0 * np.sin(np.linspace(0, 40, 512))   # много «пиков»
        coherence = np.full_like(freqs, 0.9)
        consistent = np.ones_like(freqs, dtype=bool)
        result = solve_system_eq(freqs, measured, target, coherence, consistent)
        assert len(result["filters"]) <= 8

    def test_target_presets(self):
        freqs = np.array([50.0, 500.0, 8000.0])
        edm = target_curve_db(freqs, "edm")
        flat = target_curve_db(freqs, "flat_tilt")
        assert edm[0] > flat[0]                       # EDM: больше НЧ
        assert edm[2] < 0 and flat[2] < 0             # ВЧ-спад у обоих


class TestSubTopAlignment:
    def test_finds_known_delay(self):
        # полосовой шум 60–140 Гц: у чистого тона решение периодически
        # неоднозначно (4 мс ≡ 9 мс+флип), у шума максимум уникален
        noise = RNG.normal(0, 1, 2 * SR)
        spec = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(len(noise), 1.0 / SR)
        spec[(freqs < 60) | (freqs > 140)] = 0.0
        top = np.fft.irfft(spec)
        sub = np.roll(top, -int(0.004 * SR))          # саб приходит на 4 мс раньше
        result = align_sub_top(sub, top, SR)
        assert result["delay_ms"] == pytest.approx(4.0, abs=0.3)
        assert not result["polarity_invert"]
        assert result["verified"]
        assert result["live_changeable"] is False     # запрещено в лайве

    def test_inverted_sub_detected(self):
        t = np.arange(2 * SR) / SR
        top = np.sin(2 * np.pi * 100.0 * t)
        sub = -top.copy()
        result = align_sub_top(sub, top, SR)
        # инверсия полярности ИЛИ полуволновая задержка (5 мс на 100 Гц) —
        # оба решения физически эквивалентны для тона
        assert result["polarity_invert"] or result["delay_ms"] >= 4.0
        assert result["summation_gain_db"] > 0.0


class TestFillAndLimiter:
    def test_fill_delay_with_precedence(self):
        result = solve_fill(0.020, 0.008, precedence_offset_ms=3.0)
        assert result["delay_ms"] == pytest.approx(15.0)
        assert result["level_ceiling_db"] == pytest.approx(-6.0)

    def test_precedence_capped(self):
        result = solve_fill(0.010, 0.010, precedence_offset_ms=20.0)
        assert result["delay_ms"] <= 5.0

    def test_limiter_policy(self):
        result = limiter_threshold_policy(-4.0)
        assert result["threshold_db"] == pytest.approx(-7.0)
        assert result["ratio"] == 100.0


class TestSnapshotGate:
    def test_channel_pipelines_wait_for_lock(self):
        snap = SystemSnapshot(eq_filters_by_output={}, sub_top_alignment={},
                              fill_delays={}, limiter_by_output={},
                              target_curve="live_standard")
        assert not snap.channel_pipelines_may_start()
        snap.lock()
        assert snap.channel_pipelines_may_start()


class TestLiveLanes:
    def test_hf_lane_alert_only_without_optin(self):
        lane = HFDriftLane(opt_in=False)
        out = None
        for _ in range(11):
            out = lane.step(-3.0, 0.0, mean_coherence=0.8, dt_s=60.0)
        assert out is not None and out["name"] == "HF_DROOP_ALERT"
        assert lane.applied_db == 0.0

    def test_hf_lane_optin_rate_and_cap(self):
        lane = HFDriftLane(opt_in=True)
        for _ in range(60):
            lane.step(-3.0, 0.0, mean_coherence=0.8, dt_s=60.0)
        assert lane.applied_db <= 3.0 + 1e-9          # кап +3
        # скорость ≤0.5 дБ/мин: 3 дБ не раньше чем за 6 минут после hold
        assert lane.applied_db == pytest.approx(3.0)

    def test_low_coherence_freezes(self):
        lane = HFDriftLane(opt_in=True, applied_db=1.0)
        out = lane.step(-3.0, 0.0, mean_coherence=0.4, dt_s=60.0)
        assert out["name"] == "SYSTEM_AUTOMATION_FROZEN"
        assert lane.frozen

    def test_hf_rollback(self):
        lane = HFDriftLane(opt_in=True, applied_db=2.0)
        assert lane.rollback() == -2.0
        assert lane.applied_db == 0.0

    def test_spl_predictive_alert_and_armed_action(self):
        sup = SPLSupervisor(jurisdiction="germany_din_15905_5", calibrated=True)
        sup.step(0.0, 96.0)
        sup.step(1.0, 97.0)
        out = sup.step(2.0, 98.0)                     # 99−1.5=97.5, тренд вверх
        assert out is not None and out["name"] == "SPL_PREDICTIVE_ALERT"
        assert out["action"] is None                  # не armed
        out2 = sup.step(3.0, 99.5, operator_armed=True)
        assert out2["name"] == "SPL_LIMIT_EXCEEDED"
        assert out2["action"]["master_trim_db"] == -2.0

    def test_uncalibrated_flagged(self):
        sup = SPLSupervisor(calibrated=False)
        sup.step(0.0, 97.0); sup.step(1.0, 97.5)
        out = sup.step(2.0, 98.0)
        assert out["note"] == "UNCALIBRATED_RELATIVE_ONLY"

    def test_jurisdiction_table(self):
        assert SPL_LIMITS["netherlands_vnpf"]["laeq_db"] == 103.0
        assert SPL_LIMITS["france_2017"]["window_min"] == 15

    def test_limiter_gr_alert(self):
        assert limiter_gr_alert(4.0, 6.0) is not None
        assert limiter_gr_alert(2.0, 60.0) is None

    def test_bass_excess(self):
        assert bass_excess_indicator(112.0, 95.0)["excess"]
        assert not bass_excess_indicator(105.0, 95.0)["excess"]
