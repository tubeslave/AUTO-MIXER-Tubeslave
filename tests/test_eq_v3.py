"""EQ v3: резонансы, HPF-by-f0, ERB-маскирование, A/B, live-дифференциал."""

import numpy as np
import pytest

from heuristics.resonance_eq import (
    detect_resonances,
    estimate_fundamental_hz,
    hpf_from_fundamental,
)
from heuristics.erb_masking import (
    allocate_complementary_cuts,
    band_energies_db,
    erb_band_edges,
    masking_vector_db,
)
from heuristics.eq_ab_check import (
    bell_response_db,
    eq_curve_db,
    k_weighting_curve_db,
    predictive_ab_check,
)
from heuristics.eq_v3 import (
    live_eq_differential,
    near_feedback,
    soundcheck_eq_v3,
)

SR = 48000
RNG = np.random.default_rng(42)


def _noise(n_sec=1.0, amp=0.1):
    return (amp * RNG.normal(0, 1, int(n_sec * SR))).astype(np.float64)


def _tone(freq, n_sec=1.0, amp=0.3, harmonics=None):
    t = np.arange(int(n_sec * SR)) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    for mult, h in (harmonics or {}).items():
        x += amp * h * np.sin(2 * np.pi * freq * mult * t)
    return x


def _resonant_noise(res_freq, n_sec=1.0, res_amp=0.5):
    """Шум + узкий негармонический резонанс (звон корпуса)."""
    return _noise(n_sec) + res_amp * np.sin(
        2 * np.pi * res_freq * np.arange(int(n_sec * SR)) / SR
    )


def _bursty_noise(n_sec=4.0, period_s=0.5, burst_s=0.12, amp=0.5, seed=7):
    """Пульсирующий шум: всплески с паузами (нужны ring-тесту)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.002, int(n_sec * SR))
    for start in np.arange(0.1, n_sec - burst_s, period_s):
        i = int(start * SR)
        n = int(burst_s * SR)
        x[i:i + n] += amp * rng.normal(0, 1, n) * np.hanning(n)
    return x.astype(np.float64)


class TestFundamentalAndHPF:
    def test_estimates_f0(self):
        f0 = estimate_fundamental_hz(_tone(110.0, harmonics={2: 0.5, 3: 0.3}), SR)
        assert f0 is not None
        assert f0 == pytest.approx(110.0, rel=0.03)

    def test_noise_has_no_f0(self):
        assert estimate_fundamental_hz(_noise(), SR) is None

    def test_hpf_tracks_fundamental_within_role_bounds(self):
        # вокал с f0 220 Гц: 0.75*220 = 165 -> кламп к верхней границе роли 120
        assert hpf_from_fundamental("lead_vocal", 220.0) == 120.0
        # вокал с f0 130 Гц: 0.75*130 = 97.5 внутри границ
        assert hpf_from_fundamental("lead_vocal", 130.0) == pytest.approx(97.5)
        # без f0 — нижняя граница роли
        assert hpf_from_fundamental("lead_vocal", None) == 75.0

    def test_kick_bass_no_hpf(self):
        assert hpf_from_fundamental("kick", 55.0) is None
        assert hpf_from_fundamental("bass", 41.0) is None


class TestResonanceDetection:
    def test_detects_nonharmonic_ring(self):
        cuts = detect_resonances(_resonant_noise(445.0), SR)
        assert cuts, "резонанс 445 Гц не найден"
        top = cuts[0]
        assert top.freq_hz == pytest.approx(445.0, rel=0.05)
        assert top.gain_db < 0
        assert top.gain_db >= -9.0
        assert 2.0 <= top.q <= 8.0  # валидировано #22: резать узко

    def test_harmonic_note_not_cut(self):
        """Устойчивая нота с гармониками (орган/аккордеон) — не резонанс."""
        x = _noise(1.0, amp=0.02) + _tone(440.0, harmonics={2: 0.5, 3: 0.3})
        cuts = detect_resonances(x, SR)
        assert all(abs(c.freq_hz - 440.0) / 440.0 > 0.05 for c in cuts)
        assert all(abs(c.freq_hz - 880.0) / 880.0 > 0.05 for c in cuts)

    def test_flat_noise_no_cuts(self):
        assert detect_resonances(_noise(), SR) == []

    def test_continuous_signal_unconfirmed_ring_capped(self):
        """Непрерывный сигнал: пауз нет, звон не доказан -> кэп -3
        (вопрос оператора 2026-07-06: «EQ просто давит громкое»)."""
        strong = detect_resonances(_resonant_noise(445.0, res_amp=2.0), SR)
        assert strong
        assert strong[0].gain_db == pytest.approx(-3.0)
        assert strong[0].ring_confirmed is not True

    def test_ringing_resonance_gets_deep_cut(self):
        """Звенящий резонатор (высокий Q) поверх резко глохнущего шума:
        полоса затухает медленнее широкополосного -> глубокий срез."""
        from scipy.signal import iirpeak, lfilter
        base = _bursty_noise(seed=9)              # умирает с концом всплеска
        b, a = iirpeak(445.0 / (SR / 2), Q=60)
        ring = lfilter(b, a, base) * 6.0          # звенит в паузах
        x = base + ring
        cuts = detect_resonances(x, SR)
        assert cuts, "звенящий резонанс не найден"
        top = cuts[0]
        assert top.freq_hz == pytest.approx(445.0, rel=0.06)
        assert top.ring_confirmed is True
        assert top.gain_db < -3.0

    def test_formant_without_ring_capped(self):
        """Формантоподобный пик, умирающий вместе с огибающей ноты, —
        гигиенический кэп -3, характер не выпиливается."""
        from scipy.signal import butter, sosfiltfilt
        env_noise = _bursty_noise()
        sos = butter(4, [900, 1100], btype="band", fs=SR, output="sos")
        formant = sosfiltfilt(sos, env_noise) * 12.0   # та же огибающая
        x = env_noise * 0.3 + formant
        cuts = detect_resonances(x, SR)
        for c in cuts:
            if abs(c.freq_hz - 1000.0) < 150.0:
                assert c.ring_confirmed is not True
                assert c.gain_db >= -3.0 - 1e-6


class TestERBMasking:
    def _bands(self, x):
        return band_energies_db(x, SR, erb_band_edges())

    def test_masking_vector_shows_overlap(self):
        vocal = _tone(1000.0, amp=0.05)          # тихий вокал на 1 кГц
        guitar = _tone(1000.0, amp=0.5)          # громкая гитара там же
        bands = {"voc": self._bands(vocal), "gtr": self._bands(guitar)}
        mask = masking_vector_db(bands, "voc")
        assert np.max(mask) > 6.0

    def test_lower_priority_masker_gets_cut_vocal_protected(self):
        vocal = _tone(1000.0, amp=0.05)
        guitar = _tone(1000.0, amp=0.5)
        bands = {"voc": self._bands(vocal), "gtr": self._bands(guitar)}
        cuts = allocate_complementary_cuts(
            bands, {"voc": "lead_vocal", "gtr": "guitar"})
        assert cuts, "маскинг-рез не выдан"
        assert all(c.channel == "gtr" for c in cuts)   # вокал не режется
        assert all(c.gain_db < 0 for c in cuts)         # только cut
        assert all(c.gain_db >= -6.0 for c in cuts)     # кэп Hafezi & Reiss
        assert all(c.mask_after_db <= c.mask_before_db for c in cuts)

    def test_no_cut_without_conflict(self):
        bass = _tone(60.0, amp=0.4)
        hat = _tone(6000.0, amp=0.3)
        bands = {"bass": self._bands(bass), "hat": self._bands(hat)}
        cuts = allocate_complementary_cuts(
            bands, {"bass": "bass", "hat": "hihat"})
        # разнесённые по спектру источники не конфликтуют
        assert all(abs(c.gain_db) <= 2.0 for c in cuts)

    def test_higher_priority_never_cut_for_lower(self):
        """Гитару не режут ради оверхэдов (приоритет ниже)."""
        guitar = _tone(2000.0, amp=0.5)
        oh = _tone(2000.0, amp=0.05)
        bands = {"gtr": self._bands(guitar), "oh": self._bands(oh)}
        cuts = allocate_complementary_cuts(
            bands, {"gtr": "guitar", "oh": "overheads"})
        assert all(c.channel != "gtr" or c.masked_channel != "oh" for c in cuts)

    def test_masker_identity_core_capped(self):
        """Тембровое ядро маскера (полосы у его максимума) не режется
        глубже -3 даже при конфликте с вокалом (оператор 2026-07-06:
        «EQ просто давит громкое»)."""
        vocal = _tone(1000.0, amp=0.05)
        guitar = _tone(1000.0, amp=0.5)   # 1 кГц = энергетическое ядро гитары
        bands = {"voc": self._bands(vocal), "gtr": self._bands(guitar)}
        cuts = allocate_complementary_cuts(
            bands, {"voc": "lead_vocal", "gtr": "guitar"})
        gtr_cuts = [c for c in cuts if c.channel == "gtr"]
        assert gtr_cuts, "конфликт не решался"
        assert all(c.gain_db >= -3.0 - 1e-6 for c in gtr_cuts)


class TestABCheck:
    def test_k_weighting_shape(self):
        freqs = np.array([50.0, 1000.0, 3000.0])
        kw = k_weighting_curve_db(freqs, SR)
        assert kw[0] < kw[1] < kw[2]  # HPF внизу, shelf вверху

    def test_bell_response(self):
        freqs = np.array([500.0, 1000.0, 2000.0])
        r = bell_response_db(freqs, 1000.0, -6.0, 4.0)
        assert r[1] == pytest.approx(-6.0)
        assert abs(r[0]) < 2.0

    def test_small_cut_passes(self):
        freqs, spec = np.linspace(20, 20000, 512), np.full(512, -40.0)
        res = predictive_ab_check(
            freqs, spec, [{"freq_hz": 445.0, "gain_db": -3.0, "q": 6.0}])
        assert res.allowed
        assert res.compensation_db == 0.0

    def test_big_tilt_requires_compensation(self):
        freqs = np.linspace(20, 20000, 512)
        spec = np.full(512, -40.0)
        bands = [{"freq_hz": 2000.0, "gain_db": -6.0, "q": 0.5}]
        res = predictive_ab_check(freqs, spec, bands)
        assert res.loudness_delta_lu < -0.5
        assert res.reason in ("requires_fader_compensation",
                              "loudness_bias_exceeds_safe_compensation")
        if res.allowed:
            assert res.compensation_db == pytest.approx(-res.loudness_delta_lu)


class TestSoundcheckOrchestrator:
    def test_end_to_end_multichannel(self):
        channels = {
            16: {"audio": _tone(200.0, amp=0.08, harmonics={2: 0.4, 3: 0.2}),
                 "role": "lead_vocal"},
            9: {"audio": _resonant_noise(445.0, res_amp=0.4) + _tone(200.0, amp=0.4),
                "role": "guitar"},
            1: {"audio": _noise(1.0, 0.2), "role": "kick"},
        }
        result = soundcheck_eq_v3(channels, SR)
        assert set(result) == {16, 9, 1}
        assert result[1].hpf_hz is None                # kick без HPF
        assert result[16].hpf_hz is not None           # вокал с HPF
        assert result[9].applied_bands, "гитара должна получить полосы"
        # вокал никогда не режется маскинг-резами
        assert result[16].masking_cuts == []
        # каждая полоса в пределах консольного формата
        for p in result.values():
            assert len(p.applied_bands) <= 4
            if p.ab_check is not None and p.applied_bands:
                assert p.ab_check.allowed

    def test_serializable(self):
        import json
        result = soundcheck_eq_v3(
            {1: {"audio": _resonant_noise(300.0), "role": "guitar"}}, SR)
        json.dumps(result[1].to_dict())


class TestLiveDifferential:
    SNAP = [{"freq_hz": 445.0, "gain_db": -3.0, "q": 6.0}]

    def test_deviation_clamped_to_pm3(self):
        out = live_eq_differential(
            self.SNAP, [{"freq_hz": 445.0, "gain_db": -9.0, "q": 6.0}],
            channel_level_db=-20.0)
        assert out["bands"][0]["gain_db"] == -6.0  # снапшот −3 ± 3

    def test_boost_near_feedback_blocked(self):
        out = live_eq_differential(
            self.SNAP, [{"freq_hz": 445.0, "gain_db": 0.0, "q": 6.0}],
            feedback_freqs=[450.0], channel_level_db=-20.0)
        assert out["bands"][0]["gain_db"] <= -3.0  # не выше снапшота

    def test_freeze_on_silence(self):
        out = live_eq_differential(
            self.SNAP, [{"freq_hz": 445.0, "gain_db": -1.0}],
            channel_level_db=-60.0)
        assert out["hold"] and out["reason"] == "freeze_below_gate"

    def test_hold_on_bleed(self):
        out = live_eq_differential(
            self.SNAP, [{"freq_hz": 445.0, "gain_db": -1.0}],
            channel_level_db=-20.0, bleed_ratio=0.9)
        assert out["hold"] and out["reason"] == "bleed_confidence_low"

    def test_near_feedback_octave_window(self):
        assert near_feedback(1000.0, [1050.0])
        assert not near_feedback(1000.0, [1300.0])
