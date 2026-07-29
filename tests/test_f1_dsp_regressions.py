"""Числовые регрессии Ф1 (задача #2 мастер-плана): баллистика гейта CAIG,
ratio-токены WING на транспортной границе, персистентный K-фильтр.
"""

import numpy as np
import pytest


class TestCAIGBallistics:
    """Ревью Д-1: process() без frame_samples считал баллистику в «кадрах» —
    hold 100 мс при блоке 4800 сэмплов длился ~4800 блоков (8 минут)."""

    @staticmethod
    def _controller(hold_ms=100.0, release_ms=50.0):
        from auto_gate_caig import AutoGateController, InstrumentGroup
        ctl = AutoGateController(num_channels=2, sample_rate=48000)
        ctl.configure_channel(
            0, "snare", InstrumentGroup.DRUMS,
            hold_ms=hold_ms, release_ms=release_ms, threshold_db=-40.0,
        )
        ctl.settings[0].adaptive_threshold = False
        return ctl

    def test_hold_expires_in_milliseconds_not_blocks(self):
        from auto_gate_caig import GateState
        ctl = self._controller(hold_ms=100.0)
        fs = 48000
        block = int(fs * 0.1)  # 100 мс на блок
        t = np.arange(block) / fs
        loud = 0.5 * np.sin(2 * np.pi * 200 * t)
        quiet = np.full(block, 1e-6)

        # открываем гейт
        for _ in range(2):
            decisions = ctl.process({0: loud})
        assert decisions[0].state in (GateState.OPEN, GateState.OPENING)

        # тишина: HOLD (100 мс) обязан истечь за ~1-2 блока по 100 мс
        states = []
        for _ in range(3):
            decisions = ctl.process({0: quiet})
            states.append(decisions[0].state)
        assert states[-1] in (GateState.RELEASING, GateState.CLOSED), (
            f"hold не истёк за 300 мс тишины: {[s.value for s in states]} — "
            "баллистика снова считается в блоках, а не сэмплах"
        )

    def test_gate_reaches_closed_after_release(self):
        from auto_gate_caig import GateState
        ctl = self._controller(hold_ms=50.0, release_ms=50.0)
        fs = 48000
        block = int(fs * 0.1)
        t = np.arange(block) / fs
        loud = 0.5 * np.sin(2 * np.pi * 200 * t)
        quiet = np.full(block, 1e-6)

        for _ in range(2):
            ctl.process({0: loud})
        last = None
        for _ in range(5):  # 500 мс тишины >> hold 50 + release 50
            last = ctl.process({0: quiet})[0].state
        assert last == GateState.CLOSED, f"гейт не закрылся: {last}"


class TestWingRatioTokens:
    """Ревью Д-2: '10.0'/'4.3' — невалидные enum-токены, WING отвергает молча."""

    def test_float_ratios_map_to_valid_tokens(self):
        from wing_client import ratio_to_wing_token, WING_RATIO_STRINGS
        cases = {
            10.0: "10", 20.0: "20", 50.0: "50", 100.0: "100",
            4.3: "4.0", 2.2: "2.0", 1.0: "1.1", 0.0: "1.1",
            7.0: "6.0", 9.5: "10", 1000.0: "100",
        }
        for value, expected in cases.items():
            token = ratio_to_wing_token(value)
            assert token == expected, f"{value} -> {token}, ожидалось {expected}"
            assert token in WING_RATIO_STRINGS

    def test_valid_string_passes_through(self):
        from wing_client import ratio_to_wing_token
        assert ratio_to_wing_token("10") == "10"
        assert ratio_to_wing_token("1.5") == "1.5"

    def test_invalid_string_normalized(self):
        from wing_client import ratio_to_wing_token
        assert ratio_to_wing_token("10.0") == "10"
        assert ratio_to_wing_token("4.3") == "4.0"
        assert ratio_to_wing_token("garbage") == "1.1"

    def test_set_compressor_sends_valid_token(self):
        from unittest.mock import patch, MagicMock
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient, WING_RATIO_STRINGS
            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = True
            from unittest.mock import MagicMock as MM
            client.sock = MM()
            client._osc_throttle_enabled = False
            assert client.set_compressor(1, ratio=10.0) is True
            sent = client.commanded_state.get("/ch/1/dyn/ratio")
            assert sent in WING_RATIO_STRINGS, f"на пульт ушёл токен {sent!r}"
            assert sent == "10"


class TestPersistentKFilter:
    """Ревью P1-5: пересоздание K-фильтра на каждый блок сбрасывало состояние
    HPF 38 Гц каждые ~21 мс — short-term LUFS занижался на НЧ до 0.5 дБ."""

    def test_short_term_lufs_low_freq_no_block_boundary_loss(self):
        from signal_metrics import SignalAnalyzer
        from lufs_gain_staging import KWeightingFilter
        fs = 48000
        block = 1024
        t = np.arange(fs * 4) / fs
        x = 0.25 * np.sin(2 * np.pi * 60 * t)  # kick-диапазон

        analyzer = SignalAnalyzer(channel=0, sample_rate=fs, block_size=block)
        for i in range(0, len(x) - block, block):
            analyzer.process(x[i:i + block])

        # эталон: непрерывная K-взвешенная мощность последних 3 с
        kw = KWeightingFilter(fs)
        k = kw.process(x)
        tail = k[-fs * 3:]
        expected = -0.691 + 10 * np.log10(np.mean(tail ** 2) + 1e-12)

        measured = analyzer.get_metrics().level.lufs_short_term
        assert abs(measured - expected) < 0.1, (
            f"short-term {measured:.2f} vs эталон {expected:.2f} LUFS — "
            "состояние K-фильтра снова сбрасывается на границах блоков"
        )


class TestStereoIntegratedLufs:
    """Ревью P1-4: mono-downmix занижал integrated на 3–6 дБ."""

    def test_decorrelated_noise_not_understated(self):
        from master_mix_eval import _integrated_lufs_stereo, _integrated_lufs
        fs = 48000
        rng = np.random.default_rng(7)
        left = 0.1 * rng.standard_normal(fs * 4)
        right = 0.1 * rng.standard_normal(fs * 4)
        stereo = _integrated_lufs_stereo(left, right, fs)
        mono = _integrated_lufs(0.5 * (left + right), fs)
        # декоррелированный материал: downmix теряет ~6 дБ
        assert stereo - mono > 4.0, f"stereo {stereo:.1f} vs mono {mono:.1f}"

    def test_correlated_signal_matches_bs1770(self):
        from master_mix_eval import _integrated_lufs_stereo
        from lufs_gain_staging import KWeightingFilter
        fs = 48000
        t = np.arange(fs * 4) / fs
        x = 0.1 * np.sin(2 * np.pi * 997 * t)
        # идентичные каналы: z_L+z_R = 2·z → +3.01 дБ к одноканальному
        got = _integrated_lufs_stereo(x, x, fs)
        k = KWeightingFilter(fs).process(x)
        expected = -0.691 + 10 * np.log10(2 * np.mean(k[fs:] ** 2))
        assert abs(got - expected) < 0.2, f"{got:.2f} vs {expected:.2f}"


class TestWelchCoherence:
    """Ревью C1: однокадровая когерентность ≡ 1.0 — фазовый гейт был фиктивным."""

    def test_independent_noise_low_coherence(self):
        from signal_metrics import compare_channels
        fs = 48000
        rng = np.random.default_rng(3)
        a = rng.standard_normal(fs * 2).astype(np.float32)
        b = rng.standard_normal(fs * 2).astype(np.float32)
        icm = compare_channels(a, b, fs, 0, 1)
        assert icm.coherence < 0.2, f"независимые шумы: coherence={icm.coherence:.3f}"
        assert abs(icm.cross_correlation) < 0.2, (
            f"r на лаге для независимых шумов: {icm.cross_correlation:.3f}"
        )

    def test_delayed_copy_high_coherence_and_r(self):
        from signal_metrics import compare_channels
        fs = 48000
        rng = np.random.default_rng(4)
        src = rng.standard_normal(fs * 2).astype(np.float32)
        lag = 48  # 1 мс
        a = src[lag:]
        b = src[:-lag]
        icm = compare_channels(a, b, fs, 0, 1)
        assert icm.coherence > 0.8, f"coherence={icm.coherence:.3f}"
        assert icm.cross_correlation > 0.8
        assert abs(icm.delay_ms - 1.0) < 0.1
        assert not icm.phase_inverted

    def test_inverted_delayed_copy_detected(self):
        from signal_metrics import compare_channels
        fs = 48000
        rng = np.random.default_rng(5)
        src = rng.standard_normal(fs * 2).astype(np.float32)
        lag = 28  # 0.58 мс — кейс из зафиксированного правила
        a = src[lag:]
        b = -src[:-lag]
        icm = compare_channels(a, b, fs, 0, 1)
        assert icm.cross_correlation < -0.8
        assert icm.phase_inverted, (
            "инверсия при сдвиге 0.58 мс не детектирована — r снова "
            "считается на нулевом лаге"
        )
