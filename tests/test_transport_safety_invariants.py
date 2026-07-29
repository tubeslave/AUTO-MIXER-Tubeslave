"""Транспортные инварианты безопасности (совет 2026-07-29, план Ф1).

1. Fader ceiling действует на ВСЕХ уровневых поверхностях WingClient
   (main/dca/bus/fx-return/sends), а не только на input-каналах.
2. DLiveClient: потолок фейдера на транспортной границе + все write-методы
   возвращают результат отправки (None читался manual-write gate'ом как
   applied — ложное подтверждение 48V).
3. TruePeakMeter: полоса до fs/2 — ВЧ-пики не занижаются (раньше LPF резал
   на 12 кГц: синус 0 dBFS @ 20 кГц читался как −20 dBTP).
4. auto_eq: срез остаётся срезом (отрицательный max_cut профиля не меняет
   знак коррекции) и не конвертируется позиционно в shelf.
"""

import logging
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


def _make_wing_client(**kwargs):
    with patch.dict("sys.modules", {
        "pythonosc": MagicMock(),
        "pythonosc.udp_client": MagicMock(),
        "pythonosc.dispatcher": MagicMock(),
        "pythonosc.osc_server": MagicMock(),
        "pythonosc.osc_message_builder": MagicMock(),
        "pythonosc.osc_message": MagicMock(),
    }):
        from wing_client import WingClient
        client = WingClient(ip="127.0.0.1", port=2223, **kwargs)
        client.is_connected = True
        client.sock = MagicMock()
        client._osc_throttle_enabled = False
        return client


class TestCeilingOnAllWingSurfaces:
    """DCA на 0, шина +6 — и потолок каналов ничего не значит."""

    @pytest.mark.parametrize("call,addr", [
        (lambda c: c.set_main_fader(1, 5.0), "/main/1/fdr"),
        (lambda c: c.set_dca_fader(3, 5.0), "/dca/3/fdr"),
        (lambda c: c.set_bus_fader(2, 5.0), "/bus/2/fdr"),
        (lambda c: c.set_fx_return_level(9, 5.0), "/bus/9/fdr"),
    ])
    def test_fader_surfaces_clamped_to_default_ceiling(self, call, addr):
        client = _make_wing_client()
        assert call(client) is True
        assert client.commanded_state.get(addr) == 0.0

    def test_channel_send_level_clamped(self):
        client = _make_wing_client()
        assert client.set_channel_send(4, 13, level=7.5) is True
        assert client.commanded_state.get("/ch/4/send/13/lvl") == 0.0

    def test_channel_main_send_level_clamped(self):
        client = _make_wing_client()
        assert client.set_channel_main_send(4, 1, level=7.5) is True
        assert client.commanded_state.get("/ch/4/main/1/lvl") == 0.0

    def test_opt_in_ceiling_respected_on_bus(self):
        client = _make_wing_client(fader_ceiling_db=6.0)
        assert client.set_bus_fader(2, 5.0) is True
        assert client.commanded_state.get("/bus/2/fdr") == 5.0
        # аппаратный кламп сохраняется
        assert client.set_bus_fader(2, 20.0) is True
        assert client.commanded_state.get("/bus/2/fdr") == 6.0

    def test_below_ceiling_untouched(self):
        client = _make_wing_client()
        assert client.set_dca_fader(1, -12.0) is True
        assert client.commanded_state.get("/dca/1/fdr") == -12.0


class TestDLiveTransportInvariants:
    @staticmethod
    def _make_client(monkeypatch=None, sent=None):
        from dlive_client import DLiveClient
        client = DLiveClient(ip="127.0.0.1")
        sent_msgs = sent if sent is not None else []

        def fake_send_raw(msg):
            sent_msgs.append(bytes(msg))
            return True

        client._send_raw = fake_send_raw
        return client, sent_msgs

    def test_set_fader_clamped_to_ceiling(self, caplog):
        from dlive_client import db_to_fader_value
        client, sent = self._make_client()
        with caplog.at_level(logging.WARNING):
            assert client.set_fader(1, 8.0) is True
        # NRPN data-байт соответствует потолку 0 dBFS, не запрошенным +8
        assert any("Fader ceiling enforced" in r.message for r in caplog.records)
        expected = db_to_fader_value(0.0)
        overshoot = db_to_fader_value(8.0)
        payload = sent[-1]
        assert bytes([expected & 0x7F]) in payload or expected in payload
        assert overshoot != expected  # тест не вырожден

    def test_write_methods_return_send_result(self):
        client, _ = self._make_client()
        assert client.set_phantom(1, True) is True
        assert client.set_pad(1, True) is True
        assert client.set_channel_name(1, "KICK") is True
        assert client.set_channel_colour(1, "red") is True

    def test_write_methods_propagate_failure(self):
        client, _ = self._make_client()
        client._send_raw = lambda msg: False
        assert client.set_phantom(1, True) is False
        assert client.set_pad(1, False) is False
        assert client.set_channel_name(1, "KICK") is False
        assert client.set_channel_colour(1, "red") is False
        assert client.set_fader(1, -3.0) is False

    def test_invalid_eq_band_returns_false(self):
        client, _ = self._make_client()
        assert client.set_eq_band(1, 9, 1000.0, -3.0, 2.0) is False


class TestTruePeakHighFrequency:
    """BS.1770-4: true peak честен по всей полосе, а не до 12 кГц."""

    @pytest.mark.parametrize("freq", [997, 5000, 12000, 16000, 20000, 23000])
    def test_full_scale_sine_reads_near_zero_dbtp(self, freq):
        from lufs_gain_staging import TruePeakMeter
        fs = 48000
        meter = TruePeakMeter(fs)
        t = np.arange(int(fs * 0.5)) / fs
        x = np.sin(2 * np.pi * freq * t)
        meter.process(x[: len(x) // 2])  # прогрев состояния фильтра
        tp = meter.process(x[len(x) // 2:])
        assert abs(tp) < 0.5, f"{freq} Hz: {tp:+.2f} dBTP (ожидалось ~0)"

    def test_intersample_peak_detected(self):
        """Синус fs/4 с фазой π/4: сэмплы на −3 dBFS, истинный пик 0 dBFS."""
        from lufs_gain_staging import TruePeakMeter
        fs = 48000
        meter = TruePeakMeter(fs)
        t = np.arange(int(fs * 0.5)) / fs
        x = np.sin(2 * np.pi * 12000 * t + np.pi / 4)
        sample_peak_db = 20 * np.log10(np.max(np.abs(x)))
        assert sample_peak_db < -2.5  # предусловие: межсэмпловый кейс
        meter.process(x[: len(x) // 2])
        tp = meter.process(x[len(x) // 2:])
        assert abs(tp) < 0.5, f"межсэмпловый пик не пойман: {tp:+.2f} dBTP"

    def test_true_peak_never_below_sample_peak(self):
        from lufs_gain_staging import TruePeakMeter
        rng = np.random.default_rng(42)
        meter = TruePeakMeter(48000)
        x = rng.standard_normal(4800) * 0.3
        tp = meter.process(x)
        sample_peak = 20 * np.log10(np.max(np.abs(x)))
        assert tp >= sample_peak - 1e-9


class TestAutoEQCutSign:
    """Ревью D1: deviation −8 дБ на частоте «грязи» давал +4 дБ БУСТА."""

    @staticmethod
    def _corrections_for_hump(hump_freq=250.0):
        from auto_eq import EQCorrector, InstrumentProfile, SpectralData
        corrector = EQCorrector()
        profile = InstrumentProfile(
            name="test",
            description="synthetic",
            target_curve=[(20.0, -6.0), (20000.0, -6.0)],
            cut_frequencies=[(hump_freq, -6.0, 2.0)],
            boost_frequencies=[],
        )
        freqs = np.linspace(20, 20000, 2048)
        spectrum_db = np.full_like(freqs, -52.0)
        spectrum_db += 12.0 * np.exp(-0.5 * ((freqs - hump_freq) / 60.0) ** 2)
        spectrum_linear = 10 ** (spectrum_db / 20.0)
        data = SpectralData(
            spectrum=spectrum_linear,
            frequencies=freqs,
            peak_freq=hump_freq,
            centroid=hump_freq,
            rolloff=8000.0,
            flatness=0.3,
            bandwidth=2000.0,
            peaks=[(hump_freq, float(np.max(spectrum_db)))],
        )
        return corrector.calculate_correction(data, profile)

    def test_profile_cut_is_negative_gain(self):
        corrections = self._corrections_for_hump(250.0)
        cuts = [c for c in corrections if abs(c.frequency - 250.0) < 30.0]
        assert cuts, "коррекция на 250 Гц не сгенерирована"
        for c in cuts:
            assert c.gain < 0, (
                f"срез превратился в буст: {c.frequency:.0f} Гц {c.gain:+.2f} дБ"
            )

    def test_no_positional_shelf_conversion(self):
        corrections = self._corrections_for_hump(250.0)
        for c in corrections:
            assert c.band_type == "peak", (
                f"позиционная shelf-конвертация вернулась: {c.frequency:.0f} Гц"
                f" → {c.band_type}"
            )
