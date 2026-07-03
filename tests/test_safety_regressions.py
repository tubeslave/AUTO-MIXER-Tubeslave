"""Регрессионные тесты на исправленные safety/DSP баги (июль 2026).

Каждый тест фиксирует конкретный дефект, чтобы он не вернулся.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from feedback_detector import (
    MAX_NOTCH_FILTERS,
    FeedbackDetectorConfig,
    _ChannelFeedbackState,
    FeedbackDetector,
)
from gain_staging_fixes import SignalStats
from lufs_gain_staging import TruePeakMeter
from auto_phase_gcc_phat import GCCPHATAnalyzer


# ── TruePeakMeter: состояние FIR между блоками ──────────────────────


def test_true_peak_block_split_matches_whole_signal():
    """Пик на стыке блоков не должен занижаться: измерение по блокам
    обязано совпадать с измерением всего сигнала целиком."""
    sr = 48000
    t = np.arange(sr // 10) / sr
    # Синус вблизи Найквиста/4 с пиком между сэмплами
    signal_data = 0.9 * np.sin(2 * np.pi * 11997.0 * t + 0.3)

    whole = TruePeakMeter(sample_rate=sr)
    whole_peak = whole.process(signal_data)

    blocked = TruePeakMeter(sample_rate=sr)
    block_size = 480
    for i in range(0, len(signal_data), block_size):
        blocked.process(signal_data[i:i + block_size])
    block_peak = blocked.get_max_peak_dbtp()

    assert block_peak == pytest.approx(whole_peak, abs=0.05)


def test_true_peak_reset_clears_filter_state():
    meter = TruePeakMeter(sample_rate=48000)
    meter.process(np.ones(1000) * 0.9)
    meter.reset()
    assert np.all(meter._filter_zi == 0.0)
    assert meter.get_max_peak_dbtp() == -100.0


# ── Feedback detector: слоты notch-фильтров ─────────────────────────


def _notch_config():
    return FeedbackDetectorConfig()


def test_notch_slots_unique_and_within_hardware():
    """Каждый notch занимает уникальный аппаратный слот 0..6."""
    state = _ChannelFeedbackState(channel_id=1, config=_notch_config())
    for i in range(MAX_NOTCH_FILTERS):
        nf = state.add_notch(500.0 * (i + 1) * 1.3, 0.8, now=0.0)
        assert nf is not None
    slots = [nf.slot_index for nf in state.notch_filters]
    assert sorted(slots) == list(range(MAX_NOTCH_FILTERS))


def test_notch_slot_reused_after_release():
    """После release/add индексы не растут монотонно: слот переиспользуется,
    иначе (slot % 4) перезаписывал бы чужие EQ-бэнды."""
    state = _ChannelFeedbackState(channel_id=1, config=_notch_config())
    for i in range(MAX_NOTCH_FILTERS):
        state.add_notch(500.0 * (i + 1) * 1.3, 0.8, now=0.0)
    assert state.release_notch(2) is True
    nf = state.add_notch(15000.0, 0.8, now=1.0)
    assert nf is not None
    assert nf.slot_index == 2


def test_generate_osc_commands_no_band_collision():
    """OSC-команды 7 notch-фильтров не пишут дважды в один бэнд."""
    detector = FeedbackDetector()
    state = detector._get_or_create_state(1)
    for i in range(MAX_NOTCH_FILTERS):
        state.add_notch(500.0 * (i + 1) * 1.3, 0.8, now=0.0)
    commands = detector.generate_osc_commands(1)
    freq_addresses = [addr for addr, _ in commands if addr.endswith("f")]
    assert len(freq_addresses) == MAX_NOTCH_FILTERS
    assert len(set(freq_addresses)) == MAX_NOTCH_FILTERS


# ── calculate_safe_gain: пиковый клэмп ──────────────────────────────


def test_safe_gain_never_exceeds_peak_headroom():
    """Gain не превышает запас до пикового лимита; при пике уже выше
    лимита коррекция обязана быть отрицательной (не буст)."""
    stats = SignalStats(channel_id=1)
    # Высокий crest factor: пик -1 dBFS (выше лимита -3), LUFS -18
    for _ in range(200):
        stats.update_sample(true_peak_db=-1.0, lufs=-18.0, sample_rms_db=-18.0)

    stats.calculate_safe_gain(target_lufs=-18.0, max_peak_limit=-3.0)
    delta_peak = -3.0 - stats.max_true_peak_db  # = -2.0: пик уже над лимитом
    assert stats.suggested_gain_db <= delta_peak + 1e-6
    assert stats.suggested_gain_db < 0


# ── GCC-PHAT: знак задержки и кольцевой буфер ───────────────────────


def _make_delayed_pair(sr, delay_samples, n):
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, n)
    tgt = np.zeros_like(ref)
    if delay_samples > 0:
        tgt[delay_samples:] = ref[:-delay_samples]
    elif delay_samples < 0:
        tgt[:delay_samples] = ref[-delay_samples:]
    else:
        tgt = ref.copy()
    return ref, tgt


def test_gcc_phat_ring_buffer_wraparound():
    """После оборота кольцевого буфера измерение остаётся корректным
    (окно не должно содержать склейку старых/новых данных)."""
    sr = 48000
    analyzer = GCCPHATAnalyzer(sample_rate=sr)
    delay = 480  # 10 ms
    # Пишем в 1.5 раза больше ёмкости буфера кадрами по 1024
    total = int(analyzer.buffer_size * 1.5)
    ref, tgt = _make_delayed_pair(sr, delay, total)
    for i in range(0, total - 1024, 1024):
        analyzer.add_frames(ref[i:i + 1024], tgt[i:i + 1024])
    m = analyzer.compute_delay()
    assert m.delay_ms == pytest.approx(10.0, abs=0.5)


def test_delay_applied_only_to_early_target():
    """Задержка добавляется target-каналу только когда он ОПЕРЕЖАЕТ
    reference (delay_ms < 0); отстающий target не задерживается."""
    from auto_phase_gcc_phat import AutoPhaseAligner, DelayMeasurement

    mixer = MagicMock()
    aligner = AutoPhaseAligner(mixer_client=mixer)

    # Target отстаёт (delay_ms > 0) → команды нет
    lagging = DelayMeasurement(
        delay_ms=10.0, delay_samples=480.0, correlation_peak=1.0,
        psr=30.0, snr_db=30.0, confidence=1.0, coherence=1.0,
    )
    aligner._send_delay_command(5, lagging)
    mixer.set_channel_delay.assert_not_called()

    # Target опережает (delay_ms < 0) → задержка |delay|
    leading = DelayMeasurement(
        delay_ms=-10.0, delay_samples=-480.0, correlation_peak=1.0,
        psr=30.0, snr_db=30.0, confidence=1.0, coherence=1.0,
    )
    aligner._send_delay_command(5, leading)
    mixer.set_channel_delay.assert_called_once_with(5, 10.0)


def test_max_delay_clamped_to_unambiguous_range():
    """max_delay_ms за пределами ±fft_size/2 клэмпится (заворот знака)."""
    analyzer = GCCPHATAnalyzer(sample_rate=48000, fft_size=4096, max_delay_ms=50.0)
    assert analyzer.max_delay_samples <= 4096 // 2 - 1


# ── Panner: стабильность кроссовера ─────────────────────────────────


def test_crossover_stable_no_nan():
    """LR4-кроссовер не расходится (прежние захардкоженные коэффициенты
    имели полюс |z| > 1 → выход рос до Inf/NaN)."""
    from auto_panner_adaptive import LinkwitzRileyCrossover

    xover = LinkwitzRileyCrossover(sample_rate=48000)
    rng = np.random.default_rng(3)
    for _ in range(50):  # ~1 c аудио блоками
        audio = rng.normal(0, 0.3, 1024)
        low, mid, high = xover.process(audio)
        for band in (low, mid, high):
            assert np.all(np.isfinite(band))
            assert np.max(np.abs(band)) < 10.0


def test_crossover_band_separation():
    """НЧ-тон уходит в low-полосу, ВЧ-тон — в high."""
    from auto_panner_adaptive import LinkwitzRileyCrossover

    sr = 48000
    t = np.arange(sr // 2) / sr
    low_tone = np.sin(2 * np.pi * 60.0 * t)
    xover = LinkwitzRileyCrossover(sample_rate=sr)
    low, mid, high = xover.process(low_tone)
    tail = slice(len(t) // 2, None)  # после переходного процесса
    assert np.sqrt(np.mean(low[tail] ** 2)) > 5 * np.sqrt(np.mean(high[tail] ** 2))

    high_tone = np.sin(2 * np.pi * 10000.0 * t)
    xover2 = LinkwitzRileyCrossover(sample_rate=sr)
    low2, mid2, high2 = xover2.process(high_tone)
    assert np.sqrt(np.mean(high2[tail] ** 2)) > 5 * np.sqrt(np.mean(low2[tail] ** 2))


# ── Gate: тайминги учитывают размер кадра ───────────────────────────


def test_gate_attack_completes_within_declared_time():
    """Attack 0.5 мс завершается за ~0.5 мс аудио, а не за 64× дольше."""
    from auto_gate_caig import GateProcessor, GateSettings, GateState, GateFeatures

    sr = 48000
    proc = GateProcessor(sample_rate=sr)
    settings = GateSettings()  # attack 0.5 ms
    proc.configure(settings)

    frame = 64  # сэмплов на кадр
    loud = GateFeatures(rms_db=-10.0, peak_db=-5.0,
                        crest_factor_db=5.0, lf_energy_db=-20.0)

    frames_needed = 0
    for _ in range(1000):
        decision = proc.process(loud, threshold_db=-40.0, settings=settings,
                                frame_samples=frame)
        frames_needed += 1
        if decision.state == GateState.OPEN:
            break
    elapsed_ms = frames_needed * frame / sr * 1000
    assert decision.state == GateState.OPEN
    assert elapsed_ms <= 4.0, f"attack took {elapsed_ms:.1f} ms (declared 0.5 ms)"


def test_gate_noise_floor_rises_to_real_level():
    """Трекер noise floor поднимается от -70 к реальному полу -50."""
    from auto_gate_caig import AdaptiveThreshold

    at = AdaptiveThreshold()
    # 30 секунд кадров по 64 сэмпла с полом -50 dB
    frame_sec = 64 / 48000
    for _ in range(int(30 / frame_sec)):
        at.update_noise_floor(-50.0, frame_sec=frame_sec)
    assert at.noise_floor_db > -55.0, f"floor stuck at {at.noise_floor_db:.1f}"


# ── dLive: partial read TCP и mute readback ─────────────────────────


def _dlive():
    from dlive_client import DLiveClient
    return DLiveClient()


def test_dlive_parser_survives_any_split_boundary():
    """MIDI-сообщение, разрезанное на любой границе recv(), не теряется."""
    from dlive_client import build_nrpn, fader_value_to_db

    client_ref = _dlive()
    msg = build_nrpn(0, 0x00, 4, 8000)  # fader канала 5
    client_ref._parse_incoming(msg)
    expected = client_ref.state.get("fader/5")
    assert expected == 8000

    for split in range(1, len(msg)):
        client = _dlive()
        client._parse_incoming(msg[:split])
        client._parse_incoming(msg[split:])
        assert client.state.get("fader/5") == expected, f"split at {split}"


def test_dlive_incomplete_sysex_does_not_drop_following_messages():
    """Незавершённый SysEx копится в буфере, а сообщения после его
    завершения парсятся (раньше break терял весь остаток буфера)."""
    from dlive_client import build_note_on

    client = _dlive()
    sysex_head = bytes([0xF0, 0x00, 0x00, 0x1A])
    client._parse_incoming(sysex_head)          # без F7 — ждём хвост
    tail_and_note = bytes([0xF7]) + build_note_on(0, 4, 0x7F)
    client._parse_incoming(tail_and_note)
    assert client.state.get("note/0/4") == 0x7F


def test_dlive_mute_pair_readback():
    """Пара NoteOn(0x7F)+NoteOn(0x00) = muted; хвост vel=0 не сбрасывает."""
    from dlive_client import build_note_on

    client = _dlive()
    ch = 3
    note = ch - 1
    midi_ch = client._midi_channel()

    client._parse_incoming(
        build_note_on(midi_ch, note, 0x7F) + build_note_on(midi_ch, note, 0x00)
    )
    assert client.get_mute(ch) is True

    client._parse_incoming(
        build_note_on(midi_ch, note, 0x3F) + build_note_on(midi_ch, note, 0x00)
    )
    assert client.get_mute(ch) is False


# ── Feedback detector: выбор ДОМИНИРУЮЩЕГО пика заводки ──────────────


def test_feedback_detector_picks_dominant_peak_frequency():
    """Notch ставится на частоту доминирующей (самой громкой) заводки,
    а не первого шумового пика в порядке словаря. Найдено при валидации
    на реальном аудиотракте: тон 3150 Гц давал событие на случайной
    частоте, потому что выбор шёл по dict-порядку tracked_peaks."""
    from feedback_detector import FeedbackDetector, FeedbackDetectorConfig

    sr = 48000
    fb_freq = 3150.0
    cfg = FeedbackDetectorConfig(sample_rate=sr, fft_size=4096, persistence_frames=4)
    det = FeedbackDetector(config=cfg)
    rng = np.random.default_rng(7)

    detected = []
    amp = 0.02
    for _ in range(10):
        tb = np.arange(int(0.12 * sr)) / sr
        amp *= 1.6  # растущая заводка
        tone = (amp * np.sin(2 * np.pi * fb_freq * tb)).astype(np.float32)
        background = (0.03 * rng.standard_normal(len(tb))).astype(np.float32)
        block = np.clip(tone + background, -1.0, 1.0).astype(np.float32)
        ev = det.process_audio(1, np.ascontiguousarray(block))
        if ev is not None:
            detected.append(ev.frequency_hz)

    assert detected, "заводка не обнаружена вовсе"
    # Большинство событий — на реальной частоте заводки (± один бин ~12 Гц);
    # ранние кадры до накопления persistence могут дать шум
    on_target = [f for f in detected if abs(f - fb_freq) < 25]
    assert len(on_target) >= len(detected) * 0.6, (
        f"доминирующий пик выбран неверно: {detected}"
    )
    # Последнее (устойчивое) событие обязано быть на верной частоте
    assert abs(detected[-1] - fb_freq) < 25, f"устойчивая фаза: {detected[-1]}"


# ── dLive: best-effort верификация записи фейдера ───────────────────


def test_dlive_set_fader_verified_applied_on_echo():
    """Эхо NRPN от пульта с совпадающим значением → status=applied."""
    from dlive_client import build_nrpn, db_to_fader_value

    client = _dlive()
    client.is_connected = True

    def fake_send(data):
        # Пульт эхоит применённое значение
        echo = build_nrpn(0, 0x00, 4, db_to_fader_value(-6.0))
        client._parse_incoming(echo)
        return True

    client._send_raw = fake_send
    result = client.set_fader_verified(5, -6.0, timeout_sec=0.2)
    assert result["status"] == "applied"
    assert abs(result["readback_db"] - (-6.0)) <= 0.5


def test_dlive_set_fader_verified_unconfirmed_without_echo():
    """Нет эха → status=unconfirmed (НЕ applied), запись не подтверждена."""
    client = _dlive()
    client.is_connected = True
    client._send_raw = lambda data: True
    result = client.set_fader_verified(5, -6.0, timeout_sec=0.1)
    assert result["status"] == "unconfirmed"


def test_dlive_set_fader_verified_mismatch():
    """Эхо с другим значением (пульт клэмпнул) → status=mismatch."""
    from dlive_client import build_nrpn, db_to_fader_value

    client = _dlive()
    client.is_connected = True

    def fake_send(data):
        echo = build_nrpn(0, 0x00, 4, db_to_fader_value(0.0))
        client._parse_incoming(echo)
        return True

    client._send_raw = fake_send
    result = client.set_fader_verified(5, -20.0, timeout_sec=0.2)
    assert result["status"] == "mismatch"


# ── Auth: клиент без токена не проходит ─────────────────────────────


def test_auth_requires_token_when_enabled():
    from auth import AuthManager

    auth = AuthManager(enabled=True)
    assert auth.authenticate(token=None, ip="127.0.0.1") is False
    token = auth.generate_token("client-1")
    assert auth.authenticate(token=token, ip="127.0.0.1") is True


# ── BaseAgent: start() на PAUSED не создаёт второй цикл ─────────────


def test_start_on_paused_agent_resumes_without_second_loop():
    import asyncio as aio
    from agents.base_agent import BaseAgent, AgentState

    class DummyAgent(BaseAgent):
        def __init__(self):
            super().__init__("dummy", cycle_interval=0.01)
            self.cycles = 0

        async def observe(self):
            self.cycles += 1
            return {}

        async def decide(self, observation):
            return {}

        async def act(self, decision):
            pass

    async def scenario():
        agent = DummyAgent()
        task = aio.create_task(agent.start())
        await aio.sleep(0.05)
        agent.pause()
        assert agent.state == AgentState.PAUSED
        # Повторный start() на PAUSED должен просто возобновить
        await agent.start()
        assert agent.state == AgentState.RUNNING
        await aio.sleep(0.05)
        agent.stop()
        await aio.wait_for(task, timeout=2.0)
        assert agent.cycles > 0

    aio.run(scenario())
