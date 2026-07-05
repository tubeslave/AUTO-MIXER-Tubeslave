"""FX v3: RT-бюджет, STI, темпо-синхро, калибровка, live-полосы."""

import numpy as np
import pytest

from heuristics.fx_v3 import (
    DelayRunawayGuard,
    FXSpeechState,
    FX_TOKEN_LIVE_POLICY,
    calibrate_send_step,
    ducking_config,
    estimate_rt60_schroeder,
    masking_recheck_adjustment,
    mdl_change_allowed,
    select_reverb_params,
    send_level_lane_step,
    sti_proxy,
    tempo_delay_ms,
    tempo_resync_decision,
    venue_class_from_rt,
)

SR = 48000
RNG = np.random.default_rng(51)


def _exponential_decay(rt60_s: float, n_sec: float = 3.0) -> np.ndarray:
    """Синтетическое затухание с известным RT60."""
    t = np.arange(int(n_sec * SR)) / SR
    # мощность: P(t)=exp(−t/τ); −60 дБ при t=rt60 → τ = rt60·10/(60·ln10)
    tau = rt60_s * 10.0 / (60.0 * np.log(10.0))
    return np.exp(-t / (2 * tau)) * RNG.normal(0, 1, len(t))


class TestVenueProfile:
    def test_rt60_recovered_from_synthetic_decay(self):
        result = estimate_rt60_schroeder(_exponential_decay(1.2), SR)
        assert result["rt60_s"] is not None
        assert abs(result["rt60_s"] - 1.2) < 0.4     # грубость ±0.2–0.4 с
        assert "rough" in result["quality"]

    def test_insufficient_data(self):
        assert estimate_rt60_schroeder(np.zeros(100), SR)["rt60_s"] is None

    def test_venue_classes(self):
        assert venue_class_from_rt(0.6) == "dry"
        assert venue_class_from_rt(1.0) == "medium"
        assert venue_class_from_rt(1.8) == "reverberant"

    def test_sti_proxy_monotonic(self):
        # больше RT → ниже STI; больше SNR → выше STI
        assert sti_proxy(0.6, 20.0) > sti_proxy(2.0, 20.0)
        assert sti_proxy(1.0, 25.0) > sti_proxy(1.0, 5.0)
        assert 0.0 <= sti_proxy(1.0, 15.0) <= 1.0


class TestReverbSelection:
    def test_reverberant_hall_shortens_and_darkens(self):
        dry = select_reverb_params("vocal", "pop_rock", 0.6)
        wet = select_reverb_params("vocal", "pop_rock", 2.0)
        assert wet["dcy"] < dry["dcy"]         # RT зала съедает бюджет
        assert wet["damp"] < dry["damp"]       # темнее в гулком
        assert wet["venue_class"] == "reverberant"

    def test_rt_clamped(self):
        params = select_reverb_params("vocal", "ballad", 5.0)
        assert 0.8 <= params["dcy"] <= 3.0

    def test_role_models_official_fw31(self):
        assert select_reverb_params("vocal", "pop_rock", 1.0)["mdl"] == "PLATE"
        assert select_reverb_params("drums", "pop_rock", 1.0)["mdl"] == "ROOM"

    def test_predelay_tempo_fit_clamped(self):
        slow = select_reverb_params("vocal", "ballad", 1.0, tempo_bpm=60.0)
        fast = select_reverb_params("vocal", "edm", 1.0, tempo_bpm=170.0)
        assert 15.0 <= fast["pdel"] <= slow["pdel"] <= 40.0


class TestTempoSync:
    def test_note_multipliers(self):
        d = tempo_delay_ms(120.0, "1/4")
        assert d["delay_ms"] == pytest.approx(500.0)
        assert tempo_delay_ms(120.0, "1/8d")["delay_ms"] == pytest.approx(375.0)
        assert tempo_delay_ms(120.0, "1/8t")["delay_ms"] == pytest.approx(166.7, abs=0.1)

    def test_predelay_clamp(self):
        assert tempo_delay_ms(40.0)["predelay_ms"] == 40.0   # медленно → кламп
        assert tempo_delay_ms(240.0)["predelay_ms"] == 15.6


class TestSendCalibration:
    def test_converges_to_target_ratio(self):
        send, ret, dry = -20.0, -18.0, -20.0   # возврат на +2 против цели −10
        for _ in range(20):
            new_send = calibrate_send_step(send, ret, dry)
            ret += new_send - send             # линейная модель возврата
            send = new_send
        # калибровка останавливается ВНУТРИ допуска −10±3 (не гонит до точной цели)
        assert -13.0 <= (ret - dry) <= -7.0

    def test_inside_tolerance_untouched(self):
        assert calibrate_send_step(-20.0, -31.0, -20.0) == -20.0

    def test_rate_limited_step(self):
        new = calibrate_send_step(-20.0, -10.0, -20.0)   # ошибка 20 дБ
        assert new == pytest.approx(-21.0)               # шаг ≤1 дБ

    def test_ducking_is_return_compressor(self):
        cfg = ducking_config()
        assert "compressor_on_return" in cfg["implementation"]
        assert 6.0 <= cfg["depth_db"] <= 12.0


class TestMaskingRecheck:
    def test_small_delta_no_action(self):
        assert masking_recheck_adjustment(0.5)["action"] == "none"

    def test_large_delta_reduces_sends(self):
        result = masking_recheck_adjustment(2.5)
        assert result["action"] == "reduce_sends_or_darken"
        assert result["send_reduction_db"] <= 3.0


class TestSpeechStateMachine:
    def test_speech_ducks_with_fade_not_hard_mute(self):
        state = FXSpeechState(snapshot_send_db=-12.0)
        out = state.step(0.5, speech_confidence=0.9)
        assert out["state"] == "speech_ducked"
        # фейд ≤6 дБ/с: за 0.5 с не дальше 3 дБ
        assert out["send_db"] >= -15.0 - 1e-9

    def test_low_confidence_holds(self):
        state = FXSpeechState(snapshot_send_db=-12.0)
        out = state.step(1.0, speech_confidence=0.5, singing_confidence=0.5)
        assert out["send_db"] == pytest.approx(-12.0)

    def test_unmute_needs_hysteresis(self):
        state = FXSpeechState(snapshot_send_db=-12.0)
        for _ in range(4):
            state.step(1.0, speech_confidence=0.9)      # задакано
        out = state.step(1.0, singing_confidence=0.9)   # пение началось
        assert out["state"] != "music"                  # 1 с < гистерезиса 1.5 с
        out = state.step(1.0, singing_confidence=0.9)
        assert out["state"] == "music"                  # подтверждено

    def test_silence_between_songs_mutes_and_allows_queue(self):
        state = FXSpeechState(snapshot_send_db=-12.0)
        out = state.step(1.0, silence_between_songs=True)
        assert out["queued_changes_allowed"]
        assert out["state"] == "muted_between_songs"


class TestGuards:
    def test_runaway_guard_caps_feedback(self):
        guard = DelayRunawayGuard()
        alert = None
        for i, level in enumerate([-30.0, -27.0, -24.0, -21.0]):
            alert = guard.step(i * 0.3, level, send_stable=True,
                               current_feedback_pct=60.0) or alert
        assert alert is not None and alert["name"] == "DELAY_RUNAWAY"
        assert alert["action"]["value_pct"] <= 70.0

    def test_no_alert_when_send_moving(self):
        guard = DelayRunawayGuard()
        for i, level in enumerate([-30.0, -27.0, -24.0, -21.0]):
            assert guard.step(i * 0.3, level, send_stable=False,
                              current_feedback_pct=60.0) is None

    def test_send_lane_rate_and_cap(self):
        out = send_level_lane_step(-12.0, 0.0, snapshot_send_db=-12.0, dt_s=1.0)
        assert out == pytest.approx(-11.0)   # ≤1 дБ/с
        out2 = -12.0
        for _ in range(30):
            out2 = send_level_lane_step(out2, 0.0, -12.0, 1.0)
        assert out2 <= -12.0 + 3.0 + 1e-9    # кламп ±3 от снапшота


class TestTempoResync:
    def test_low_confidence_falls_back(self):
        d = tempo_resync_decision(128.0, 120.0, beat_confidence=0.5,
                                  pause_duration_s=5.0)
        assert d["action"] == "none" and "fallback" in d["reason"]

    def test_small_drift_no_action(self):
        d = tempo_resync_decision(121.0, 120.0, beat_confidence=0.9,
                                  pause_duration_s=5.0)
        assert d["action"] == "none"

    def test_resync_only_in_pause(self):
        d = tempo_resync_decision(130.0, 120.0, beat_confidence=0.9,
                                  pause_duration_s=0.5)
        assert d["action"] == "propose_resync_wait_for_pause"
        d2 = tempo_resync_decision(130.0, 120.0, beat_confidence=0.9,
                                   pause_duration_s=3.0)
        assert d2["action"] == "resync_now"
        assert d2["new_delay"]["delay_ms"] > 0


class TestWingDoctrines:
    def test_mdl_change_gated(self):
        assert not mdl_change_allowed(silence_detected=False)
        assert mdl_change_allowed(silence_detected=True)
        assert mdl_change_allowed(silence_detected=False,
                                  glitch_test_passed=True)

    def test_token_policy_table(self):
        assert FX_TOKEN_LIVE_POLICY["mdl"] == "silence_only"
        assert FX_TOKEN_LIVE_POLICY["fxmix"] == "keep_100_percent"
        assert FX_TOKEN_LIVE_POLICY["send_level"] == "safe_rate_limited"
