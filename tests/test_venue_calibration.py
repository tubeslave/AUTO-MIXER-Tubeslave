"""Ф5б: venue-калибровка порогов feedback-детектора по шуму зала."""

import numpy as np
import pytest

from venue_calibration import (
    HEIGHT_CLAMP_DB,
    VenueNoiseProfile,
    calibrate_feedback_thresholds,
    profile_venue_noise,
)

FS = 48000


def _noise(level_dbfs, seconds=3, seed=1):
    rng = np.random.default_rng(seed)
    amp = 10 ** (level_dbfs / 20.0)
    return (rng.standard_normal(FS * seconds) * amp).astype(np.float32)


class TestNoiseProfile:
    def test_floor_measured_robustly(self):
        """Пол — по тихим блокам: случайный стук в «тишине» не завышает."""
        x = _noise(-60.0)
        x[FS:FS + 4800] += 0.5  # одиночный стук
        profile = profile_venue_noise({1: x}, FS)
        assert profile.channel_floor_dbfs[1] == pytest.approx(-60.0, abs=2.0)

    def test_hum_detected(self):
        t = np.arange(FS * 3) / FS
        x = _noise(-70.0) + 0.003 * np.sin(2 * np.pi * 50.0 * t)
        profile = profile_venue_noise({1: x.astype(np.float32)}, FS)
        assert profile.hum_detected and profile.hum_freq_hz == 50.0

    def test_clean_quiet_room_no_hum(self):
        profile = profile_venue_noise({1: _noise(-80.0)}, FS)
        assert not profile.hum_detected


class TestCalibration:
    def test_quiet_room_keeps_base_thresholds(self):
        profile = profile_venue_noise({1: _noise(-85.0)}, FS)
        calib = calibrate_feedback_thresholds(profile)
        assert calib.peak_height_db == pytest.approx(-40.0)
        assert calib.steady_state_min_level_dbfs == pytest.approx(-30.0)
        assert calib.peak_prominence_bump_db == 0.0

    def test_noisy_room_raises_thresholds(self):
        """Пол −50 dBFS: порог пика поднимается до −30 (пол+20)."""
        profile = profile_venue_noise({1: _noise(-50.0)}, FS)
        calib = calibrate_feedback_thresholds(profile)
        assert calib.peak_height_db == pytest.approx(-30.0, abs=2.0)
        assert calib.steady_state_min_level_dbfs == pytest.approx(-24.0, abs=2.0)
        assert calib.notes

    def test_clamps_hold_for_extreme_noise(self):
        profile = VenueNoiseProfile(overall_floor_dbfs=-10.0)
        calib = calibrate_feedback_thresholds(profile)
        assert calib.peak_height_db <= HEIGHT_CLAMP_DB[1]

    def test_hum_bumps_prominence(self):
        profile = VenueNoiseProfile(
            overall_floor_dbfs=-70.0, hum_detected=True,
            hum_freq_hz=50.0, hum_prominence_db=18.0,
        )
        calib = calibrate_feedback_thresholds(profile)
        assert calib.peak_prominence_bump_db == 2.0


class TestEngineIntegration:
    def test_detector_config_uses_calibration(self):
        from auto_soundcheck_engine import AutoSoundcheckEngine
        engine = AutoSoundcheckEngine(num_channels=1, auto_discover=False)
        base = engine._build_feedback_detector_config()
        profile = VenueNoiseProfile(overall_floor_dbfs=-50.0)
        engine.venue_calibration = calibrate_feedback_thresholds(profile)
        calibrated = engine._build_feedback_detector_config()
        assert calibrated.peak_height_db > base.peak_height_db
        assert calibrated.steady_state_min_level_dbfs > -30.0

    def test_run_venue_calibration_failure_is_non_fatal(self):
        from auto_soundcheck_engine import AutoSoundcheckEngine
        engine = AutoSoundcheckEngine(num_channels=1, auto_discover=False)
        engine.audio_capture = None
        engine._run_venue_calibration()  # не должно бросать
        assert getattr(engine, "venue_calibration", None) is None
