"""Tests for ml.eq_normalization -- corrective EQ to match target spectral profiles."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))


class TestTargetCurves:
    """Tests for the TARGET_CURVES constant dictionary."""

    def test_target_curves_not_empty(self):
        from ml.eq_normalization import TARGET_CURVES
        assert len(TARGET_CURVES) > 0

    def test_flat_curve_exists(self):
        from ml.eq_normalization import TARGET_CURVES
        assert 'flat' in TARGET_CURVES

    def test_flat_curve_all_zeros(self):
        from ml.eq_normalization import TARGET_CURVES
        flat = TARGET_CURVES['flat']
        for freq, gain in flat:
            assert gain == 0.0, f"Flat curve should be 0 dB at {freq} Hz"

    def test_known_curves_present(self):
        from ml.eq_normalization import TARGET_CURVES
        expected = ['flat', 'warm', 'bright', 'vocal_presence', 'kick_drum']
        for name in expected:
            assert name in TARGET_CURVES

    def test_curve_format(self):
        from ml.eq_normalization import TARGET_CURVES
        for name, points in TARGET_CURVES.items():
            assert isinstance(points, list)
            for point in points:
                assert len(point) == 2, f"Each point in {name} should be (freq, gain)"
                assert point[0] > 0, f"Frequency should be positive in {name}"


class TestEQBand:
    """Tests for the EQBand dataclass."""

    def test_eq_band_creation(self):
        from ml.eq_normalization import EQBand
        band = EQBand(frequency=1000.0, gain_db=3.0, q=1.4)
        assert band.frequency == 1000.0
        assert band.gain_db == 3.0
        assert band.q == 1.4
        assert band.band_type == 'peaking'

    def test_eq_band_custom_type(self):
        from ml.eq_normalization import EQBand
        band = EQBand(frequency=80.0, gain_db=4.0, q=0.7, band_type='low_shelf')
        assert band.band_type == 'low_shelf'


class TestEQProfile:
    """Tests for the EQProfile dataclass."""

    def test_eq_profile_creation(self):
        from ml.eq_normalization import EQProfile, EQBand
        bands = [
            EQBand(100.0, 2.0, 1.0),
            EQBand(1000.0, -3.0, 1.4),
        ]
        profile = EQProfile(bands=bands, channel_id=1, label='test')
        assert len(profile.bands) == 2
        assert profile.channel_id == 1
        assert profile.label == 'test'


class TestEQNormalizer:
    """Tests for the EQNormalizer."""

    def test_instantiation(self):
        from ml.eq_normalization import EQNormalizer
        norm = EQNormalizer(sample_rate=48000, fft_size=4096, n_bands=6)
        assert norm.sample_rate == 48000
        assert norm.fft_size == 4096
        assert norm.n_bands == 6

    def test_analyze_spectrum_shape(self):
        from ml.eq_normalization import EQNormalizer
        norm = EQNormalizer(sample_rate=48000, fft_size=2048)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        spectrum = norm.analyze_spectrum(audio)
        assert spectrum.shape == (2048 // 2 + 1,)

    def test_analyze_spectrum_short_audio(self):
        from ml.eq_normalization import EQNormalizer
        norm = EQNormalizer(sample_rate=48000, fft_size=4096)
        # Audio shorter than fft_size should be padded
        short_audio = np.random.randn(1000).astype(np.float32) * 0.1
        spectrum = norm.analyze_spectrum(short_audio)
        assert spectrum.shape == (4096 // 2 + 1,)

    def test_interpolate_target_flat(self):
        from ml.eq_normalization import EQNormalizer
        norm = EQNormalizer(sample_rate=48000, fft_size=4096)
        target = norm.interpolate_target('flat')
        assert target.shape == (4096 // 2 + 1,)
        # Flat target should be all zeros
        assert np.allclose(target[1:], 0.0, atol=0.1)

    def test_interpolate_target_unknown_falls_back_to_flat(self):
        from ml.eq_normalization import EQNormalizer
        norm = EQNormalizer(sample_rate=48000, fft_size=4096)
        target = norm.interpolate_target('nonexistent_curve')
        flat = norm.interpolate_target('flat')
        assert np.allclose(target, flat)

    def test_compute_correction_returns_profile(self):
        from ml.eq_normalization import EQNormalizer, EQProfile
        norm = EQNormalizer(sample_rate=48000, fft_size=2048, n_bands=4)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        profile = norm.compute_correction(audio, target_curve='flat')
        assert isinstance(profile, EQProfile)
        assert profile.label == 'correction_flat'

    def test_compute_correction_band_gains_clamped(self):
        from ml.eq_normalization import EQNormalizer
        norm = EQNormalizer(sample_rate=48000, fft_size=2048, n_bands=4)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        profile = norm.compute_correction(audio, target_curve='warm', max_gain_db=6.0)
        for band in profile.bands:
            assert abs(band.gain_db) <= 6.1  # small tolerance for rounding

    def test_apply_profile_to_osc(self):
        from ml.eq_normalization import EQNormalizer, EQProfile, EQBand
        norm = EQNormalizer()
        bands = [
            EQBand(100.0, 2.0, 1.0, 'low_shelf'),
            EQBand(3000.0, -1.5, 1.4, 'peaking'),
        ]
        profile = EQProfile(bands=bands, label='test')
        commands = norm.apply_profile_to_osc(profile)
        assert len(commands) == 2
        assert commands[0]['band'] == 1
        assert commands[0]['frequency'] == 100.0
        assert commands[0]['type'] == 'low_shelf'
        assert commands[1]['band'] == 2
        assert commands[1]['gain_db'] == -1.5
