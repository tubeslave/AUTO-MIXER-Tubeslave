"""Tests for ml.reference_profiles -- reference mix profiles for style matching."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))


class TestBuiltinProfiles:
    """Tests for the BUILTIN_PROFILES constant."""

    def test_builtin_profiles_not_empty(self):
        from ml.reference_profiles import BUILTIN_PROFILES
        assert len(BUILTIN_PROFILES) > 0

    def test_known_profiles_present(self):
        from ml.reference_profiles import BUILTIN_PROFILES
        expected = ['modern_rock', 'jazz_live', 'pop_broadcast', 'worship', 'edm_festival']
        for name in expected:
            assert name in BUILTIN_PROFILES

    def test_profile_required_keys(self):
        from ml.reference_profiles import BUILTIN_PROFILES
        required_keys = [
            'genre', 'lufs_integrated', 'lufs_range', 'true_peak_db',
            'dynamic_range_db', 'crest_factor_db',
        ]
        for name, data in BUILTIN_PROFILES.items():
            for key in required_keys:
                assert key in data, f"Profile {name} missing key {key}"

    def test_profile_lufs_negative(self):
        from ml.reference_profiles import BUILTIN_PROFILES
        for name, data in BUILTIN_PROFILES.items():
            assert data['lufs_integrated'] < 0, f"{name} LUFS should be negative"

    def test_band_energies_present(self):
        from ml.reference_profiles import BUILTIN_PROFILES
        for name, data in BUILTIN_PROFILES.items():
            assert 'band_energies' in data
            assert isinstance(data['band_energies'], dict)
            assert len(data['band_energies']) > 0


class TestReferenceProfile:
    """Tests for the ReferenceProfile dataclass."""

    def test_creation(self):
        from ml.reference_profiles import ReferenceProfile
        profile = ReferenceProfile(
            name='test', genre='rock',
            spectral_envelope=np.zeros(100),
            frequencies=np.linspace(0, 24000, 100),
            lufs_integrated=-16.0, lufs_range=6.0,
            true_peak_db=-1.0, dynamic_range_db=8.0,
            crest_factor_db=6.0,
        )
        assert profile.name == 'test'
        assert profile.genre == 'rock'
        assert profile.lufs_integrated == -16.0

    def test_default_fields(self):
        from ml.reference_profiles import ReferenceProfile
        profile = ReferenceProfile(
            name='x', genre='pop',
            spectral_envelope=np.zeros(10),
            frequencies=np.zeros(10),
            lufs_integrated=-14.0, lufs_range=5.0,
            true_peak_db=-0.5, dynamic_range_db=6.0,
            crest_factor_db=5.0,
        )
        assert profile.spectral_centroid == 0.0
        assert profile.spectral_rolloff == 0.0
        assert profile.stereo_width == 0.0
        assert profile.description == ''
        assert profile.band_energies == {}


class TestReferenceProfileManager:
    """Tests for the ReferenceProfileManager."""

    def test_instantiation_loads_builtins(self):
        from ml.reference_profiles import ReferenceProfileManager, BUILTIN_PROFILES
        mgr = ReferenceProfileManager(sample_rate=48000, fft_size=4096)
        assert len(mgr.profiles) == len(BUILTIN_PROFILES)

    def test_list_profiles(self):
        from ml.reference_profiles import ReferenceProfileManager, BUILTIN_PROFILES
        mgr = ReferenceProfileManager()
        names = mgr.list_profiles()
        for name in BUILTIN_PROFILES:
            assert name in names

    def test_get_profile_existing(self):
        from ml.reference_profiles import ReferenceProfileManager, ReferenceProfile
        mgr = ReferenceProfileManager()
        profile = mgr.get_profile('modern_rock')
        assert profile is not None
        assert isinstance(profile, ReferenceProfile)
        assert profile.name == 'modern_rock'
        assert profile.genre == 'rock'

    def test_get_profile_nonexistent(self):
        from ml.reference_profiles import ReferenceProfileManager
        mgr = ReferenceProfileManager()
        result = mgr.get_profile('nonexistent_profile')
        assert result is None

    def test_builtin_profile_spectral_envelope_shape(self):
        from ml.reference_profiles import ReferenceProfileManager
        mgr = ReferenceProfileManager(sample_rate=48000, fft_size=4096)
        profile = mgr.get_profile('modern_rock')
        expected_bins = 4096 // 2 + 1
        assert profile.spectral_envelope.shape == (expected_bins,)

    def test_create_from_audio(self):
        from ml.reference_profiles import ReferenceProfileManager, ReferenceProfile
        mgr = ReferenceProfileManager(sample_rate=48000, fft_size=2048)
        # Generate a simple sine wave
        t = np.linspace(0, 1.0, 48000)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        profile = mgr.create_from_audio('test_sine', audio, genre='test')
        assert isinstance(profile, ReferenceProfile)
        assert profile.name == 'test_sine'
        assert profile.genre == 'test'
        assert 'test_sine' in mgr.list_profiles()

    def test_create_from_audio_fields_populated(self):
        from ml.reference_profiles import ReferenceProfileManager
        mgr = ReferenceProfileManager(sample_rate=48000, fft_size=2048)
        t = np.linspace(0, 1.0, 48000)
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        profile = mgr.create_from_audio('test_1k', audio, genre='test')
        assert profile.lufs_integrated < 0
        assert profile.spectral_centroid > 0
        assert profile.spectral_rolloff > 0
        assert len(profile.band_energies) > 0

    def test_compute_distance_same_profile(self):
        from ml.reference_profiles import ReferenceProfileManager
        mgr = ReferenceProfileManager()
        dist = mgr.compute_distance('modern_rock', 'modern_rock')
        assert dist == 0.0

    def test_compute_distance_different_profiles(self):
        from ml.reference_profiles import ReferenceProfileManager
        mgr = ReferenceProfileManager()
        dist = mgr.compute_distance('modern_rock', 'jazz_live')
        assert dist > 0.0
        assert np.isfinite(dist)

    def test_compute_distance_nonexistent_profile(self):
        from ml.reference_profiles import ReferenceProfileManager
        mgr = ReferenceProfileManager()
        dist = mgr.compute_distance('modern_rock', 'nonexistent')
        assert dist == float('inf')

    def test_compute_distance_symmetry(self):
        from ml.reference_profiles import ReferenceProfileManager
        mgr = ReferenceProfileManager()
        d1 = mgr.compute_distance('modern_rock', 'jazz_live')
        d2 = mgr.compute_distance('jazz_live', 'modern_rock')
        assert abs(d1 - d2) < 1e-6
