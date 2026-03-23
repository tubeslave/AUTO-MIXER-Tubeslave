"""Tests for ml.lufs_targets -- genre-aware LUFS target management."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))


class TestGenreTargets:
    """Tests for the GENRE_TARGETS constant dictionary."""

    def test_genre_targets_not_empty(self):
        from ml.lufs_targets import GENRE_TARGETS
        assert len(GENRE_TARGETS) > 0

    def test_default_genre_exists(self):
        from ml.lufs_targets import GENRE_TARGETS
        assert 'default' in GENRE_TARGETS

    def test_genre_target_fields(self):
        from ml.lufs_targets import GENRE_TARGETS
        target = GENRE_TARGETS['rock']
        assert hasattr(target, 'integrated')
        assert hasattr(target, 'short_term_max')
        assert hasattr(target, 'momentary_max')
        assert hasattr(target, 'true_peak_max')
        assert hasattr(target, 'loudness_range')
        assert hasattr(target, 'tolerance')

    def test_integrated_values_negative(self):
        from ml.lufs_targets import GENRE_TARGETS
        for genre, target in GENRE_TARGETS.items():
            assert target.integrated < 0, f"Genre {genre} integrated LUFS should be negative"

    def test_known_genres_present(self):
        from ml.lufs_targets import GENRE_TARGETS
        expected_genres = ['rock', 'pop', 'jazz', 'classical', 'edm', 'metal']
        for genre in expected_genres:
            assert genre in GENRE_TARGETS


class TestInstrumentProfiles:
    """Tests for the INSTRUMENT_PROFILES dictionary."""

    def test_profiles_not_empty(self):
        from ml.lufs_targets import INSTRUMENT_PROFILES
        assert len(INSTRUMENT_PROFILES) > 0

    def test_lead_vocal_highest_priority(self):
        from ml.lufs_targets import INSTRUMENT_PROFILES
        lead = INSTRUMENT_PROFILES['lead_vocal']
        assert lead.priority == 1

    def test_unknown_profile_exists(self):
        from ml.lufs_targets import INSTRUMENT_PROFILES
        assert 'unknown' in INSTRUMENT_PROFILES

    def test_profile_fields(self):
        from ml.lufs_targets import INSTRUMENT_PROFILES
        profile = INSTRUMENT_PROFILES['kick']
        assert hasattr(profile, 'relative_db')
        assert hasattr(profile, 'priority')
        assert hasattr(profile, 'duck_amount')

    def test_click_track_very_quiet(self):
        from ml.lufs_targets import INSTRUMENT_PROFILES
        click = INSTRUMENT_PROFILES['click_track']
        assert click.relative_db <= -50.0


class TestLUFSTargetManager:
    """Tests for LUFSTargetManager."""

    def test_default_instantiation(self):
        from ml.lufs_targets import LUFSTargetManager
        mgr = LUFSTargetManager()
        assert mgr.genre == 'default'
        assert mgr.venue == 'medium_venue'

    def test_target_property_returns_lufs_target(self):
        from ml.lufs_targets import LUFSTargetManager, LUFSTarget
        mgr = LUFSTargetManager(genre='rock', venue='medium_venue')
        target = mgr.target
        assert isinstance(target, LUFSTarget)
        assert target.integrated == -18.0  # rock base with medium_venue (0 offset)

    def test_venue_offset_applied(self):
        from ml.lufs_targets import LUFSTargetManager
        mgr_medium = LUFSTargetManager(genre='rock', venue='medium_venue')
        mgr_arena = LUFSTargetManager(genre='rock', venue='large_arena')
        # Arena has -2.0 offset, medium is 0.0
        assert mgr_arena.target.integrated == mgr_medium.target.integrated - 2.0

    def test_get_channel_target(self):
        from ml.lufs_targets import LUFSTargetManager
        mgr = LUFSTargetManager(genre='default', venue='medium_venue')
        vocal_target = mgr.get_channel_target('lead_vocal')
        kick_target = mgr.get_channel_target('kick')
        # Lead vocal relative_db=0, kick relative_db=-6
        assert vocal_target > kick_target

    def test_get_gain_adjustment_within_bounds(self):
        from ml.lufs_targets import LUFSTargetManager
        mgr = LUFSTargetManager()
        adj = mgr.get_gain_adjustment('lead_vocal', current_lufs=-24.0)
        assert -12.0 <= adj <= 12.0

    def test_get_gain_adjustment_silent_signal(self):
        from ml.lufs_targets import LUFSTargetManager
        mgr = LUFSTargetManager()
        adj = mgr.get_gain_adjustment('kick', current_lufs=-80.0)
        assert adj == 0.0, "Silent signals should get no adjustment"

    def test_set_custom_target(self):
        from ml.lufs_targets import LUFSTargetManager, LUFSTarget
        mgr = LUFSTargetManager(genre='custom_genre')
        custom = LUFSTarget(-10.0, -6.0, -3.0, -0.5, 4.0)
        mgr.set_custom_target('custom_genre', custom)
        assert mgr.target.integrated == -10.0

    def test_get_duck_amount(self):
        from ml.lufs_targets import LUFSTargetManager
        mgr = LUFSTargetManager()
        duck = mgr.get_duck_amount('ambient_mic')
        assert duck > 0.0
        duck_vocal = mgr.get_duck_amount('lead_vocal')
        assert duck_vocal == 0.0

    def test_get_priority(self):
        from ml.lufs_targets import LUFSTargetManager
        mgr = LUFSTargetManager()
        assert mgr.get_priority('lead_vocal') == 1
        assert mgr.get_priority('unknown') == 3
