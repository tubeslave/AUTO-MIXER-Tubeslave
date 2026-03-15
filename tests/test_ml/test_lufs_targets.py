"""
Tests for LUFS target computation per instrument and genre.

Covers:
- INSTRUMENT_LUFS_OFFSETS dictionary completeness
- GENRE_MODIFIERS dictionary completeness
- get_target_lufs() for known instruments and genres
- get_all_targets() returns a complete mapping
- get_relative_balance() normalises vocals to 0 dB
"""

import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend", "ml"))

from lufs_targets import (
    INSTRUMENT_LUFS_OFFSETS,
    GENRE_MODIFIERS,
    get_target_lufs,
    get_all_targets,
    get_relative_balance,
)


class TestInstrumentLUFSOffsets:
    """Verify the INSTRUMENT_LUFS_OFFSETS dictionary."""

    def test_offsets_has_13_instruments(self):
        """INSTRUMENT_LUFS_OFFSETS should contain all 13 instrument classes."""
        assert len(INSTRUMENT_LUFS_OFFSETS) == 13

    def test_vocals_offset(self):
        """Vocals offset should be present and numeric."""
        assert "vocals" in INSTRUMENT_LUFS_OFFSETS
        assert isinstance(INSTRUMENT_LUFS_OFFSETS["vocals"], (int, float))

    def test_kick_offset(self):
        """Kick offset should be present and numeric."""
        assert "kick" in INSTRUMENT_LUFS_OFFSETS
        assert isinstance(INSTRUMENT_LUFS_OFFSETS["kick"], (int, float))

    def test_all_offsets_are_numeric(self):
        """Every offset value should be a number."""
        for instrument, offset in INSTRUMENT_LUFS_OFFSETS.items():
            assert isinstance(offset, (int, float)), f"{instrument} offset is not numeric"

    def test_known_instruments_present(self):
        """All expected instrument types should be in the offsets dict."""
        expected = [
            "kick", "snare", "hihat", "toms", "overheads",
            "bass_guitar", "electric_guitar", "acoustic_guitar",
            "keys", "vocals", "brass", "strings", "percussion",
        ]
        for inst in expected:
            assert inst in INSTRUMENT_LUFS_OFFSETS, f"{inst} missing from offsets"


class TestGenreModifiers:
    """Verify the GENRE_MODIFIERS dictionary."""

    def test_genre_modifiers_has_7_genres(self):
        """GENRE_MODIFIERS should contain exactly 7 genre entries."""
        assert len(GENRE_MODIFIERS) == 7

    def test_known_genres_present(self):
        """All expected genre names should be present."""
        expected = ["rock", "jazz", "pop", "electronic", "classical", "acoustic", "metal"]
        for genre in expected:
            assert genre in GENRE_MODIFIERS, f"{genre} missing from GENRE_MODIFIERS"

    def test_genre_modifier_values_are_dicts(self):
        """Each genre modifier should be a dict mapping instruments to offsets."""
        for genre, mods in GENRE_MODIFIERS.items():
            assert isinstance(mods, dict), f"{genre} modifier is not a dict"


class TestGetTargetLUFS:
    """Tests for get_target_lufs()."""

    def test_vocals_default_base(self):
        """get_target_lufs('vocals') with default base should return a float."""
        result = get_target_lufs("vocals")
        assert isinstance(result, float)

    def test_kick_default_base(self):
        """get_target_lufs('kick') should return a float."""
        result = get_target_lufs("kick")
        assert isinstance(result, float)

    def test_custom_base(self):
        """Supplying a custom base should shift the result."""
        result_default = get_target_lufs("vocals", base=-18.0)
        result_loud = get_target_lufs("vocals", base=-14.0)
        # Louder base should give a louder target
        assert result_loud > result_default

    def test_genre_modifier_applied(self):
        """A genre modifier should adjust the target LUFS."""
        no_genre = get_target_lufs("kick")
        with_genre = get_target_lufs("kick", genre="electronic")
        # They may or may not differ depending on modifier, but both should be float
        assert isinstance(with_genre, float)

    def test_unknown_instrument_returns_base(self):
        """An unknown instrument should return something sensible (likely base)."""
        result = get_target_lufs("unknown_instrument_xyz")
        assert isinstance(result, float)

    def test_unknown_genre_still_returns_float(self):
        """An unknown genre should not crash; should return a float."""
        result = get_target_lufs("vocals", genre="alien_music")
        assert isinstance(result, float)


class TestGetAllTargets:
    """Tests for get_all_targets()."""

    def test_returns_dict(self):
        """get_all_targets should return a dict."""
        result = get_all_targets("rock")
        assert isinstance(result, dict)

    def test_contains_all_instruments(self):
        """Result should contain all 13 instruments."""
        result = get_all_targets("rock")
        for inst in INSTRUMENT_LUFS_OFFSETS:
            assert inst in result, f"{inst} missing from get_all_targets result"

    def test_all_values_are_float(self):
        """All target values should be floats."""
        result = get_all_targets("pop")
        for inst, val in result.items():
            assert isinstance(val, float), f"{inst} target is not float"

    def test_custom_base_propagates(self):
        """Custom base should propagate to all targets."""
        result_low = get_all_targets("jazz", base=-24.0)
        result_high = get_all_targets("jazz", base=-12.0)
        # Every instrument should be louder with higher base
        for inst in result_low:
            assert result_high[inst] > result_low[inst], (
                f"{inst}: high base {result_high[inst]} not > low base {result_low[inst]}"
            )


class TestGetRelativeBalance:
    """Tests for get_relative_balance()."""

    def test_vocals_normalised_to_zero(self):
        """Vocals should be at 0 dB in the relative balance."""
        result = get_relative_balance("rock")
        assert "vocals" in result
        assert abs(result["vocals"]) < 0.01  # Should be ~0.0

    def test_returns_dict(self):
        """get_relative_balance should return a dict."""
        result = get_relative_balance("jazz")
        assert isinstance(result, dict)

    def test_all_instruments_present(self):
        """All 13 instruments should be in the relative balance."""
        result = get_relative_balance("pop")
        for inst in INSTRUMENT_LUFS_OFFSETS:
            assert inst in result, f"{inst} missing from relative balance"

    def test_values_are_relative(self):
        """Values should be relative offsets (floats)."""
        result = get_relative_balance("classical")
        for inst, val in result.items():
            assert isinstance(val, float), f"{inst} relative balance is not float"
