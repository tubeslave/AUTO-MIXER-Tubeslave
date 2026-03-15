"""
Tests for backend.ml.style_transfer — style extraction, application,
preset save/load, and OSC command generation.

All tests use numpy-generated audio. No mixer hardware required.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from backend.ml.style_transfer import (
    InstrumentStyle,
    StyleProfile,
    StyleTransfer,
    SPECTRAL_BANDS,
    INSTRUMENT_TYPES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stereo_mix():
    """1-second stereo mix (440 Hz L, 880 Hz R) at 48 kHz."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    left = np.sin(2 * np.pi * 440 * t) * 0.5
    right = np.sin(2 * np.pi * 880 * t) * 0.3
    return np.stack([left, right], axis=0)  # (2, samples)


@pytest.fixture
def mono_signal():
    """1-second 440 Hz mono signal at 48 kHz."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t) * 0.5


@pytest.fixture
def style_transfer():
    return StyleTransfer(fft_size=2048, hop_size=512)


# ---------------------------------------------------------------------------
# InstrumentStyle / StyleProfile dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses:

    def test_instrument_style_defaults(self):
        style = InstrumentStyle(instrument_type="vocals")
        assert style.gain_db == 0.0
        assert style.pan == 0.0
        assert style.compression_ratio == 1.0

    def test_instrument_style_to_dict(self):
        style = InstrumentStyle(instrument_type="kick", gain_db=-3.0)
        d = style.to_dict()
        assert d["instrument_type"] == "kick"
        assert d["gain_db"] == -3.0

    def test_style_profile_to_dict(self):
        profile = StyleProfile(
            name="rock",
            spectral_balance={"bass": -5.0, "mid": -8.0},
            dynamic_range=15.0,
            stereo_width=0.4,
            loudness_lufs=-14.0,
        )
        d = profile.to_dict()
        assert d["name"] == "rock"
        assert d["dynamic_range"] == 15.0
        assert isinstance(d["spectral_balance"], dict)

    def test_style_profile_per_instrument(self):
        profile = StyleProfile(name="jazz")
        profile.per_instrument_settings["vocals"] = InstrumentStyle(
            instrument_type="vocals", gain_db=2.0
        )
        d = profile.to_dict()
        assert "vocals" in d["per_instrument_settings"]
        assert d["per_instrument_settings"]["vocals"]["gain_db"] == 2.0


# ---------------------------------------------------------------------------
# Style extraction
# ---------------------------------------------------------------------------

class TestExtractStyle:

    def test_extract_mono(self, mono_signal, style_transfer):
        profile = style_transfer.extract_style(mono_signal, sr=48000, name="test_mono")
        assert profile.name == "test_mono"
        assert isinstance(profile.spectral_balance, dict)
        assert profile.stereo_width == 0.0  # mono input → zero width
        assert profile.dynamic_range >= 0

    def test_extract_stereo(self, stereo_mix, style_transfer):
        profile = style_transfer.extract_style(stereo_mix, sr=48000, name="test_stereo")
        assert profile.name == "test_stereo"
        assert profile.stereo_width > 0  # L and R are different signals
        assert set(SPECTRAL_BANDS.keys()).issubset(set(profile.spectral_balance.keys()))

    def test_spectral_balance_values_are_negative_db(self, mono_signal, style_transfer):
        """All spectral balance values should be negative (relative to total)."""
        profile = style_transfer.extract_style(mono_signal, sr=48000)
        for band_name, db_val in profile.spectral_balance.items():
            assert db_val <= 0, f"{band_name}={db_val} should be <= 0 dB (relative energy)"

    def test_crest_factor_positive(self, mono_signal, style_transfer):
        profile = style_transfer.extract_style(mono_signal, sr=48000)
        assert profile.crest_factor > 0, "Sine wave should have positive crest factor"

    def test_lufs_reasonable(self, mono_signal, style_transfer):
        profile = style_transfer.extract_style(mono_signal, sr=48000)
        # A 0.5-amplitude sine at 48 kHz should have LUFS somewhere between -30 and 0
        assert -50 < profile.loudness_lufs < 0


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

class TestApplyStyle:

    def test_apply_returns_params_for_all_channels(self, mono_signal, style_transfer):
        profile = StyleProfile(
            name="test",
            spectral_balance={b: -10.0 for b in SPECTRAL_BANDS},
            dynamic_range=15.0,
            loudness_lufs=-14.0,
        )
        channel_audios = {"ch1": mono_signal, "ch2": mono_signal * 0.3}
        channel_types = {"ch1": "vocals", "ch2": "kick"}
        result = style_transfer.apply_style(profile, channel_audios, channel_types, sr=48000)
        assert "ch1" in result and "ch2" in result

    def test_apply_params_have_expected_keys(self, mono_signal, style_transfer):
        profile = StyleProfile(
            name="test",
            spectral_balance={b: -10.0 for b in SPECTRAL_BANDS},
            dynamic_range=15.0,
            loudness_lufs=-14.0,
        )
        channel_audios = {"vox": mono_signal}
        channel_types = {"vox": "vocals"}
        result = style_transfer.apply_style(profile, channel_audios, channel_types, sr=48000)
        params = result["vox"]
        for key in ("fader_db", "pan", "eq_bands", "compression", "gate_threshold"):
            assert key in params, f"Missing key '{key}' in channel params"

    def test_fader_within_bounds(self, mono_signal, style_transfer):
        profile = StyleProfile(
            name="test",
            spectral_balance={b: -10.0 for b in SPECTRAL_BANDS},
            dynamic_range=15.0,
            loudness_lufs=-14.0,
        )
        channel_audios = {"bass": mono_signal}
        channel_types = {"bass": "bass"}
        result = style_transfer.apply_style(profile, channel_audios, channel_types, sr=48000)
        fader = result["bass"]["fader_db"]
        assert -96.0 <= fader <= 10.0


# ---------------------------------------------------------------------------
# Preset save / load round-trip
# ---------------------------------------------------------------------------

class TestPresetIO:

    def test_save_and_load_roundtrip(self, style_transfer):
        profile = StyleProfile(
            name="roundtrip",
            spectral_balance={"bass": -8.0, "mid": -12.0},
            dynamic_range=20.0,
            stereo_width=0.35,
            loudness_lufs=-16.0,
            crest_factor=9.5,
        )
        profile.per_instrument_settings["vocals"] = InstrumentStyle(
            instrument_type="vocals", gain_db=1.5, pan=-0.2
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_preset.json")
            style_transfer.save_preset(profile, path)
            assert os.path.isfile(path)

            loaded = style_transfer.load_preset(path)
            assert loaded.name == "roundtrip"
            assert loaded.dynamic_range == pytest.approx(20.0, abs=0.1)
            assert loaded.stereo_width == pytest.approx(0.35, abs=0.01)
            assert "vocals" in loaded.per_instrument_settings

    def test_saved_json_valid(self, style_transfer):
        profile = StyleProfile(name="json_check")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "check.json")
            style_transfer.save_preset(profile, path)
            with open(path) as f:
                data = json.load(f)
            assert data["name"] == "json_check"

    def test_load_missing_fields(self, style_transfer):
        """Load a minimal JSON and verify defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "minimal.json")
            with open(path, "w") as f:
                json.dump({"name": "minimal"}, f)
            loaded = style_transfer.load_preset(path)
            assert loaded.name == "minimal"
            assert loaded.loudness_lufs == -14.0  # default


# ---------------------------------------------------------------------------
# OSC command generation
# ---------------------------------------------------------------------------

class TestGenerateWingOSC:

    def test_generates_fader_and_pan(self, style_transfer):
        mixing_params = {
            "ch1": {
                "fader_db": -3.0,
                "pan": 0.5,
                "eq_bands": [],
                "compression": {},
                "gate_threshold": -60.0,
                "bus_send": {},
            }
        }
        cmds = style_transfer.generate_wing_osc(mixing_params, channel_map={"ch1": 1})
        addresses = [addr for addr, _ in cmds]
        assert any("fader" in a for a in addresses)
        assert any("pan" in a for a in addresses)

    def test_auto_channel_map(self, style_transfer):
        mixing_params = {
            "vox": {"fader_db": 0, "pan": 0, "eq_bands": [], "compression": {},
                    "gate_threshold": -60, "bus_send": {}},
            "bass": {"fader_db": -2, "pan": 0, "eq_bands": [], "compression": {},
                     "gate_threshold": -60, "bus_send": {}},
        }
        cmds = style_transfer.generate_wing_osc(mixing_params)
        addresses = [addr for addr, _ in cmds]
        # First channel should use 01, second 02
        assert any("/01/" in a for a in addresses)
        assert any("/02/" in a for a in addresses)

    def test_db_to_wing_fader_boundaries(self):
        assert StyleTransfer._db_to_wing_fader(-144.0) == 0.0
        assert StyleTransfer._db_to_wing_fader(-200.0) == 0.0
        assert StyleTransfer._db_to_wing_fader(10.0) == 1.0
        assert StyleTransfer._db_to_wing_fader(20.0) == 1.0
        # 0 dB → 0.75
        assert StyleTransfer._db_to_wing_fader(0.0) == pytest.approx(0.75, abs=0.01)
