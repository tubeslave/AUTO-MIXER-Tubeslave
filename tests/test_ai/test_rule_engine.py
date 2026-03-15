"""
Tests for backend.ai.rule_engine — deterministic mixing rules for HPF, EQ,
compressor, gate, pan, gain staging, feedback handling, and full presets.

No optional dependencies required. All tests are self-contained.
"""

import math

import pytest

from backend.ai.rule_engine import (
    EQBand,
    RuleEngine,
    HPF_FREQUENCIES,
    DEFAULT_EQ,
    DEFAULT_COMPRESSOR,
    DEFAULT_GATE,
    DEFAULT_PAN,
    FEEDBACK_BANDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return RuleEngine()


# ---------------------------------------------------------------------------
# EQBand dataclass
# ---------------------------------------------------------------------------

class TestEQBand:

    def test_defaults(self):
        band = EQBand(freq=1000, gain=-3.0)
        assert band.freq == 1000
        assert band.gain == -3.0
        assert band.q == 1.0
        assert band.band_type == "peaking"

    def test_to_dict(self):
        band = EQBand(freq=250, gain=2.0, q=1.5, band_type="lowshelf")
        d = band.to_dict()
        assert d == {"freq": 250, "gain": 2.0, "q": 1.5, "type": "lowshelf"}

    def test_custom_q(self):
        band = EQBand(freq=5000, gain=-6.0, q=8.0)
        assert band.q == 8.0


# ---------------------------------------------------------------------------
# HPF lookup
# ---------------------------------------------------------------------------

class TestApplyHPF:

    def test_known_instrument(self, engine):
        freq = engine.apply_hpf("kick")
        assert freq == 30.0

    def test_vocals_hpf(self, engine):
        freq = engine.apply_hpf("leadVocal")
        assert freq == 80.0

    def test_unknown_instrument_returns_default(self, engine):
        freq = engine.apply_hpf("didgeridoo")
        assert freq == 100.0  # Default fallback

    def test_case_insensitive_match(self, engine):
        freq = engine.apply_hpf("KICK")
        assert isinstance(freq, float)
        assert freq > 0

    def test_all_known_instruments_have_positive_hpf(self, engine):
        for instrument in HPF_FREQUENCIES:
            freq = engine.apply_hpf(instrument)
            assert freq > 0, f"HPF for {instrument} should be positive"


# ---------------------------------------------------------------------------
# Default EQ presets
# ---------------------------------------------------------------------------

class TestDefaultEQ:

    def test_kick_eq_has_bands(self, engine):
        eq = engine.get_default_eq("kick")
        assert len(eq) > 0
        assert all(isinstance(b, dict) for b in eq)

    def test_eq_band_has_required_keys(self, engine):
        eq = engine.get_default_eq("snare")
        for band in eq:
            assert "freq" in band
            assert "gain" in band
            assert "q" in band
            assert "type" in band

    def test_unknown_instrument_returns_empty(self, engine):
        eq = engine.get_default_eq("theremin")
        assert eq == []

    def test_returns_copies(self, engine):
        """Ensure returned EQ bands are copies, not references to the originals."""
        eq1 = engine.get_default_eq("kick")
        eq2 = engine.get_default_eq("kick")
        eq1[0]["gain"] = 999.0
        assert eq2[0]["gain"] != 999.0


# ---------------------------------------------------------------------------
# Default compressor presets
# ---------------------------------------------------------------------------

class TestDefaultCompressor:

    def test_kick_compressor(self, engine):
        comp = engine.get_default_compressor("kick")
        assert comp["ratio"] >= 1.0
        assert comp["threshold"] < 0
        assert comp["attack"] > 0
        assert comp["release"] > 0

    def test_compressor_has_all_keys(self, engine):
        comp = engine.get_default_compressor("leadVocal")
        expected_keys = {"threshold", "ratio", "attack", "release", "knee", "makeup_gain"}
        assert set(comp.keys()) == expected_keys

    def test_unknown_instrument_gets_gentle_defaults(self, engine):
        comp = engine.get_default_compressor("kazoo")
        assert comp["ratio"] == 2.0
        assert comp["threshold"] == -20.0

    def test_returns_copy(self, engine):
        comp1 = engine.get_default_compressor("kick")
        comp2 = engine.get_default_compressor("kick")
        comp1["ratio"] = 999.0
        assert comp2["ratio"] != 999.0


# ---------------------------------------------------------------------------
# Default gate presets
# ---------------------------------------------------------------------------

class TestDefaultGate:

    def test_kick_gate(self, engine):
        gate = engine.get_default_gate("kick")
        assert gate["threshold"] < 0
        assert gate["attack"] >= 0
        assert gate["hold"] > 0
        assert gate["release"] > 0

    def test_gate_has_all_keys(self, engine):
        gate = engine.get_default_gate("snare")
        expected_keys = {"threshold", "attack", "hold", "release", "range"}
        assert set(gate.keys()) == expected_keys

    def test_unknown_instrument_gets_open_defaults(self, engine):
        gate = engine.get_default_gate("djembe")
        assert gate["threshold"] == -60.0

    def test_returns_copy(self, engine):
        gate1 = engine.get_default_gate("kick")
        gate2 = engine.get_default_gate("kick")
        gate1["threshold"] = 999.0
        assert gate2["threshold"] != 999.0


# ---------------------------------------------------------------------------
# Default pan presets
# ---------------------------------------------------------------------------

class TestDefaultPan:

    def test_kick_centered(self, engine):
        pan = engine.get_default_pan("kick")
        assert pan == 0.0

    def test_bass_centered(self, engine):
        pan = engine.get_default_pan("bass")
        assert pan == 0.0

    def test_lead_vocal_centered(self, engine):
        pan = engine.get_default_pan("leadVocal")
        assert pan == 0.0

    def test_hihat_panned(self, engine):
        pan = engine.get_default_pan("hihat")
        assert pan != 0.0  # Should be panned

    def test_unknown_instrument_centered(self, engine):
        pan = engine.get_default_pan("theremin")
        assert pan == 0.0


# ---------------------------------------------------------------------------
# Gain staging
# ---------------------------------------------------------------------------

class TestGainStaging:

    def test_exact_target(self, engine):
        """When current matches target, no adjustment needed."""
        adj = engine.gain_stage_to_target(-20.0, target_lufs=-20.0)
        assert adj == 0.0

    def test_boost_needed(self, engine):
        adj = engine.gain_stage_to_target(-25.0, target_lufs=-20.0)
        assert adj == 5.0

    def test_cut_needed(self, engine):
        adj = engine.gain_stage_to_target(-15.0, target_lufs=-20.0)
        assert adj == -5.0

    def test_clamped_max_boost(self, engine):
        adj = engine.gain_stage_to_target(-50.0, target_lufs=-20.0)
        assert adj == 12.0  # Clamped to max boost

    def test_clamped_max_cut(self, engine):
        adj = engine.gain_stage_to_target(10.0, target_lufs=-20.0)
        assert adj == -24.0  # Clamped to max cut

    def test_non_finite_returns_zero(self, engine):
        assert engine.gain_stage_to_target(float("inf")) == 0.0
        assert engine.gain_stage_to_target(float("-inf")) == 0.0
        assert engine.gain_stage_to_target(float("nan")) == 0.0

    def test_result_rounded(self, engine):
        adj = engine.gain_stage_to_target(-22.33, target_lufs=-20.0)
        # Should be rounded to 1 decimal place
        assert adj == round(adj, 1)


# ---------------------------------------------------------------------------
# Feedback handling
# ---------------------------------------------------------------------------

class TestFeedbackHandling:

    def test_returns_required_keys(self, engine):
        result = engine.handle_feedback(1000.0)
        required_keys = {"freq", "gain", "q", "type", "severity", "nearest_band"}
        assert set(result.keys()) == required_keys

    def test_gain_is_negative(self, engine):
        result = engine.handle_feedback(2500.0)
        assert result["gain"] < 0  # Should be a cut

    def test_type_is_peaking(self, engine):
        result = engine.handle_feedback(800.0)
        assert result["type"] == "peaking"

    def test_frequency_clamped_low(self, engine):
        result = engine.handle_feedback(5.0)
        assert result["freq"] >= 20.0

    def test_frequency_clamped_high(self, engine):
        result = engine.handle_feedback(30000.0)
        assert result["freq"] <= 20000.0

    def test_near_known_band_is_high_severity(self, engine):
        # 1000 Hz is an exact FEEDBACK_BANDS entry
        result = engine.handle_feedback(1000.0)
        assert result["severity"] == "high"
        assert result["gain"] < -6.0  # Deeper cut for known problem freq

    def test_far_from_known_band_is_moderate(self, engine):
        # 1300 Hz is not close to any standard band
        result = engine.handle_feedback(1300.0)
        assert result["severity"] == "moderate"

    def test_nearest_band_label(self, engine):
        result = engine.handle_feedback(800.0)
        assert result["nearest_band"] == "honk"


# ---------------------------------------------------------------------------
# Full channel preset
# ---------------------------------------------------------------------------

class TestFullChannelPreset:

    def test_has_all_sections(self, engine):
        preset = engine.get_full_channel_preset("kick")
        expected_keys = {"instrument", "hpf", "eq", "compressor", "gate", "pan"}
        assert set(preset.keys()) == expected_keys

    def test_instrument_field_matches(self, engine):
        preset = engine.get_full_channel_preset("snare")
        assert preset["instrument"] == "snare"

    def test_hpf_is_float(self, engine):
        preset = engine.get_full_channel_preset("leadVocal")
        assert isinstance(preset["hpf"], float)

    def test_eq_is_list(self, engine):
        preset = engine.get_full_channel_preset("kick")
        assert isinstance(preset["eq"], list)

    def test_compressor_is_dict(self, engine):
        preset = engine.get_full_channel_preset("bass")
        assert isinstance(preset["compressor"], dict)


# ---------------------------------------------------------------------------
# should_enable_gate / should_enable_compressor
# ---------------------------------------------------------------------------

class TestGateAndCompressorRecommendation:

    def test_gate_recommended_for_drums(self):
        assert RuleEngine.should_enable_gate("kick") is True
        assert RuleEngine.should_enable_gate("snare") is True
        assert RuleEngine.should_enable_gate("tom") is True

    def test_gate_recommended_for_vocals(self):
        assert RuleEngine.should_enable_gate("leadVocal") is True
        assert RuleEngine.should_enable_gate("backVocal") is True

    def test_gate_not_recommended_for_ambient(self):
        assert RuleEngine.should_enable_gate("overheads") is False
        assert RuleEngine.should_enable_gate("room") is False
        assert RuleEngine.should_enable_gate("synth") is False

    def test_compressor_recommended_for_most(self):
        assert RuleEngine.should_enable_compressor("kick") is True
        assert RuleEngine.should_enable_compressor("leadVocal") is True
        assert RuleEngine.should_enable_compressor("bass") is True

    def test_compressor_not_recommended_for_playback(self):
        assert RuleEngine.should_enable_compressor("playback") is False
        assert RuleEngine.should_enable_compressor("djTrack") is False
