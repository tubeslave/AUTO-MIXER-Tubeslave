"""Unit tests for Ableton EQ Eight parameter name parsing."""

from ableton_client import AbletonClient


def test_parse_eq_eight_live_style_names():
    """Live-style English names: Filter rows skipped; Freq / Gain / Resonance per band."""
    names = [
        "Device On",
        "1 Filter On A",
        "1 Filter Type A",
        "1 Frequency A",
        "1 Gain A",
        "1 Resonance A",
        "2 Filter On A",
        "2 Filter Type A",
        "2 Frequency A",
        "2 Gain A",
        "2 Resonance A",
        "3 Filter On A",
        "3 Filter Type A",
        "3 Frequency A",
        "3 Gain A",
        "3 Resonance A",
        "4 Filter On A",
        "4 Filter Type A",
        "4 Frequency A",
        "4 Gain A",
        "4 Resonance A",
    ]
    m = AbletonClient._parse_eq_eight_parameter_names(names)
    # Tuple order: (gain_idx, freq_idx, q_idx) — indices follow list order in Live
    assert m[1] == (4, 3, 5)
    assert m[2] == (9, 8, 10)
    assert m[3] == (14, 13, 15)
    assert m[4] == (19, 18, 20)


def test_parse_eq_eight_fallback_when_empty():
    m = AbletonClient._parse_eq_eight_parameter_names(["Device On"])
    fb = AbletonClient._eq_eight_band_fallback()
    for b in range(1, 9):
        assert m[b] == fb[b]


def test_eq_eight_fallback_extrapolates_bands_5_8():
    fb = AbletonClient._eq_eight_band_fallback()
    assert fb[5][0] == fb[4][0] + AbletonClient._EQ8_FALLBACK_STEP
    assert fb[6][0] == fb[5][0] + AbletonClient._EQ8_FALLBACK_STEP
