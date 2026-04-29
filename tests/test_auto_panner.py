"""Tests for spectral auto-panning heuristics."""

from auto_panner import AutoPanner


def test_low_frequency_dominant_band_is_centered():
    panner = AutoPanner()

    decisions = panner.calculate_panning(
        channels=[1],
        instrument_types={1: "unknown"},
        spectral_centroids={1: 1000.0},
        channel_band_energy={1: {"sub": -10.0, "mid": -30.0}},
    )

    assert decisions[1].pan_percent == 0.0
    assert decisions[1].dominant_band == "sub"


def test_same_dominant_band_is_distributed_with_input_priority():
    panner = AutoPanner(max_pan_percent=80.0)

    channel_band_energy = {
        5: {"high_mid": -10.0, "bass": -40.0},
        2: {"high_mid": -11.0, "bass": -40.0},
        8: {"high_mid": -12.0, "bass": -40.0},
        4: {"high_mid": -13.0, "bass": -40.0},
    }
    decisions = panner.calculate_panning(
        channels=[5, 2, 8, 4],
        instrument_types={ch: "guitar" for ch in channel_band_energy},
        spectral_centroids={ch: 3000.0 for ch in channel_band_energy},
        channel_band_energy=channel_band_energy,
    )

    pans = {ch: decision.pan_percent for ch, decision in decisions.items()}
    assert any(value < 0 for value in pans.values())
    assert any(value > 0 for value in pans.values())
    assert abs(pans[2]) < abs(pans[5])
    assert abs(pans[4]) < abs(pans[8])
