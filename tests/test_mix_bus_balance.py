"""Tests for bus-level balance heuristics."""

import pytest

from mix_bus_balance import calculate_drum_bus_gain


def test_drum_bus_gain_hits_middle_target():
    decision = calculate_drum_bus_gain(drum_lufs=-20.0, music_lufs=-20.0)

    assert decision.current_delta_lu == pytest.approx(0.0)
    assert decision.target_delta_lu == pytest.approx(1.5)
    assert decision.gain_db == pytest.approx(1.5)


def test_drum_bus_gain_clamps_boost():
    decision = calculate_drum_bus_gain(drum_lufs=-26.0, music_lufs=-20.0)

    assert decision.current_delta_lu == pytest.approx(-6.0)
    assert decision.gain_db == pytest.approx(4.0)


def test_drum_bus_gain_clamps_cut():
    decision = calculate_drum_bus_gain(drum_lufs=-16.0, music_lufs=-20.0)

    assert decision.current_delta_lu == pytest.approx(4.0)
    assert decision.gain_db == pytest.approx(-2.0)


def test_drum_bus_gain_rejects_bad_bounds():
    with pytest.raises(ValueError):
        calculate_drum_bus_gain(
            drum_lufs=-20.0,
            music_lufs=-20.0,
            min_gain_db=4.0,
            max_gain_db=-2.0,
        )
