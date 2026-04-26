"""Tests for Dugan-style NOM automix heuristics."""

import pytest

from auto_fader_v2.balance.dugan_automixer import (
    DuganAutomixer,
    DuganAutomixSettings,
)


def test_dugan_nom_attenuation_for_equal_open_mics():
    automixer = DuganAutomixer(
        DuganAutomixSettings(
            active_threshold_db=-70.0,
            auto_mix_depth_db=24.0,
            last_hold_enabled=False,
        )
    )

    one = automixer.calculate_target_gains({1: -20.0})
    assert one[1] == pytest.approx(0.0, abs=0.01)

    two = automixer.calculate_target_gains({1: -20.0, 2: -20.0})
    assert two[1] == pytest.approx(-3.01, abs=0.02)
    assert two[2] == pytest.approx(-3.01, abs=0.02)

    four = automixer.calculate_target_gains({
        1: -20.0,
        2: -20.0,
        3: -20.0,
        4: -20.0,
    })
    assert four[1] == pytest.approx(-6.02, abs=0.02)
    assert four[4] == pytest.approx(-6.02, abs=0.02)


def test_dugan_last_hold_keeps_last_active_channel_open():
    automixer = DuganAutomixer(
        DuganAutomixSettings(
            active_threshold_db=-50.0,
            auto_mix_depth_db=18.0,
            last_hold_enabled=True,
        )
    )

    active = automixer.calculate_target_gains({1: -80.0, 2: -25.0})
    assert active[2] == pytest.approx(0.0, abs=0.01)

    silent = automixer.calculate_target_gains({1: -80.0, 2: -80.0})
    assert silent[2] == pytest.approx(0.0, abs=0.01)
    assert silent[1] == pytest.approx(-18.0, abs=0.01)


def test_dugan_auto_mix_depth_clamps_inactive_and_weak_channels():
    automixer = DuganAutomixer(
        DuganAutomixSettings(
            active_threshold_db=-90.0,
            auto_mix_depth_db=12.0,
            last_hold_enabled=False,
        )
    )

    targets = automixer.calculate_target_gains({1: -10.0, 2: -80.0})
    assert targets[1] == pytest.approx(0.0, abs=0.01)
    assert targets[2] == pytest.approx(-12.0, abs=0.01)


def test_dugan_gain_limiting_allows_configured_full_gain_mics():
    automixer = DuganAutomixer(
        DuganAutomixSettings(
            active_threshold_db=-70.0,
            auto_mix_depth_db=24.0,
            max_full_gain_mics=4,
            last_hold_enabled=False,
        )
    )

    four = automixer.calculate_target_gains({ch: -20.0 for ch in range(1, 5)})
    assert all(value == pytest.approx(0.0, abs=0.01) for value in four.values())

    eight = automixer.calculate_target_gains({ch: -20.0 for ch in range(1, 9)})
    assert all(value == pytest.approx(-3.01, abs=0.02) for value in eight.values())
