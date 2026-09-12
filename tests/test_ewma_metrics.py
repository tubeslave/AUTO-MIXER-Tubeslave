"""Tests for per-stem MuQ EWMA drift metrics."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ewma_metrics import DriftState, EwmaDrift, StemEwmaDriftMonitor


def _detector(**overrides):
    params = {
        "ewma": 0.90,
        "baseline": 0.90,
        "tau": 0.20,
        "warn_T": 0.05,
        "crit_T": 0.10,
        "debounce_warn": 0.30,
        "debounce_crit": 0.60,
    }
    params.update(overrides)
    return EwmaDrift(**params)


def test_noise_below_threshold_does_not_warn():
    detector = _detector(tau=1.0)
    noisy_scores = [0.895, 0.902, 0.891, 0.900, 0.894] * 10

    states = [detector.update(score, 0.1)["state"] for score in noisy_scores]

    assert set(states) == {DriftState.NORMAL.value}
    assert detector.warn_timer == 0.0
    assert detector.crit_timer == 0.0


def test_step_drop_warns_and_then_crits_after_debounce_delay():
    detector = _detector()
    states = []

    for _ in range(12):
        states.append(detector.update(0.70, 0.1)["state"])

    warn_index = states.index(DriftState.WARN.value)
    crit_index = states.index(DriftState.CRIT.value)

    assert warn_index >= 2
    assert crit_index >= 5
    assert warn_index < crit_index


def test_short_spike_is_debounced_and_resets_timer():
    detector = _detector(tau=0.001, debounce_warn=0.5, debounce_crit=1.0)

    assert detector.update(0.70, 0.1)["state"] == DriftState.NORMAL.value
    assert detector.warn_timer > 0.0
    assert detector.update(0.90, 0.1)["state"] == DriftState.NORMAL.value
    assert detector.warn_timer == 0.0


def test_warn_state_uses_hysteresis_before_clearing():
    detector = _detector()
    detector._state = DriftState.WARN
    detector.warn_timer = 1.0
    detector.ewma = 0.855

    held = detector.update(0.855, 0.1)
    assert held["state"] == DriftState.WARN.value

    detector.ewma = 0.89
    cleared = detector.update(0.90, 0.1)
    assert cleared["state"] == DriftState.NORMAL.value
    assert "clear_osc_highlight" in cleared["actions"]


def test_snapshot_and_restore_last_known_good_params():
    detector = _detector()
    snapshot = detector.snapshot_params({"fader_db": -8.0, "eq": {"band": 2}})

    snapshot["eq"]["band"] = 4
    restored = detector.restore_last_good()

    assert restored == {"fader_db": -8.0, "eq": {"band": 2}}


def test_stem_monitor_freezes_restores_and_releases_after_normal_window():
    monitor = StemEwmaDriftMonitor(
        {
            "enabled": True,
            "freeze_normal_seconds": 0.2,
            "default": {
                "tau": 0.001,
                "warn_T": 0.02,
                "crit_T": 0.04,
                "debounce_warn": 0.1,
                "debounce_crit": 0.1,
            },
            "groups": {
                "vox": {
                    "tau": 0.001,
                    "warn_T": 0.02,
                    "crit_T": 0.04,
                    "debounce_warn": 0.1,
                    "debounce_crit": 0.1,
                }
            },
        }
    )

    monitor.update_batch(
        {"lead_vocal": {"score": 0.90, "group": "vox"}},
        0.1,
        params_by_stem={"lead_vocal": {"fader_db": -7.0}},
        timestamp=1.0,
    )
    crit = monitor.update_batch(
        {"lead_vocal": {"score": 0.82, "group": "vox"}},
        0.1,
        timestamp=1.1,
    )

    stem = crit["stems"]["lead_vocal"]
    full_band = stem["masks"]["full_band"]
    assert stem["state"] == DriftState.CRIT.value
    assert monitor.is_stem_frozen("lead_vocal") is True
    assert full_band["restored_params"] == {"fader_db": -7.0}

    monitor.update_batch({"lead_vocal": {"score": 0.90, "group": "vox"}}, 0.1, timestamp=1.2)
    assert monitor.is_stem_frozen("lead_vocal") is True
    monitor.update_batch({"lead_vocal": {"score": 0.90, "group": "vox"}}, 0.2, timestamp=1.4)
    assert monitor.is_stem_frozen("lead_vocal") is False


def test_frequency_mask_uses_group_specific_thresholds():
    monitor = StemEwmaDriftMonitor(
        {
            "enabled": True,
            "default": {
                "tau": 0.001,
                "warn_T": 0.20,
                "crit_T": 0.30,
                "debounce_warn": 0.1,
                "debounce_crit": 0.1,
            },
            "frequency_masks": {
                "vox": {
                    "low_mid_300_800": {
                        "tau": 0.001,
                        "warn_T": 0.02,
                        "crit_T": 0.04,
                        "debounce_warn": 0.1,
                        "debounce_crit": 0.1,
                    }
                }
            },
        }
    )

    monitor.update_batch(
        {"vox": {"score": 0.90, "bands": {"low_mid_300_800": 0.90}}},
        0.1,
    )
    result = monitor.update_batch(
        {"vox": {"score": 0.86, "bands": {"low_mid_300_800": 0.84}}},
        0.1,
    )

    stem = result["stems"]["vox"]
    assert stem["masks"]["full_band"]["state"] == DriftState.NORMAL.value
    assert stem["masks"]["low_mid_300_800"]["state"] == DriftState.CRIT.value
