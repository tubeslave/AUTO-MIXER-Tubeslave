"""Ableton EQ reset uses direct OSC path (not throttled send shim)."""

from unittest.mock import patch

import pytest

from ableton_client import AbletonClient


@pytest.fixture
def client():
    c = AbletonClient()
    c.is_connected = True
    c.sock = object()
    c._eq_band_param_indices = {
        1: (7, 6, 8),
        2: (17, 16, 18),
        3: (27, 26, 28),
        4: (37, 36, 38),
        5: (47, 46, 48),
        6: (57, 56, 58),
        7: (67, 66, 68),
        8: (77, 76, 78),
    }
    return c


def test_reset_channel_eq_gains_zero_invokes_set_eq_on_and_gains(client):
    with patch.object(client, "set_eq_on") as mock_on, patch.object(
        client, "set_eq_band_gain"
    ) as mock_gain, patch.object(
        client, "invalidate_eq_eight_indices"
    ), patch.object(
        client, "_ensure_eq_eight_indices", return_value=True
    ), patch.object(
        client, "_zero_eq_eight_gains_by_param_names", return_value=0
    ), patch.object(
        client, "_set_eq_eight_physical_band"
    ) as mock_phys:
        assert client.reset_channel_eq_gains_zero(3) is True
    mock_on.assert_called_once_with(3, 1)
    assert mock_gain.call_count == 6
    mock_gain.assert_any_call(3, "lg", 0.0)
    mock_gain.assert_any_call(3, "1g", 0.0)
    mock_gain.assert_any_call(3, "4g", 0.0)
    mock_gain.assert_any_call(3, "hg", 0.0)
    assert mock_phys.call_count == 2
    mock_phys.assert_any_call(3, 7, gain=0.0, log_label="reset_flat")
    mock_phys.assert_any_call(3, 8, gain=0.0, log_label="reset_flat")


def test_reset_channel_eq_gains_zero_not_connected(client):
    client.is_connected = False
    assert client.reset_channel_eq_gains_zero(1) is False


def test_zero_eq_eight_gains_by_param_names_matches_english_gain_rows():
    c = AbletonClient()
    c._eq_eight_last_param_names = [
        "Device On",
        "1 Filter On A",
        "1 Gain A",
        "1 Frequency A",
        "2 Gain A",
    ]
    c.eq_eight_device_index = 1
    c.is_connected = True
    c.sock = object()
    with patch.object(c, "_send_osc", return_value=True) as osc:
        n = c._zero_eq_eight_gains_by_param_names(0)
    assert n == 2
    assert osc.call_count == 2
    osc.assert_any_call(
        "/live/device/set/parameter/value", 0, 1, 2, (0.0 + 15.0) / 30.0
    )
    osc.assert_any_call(
        "/live/device/set/parameter/value", 0, 1, 4, (0.0 + 15.0) / 30.0
    )
