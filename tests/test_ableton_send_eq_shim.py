"""Wing-style /ch/N/eq/*g paths → Ableton EQ Eight (reset_eq compatibility)."""

from unittest.mock import patch

import pytest

from ableton_client import AbletonClient


@pytest.fixture
def client():
    c = AbletonClient()
    c.is_connected = True
    c.sock = object()
    c._osc_throttle_enabled = False
    return c


def test_send_translates_eq_low_high_and_peq_gains(client):
    with patch.object(client, "set_eq_band_gain") as mock_gain:
        assert client.send("/ch/2/eq/lg", 0.0) is True
        mock_gain.assert_called_with(2, "lg", 0.0)

        mock_gain.reset_mock()
        assert client.send("/ch/2/eq/hg", -1.5) is True
        mock_gain.assert_called_with(2, "hg", -1.5)

        mock_gain.reset_mock()
        assert client.send("/ch/5/eq/3g", 2.0) is True
        mock_gain.assert_called_with(5, "3g", 2.0)


def test_send_does_not_translate_non_gain_eq_paths(client):
    with patch.object(client, "_send_osc") as mock_raw:
        client.send("/ch/1/eq/lf", 100.0)
        mock_raw.assert_called_once()
        args = mock_raw.call_args[0]
        assert args[0] == "/ch/1/eq/lf"
