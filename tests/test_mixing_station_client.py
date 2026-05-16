"""
Tests for MixingStationClient live-apply/readback contract.
"""

import os
import sys
from unittest.mock import MagicMock, patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


PYTHONOSC_MODULES = (
    "pythonosc",
    "pythonosc.osc_message_builder",
    "pythonosc.osc_message",
)


def _mock_pythonosc_modules():
    return {module: MagicMock() for module in PYTHONOSC_MODULES}


class TestMixingStationLiveApplyContract:
    def test_state_address_maps_preamp_gain_for_readback(self):
        with patch.dict("sys.modules", _mock_pythonosc_modules()):
            from mixing_station_client import MixingStationClient

            client = MixingStationClient()
            assert client._ms_to_state_address("/con/v/ch.0.preamp.gain") == "/ch/1/gain"

    def test_live_apply_contract_reports_query_confirmed_fader_and_gain(self):
        with patch.dict("sys.modules", _mock_pythonosc_modules()):
            from mixing_station_client import MixingStationClient

            client = MixingStationClient()
            contract = client.get_live_apply_contract()

            assert contract["contract_version"] == "live_apply_v1"
            assert contract["mixer_family"] == "mixing_station"
            assert contract["value_types"]["fader"]["supported"] is True
            assert contract["value_types"]["gain"]["supported"] is True
            assert contract["value_types"]["fader"]["mode"] == "query_confirmed"

    def test_apply_live_change_confirms_fader_readback(self):
        with patch.dict("sys.modules", _mock_pythonosc_modules()):
            from mixing_station_client import MixingStationClient

            client = MixingStationClient()
            with patch.object(client, "read_live_value", side_effect=[
                {
                    "status": "ok",
                    "channel": 7,
                    "value_type": "fader",
                    "value": 0.0,
                    "readback_source": "mixing_station_query",
                    "reason": "ok",
                },
                {
                    "status": "ok",
                    "channel": 7,
                    "value_type": "fader",
                    "value": -4.5,
                    "readback_source": "mixing_station_query",
                    "reason": "ok",
                },
            ]), patch.object(client, "set_channel_fader", return_value=True) as set_fader:
                result = client.apply_live_change(
                    channel=7,
                    value=-4.5,
                    value_type="fader",
                    source="test",
                    reason="manual live apply",
                    approved=True,
                )

        assert result["status"] == "applied"
        assert result["old_value"] == 0.0
        assert result["readback_value"] == -4.5
        assert result["readback_source"] == "mixing_station_query"
        assert result["contract_mode"] == "query_confirmed"
        set_fader.assert_called_once_with(7, -4.5)

    def test_apply_live_change_blocks_when_pre_read_is_unavailable(self):
        with patch.dict("sys.modules", _mock_pythonosc_modules()):
            from mixing_station_client import MixingStationClient

            client = MixingStationClient()
            with patch.object(client, "read_live_value", return_value={
                "status": "unavailable",
                "channel": 7,
                "value_type": "gain",
                "value": None,
                "readback_source": "mixing_station_query",
                "reason": "readback_unavailable",
            }):
                result = client.apply_live_change(
                    channel=7,
                    value=2.0,
                    value_type="gain",
                    source="test",
                    reason="manual live apply",
                    approved=True,
                )

        assert result["status"] == "blocked"
        assert result["reason"] == "pre_read_unavailable"
