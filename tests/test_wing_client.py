"""
Tests for WingClient OSC communication (throttle, state management).

Uses mocking to avoid actual network connections to a Behringer Wing mixer.

Covers:
- OSC throttle rate limiting
- State management (state dict updates, callbacks)
- Connection and disconnection logic
"""

import time
from unittest.mock import patch, MagicMock, PropertyMock

import pytest


class TestOSCThrottle:
    """Tests for OSC message rate throttle."""

    def test_osc_throttle_default_enabled(self):
        """WingClient should have throttle enabled by default at 10 Hz."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)

            assert client._osc_throttle_enabled is True
            assert client._osc_throttle_hz == 10.0

    def test_osc_throttle_set(self):
        """set_osc_throttle should update throttle settings."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)

            client.set_osc_throttle(enabled=False, hz=20.0)
            assert client._osc_throttle_enabled is False
            assert client._osc_throttle_hz == 20.0

            client.set_osc_throttle(enabled=True, hz=5.0)
            assert client._osc_throttle_enabled is True
            assert client._osc_throttle_hz == 5.0

    def test_osc_throttle_queues_rapid_sends(self):
        """Rapid sends to the same address are queued, not dropped."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = True
            client.sock = MagicMock()
            client.set_osc_throttle(enabled=True, hz=10.0)

            result1 = client.send("/ch/1/fdr", 0.5)
            # Immediate second send is queued as trailing edge, reported success
            result2 = client.send("/ch/1/fdr", 0.6)

            assert result1 is True
            assert result2 is True
            assert client._osc_pending_sends.get("/ch/1/fdr") == (0.6,)
            client.disconnect()

    def test_osc_throttle_trailing_edge_flushes_last_value(self):
        """The latest throttled value must reach the socket after the window."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = True
            client.sock = MagicMock()
            client.set_osc_throttle(enabled=True, hz=20.0)  # 50 ms window

            client.send("/ch/1/fdr", -10.0)
            client.send("/ch/1/fdr", -8.0)
            client.send("/ch/1/fdr", -6.0)  # latest value wins

            # Only the first send hits the socket immediately
            assert client.sock.sendto.call_count == 1

            deadline = time.time() + 1.0
            while client.sock.sendto.call_count < 2 and time.time() < deadline:
                time.sleep(0.01)

            assert client.sock.sendto.call_count == 2
            assert client.state["/ch/1/fdr"] == -6.0
            assert client._osc_pending_sends == {}
            client.disconnect()

    def test_disconnect_cancels_pending_flush(self):
        """disconnect() must cancel queued trailing-edge sends."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = True
            client.sock = MagicMock()
            sock = client.sock
            client.set_osc_throttle(enabled=True, hz=20.0)

            client.send("/ch/1/fdr", -10.0)
            client.send("/ch/1/fdr", -6.0)
            client.disconnect()
            time.sleep(0.1)

            assert sock.sendto.call_count == 1
            assert client._osc_pending_sends == {}

    def test_osc_throttle_allows_queries(self):
        """Queries (sends without values) should not be throttled."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = True
            client.sock = MagicMock()
            client.set_osc_throttle(enabled=True, hz=1.0)

            # Queries (no values) should not be throttled
            client.send("/ch/1/fdr")
            client.send("/ch/1/fdr")
            # No assertion needed — just ensure no exception is raised


class TestStateManagement:
    """Tests for WingClient state dict and callback management."""

    def test_state_management_handle_message(self):
        """_handle_message should update internal state dict."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)

            # Single-arg message
            client._handle_message("/ch/1/fdr", 0.75)
            assert client.state["/ch/1/fdr"] == 0.75

            # Three-arg message (Wing format: display, normalized, actual)
            client._handle_message("/ch/2/fdr", "-10.0 dB", 0.5, -10.0)
            assert client.state["/ch/2/fdr"] == -10.0  # Actual value

    def test_state_management_callbacks(self):
        """Registered callbacks should be called on matching messages."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)

            received = []
            client.callbacks["/ch/1/fdr"] = [
                lambda addr, *args: received.append((addr, args))
            ]

            client._handle_message("/ch/1/fdr", 0.5)
            assert len(received) == 1
            assert received[0][0] == "/ch/1/fdr"

    def test_state_management_global_callback(self):
        """Wildcard '*' callbacks should receive all messages."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)

            all_msgs = []
            client.callbacks["*"] = [
                lambda addr, *args: all_msgs.append(addr)
            ]

            client._handle_message("/ch/1/fdr", 0.5)
            client._handle_message("/ch/2/mute", 1)

            assert len(all_msgs) == 2

    def test_state_management_name_handling(self):
        """Name-related messages should store the string value."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)

            client._handle_message("/ch/1/$name", "Vocals")
            assert client.state["/ch/1/$name"] == "Vocals"


class TestRoutingState:
    """Tests for WING channel/BUS/DCA routing snapshots."""

    def test_dca_tags_are_parsed_from_wing_tags(self):
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient

            assert WingClient._parse_dca_assignments("#D3 vocal #D12") == [3, 12]
            assert WingClient._parse_dca_assignments(["#D1", "band", "#D16"]) == [1, 16]

    def test_get_channel_settings_reads_bus_sends_and_dca_assignments(self):
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient

            client = WingClient(ip="127.0.0.1", port=2223)
            sent = []
            client.send = lambda address, *values: sent.append(address) or True
            client.state.update({
                "/ch/7/$name": "Lead Vox",
                "/ch/7/fdr": -6.0,
                "/ch/7/mute": 0,
                "/ch/7/tags": "#D2 #VOCAL",
                "/ch/7/send/5/on": 1,
                "/ch/7/send/5/lvl": -12.0,
                "/ch/7/send/5/mode": "POST",
                "/ch/7/send/8/on": 0,
                "/ch/7/send/8/lvl": -10.0,
            })

            with patch("wing_client.time.sleep", lambda _seconds: None):
                settings = client.get_channel_settings(7)

            assert "/ch/7/tags" in sent
            assert "/ch/7/send/5/lvl" in sent
            assert settings["dca_assignments"] == [2]
            assert settings["active_bus_sends"] == [
                {
                    "bus": 5,
                    "on": 1,
                    "level_db": -12.0,
                    "pre_on": None,
                    "mode": "POST",
                    "plink": None,
                    "pan": None,
                    "active": True,
                }
            ]

    def test_get_bus_and_dca_settings_read_processing_state(self):
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient

            client = WingClient(ip="127.0.0.1", port=2223)
            sent = []
            client.send = lambda address, *values: sent.append(address) or True
            client.state.update({
                "/bus/4/$name": "VOC BUS",
                "/bus/4/fdr": -4.0,
                "/bus/4/mute": 0,
                "/bus/4/eq/on": 1,
                "/bus/4/dyn/on": 1,
                "/bus/4/dyn/thr37": -18.0,
                "/bus/4/tags": "#D6",
                "/dca/6/name": "Vocals",
                "/dca/6/fdr": -2.0,
                "/dca/6/mute": 0,
            })

            with patch("wing_client.time.sleep", lambda _seconds: None):
                bus_settings = client.get_bus_settings(4)
                dca_settings = client.get_dca_settings(6)

            assert "/bus/4/dyn/thr37" in sent
            assert "/dca/6/fdr" in sent
            assert bus_settings["name"] == "VOC BUS"
            assert bus_settings["compressor_enabled"] == 1
            assert bus_settings["dca_assignments"] == [6]
            assert dca_settings["name"] == "Vocals"
            assert dca_settings["fader_db"] == -2.0


class TestConnectionLifecycle:
    """Tests for connect/disconnect logic."""

    def test_initial_state(self):
        """WingClient should initialize with correct defaults."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="10.0.0.1", port=2223)

            assert client.ip == "10.0.0.1"
            assert client.port == 2223
            assert client.is_connected is False
            assert client.state == {}
            assert client.callbacks == {}

    def test_disconnect(self):
        """disconnect() should set is_connected to False."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = True
            client.sock = MagicMock()

            client.disconnect()

            assert client.is_connected is False

    def test_send_when_not_connected(self):
        """send() should return False when not connected."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = False

            result = client.send("/ch/1/fdr", 0.5)
            assert result is False

    def test_setters_report_failure_when_disconnected(self):
        """Setters must propagate send() failure instead of returning None."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = False

            assert client.set_channel_fader(1, -6.0) is False
            assert client.set_fader(1, -6.0) is False
            assert client.set_eq_band(1, 2, freq=1000.0, gain=-3.0, q=2.0) is False
            assert client.set_main_fader(1, -3.0) is False
            assert client.set_bus_fader(1, -3.0) is False
            assert client.set_dca_fader(1, -3.0) is False

    def test_main_fader_clamped_to_hardware_range(self):
        """set_main_fader must clamp to the WING -144..10 dB range."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = True
            client.sock = MagicMock()
            client._osc_throttle_enabled = False

            assert client.set_main_fader(1, 99.0) is True
            assert client.state["/main/1/fdr"] == 10.0
            assert client.set_main_fader(1, -500.0) is True
            assert client.state["/main/1/fdr"] == -144.0

    def test_send_updates_local_state(self):
        """Sending a value should update the local state cache."""
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = True
            client.sock = MagicMock()
            client._osc_throttle_enabled = False  # Disable throttle

            client.send("/ch/1/fdr", 0.75)
            assert client.state.get("/ch/1/fdr") == 0.75


class TestAutoSoundcheckBridge:
    """Tests for the generic control methods used by AutoSoundcheckEngine."""

    def test_compressor_accepts_engine_aliases(self):
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = True
            client.sock = MagicMock()
            client._osc_throttle_enabled = False

            with patch("wing_client.time.sleep", return_value=None):
                client.set_compressor(
                    1,
                    threshold_db=-18.0,
                    ratio=4,
                    attack_ms=12.0,
                    release_ms=120.0,
                    makeup_db=3.0,
                    enabled=True,
                )

            assert client.state["/ch/1/dyn/on"] == 1
            assert client.state["/ch/1/dyn/thr26"] == -18.0
            assert client.state["/ch/1/dyn/ratio"] == "4.0"
            assert client.state["/ch/1/dyn/att"] == 12.0
            assert client.state["/ch/1/dyn/rel"] == 120.0
            assert client.state["/ch/1/dyn/gain"] == 3.0

    def test_generic_engine_methods_map_to_wing_commands(self):
        with patch.dict("sys.modules", {
            "pythonosc": MagicMock(),
            "pythonosc.udp_client": MagicMock(),
            "pythonosc.dispatcher": MagicMock(),
            "pythonosc.osc_server": MagicMock(),
            "pythonosc.osc_message_builder": MagicMock(),
            "pythonosc.osc_message": MagicMock(),
        }):
            from wing_client import WingClient
            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = True
            client.sock = MagicMock()
            client._osc_throttle_enabled = False

            client.set_hpf(2, 90.0, enabled=True)
            client.set_polarity(2, True)
            client.set_delay(2, 2.5, enabled=False)

            assert client.state["/ch/2/flt/lc"] == 1
            assert client.state["/ch/2/flt/lcf"] == 90.0
            assert client.state["/ch/2/in/set/inv"] == 1
            assert client.state["/ch/2/in/set/dlyon"] == 0
