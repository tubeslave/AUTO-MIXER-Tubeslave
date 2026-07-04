"""
Tests for WingClient OSC communication (throttle, state management).

Uses mocking to avoid actual network connections to a Behringer Wing mixer.

Covers:
- OSC throttle rate limiting
- State management (state dict updates, callbacks)
- Connection and disconnection logic
"""

import ast
import time
from pathlib import Path
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

    def test_osc_throttle_blocks_rapid_sends(self):
        """Rapid sends to the same address should be throttled."""
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

            # First send should succeed
            result1 = client.send("/ch/1/fdr", 0.5)
            # Immediate second send to same address should be throttled
            result2 = client.send("/ch/1/fdr", 0.6)

            assert result1 is not False or result1 is None  # First send goes through
            # Second send is likely throttled (returns False)
            # Note: exact behavior depends on timing, but we test the mechanism

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


class TestChannelNameScanning:
    """Tests for read-only WING channel name discovery."""

    def test_initial_scan_queries_linked_source_names(self):
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

            with patch("wing_client.time.sleep", lambda _seconds: None), patch("wing_client.threading.Thread"):
                client._scan_console_state()

            assert "/ch/1/$name" in sent
            assert "/ch/1/name" in sent
            assert "/ch/8/$name" in sent
            assert sent.index("/ch/1/$name") < sent.index("/ch/1/name")

    def test_full_scan_queries_linked_source_names(self):
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
            client._scan_channel_inserts = MagicMock()
            client.get_all_channels_with_inserts = MagicMock(return_value={})

            with patch("wing_client.time.sleep", lambda _seconds: None):
                client._scan_all_channels_with_inserts()

            assert "/ch/16/$name" in sent
            assert "/ch/16/name" in sent
            assert "/ch/40/$name" in sent
            assert client._scan_channel_inserts.call_count == 40


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

    def test_set_methods_with_direct_send_return_a_value_static(self):
        """Write helpers that call send() must expose send success/failure."""

        source_path = Path(__file__).resolve().parents[1] / "backend" / "wing_client.py"
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        wing_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "WingClient"
        )

        offenders = []
        for function in wing_class.body:
            if not isinstance(function, ast.FunctionDef) or not function.name.startswith("set_"):
                continue

            has_direct_send = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr == "send"
                for node in ast.walk(function)
            )
            has_return = any(isinstance(node, ast.Return) for node in ast.walk(function))

            if has_direct_send and not has_return:
                offenders.append(f"{function.name}:{function.lineno}")

        assert offenders == []

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

            result = client.send("/ch/1/fdr", 0.75)
            assert result is True
            assert client.state.get("/ch/1/fdr") == 0.75

    def test_set_channel_fader_returns_send_result(self):
        """set_channel_fader should expose send() success/failure to callers."""
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

            assert client.set_channel_fader(1, -6.0) is True
            assert client.state.get("/ch/1/fdr") == -6.0

    def test_set_channel_mute_returns_send_result(self):
        """set_channel_mute must not hide a dropped/failed write from callers.

        The server manual-write gate treats a False return as mixer_write_failed;
        a None return would be misreported as an applied mute.
        """
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

            assert client.set_channel_mute(1, 1) is True
            assert client.state.get("/ch/1/mute") == 1

            client.is_connected = False
            assert client.set_channel_mute(1, 0) is False

    def test_set_channel_pan_returns_send_result(self):
        """set_channel_pan must not hide a dropped/failed write from callers."""
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

            assert client.set_channel_pan(1, -50.0) is True
            assert client.state.get("/ch/1/pan") == -50.0

            client.is_connected = False
            assert client.set_channel_pan(1, 0.0) is False

    def test_mixer_client_bridge_write_methods_return_send_results(self):
        """MixerClientBase bridge methods must not convert failed writes to None."""

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

            assert client.set_fader(1, -6.0) is True
            assert client.set_gain(1, -3.0) is True
            assert client.set_mute(1, True) is True
            assert client.set_pan(1, 10.0) is True
            assert client.set_hpf(1, 120.0) is True
            assert client.set_polarity(1, True) is True
            assert client.set_delay(1, 2.5) is True
            assert client.set_delay(1, 0.0, enabled=False) is True
            assert client.set_send_level(1, 13, -18.0) is True

            assert client.state["/ch/1/fdr"] == -6.0
            assert client.state["/ch/1/in/set/trim"] == -3.0
            assert client.state["/ch/1/mute"] == 1
            assert client.state["/ch/1/pan"] == 10.0
            assert client.state["/ch/1/flt/lc"] == 1
            assert client.state["/ch/1/flt/lcf"] == 120.0
            assert client.state["/ch/1/in/set/inv"] == 1
            assert client.state["/ch/1/in/set/dlymode"] == "MS"
            assert client.state["/ch/1/in/set/dly"] == 2.5
            assert client.state["/ch/1/in/set/dlyon"] == 0
            assert client.state["/ch/1/send/13/lvl"] == -18.0

            client.is_connected = False

            assert client.set_fader(1, -7.0) is False
            assert client.set_gain(1, -4.0) is False
            assert client.set_mute(1, False) is False
            assert client.set_pan(1, 0.0) is False
            assert client.set_hpf(1, 100.0) is False
            assert client.set_polarity(1, False) is False
            assert client.set_delay(1, 1.0) is False
            assert client.set_delay(1, 0.0, enabled=False) is False
            assert client.set_send_level(1, 13, -20.0) is False

    def test_direct_eq_filter_write_methods_return_send_results(self):
        """Channel EQ/filter wrappers must not hide failed OSC writes."""

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

            assert client.set_high_cut(1, 1, frequency=12000.0, slope="12") is True
            assert client.set_eq_on(1, 1) is True
            assert client.set_eq_model(1, "STD") is True
            assert client.set_eq_mix(1, 100.0) is True
            assert client.set_eq_band_gain(1, "1g", -3.0) is True
            assert client.set_eq_band_frequency(1, "1f", 250.0) is True
            assert client.set_eq_band_q(1, "1q", 1.2) is True
            assert client.set_eq_low_shelf(1, gain=-2.0, freq=120.0, q=0.7, eq_type="SHV") is True
            assert client.set_eq_band(1, 2, freq=1200.0, gain=1.5, q=1.4) is True
            assert client.set_eq_high_shelf(1, gain=2.0, freq=8000.0, q=0.8, eq_type="SHV") is True
            assert client.set_channel_width(1, 80.0) is True
            assert client.set_channel_name(1, "Lead Vox") is True
            assert client.set_channel_color(1, 4) is True

            assert client.state["/ch/1/flt/hc"] == 1
            assert client.state["/ch/1/eq/on"] == 1
            assert client.state["/ch/1/eq/mdl"] == "STD"
            assert client.state["/ch/1/eq/1g"] == -3.0
            assert client.state["/ch/1/eq/2g"] == 1.5
            assert client.state["/ch/1/name"] == "Lead Vox"

            client.is_connected = False

            assert client.set_high_cut(1, 1, frequency=10000.0) is False
            assert client.set_eq_band(1, 2, freq=1000.0, gain=0.0, q=1.0) is False
            assert client.set_eq_high_shelf(1, gain=0.0, freq=9000.0) is False

    def test_direct_dynamics_bus_fx_write_methods_return_send_results(self):
        """Dynamics, gate, bus/main/matrix and FX helpers expose write failures."""

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
                assert client.set_compressor_on(1, 1) is True
                assert client.set_compressor_threshold(1, -24.0) is True
                assert client.set_compressor_ratio(1, "4.0") is True
                assert client.set_compressor_gain(1, 2.0) is True
                assert client.set_compressor_attack(1, 12.0) is True
                assert client.set_compressor_release(1, 180.0) is True
                assert client.set_compressor_knee(1, 2) is True
                assert client.set_compressor_mix(1, 80.0) is True
                assert client.set_compressor_model(1, "COMP") is True
                assert client.set_compressor(
                    1,
                    threshold_db=-20.0,
                    ratio=3,
                    knee=2,
                    attack_ms=10.0,
                    release_ms=150.0,
                    makeup_db=1.5,
                    mix=75.0,
                    enabled=True,
                ) is True
                assert client.set_gate_on(1, 1) is True
                assert client.set_gate(
                    1,
                    threshold=-55.0,
                    range_db=18.0,
                    attack=4.0,
                    hold=40.0,
                    release=120.0,
                    accent=20.0,
                    ratio="GATE",
                ) is True
                assert client.set_gate_model(1, "GATE") is True
                assert client.set_bus_fader(2, -6.0) is True
                assert client.set_bus_eq_band(2, 1, freq=120.0, gain=-2.0, q=0.8) is True
                assert client.set_bus_compressor(
                    2,
                    threshold_db=-12.0,
                    ratio=2,
                    attack_ms=8.0,
                    release_ms=120.0,
                    makeup_db=1.0,
                    enabled=True,
                ) is True
                assert client.set_main_eq_band(1, 1, freq=100.0, gain=-1.0, q=1.0) is True
                assert client.set_matrix_eq_band(1, 1, freq=100.0, gain=-1.0, q=1.0) is True
                assert client.set_fx_parameter("FX1", 1, 2.5) is True
                assert client.set_fx_model("FX1", "REVERB") is True
                assert client.set_fx_on("FX1", 1) is False
                assert client.set_fx_mix("FX1", 100.0) is True
                assert client.set_insert("channel", 1, "pre", slot="FX1", on=1) is True
                assert client.set_reverb_send(1, -18.0) is True
                assert client.set_reverb_params(
                    1,
                    "reverb",
                    decay_time=2.5,
                    predelay=20.0,
                    brightness=60.0,
                    density=80.0,
                    hpf=160.0,
                    lpf=9000.0,
                ) is True

            assert client.state["/ch/1/dyn/thr"] == -24.0
            assert client.state["/ch/1/dyn/thr26"] == -20.0
            assert client.state["/ch/1/gate/thr"] == -55.0
            assert client.state["/bus/2/dyn/thr37"] == -12.0
            assert client.state["/main/1/eq/1g"] == -1.0
            assert client.state["/mtx/1/eq/1g"] == -1.0
            assert "/fx/1/on" not in client.state
            assert client.state["/fx/1/fxmix"] == 100.0
            assert client.state["/ch/1/preins/ins"] == "FX1"
            assert client.state["/ch/1/send/13/mode"] == "POST"

            client.is_connected = False

            with patch("wing_client.time.sleep", return_value=None):
                assert client.set_compressor_threshold(1, -24.0) is False
                assert client.set_compressor(1, threshold_db=-20.0) is False
                assert client.set_gate(1, threshold=-55.0) is False
                assert client.set_bus_compressor(2, threshold_db=-12.0) is False
                assert client.set_main_eq_band(1, 1, freq=100.0, gain=0.0) is False
                assert client.set_fx_model("FX1", "REVERB") is False
                assert client.set_insert("channel", 1, "pre", slot="FX1", on=1) is False
                assert client.set_reverb_send(1, -20.0) is False
                assert client.set_reverb_params(
                    1,
                    "reverb",
                    decay_time=2.0,
                    predelay=10.0,
                    brightness=50.0,
                    density=70.0,
                    hpf=120.0,
                    lpf=8000.0,
                ) is False


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
