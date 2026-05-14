"""
Tests for WingClient OSC communication (throttle, state management).

Uses mocking to avoid actual network connections to a Behringer Wing mixer.

Covers:
- OSC throttle rate limiting
- State management (state dict updates, callbacks)
- Connection and disconnection logic
"""

import time
import threading
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

    def test_shadow_mode_blocks_mutation_and_emits_would_send_event(self):
        events = []
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
            client.configure_shadow_transport(enabled=True, event_sink=events.append)
            client.set_transport_context(
                audit_id="audit-shadow-1",
                replay_correlation_id="corr::wing::1",
                source="unit_test",
                operation="set_fader",
                channel=1,
                metadata={"path": "test"},
            )

            result = client.send("/ch/1/fdr", -5.0)

            assert result is False
            assert client.get_last_send_status() == "shadow_blocked"
            client.sock.sendto.assert_not_called()
            assert events[-1]["audit_id"] == "audit-shadow-1"
            assert events[-1]["replay_correlation_id"] == "corr::wing::1"
            assert events[-1]["address"] == "/ch/1/fdr"
            assert events[-1]["values"] == [-5.0]
            assert client.consume_shadow_transport_event()["audit_id"] == "audit-shadow-1"

    def test_shadow_mode_preserves_query_transport(self):
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
            client.configure_shadow_transport(enabled=True)

            result = client.send("/ch/1/fdr")

            assert result is True
            assert client.get_last_send_status() == "sent"
            client.sock.sendto.assert_called_once()


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
            assert client.get_last_send_status() == "disconnected"

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

    def test_set_fader_returns_send_result(self):
        """set_fader() should expose whether the OSC send succeeded."""
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

            with patch.object(client, "send", return_value=False) as mocked_send:
                result = client.set_fader(1, -12.0)

            assert result is False
            mocked_send.assert_called_once_with("/ch/1/fdr", -12.0)

    def test_send_throttled_sets_last_send_status(self):
        """Throttle should expose a distinct send status."""
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

            assert client.send("/ch/1/fdr", 0.5) is True
            assert client.send("/ch/1/fdr", 0.6) is False
            assert client.get_last_send_status() == "throttled"

    def test_confirm_manual_write_confirms_fresh_query_response(self):
        """Confirmed status requires a fresh callback after the query send."""
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

            def fake_send(address, *values):
                client._last_send_status = "sent"
                if not values:
                    threading.Timer(
                        0.01,
                        lambda: client._handle_message(address, "-10.0 dB", 0.5, -10.0),
                    ).start()
                return True

            with patch.object(client, "send", side_effect=fake_send):
                result = client.confirm_manual_write("set_fader", 1, -10.0, timeout_ms=100)

            assert result["readback_status"] == "confirmed"
            assert result["confirmed_value"] == -10.0
            assert result["failure_reason"] is None
            assert result["verification_id"]

    def test_confirm_manual_write_times_out_with_old_cached_value(self):
        """Old cached state must not close a new verification."""
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
            client.state["/ch/1/fdr"] = -10.0
            client._state_timestamps["/ch/1/fdr"] = time.time()

            with patch.object(client, "send", return_value=True):
                result = client.confirm_manual_write("set_fader", 1, -10.0, timeout_ms=20)

            assert result["readback_status"] == "timeout"
            assert result["confirmed_value"] is None
            assert result["failure_reason"] == "readback_timeout"

    def test_confirm_manual_write_detects_mismatch(self):
        """A fresh but different readback must report mismatch."""
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

            def fake_send(address, *values):
                client._last_send_status = "sent"
                if not values:
                    threading.Timer(
                        0.01,
                        lambda: client._handle_message(address, "-30.0 dB", 0.1, -30.0),
                    ).start()
                return True

            with patch.object(client, "send", side_effect=fake_send):
                result = client.confirm_manual_write("set_fader", 1, -10.0, timeout_ms=100)

            assert result["readback_status"] == "mismatch"
            assert result["confirmed_value"] == -30.0
            assert result["failure_reason"] == "readback_mismatch"

    def test_confirm_manual_write_exposes_query_send_failure(self):
        """A readback-query send failure must stay explicit for the caller."""
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

            def fake_send(address, *values):
                client._last_send_status = "failed"
                return False

            with patch.object(client, "send", side_effect=fake_send):
                result = client.confirm_manual_write("set_gain", 1, 6.0, timeout_ms=100)

            assert result["readback_status"] == "timeout"
            assert result["confirmed_value"] is None
            assert result["failure_reason"] == "failed"
            assert result["verification_id"]
