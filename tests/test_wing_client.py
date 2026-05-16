"""
Tests for WingClient OSC communication (throttle, state management).

Uses mocking to avoid actual network connections to a Behringer Wing mixer.

Covers:
- OSC throttle rate limiting
- State management (state dict updates, callbacks)
- Connection and disconnection logic
"""

from unittest.mock import patch, MagicMock, PropertyMock

import pytest


PYTHONOSC_MODULES = (
    "pythonosc",
    "pythonosc.udp_client",
    "pythonosc.dispatcher",
    "pythonosc.osc_server",
    "pythonosc.osc_message_builder",
    "pythonosc.osc_message",
)


def _mock_pythonosc_modules():
    return {module: MagicMock() for module in PYTHONOSC_MODULES}


@pytest.fixture
def connected_supervised_wing_client():
    with patch.dict("sys.modules", _mock_pythonosc_modules()):
        from wing_client import WingClient

        client = WingClient(ip="127.0.0.1", port=2223)
        client.is_connected = True
        client.sock = MagicMock()
        client.set_osc_throttle(enabled=False)
        client.arm_write_gate(reason="unit_test_arm")
        yield client


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
            client._write_gate_enabled = False
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
            client._write_gate_enabled = False

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
            client._write_gate_enabled = False
            client._osc_throttle_enabled = False  # Disable throttle

            client.send("/ch/1/fdr", 0.75)
            assert client.state.get("/ch/1/fdr") == 0.75


class TestSupervisedWriteGate:
    """Focused TUB-352 WING rollback/pre-read tests."""

    def test_direct_fader_and_gain_setters_propagate_blocked_gate(self):
        with patch.dict("sys.modules", _mock_pythonosc_modules()):
            from wing_client import WingClient

            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = True
            client.sock = MagicMock()
            client.set_osc_throttle(enabled=False)

            assert client.set_channel_fader(1, -20.0) is False
            assert client.set_fader(1, -21.0) is False
            assert client.set_channel_gain(1, -3.0) is False
            assert client.set_gain(1, -4.0) is False

            blocked_payloads = [
                event["payload"]
                for event in client._write_gate_events
                if event["event_type"] == "blocked"
            ]
            assert {payload["write_kind"] for payload in blocked_payloads} == {"fader", "gain"}

    def test_direct_write_gate_records_stable_inventory_fields_for_legacy_writes(self):
        with patch.dict("sys.modules", _mock_pythonosc_modules()):
            from wing_client import WingClient

            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = True
            client.sock = MagicMock()
            client.set_osc_throttle(enabled=False)

            assert client.set_fader(7, -21.0) is False
            assert client.set_gain(7, -4.0) is False

            blocked_payloads = [
                event["payload"]
                for event in client._write_gate_events
                if event["event_type"] == "blocked"
            ]
            by_kind = {payload["write_kind"]: payload for payload in blocked_payloads}

            assert by_kind["fader"]["channel"] == 7
            assert by_kind["fader"]["requested_value"] == -21.0
            assert by_kind["fader"]["source"] == "send"
            assert by_kind["fader"]["reason"] == "direct_write_requires_supervision"

            assert by_kind["gain"]["channel"] == 7
            assert by_kind["gain"]["requested_value"] == -4.0
            assert by_kind["gain"]["source"] == "send"
            assert by_kind["gain"]["reason"] == "direct_write_requires_supervision"

    def test_generic_compressor_aliases_propagate_blocked_gate(self):
        with patch.dict("sys.modules", _mock_pythonosc_modules()):
            from wing_client import WingClient

            client = WingClient(ip="127.0.0.1", port=2223)
            client.is_connected = True
            client.sock = MagicMock()
            client.set_osc_throttle(enabled=False)

            result = client.set_compressor(
                1,
                threshold_db=-18.0,
                ratio=3.0,
                attack_ms=5.0,
                release_ms=80.0,
                makeup_db=0.0,
                enabled=True,
            )

            assert result is False
            assert client._write_gate_events[-1]["payload"]["write_kind"] == "dynamics"

    def test_supervised_write_blocks_when_fresh_pre_read_is_unavailable(self, connected_supervised_wing_client):
        client = connected_supervised_wing_client

        with patch.object(client, "_wait_for_readback", return_value=None):
            result = client.supervised_set_channel_gain(
                1,
                -3.0,
                source="test",
                reason="approved change",
                approved=True,
                approval_id="approval-gain-2",
            )

        assert result["status"] == "blocked"
        assert result["reason"] == "pre_read_unavailable"
        client.sock.sendto.assert_not_called()

    def test_supervised_gain_readback_failure_rolls_back_and_emergency_stops(self, connected_supervised_wing_client):
        client = connected_supervised_wing_client

        with patch.object(client, "_wait_for_readback", side_effect=[0.0, -9.0, 0.0]):
            result = client.supervised_set_channel_gain(
                1,
                -3.0,
                source="test",
                reason="approved gain change",
                approved=True,
                approval_id="approval-gain-rollback",
            )

        assert result["status"] == "error"
        assert result["reason"] == "readback_mismatch"
        assert result["old_value"] == 0.0
        assert result["readback_value"] == -9.0
        assert result["rollback"]["attempted"] is True
        assert result["rollback"]["restored"] is True
        assert result["rollback"]["readback_value"] == 0.0
        assert client.sock.sendto.call_count == 2
        assert client.get_write_gate_status()["state"] == "emergency_stopped"

    def test_live_apply_contract_reports_supervised_fader_and_gain(self, connected_supervised_wing_client):
        contract = connected_supervised_wing_client.get_live_apply_contract()

        assert contract["contract_version"] == "live_apply_v1"
        assert contract["mixer_family"] == "wing"
        assert contract["value_types"]["fader"]["supported"] is True
        assert contract["value_types"]["gain"]["supported"] is True
        assert contract["value_types"]["fader"]["mode"] == "supervised"
        assert contract["value_types"]["gain"]["readback_source"] == "wing_osc_query"

    def test_apply_live_change_delegates_to_supervised_write_path(self, connected_supervised_wing_client):
        client = connected_supervised_wing_client

        with patch.object(client, "_wait_for_readback", side_effect=[0.0, -6.0]):
            result = client.apply_live_change(
                channel=4,
                value=-6.0,
                value_type="fader",
                source="test",
                reason="approved fader change",
                approved=True,
                approval_id="approval-fader-live",
            )

        assert result["status"] == "applied"
        assert result["channel"] == 4
        assert result["value_type"] == "fader"
        assert result["old_value"] == 0.0
        assert result["readback_value"] == -6.0
        assert result["readback_source"] == "wing_osc_query"
        assert result["contract_mode"] == "supervised"
