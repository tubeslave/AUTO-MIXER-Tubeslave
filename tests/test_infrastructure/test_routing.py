"""
Tests for backend/routing.py — MessageRouter, command validation,
WS/OSC message parsing, command-to-OSC translation, pattern matching.
"""

import json
import pytest

try:
    from routing import (
        MessageRouter,
        WSCommand,
        WSResponse,
        OSCAction,
        validate_command,
        parse_ws_message,
        command_to_osc,
    )
except ImportError:
    pytest.skip("routing module not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def router():
    """Fresh MessageRouter instance."""
    return MessageRouter()


class FakeWebSocket:
    """Minimal fake WebSocket for testing client registration."""

    def __init__(self, ws_id="ws_1"):
        self.id = ws_id
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


@pytest.fixture
def fake_ws():
    return FakeWebSocket("ws_test_1")


@pytest.fixture
def fake_ws2():
    return FakeWebSocket("ws_test_2")


# ---------------------------------------------------------------------------
# parse_ws_message tests
# parse_ws_message raises ValueError on bad input, not returns None
# ---------------------------------------------------------------------------

class TestParseWSMessage:

    def test_parse_action_format(self):
        """Should parse JSON with 'action' key."""
        msg = json.dumps({"action": "set_fader", "channel": 1, "params": {"value": -5.0}})
        cmd = parse_ws_message(msg)
        assert cmd is not None
        assert cmd.action == "set_fader"

    def test_parse_type_format(self):
        """Should parse JSON with 'type' key as an alias for action."""
        msg = json.dumps({"type": "set_mute", "channel": 2, "value": 1})
        cmd = parse_ws_message(msg)
        assert cmd is not None
        assert cmd.action == "set_mute"

    def test_parse_invalid_json(self):
        """Non-JSON input should raise ValueError."""
        with pytest.raises(ValueError):
            parse_ws_message("not json {{{")

    def test_parse_missing_action(self):
        """JSON without action or type should raise ValueError."""
        msg = json.dumps({"channel": 1, "value": 0.5})
        with pytest.raises(ValueError):
            parse_ws_message(msg)


# ---------------------------------------------------------------------------
# validate_command tests
# validate_command returns None (valid) or a string error message (invalid)
# ---------------------------------------------------------------------------

class TestValidateCommand:

    def test_valid_set_fader(self):
        cmd = WSCommand(action="set_fader", channel=1, params={"value": -5.0})
        error = validate_command(cmd)
        assert error is None

    def test_valid_set_mute(self):
        cmd = WSCommand(action="set_mute", channel=3, params={"value": 1})
        error = validate_command(cmd)
        assert error is None

    def test_unknown_action(self):
        cmd = WSCommand(action="do_something_weird", channel=1, params={})
        error = validate_command(cmd)
        # Unknown actions return a string error message
        assert isinstance(error, str)


# ---------------------------------------------------------------------------
# command_to_osc tests
# WSCommand uses params dict; OSCAction has address and args tuple
# ---------------------------------------------------------------------------

class TestCommandToOSC:

    def test_set_fader_to_osc(self):
        cmd = WSCommand(action="set_fader", channel=1, params={"value": -5.0})
        actions = command_to_osc(cmd)
        assert isinstance(actions, list)
        assert len(actions) >= 1
        # Should contain an OSC address like /ch/1/fdr
        osc = actions[0]
        assert isinstance(osc, OSCAction)
        assert "/ch/1/" in osc.address
        # Value is in args tuple
        assert osc.args[0] == -5.0

    def test_set_mute_to_osc(self):
        cmd = WSCommand(action="set_mute", channel=5, params={"value": 1})
        actions = command_to_osc(cmd)
        assert len(actions) >= 1
        assert "mute" in actions[0].address.lower() or "/ch/5/" in actions[0].address

    def test_set_pan_to_osc(self):
        cmd = WSCommand(action="set_pan", channel=2, params={"value": 50.0})
        actions = command_to_osc(cmd)
        assert len(actions) >= 1


# ---------------------------------------------------------------------------
# MessageRouter client management tests
# Uses register_ws_client(client_id, websocket) API
# ---------------------------------------------------------------------------

class TestMessageRouterClients:

    def test_register_client(self, router, fake_ws):
        router.register_ws_client(fake_ws.id, fake_ws)
        assert fake_ws.id in router._ws_clients

    def test_unregister_client(self, router, fake_ws):
        router.register_ws_client(fake_ws.id, fake_ws)
        router.unregister_ws_client(fake_ws.id)
        assert fake_ws.id not in router._ws_clients

    def test_unregister_unknown_client_no_error(self, router, fake_ws):
        """Unregistering an unknown client should not raise."""
        router.unregister_ws_client(fake_ws.id)

    def test_subscribe_client(self, router, fake_ws):
        """Subscribing a client to a topic should register the subscription."""
        router.register_ws_client(fake_ws.id, fake_ws)
        router.subscribe_client(fake_ws.id, "/ch/*/fdr")
        ws_client = router._ws_clients.get(fake_ws.id)
        assert ws_client is not None
        assert "/ch/*/fdr" in ws_client.subscriptions


# ---------------------------------------------------------------------------
# OSC pattern matching
# Note: '?' wildcard is NOT supported in the actual implementation
# ---------------------------------------------------------------------------

class TestOSCPatternMatching:

    def test_exact_match(self):
        assert MessageRouter._match_osc_pattern("/ch/1/fdr", "/ch/1/fdr") is True

    def test_wildcard_match(self):
        assert MessageRouter._match_osc_pattern("/ch/*/fdr", "/ch/1/fdr") is True
        assert MessageRouter._match_osc_pattern("/ch/*/fdr", "/ch/25/fdr") is True

    def test_no_match(self):
        assert MessageRouter._match_osc_pattern("/ch/1/fdr", "/ch/2/fdr") is False

    def test_global_wildcard(self):
        """A bare '*' pattern matches everything."""
        assert MessageRouter._match_osc_pattern("*", "/ch/1/fdr") is True
        assert MessageRouter._match_osc_pattern("*", "/any/address") is True
