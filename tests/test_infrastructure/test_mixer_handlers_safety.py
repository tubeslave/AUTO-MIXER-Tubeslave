"""Tests for mixer_handlers safety limits (fader/gain clamping)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

try:
    from handlers.mixer_handlers import (
        register_handlers,
        _get_safety_limits,
        _validate_channel,
        _validate_float,
        CHANNEL_MIN,
        CHANNEL_MAX,
    )
except ImportError:
    pytest.skip("handlers.mixer_handlers not importable", allow_module_level=True)


class TestSafetyLimits:
    """Test _get_safety_limits and safety application."""

    def test_safety_disabled_returns_none(self):
        server = MagicMock()
        server.config = {"safety": {"enable_limits": False}}
        assert _get_safety_limits(server) is None

    def test_safety_enabled_returns_limits(self):
        server = MagicMock()
        server.config = {
            "safety": {"enable_limits": True, "max_fader": 0, "max_gain": 12}
        }
        limits = _get_safety_limits(server)
        assert limits == (0, 12)

    def test_safety_defaults_when_missing(self):
        server = MagicMock()
        server.config = {"safety": {"enable_limits": True}}
        limits = _get_safety_limits(server)
        assert limits[0] == 0  # max_fader default 0 dBFS
        assert limits[1] == 18  # max_gain default 18


class TestValidateChannel:
    """Test _validate_channel."""

    def test_valid_channel(self):
        for ch in [1, 20, 40]:
            ok, err = _validate_channel(ch)
            assert ok is True
            assert err == ""

    def test_missing_channel(self):
        ok, err = _validate_channel(None)
        assert ok is False
        assert "Missing" in err

    def test_invalid_type(self):
        ok, err = _validate_channel("abc")
        assert ok is False
        assert "Invalid" in err

    def test_out_of_range_low(self):
        ok, err = _validate_channel(0)
        assert ok is False
        assert f"{CHANNEL_MIN}..{CHANNEL_MAX}" in err

    def test_out_of_range_high(self):
        ok, err = _validate_channel(41)
        assert ok is False


class TestValidateFloat:
    """Test _validate_float."""

    def test_valid_float(self):
        ok, val, err = _validate_float(3.14, "value")
        assert ok is True
        assert val == 3.14
        assert err == ""

    def test_int_accepted(self):
        ok, val, err = _validate_float(5, "value")
        assert ok is True
        assert val == 5.0

    def test_missing_value(self):
        ok, val, err = _validate_float(None, "value")
        assert ok is False
        assert val is None
        assert "Missing" in err

    def test_invalid_type(self):
        ok, val, err = _validate_float("not_a_number", "value")
        assert ok is False
        assert val is None
        assert "Invalid" in err


@pytest.mark.asyncio
class TestHandlerSafetyClamping:
    """Test that handlers apply safety limits before calling mixer."""

    async def test_set_fader_clamped_when_safety_enabled(self):
        server = MagicMock()
        server.config = {"safety": {"enable_limits": True, "max_fader": 0, "max_gain": 18}}
        mock_client = MagicMock()
        server.mixer_client = mock_client

        handlers = register_handlers(server)
        handle = handlers["set_fader"]
        ws = AsyncMock()

        await handle(ws, {"channel": 1, "value": 5.0})

        # Value 5 should be clamped to 0
        mock_client.set_channel_fader.assert_called_once_with(1, 0.0)

    async def test_set_gain_clamped_when_safety_enabled(self):
        server = MagicMock()
        server.config = {"safety": {"enable_limits": True, "max_fader": 0, "max_gain": 12}}
        mock_client = MagicMock()
        server.mixer_client = mock_client

        handlers = register_handlers(server)
        handle = handlers["set_gain"]
        ws = AsyncMock()

        await handle(ws, {"channel": 5, "value": 15.0})

        # Value 15 should be clamped to 12
        mock_client.set_channel_gain.assert_called_once_with(5, 12.0)

    async def test_set_fader_no_clamp_when_safety_disabled(self):
        server = MagicMock()
        server.config = {"safety": {"enable_limits": False}}
        mock_client = MagicMock()
        server.mixer_client = mock_client

        handlers = register_handlers(server)
        handle = handlers["set_fader"]
        ws = AsyncMock()

        await handle(ws, {"channel": 1, "value": 5.0})

        # Value 5 passed through (handler does not clamp when disabled;
        # WingClient would still apply hardware limit 10)
        mock_client.set_channel_fader.assert_called_once_with(1, 5.0)

    async def test_set_fader_rejects_invalid_channel(self):
        server = MagicMock()
        server.config = {"safety": {"enable_limits": True}}
        mock_client = MagicMock()
        server.mixer_client = mock_client
        server.send_to_client = AsyncMock()

        handlers = register_handlers(server)
        handle = handlers["set_fader"]
        ws = AsyncMock()

        await handle(ws, {"channel": 99, "value": 0.0})

        mock_client.set_channel_fader.assert_not_called()
        server.send_to_client.assert_called_once()
        call_args = server.send_to_client.call_args[0][1]
        assert call_args.get("type") == "error"
        assert "error" in call_args

    async def test_set_fader_rejects_missing_value(self):
        server = MagicMock()
        server.config = {"safety": {"enable_limits": True}}
        mock_client = MagicMock()
        server.mixer_client = mock_client
        server.send_to_client = AsyncMock()

        handlers = register_handlers(server)
        handle = handlers["set_fader"]
        ws = AsyncMock()

        await handle(ws, {"channel": 1})  # no value

        mock_client.set_channel_fader.assert_not_called()
        server.send_to_client.assert_called_once()
        call_args = server.send_to_client.call_args[0][1]
        assert call_args.get("type") == "error"
