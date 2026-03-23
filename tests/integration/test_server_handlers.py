"""
Integration tests: WebSocket server + handlers with mock mixer.

Starts the real server, connects a WebSocket client, sends messages,
and verifies handler behavior. Uses mock mixer_client (no hardware).
"""

import asyncio
import json
import pytest
from unittest.mock import MagicMock

try:
    import websockets
except ImportError:
    pytest.skip("websockets not installed", allow_module_level=True)

try:
    from server import AutoMixerServer
except ImportError:
    pytest.skip("server module not importable", allow_module_level=True)


def _make_mock_mixer():
    """Create a mock mixer client with required interface."""
    mock = MagicMock()
    mock.is_connected = True
    mock.get_state.return_value = {"channels": {}, "connected": True}
    mock.get_channel_fader.return_value = 0.0
    mock.set_channel_fader = MagicMock()
    mock.set_channel_gain = MagicMock()
    mock.set_eq_band = MagicMock()
    mock.set_compressor = MagicMock()
    return mock


@pytest.mark.asyncio
async def test_set_fader_with_mock_mixer():
    """set_fader handler should call mixer_client.set_channel_fader when connected."""
    server = AutoMixerServer(ws_host="127.0.0.1", ws_port=18765)
    mock_mixer = _make_mock_mixer()
    server.mixer_client = mock_mixer
    server.connection_mode = "wing"

    # Disable safety limits for this test so value passes through
    server.config = {"safety": {"enable_limits": False}}

    async def run_server():
        await server.start()

    server_task = asyncio.create_task(run_server())
    await asyncio.sleep(0.3)  # Allow server to bind

    try:
        async with websockets.connect("ws://127.0.0.1:18765") as ws:
            await ws.send('{"type": "set_fader", "channel": 1, "value": -6.0}')
            # Handler runs synchronously; no response for set_fader
            await asyncio.sleep(0.1)

        mock_mixer.set_channel_fader.assert_called_once_with(1, -6.0)
    finally:
        server._shutdown_event.set()
        await asyncio.wait_for(server_task, timeout=2.0)


@pytest.mark.asyncio
async def test_get_state_with_mock_mixer():
    """get_state handler should return mixer state when connected."""
    server = AutoMixerServer(ws_host="127.0.0.1", ws_port=18766)
    mock_mixer = _make_mock_mixer()
    mock_mixer.get_state.return_value = {"channels": {"1": {"fader": 0.0}}, "connected": True}
    server.mixer_client = mock_mixer
    server.connection_mode = "wing"

    async def run_server():
        await server.start()

    server_task = asyncio.create_task(run_server())
    await asyncio.sleep(0.3)

    try:
        async with websockets.connect("ws://127.0.0.1:18766") as ws:
            await ws.send('{"type": "get_state"}')
            state_msg = None
            for _ in range(5):
                response = await asyncio.wait_for(ws.recv(), timeout=0.5)
                data = json.loads(response)
                if data.get("type") == "state_update":
                    state_msg = data
                    break
            assert state_msg is not None, "Expected state_update (got connection_status first)"
            assert state_msg.get("mode") == "wing"
            assert state_msg.get("state", {}).get("connected") is True
    finally:
        server._shutdown_event.set()
        await asyncio.wait_for(server_task, timeout=2.0)


@pytest.mark.asyncio
async def test_set_fader_rejects_when_mixer_disconnected():
    """set_fader should send error when mixer not connected."""
    server = AutoMixerServer(ws_host="127.0.0.1", ws_port=18767)
    server.mixer_client = None

    async def run_server():
        await server.start()

    server_task = asyncio.create_task(run_server())
    await asyncio.sleep(0.3)

    try:
        async with websockets.connect("ws://127.0.0.1:18767") as ws:
            await ws.send('{"type": "set_fader", "channel": 1, "value": 0.0}')
            response = await asyncio.wait_for(ws.recv(), timeout=1.0)

        import json
        data = json.loads(response)
        assert data.get("type") == "error"
        assert "not connected" in data.get("error", "").lower()
    finally:
        server._shutdown_event.set()
        await asyncio.wait_for(server_task, timeout=2.0)


@pytest.mark.asyncio
async def test_set_fader_safety_clamp():
    """set_fader should clamp to max_fader when safety enabled."""
    server = AutoMixerServer(ws_host="127.0.0.1", ws_port=18768)
    mock_mixer = _make_mock_mixer()
    server.mixer_client = mock_mixer
    server.connection_mode = "wing"
    server.config = {"safety": {"enable_limits": True, "max_fader": 0, "max_gain": 18}}

    async def run_server():
        await server.start()

    server_task = asyncio.create_task(run_server())
    await asyncio.sleep(0.3)

    try:
        async with websockets.connect("ws://127.0.0.1:18768") as ws:
            await ws.send('{"type": "set_fader", "channel": 1, "value": 5.0}')
            await asyncio.sleep(0.1)

        # Value 5 should be clamped to 0
        mock_mixer.set_channel_fader.assert_called_once_with(1, 0.0)
    finally:
        server._shutdown_event.set()
        await asyncio.wait_for(server_task, timeout=2.0)
