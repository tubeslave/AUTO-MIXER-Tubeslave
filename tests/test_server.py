"""
Tests for backend/server.py — convert_numpy_types utility and
AutoMixerServer basic initialization.

All tests work without hardware or network (all external connections mocked).
"""

import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import importlib

try:
    from server import convert_numpy_types, AutoMixerServer
except ImportError:
    pytest.skip("server module not importable", allow_module_level=True)

server_module = importlib.import_module("server")


# ---------------------------------------------------------------------------
# convert_numpy_types tests
# ---------------------------------------------------------------------------

class TestConvertNumpyTypes:

    def test_int_types(self):
        for dtype in [np.int8, np.int16, np.int32, np.int64,
                      np.uint8, np.uint16, np.uint32, np.uint64]:
            val = dtype(42)
            result = convert_numpy_types(val)
            assert isinstance(result, int)
            assert result == 42

    def test_float_types(self):
        for dtype in [np.float32, np.float64]:
            val = dtype(3.14)
            result = convert_numpy_types(val)
            assert isinstance(result, float)
            assert abs(result - 3.14) < 1e-5

    def test_bool_type(self):
        val = np.bool_(True)
        result = convert_numpy_types(val)
        assert isinstance(result, bool)
        assert result is True

    def test_ndarray(self):
        arr = np.array([1, 2, 3])
        result = convert_numpy_types(arr)
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_dict_recursive(self):
        data = {"a": np.int32(1), "b": np.float64(2.5), "c": "hello"}
        result = convert_numpy_types(data)
        assert isinstance(result["a"], int)
        assert isinstance(result["b"], float)
        assert result["c"] == "hello"

    def test_list_recursive(self):
        data = [np.int32(1), np.float64(2.5), "hello"]
        result = convert_numpy_types(data)
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)
        assert result[2] == "hello"

    def test_nested_dict(self):
        data = {"outer": {"inner": np.int64(10)}}
        result = convert_numpy_types(data)
        assert isinstance(result["outer"]["inner"], int)

    def test_native_types_unchanged(self):
        assert convert_numpy_types(42) == 42
        assert convert_numpy_types(3.14) == 3.14
        assert convert_numpy_types("hello") == "hello"
        assert convert_numpy_types(True) is True
        assert convert_numpy_types(None) is None

    def test_tuple_recursive(self):
        data = (np.int32(1), np.float64(2.0))
        result = convert_numpy_types(data)
        assert isinstance(result, list)
        assert all(isinstance(x, (int, float)) for x in result)

    def test_json_serializable(self):
        """Result should be JSON-serializable."""
        data = {
            "level": np.float32(-5.0),
            "channel": np.int64(3),
            "muted": np.bool_(False),
            "values": np.array([1.0, 2.0, 3.0]),
        }
        result = convert_numpy_types(data)
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# AutoMixerServer initialization tests (with mocks)
# ---------------------------------------------------------------------------

class TestAutoMixerServerInit:

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_basic_init(self, mock_config, mock_bleed):
        """Server should initialize with default host and port."""
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        assert server.ws_host == "localhost"
        assert server.ws_port == 8765
        assert server.mixer_client is None
        assert server.connection_mode is None
        assert server.connected_clients == set()
        assert server.live_mode is False

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_custom_host_port(self, mock_config, mock_bleed):
        """Server should accept custom host and port."""
        server = AutoMixerServer(ws_host="0.0.0.0", ws_port=9999)
        assert server.ws_host == "0.0.0.0"
        assert server.ws_port == 9999

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_controllers_none_initially(self, mock_config, mock_bleed):
        """All controllers should be None after initialization."""
        server = AutoMixerServer()
        assert server.voice_control is None
        assert server.gain_staging is None
        assert server.auto_eq_controller is None
        assert server.phase_alignment_controller is None
        assert server.auto_fader_controller is None
        assert server.auto_compressor_controller is None

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_auto_soundcheck_not_running(self, mock_config, mock_bleed):
        """Auto soundcheck should not be running initially."""
        server = AutoMixerServer()
        assert server.auto_soundcheck_running is False
        assert server.auto_soundcheck_task is None

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_shutdown_flags(self, mock_config, mock_bleed):
        """Shutdown flags should be properly initialized."""
        server = AutoMixerServer()
        assert server._is_shutting_down is False
        assert server._shutdown_event is None


class DummyWingLike:
    def __init__(self):
        self.set_channel_fader = MagicMock()
        self.set_channel_gain = MagicMock()
        self.supervised_set_channel_fader = MagicMock(side_effect=self._supervised_fader)
        self.supervised_set_channel_gain = MagicMock(side_effect=self._supervised_gain)
        self.configure_write_gate = MagicMock()
        self.arm_write_gate = MagicMock()
        self.disarm_write_gate = MagicMock()
        self.emergency_stop_writes = MagicMock()
        self.get_write_gate_status = MagicMock(return_value={"state": "disarmed"})
        self.send = MagicMock()
        self.get_channel_fader = MagicMock(return_value=0.0)
        self._fader_value = -5.0
        self._gain_value = 0.0

    def _supervised_fader(self, channel, value, **kwargs):
        old_value = self._fader_value
        self._fader_value = value
        return {
            "status": "applied",
            "channel": channel,
            "effective_value": value,
            "old_value": old_value,
            "readback_value": value,
            **kwargs,
        }

    def _supervised_gain(self, channel, value, **kwargs):
        old_value = self._gain_value
        self._gain_value = value
        return {
            "status": "applied",
            "channel": channel,
            "effective_value": value,
            "old_value": old_value,
            "readback_value": value,
            **kwargs,
        }


@pytest.fixture
def patch_wing_type_for_isinstance():
    with patch.object(server_module, "WingClient", DummyWingLike), patch.object(server_module, "EnhancedOSCClient", DummyWingLike):
        yield


@pytest.fixture
def dummy_wing_server(patch_wing_type_for_isinstance):
    server = AutoMixerServer(ws_host="localhost", ws_port=8765)
    server.connection_mode = "wing"
    server.mixer_client = DummyWingLike()
    server._manual_write_supervision_enabled = True
    server._manual_writes_armed = True
    server._manual_write_cooldown_sec = 0.0
    return server


@pytest.fixture
def real_wing_server():
    server = AutoMixerServer(ws_host="localhost", ws_port=8765)
    server.connection_mode = "wing"
    client = server_module.WingClient(ip="127.0.0.1", port=2223)
    client.is_connected = True
    client.sock = MagicMock()
    client.set_osc_throttle(enabled=False)
    client.arm_write_gate(reason="unit_test_arm")
    server.mixer_client = client
    server._manual_write_supervision_enabled = True
    server._manual_writes_armed = True
    server._manual_write_cooldown_sec = 0.0
    return server, client


class TestManualWriteRollbackSession:

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_manual_write_session_records_steps_and_rolls_back_last_safe_value(
        self,
        mock_bleed_service,
        mock_load_config,
        dummy_wing_server,
    ):
        server = dummy_wing_server
        session = server.start_manual_write_session(session_label="test-session", reason="unit test")
        assert session["status"] == "started"

        await server._apply_manual_console_write(
            None,
            channel=1,
            value=-10.0,
            value_type="fader",
            source="manual:first",
            reason="first approved write",
            approved=True,
            approval_id="approval-fader-1",
        )
        await server._apply_manual_console_write(
            None,
            channel=1,
            value=2.5,
            value_type="gain",
            source="manual:second",
            reason="second approved write",
            approved=True,
            approval_id="approval-gain-1",
        )

        assert server.get_manual_write_session_status()["step_count"] == 2

        rollback = await server.rollback_manual_write_session(
            None,
            rollback_mode="rollback_last",
            reason="rollback last",
            approved=True,
            approval_id="rollback-1",
        )

        assert rollback["status"] == "applied"
        assert rollback["rolled_back_count"] == 1
        assert rollback["remaining_step_count"] == 1
        assert rollback["rolled_back_steps"][0]["original_step"]["value_type"] == "gain"
        assert rollback["rolled_back_steps"][0]["original_step"]["old_value"] == 0.0
        assert rollback["rolled_back_steps"][0]["rollback_result"]["readback_value"] == 0.0
        assert server.get_manual_write_session_status()["step_count"] == 1
        assert server.mixer_client.supervised_set_channel_gain.call_args_list[-1].kwargs["bypass_cooldown"] is True

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_manual_write_session_gain_rollback_succeeds_with_real_wing_client_pre_read(
        self,
        mock_bleed_service,
        mock_load_config,
        real_wing_server,
    ):
        server, client = real_wing_server
        server.start_manual_write_session(session_label="test-session", reason="unit test")

        with patch.object(client, "_wait_for_readback", side_effect=[0.0, -3.0, -3.0, 0.0]):
            applied = await server._apply_manual_console_write(
                None,
                channel=1,
                value=-3.0,
                value_type="gain",
                source="manual:first",
                reason="first approved write",
                approved=True,
                approval_id="approval-gain-1",
            )
            rollback = await server.rollback_manual_write_session(
                None,
                rollback_mode="rollback_last",
                reason="rollback last",
                approved=True,
                approval_id="rollback-gain-1",
            )

        assert applied["status"] == "applied"
        assert applied["old_value"] == 0.0
        assert rollback["status"] == "applied"
        assert rollback["rolled_back_count"] == 1
        assert rollback["remaining_step_count"] == 0
        assert rollback["rolled_back_steps"][0]["original_step"]["value_type"] == "gain"
        assert rollback["rolled_back_steps"][0]["rollback_result"]["effective_value"] == 0.0
        assert rollback["rolled_back_steps"][0]["rollback_result"]["readback_value"] == 0.0
        assert client.sock.sendto.call_count == 2
        assert server.get_manual_write_session_status()["active"] is False
        assert server.get_manual_write_session_status()["step_count"] == 0

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_manual_write_session_gain_rollback_blocks_when_real_wing_pre_read_is_unavailable(
        self,
        mock_bleed_service,
        mock_load_config,
        real_wing_server,
    ):
        server, client = real_wing_server
        server.start_manual_write_session(session_label="test-session", reason="unit test")

        with patch.object(client, "send", wraps=client.send) as send_spy:
            with patch.object(client, "_wait_for_readback", side_effect=[0.0, -3.0, None]):
                applied = await server._apply_manual_console_write(
                    None,
                    channel=1,
                    value=-3.0,
                    value_type="gain",
                    source="manual:first",
                    reason="first approved write",
                    approved=True,
                    approval_id="approval-gain-1",
                )
                rollback = await server.rollback_manual_write_session(
                    None,
                    rollback_mode="rollback_last",
                    reason="rollback last",
                    approved=True,
                    approval_id="rollback-gain-2",
                )

        write_calls = [call_args for call_args in send_spy.call_args_list if len(call_args.args) > 1]
        assert applied["status"] == "applied"
        assert rollback["status"] == "blocked"
        assert rollback["reason"] == "pre_read_unavailable"
        assert rollback["rolled_back_count"] == 0
        assert rollback["remaining_step_count"] == 1
        assert rollback["failed_step"]["original_step"]["value_type"] == "gain"
        assert rollback["failed_step"]["rollback_result"]["status"] == "blocked"
        assert rollback["failed_step"]["rollback_result"]["reason"] == "pre_read_unavailable"
        assert len(write_calls) == 1
        assert write_calls[0].args == ("/ch/1/in/set/trim", -3.0)
        assert server.get_manual_write_session_status()["active"] is True
        assert server.get_manual_write_session_status()["step_count"] == 1
