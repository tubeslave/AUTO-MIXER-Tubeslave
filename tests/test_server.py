"""
Tests for backend/server.py — convert_numpy_types utility and
AutoMixerServer basic initialization.

All tests work without hardware or network (all external connections mocked).
"""

import asyncio
import json
import os
import numpy as np
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

try:
    from server import convert_numpy_types, AutoMixerServer
except ImportError:
    pytest.skip("server module not importable", allow_module_level=True)


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
        assert server.mixing_agent is None
        assert server.mixing_agent_task is None

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


class TestAutoMixerServerConfigNormalization:

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_normalize_wing_mixer_settings_accepts_ui_aliases(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        normalized = server._normalize_mixer_user_settings({
            "mixerType": "wing",
            "mixerIp": "10.0.0.5",
            "mixerPort": 2223,
        })
        assert normalized["type"] == "wing"
        assert normalized["ip"] == "10.0.0.5"
        assert normalized["port"] == 2223
        assert normalized["send_port"] == 2222
        assert normalized["receive_port"] == 2223

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_normalize_wing_mixer_settings_preserves_send_receive_ports(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        normalized = server._normalize_mixer_user_settings({
            "type": "wing",
            "ip": "10.0.0.9",
            "send_port": 2222,
            "receive_port": 2223,
        })
        assert normalized["mixerIp"] == "10.0.0.9"
        assert normalized["mixerSendPort"] == 2222
        assert normalized["mixerReceivePort"] == 2223
        assert normalized["mixerPort"] == 2223

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_normalize_dlive_mixer_settings_accepts_canonical_keys(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        normalized = server._normalize_mixer_user_settings({
            "type": "dlive",
            "ip": "192.168.3.70",
            "port": 51328,
            "tls": True,
            "midi_base_channel": 2,
        })
        assert normalized["mixerType"] == "dlive"
        assert normalized["dliveIp"] == "192.168.3.70"
        assert normalized["dlivePort"] == 51328
        assert normalized["dliveTls"] is True
        assert normalized["dliveMidiChannel"] == 2

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_ready_for_live_requires_active_auto_fader_status(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.mixer_client = MagicMock()
        server.mixer_client.is_connected = True
        server.gain_staging = MagicMock()
        server.auto_fader_controller = MagicMock()
        server.auto_fader_controller.get_status.return_value = {"active": False, "realtime_enabled": False}

        status = server.get_ready_for_live_status()

        assert status["ready"] is False
        assert status["checks"]["auto_fader_available"] is False

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_ready_for_live_accepts_collecting_auto_fader_status(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.mixer_client = MagicMock()
        server.mixer_client.is_connected = True
        server.gain_staging = MagicMock()
        server.auto_fader_controller = MagicMock()
        server.auto_fader_controller.get_status.return_value = {"collecting": True}

        status = server.get_ready_for_live_status()

        assert status["ready"] is True
        assert status["checks"]["auto_fader_available"] is True

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.backup_channel")
    async def test_create_snapshot_reports_failure_when_all_channels_fail(self, mock_backup_channel, mock_config, mock_bleed, tmp_path):
        server = AutoMixerServer()
        server.mixer_client = MagicMock()
        server.mixer_client.is_connected = True
        server.send_to_client = AsyncMock()

        mock_backup_channel.side_effect = RuntimeError("backup failed")
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            await server.create_snapshot("ws", [1, 2])
        finally:
            os.chdir(cwd)

        payload = server.send_to_client.await_args.args[1]
        assert payload["type"] == "snapshot_result"
        assert payload["success"] is False
        assert payload["success_count"] == 0
        assert payload["failed_channels"] == [1, 2]
        assert payload["error"] == "Failed to back up any selected channels"
        assert server._last_snapshot_path is None

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.backup_channel")
    async def test_create_snapshot_preserves_partial_success_details(self, mock_backup_channel, mock_config, mock_bleed, tmp_path):
        server = AutoMixerServer()
        server.mixer_client = MagicMock()
        server.mixer_client.is_connected = True
        server.send_to_client = AsyncMock()

        def backup_side_effect(client, channel):
            if channel == 2:
                raise RuntimeError("backup failed")
            return {"channel": channel, "ok": True}

        mock_backup_channel.side_effect = backup_side_effect
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            await server.create_snapshot("ws", [1, 2])
        finally:
            os.chdir(cwd)

        payload = server.send_to_client.await_args.args[1]
        assert payload["success"] is True
        assert payload["success_count"] == 1
        assert payload["failed_channels"] == [2]
        assert payload["error"] is None
        assert server._last_snapshot_path is not None

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_stop_voice_control_reports_not_running(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock()
        server.connected_clients.add("ws")

        await server.stop_voice_control()

        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "voice_control_status",
            "active": False,
            "message": "Voice control is not running"
        })

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_stop_voice_control_reports_error_to_client(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock()
        server.connected_clients.add("ws")
        server.voice_control = MagicMock()
        server.voice_control.stop_listening.side_effect = RuntimeError("stop failed")

        await server.stop_voice_control()

        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "voice_control_status",
            "active": False,
            "error": "stop failed"
        })

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_stop_realtime_fader_requires_controller(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock()

        await server.stop_realtime_fader("ws")

        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "auto_fader_status",
            "status_type": "error",
            "error": "Auto Fader controller not initialized"
        })

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_cancel_auto_balance_requires_controller(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock()

        await server.cancel_auto_balance("ws")

        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "auto_fader_status",
            "status_type": "error",
            "error": "Auto Fader controller not initialized"
        })

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_start_voice_control_clears_controller_after_failed_start(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.broadcast = AsyncMock()

        failed_voice_control = MagicMock()
        failed_voice_control.model = object()
        failed_voice_control.recognizer = None
        failed_voice_control.is_listening = False
        failed_voice_control.start_listening.side_effect = RuntimeError("mic failed")

        with patch("server.VoiceControlSherpa", return_value=failed_voice_control):
            await server.start_voice_control()

        assert server.voice_control is None
        server.broadcast.assert_awaited_with({
            "type": "voice_control_status",
            "active": False,
            "error": "mic failed"
        })

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_start_voice_control_keeps_active_instance_on_broadcast_failure(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        active_voice_control = MagicMock()
        active_voice_control.model = object()
        active_voice_control.recognizer = None
        active_voice_control.is_listening = True
        active_voice_control.start_listening.return_value = None
        server.broadcast = AsyncMock(side_effect=RuntimeError("send failed"))

        with patch("server.VoiceControlSherpa", return_value=active_voice_control):
            await server.start_voice_control()

        assert server.voice_control is active_voice_control

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_restore_snapshot_reports_when_snapshot_has_no_restorable_channels(self, mock_config, mock_bleed, tmp_path):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock()
        server.mixer_client = MagicMock()
        server.mixer_client.is_connected = True

        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps({
            "channels": {
                "1": {"error": "backup failed"},
                "2": {"error": "backup failed"},
            }
        }), encoding="utf-8")

        await server.restore_snapshot("ws", str(snapshot_path))

        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "restore_result",
            "success": False,
            "error": "Snapshot does not contain any restorable channels",
            "path": str(snapshot_path),
        })

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_rescan_channel_names_reports_not_available(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock()
        server.connected_clients.add("ws")

        await server.rescan_channel_names()

        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "voice_control_status",
            "active": False,
            "error": "Rescan only available for Wing mixer",
        })

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_rescan_channel_names_reports_success(self, mock_config, mock_bleed):
        WingClientStub = type("WingClientStub", (), {})
        server = AutoMixerServer()
        server.send_to_client = AsyncMock()
        server.connected_clients.add("ws")
        server.voice_control = MagicMock()
        server.voice_control.is_listening = True
        client = WingClientStub()
        client.send = MagicMock()
        client.get_all_channel_names = MagicMock(return_value={1: "Kick"})
        server.mixer_client = client

        with patch("server.WingClient", new=WingClientStub), \
             patch("server.EnhancedOSCClient", new=type("EnhancedOSCClientStub", (), {})):
            await server.rescan_channel_names()

        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "voice_control_status",
            "active": True,
            "message": "Channel names rescanned successfully",
        })


class TestAutoMixerServerCleanup:

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_stop_auto_soundcheck_continues_when_controller_stop_fails(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock()
        server.auto_soundcheck_running = True
        server.auto_soundcheck_task = asyncio.create_task(asyncio.sleep(10))

        server.gain_staging = MagicMock()
        server.gain_staging.stop.side_effect = RuntimeError("gain stop failed")
        server.phase_alignment_controller = MagicMock()
        server.multi_channel_auto_eq_controller = MagicMock()
        server.multi_channel_auto_eq_controller.stop_all.side_effect = RuntimeError("eq stop failed")
        server.auto_fader_controller = MagicMock()

        await server.stop_auto_soundcheck("ws")

        assert server.auto_soundcheck_running is False
        assert server.auto_soundcheck_observe_only is False
        assert server.auto_soundcheck_task is None
        server.gain_staging.stop.assert_called_once()
        server.phase_alignment_controller.stop.assert_called_once()
        server.multi_channel_auto_eq_controller.stop_all.assert_called_once()
        server.auto_fader_controller.stop.assert_called_once()
        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "auto_soundcheck_status",
            "is_running": False,
            "message": "Auto soundcheck stopped"
        })

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_cleanup_all_controllers_clears_references_even_when_stop_raises(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.gain_staging = MagicMock()
        server.gain_staging.stop.side_effect = RuntimeError("gain stop failed")
        server.multi_channel_auto_eq_controller = MagicMock()
        server.multi_channel_auto_eq_controller.stop_all.side_effect = RuntimeError("eq stop failed")
        server.phase_alignment_controller = MagicMock()
        server.phase_alignment_controller.stop.side_effect = RuntimeError("phase stop failed")
        server.auto_fader_controller = MagicMock()
        server.auto_fader_controller.stop.side_effect = RuntimeError("fader stop failed")

        server.cleanup_all_controllers()

        assert server.gain_staging is None
        assert server.multi_channel_auto_eq_controller is None
        assert server.phase_alignment_controller is None
        assert server.auto_fader_controller is None


class TestAutoMixerServerRuntimeErrors:

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_handle_client_message_reraises_cancelled_error(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock()
        cancelling_handler = AsyncMock(side_effect=asyncio.CancelledError())
        server._dispatch["cancel_me"] = cancelling_handler

        with pytest.raises(asyncio.CancelledError):
            await server.handle_client_message("ws", '{"type": "cancel_me"}')

        server.send_to_client.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_handle_client_message_ignores_closed_socket_when_reporting_error(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        failing_handler = AsyncMock(side_effect=RuntimeError("boom"))
        server._dispatch["explode"] = failing_handler
        server.send_to_client = AsyncMock(side_effect=Exception("connection closed"))

        await server.handle_client_message("ws", '{"type": "explode"}')

        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "error",
            "error": "boom",
        })

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_schedule_mixer_update_broadcast_tolerates_closed_loop(self, mock_config, mock_bleed):
        server = AutoMixerServer()

        class ClosedLoop:
            def call_soon_threadsafe(self, callback):
                raise RuntimeError("Event loop is closed")

        server._schedule_mixer_update_broadcast(ClosedLoop(), "/ch/1/fdr", (0.5,))

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_send_soundcheck_status_reraises_cancelled_error(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await server._send_soundcheck_status("ws", "reset", 10, 0, "Updating")

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_reset_trim_reraises_cancelled_error(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock(side_effect=asyncio.CancelledError())
        server.mixer_client = MagicMock()
        server.mixer_client.is_connected = True
        server.mixer_client.set_channel_gain.return_value = True

        with pytest.raises(asyncio.CancelledError):
            await server.reset_trim("ws", [1])

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_bypass_mixer_reraises_cancelled_error(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock(side_effect=asyncio.CancelledError())
        WingClientStub = type("WingClientStub", (), {})
        EnhancedOSCClientStub = type("EnhancedOSCClientStub", (), {})
        client = WingClientStub()
        client.is_connected = True
        client.set_channel_fader = MagicMock(return_value=True)
        client.set_eq_on = MagicMock(return_value=True)
        client.send = MagicMock(return_value=True)
        client.set_compressor_on = MagicMock(return_value=True)
        client.set_gate_on = MagicMock(return_value=True)
        client.set_low_cut = MagicMock(return_value=True)
        client.set_high_cut = MagicMock(return_value=True)
        server.mixer_client = client

        with patch("server.WingClient", new=WingClientStub), \
             patch("server.EnhancedOSCClient", new=EnhancedOSCClientStub):
            with pytest.raises(asyncio.CancelledError):
                await server.bypass_mixer("ws")

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_execute_voice_command_reraises_cancelled_error(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.mixer_client = MagicMock()
        server.broadcast = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await server._execute_voice_command({
                "type": "set_fader",
                "channel": 1,
                "value": 0.5,
            })

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_run_auto_soundcheck_cycle_reraises_cancelled_error_and_cleans_up(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.auto_soundcheck_running = True
        server.auto_soundcheck_observe_only = True
        server.auto_soundcheck_task = object()
        server.auto_soundcheck_websocket = "ws"
        server._send_soundcheck_status = AsyncMock(return_value=None)
        server.reset_all_functions_to_defaults = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await server._run_auto_soundcheck_cycle(
                "ws",
                "device-1",
                [1, 2],
                {},
                {},
                {"gain_staging": 1, "phase_alignment": 1, "auto_eq": 1, "auto_fader": 1},
                observe_only=False,
            )

        assert server.auto_soundcheck_running is False
        assert server.auto_soundcheck_observe_only is False
        assert server.auto_soundcheck_task is None
        assert server.auto_soundcheck_websocket is None

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_lock_auto_balance_channel_reports_invalid_channel(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.auto_fader_controller = MagicMock()
        server.send_to_client = AsyncMock()

        await server.lock_auto_balance_channel("ws", "bad")

        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "auto_fader_status",
            "status_type": "error",
            "error": "Invalid channel: bad"
        })

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_unlock_auto_balance_channel_reports_controller_error(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.auto_fader_controller = MagicMock()
        server.auto_fader_controller.unlock_channel.side_effect = RuntimeError("unlock failed")
        server.send_to_client = AsyncMock()

        await server.unlock_auto_balance_channel("ws", 7)

        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "auto_fader_status",
            "status_type": "error",
            "error": "unlock failed"
        })

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_set_auto_fader_profile_requires_controller(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock()

        await server.set_auto_fader_profile("ws", "rock")

        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "auto_fader_status",
            "status_type": "error",
            "error": "Auto Fader controller not initialized"
        })

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_update_auto_fader_settings_requires_controller(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock()

        await server.update_auto_fader_settings("ws", {"targetLufs": -16})

        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "auto_fader_status",
            "status_type": "error",
            "error": "Auto Fader controller not initialized"
        })

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.AutoFaderController")
    async def test_start_realtime_fader_clears_controller_when_realtime_start_fails(
        self, mock_auto_fader_cls, mock_config, mock_bleed
    ):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock()
        server.mixer_client = MagicMock()
        server.mixer_client.is_connected = True

        controller = MagicMock()
        controller.start.return_value = True
        controller.start_realtime_fader.return_value = False
        mock_auto_fader_cls.return_value = controller

        await server.start_realtime_fader("ws", device_id="1", channels=[1], channel_settings={}, channel_mapping={})

        controller.stop.assert_called()
        assert server.auto_fader_controller is None
        server.send_to_client.assert_awaited_with("ws", {
            "type": "auto_fader_status",
            "status_type": "error",
            "error": "Failed to start real-time fader"
        })

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.AutoFaderController")
    async def test_start_auto_balance_clears_controller_when_audio_capture_start_fails(
        self, mock_auto_fader_cls, mock_config, mock_bleed
    ):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock()
        server.mixer_client = MagicMock()
        server.mixer_client.is_connected = True

        controller = MagicMock()
        controller.start.return_value = False
        mock_auto_fader_cls.return_value = controller

        await server.start_auto_balance("ws", device_id="1", channels=[1], channel_settings={}, channel_mapping={})

        controller.stop.assert_called()
        assert server.auto_fader_controller is None
        server.send_to_client.assert_awaited_with("ws", {
            "type": "auto_fader_status",
            "status_type": "error",
            "active": False,
            "error": "Failed to start audio capture"
        })
