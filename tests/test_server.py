"""
Tests for backend/server.py — convert_numpy_types utility and
AutoMixerServer basic initialization.

All tests work without hardware or network (all external connections mocked).
"""

import asyncio
import json
import os
import time
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from handlers import automation_handlers, fx_handlers, mixer_handlers, routing_handlers, snapshot_handlers
from backend.operator_mode_policy import SUPPORTED_MANUAL_LIVE_WRITE_KINDS

try:
    from server import convert_numpy_types, AutoMixerServer
except ImportError:
    pytest.skip("server module not importable", allow_module_level=True)
try:
    from ai.agent import AgentMode
except ImportError:
    AgentMode = None


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


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_connect_wing_failed_attempt_disconnects_failed_client(mock_config, mock_bleed):
    class FailingEnhancedOSCClient:
        instances = []

        def __init__(self, ip, port):
            self.ip = ip
            self.port = port
            self.disconnected = False
            FailingEnhancedOSCClient.instances.append(self)

        def subscribe(self, address_pattern, callback):
            self.address_pattern = address_pattern

        def connect(self):
            return False

        def disconnect(self):
            self.disconnected = True

    server = AutoMixerServer()
    server.broadcast = AsyncMock()

    with patch("server.EnhancedOSCClient", new=FailingEnhancedOSCClient):
        await server.connect_wing("192.168.1.102", send_port=2222, receive_port=2223)

    assert FailingEnhancedOSCClient.instances[0].disconnected is True
    assert server.mixer_client is None
    assert server.connection_mode is None
    server.broadcast.assert_awaited_with({
        "type": "connection_status",
        "connected": False,
        "error": "Connection failed - Wing did not respond",
    })


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

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_operator_apply_idempotency_key_records_utc_first_seen(self, mock_config, mock_bleed):
        server = AutoMixerServer()

        first = server.reserve_operator_apply_idempotency_key(
            "apply-1",
            proposal_id="proposal-1",
            approval_id="approval-1",
        )
        duplicate = server.reserve_operator_apply_idempotency_key(
            "apply-1",
            proposal_id="proposal-2",
            approval_id="approval-2",
        )

        assert first == {
            "reserved": True,
            "reason": "reserved",
            "idempotency_key": "apply-1",
        }
        assert duplicate["reserved"] is False
        assert duplicate["reason"] == "duplicate_idempotency_key"
        assert duplicate["idempotency_key"] == "apply-1"
        assert duplicate["first_seen"]["proposal_id"] == "proposal-1"
        assert duplicate["first_seen"]["approval_id"] == "approval-1"
        assert duplicate["first_seen"]["created_at"].endswith("Z")
        assert "+00:00" not in duplicate["first_seen"]["created_at"]

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_manual_write_event_records_utc_timestamp(self, mock_config, mock_bleed, tmp_path):
        server = AutoMixerServer()
        server._manual_write_artifacts_dir = tmp_path

        artifact_path = server._record_manual_write_event(
            "manual_write_blocked",
            {
                "status": "blocked",
                "reason": "unit_test",
                "value": np.float64(-12.5),
            },
        )

        assert artifact_path == str(tmp_path / "manual_write_events.jsonl")
        event = json.loads((tmp_path / "manual_write_events.jsonl").read_text(encoding="utf-8"))
        assert event["event_type"] == "manual_write_blocked"
        assert event["timestamp"].endswith("Z")
        assert "+00:00" not in event["timestamp"]
        assert event["result"]["value"] == -12.5


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

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_coerce_agent_mode_normalizes_aliases(self, mock_config, mock_bleed):
        if AgentMode is None:
            pytest.skip("ai.agent module not available in this test environment")

        server = AutoMixerServer()
        checks = [
            ("analysis", AgentMode.SUGGEST),
            ("analysis_mode", AgentMode.SUGGEST),
            ("analysis mode", AgentMode.SUGGEST),
            ("analysis-mode", AgentMode.SUGGEST),
            ("assist", AgentMode.SUGGEST),
            ("assist mode", AgentMode.SUGGEST),
            ("assisted mode", AgentMode.SUGGEST),
            ("suggestion", AgentMode.SUGGEST),
            ("suggest mode", AgentMode.SUGGEST),
            ("supervised automix mode", AgentMode.SUGGEST),
            ("supervised_auto_mix", AgentMode.SUGGEST),
            ("full autonomous mode", AgentMode.AUTO),
            ("full autonomous mixing mode", AgentMode.AUTO),
            ("full_autonomous_mode", AgentMode.AUTO),
            ("auto mode", AgentMode.AUTO),
            ("manual_mode", AgentMode.MANUAL),
            ("full manual", AgentMode.MANUAL),
            ("manual mode", AgentMode.MANUAL),
            (None, AgentMode.AUTO),
        ]

        for raw, expected in checks:
            assert server._coerce_agent_mode(raw).value == expected.value

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

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_ready_for_live_blocks_on_manual_write_emergency_stop(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.mixer_client = MagicMock()
        server.mixer_client.is_connected = True
        server.gain_staging = MagicMock()
        server.auto_fader_controller = MagicMock()
        server.auto_fader_controller.get_status.return_value = {"collecting": True}
        server._manual_write_emergency_stopped = True

        status = server.get_ready_for_live_status()

        assert status["ready"] is False
        assert status["checks"]["manual_write_emergency_clear"] is False
        assert "manual_write_emergency_stop_active" in status["blocked_reasons"]

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
    async def test_execute_voice_command_mute_channel_applies_mute(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        client = MagicMock()
        client.set_channel_mute = MagicMock()
        client.is_connected = True
        client.get_channel_fader = MagicMock(return_value=0.0)
        server.mixer_client = client
        server.broadcast = AsyncMock()

        await server._execute_voice_command({
            "type": "mute_channel",
            "channel": 1,
            "muted": True,
        })

        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_executed"
        assert payload["command"] == "mute_channel"
        assert payload["channel"] == 1
        assert payload["muted"] is True
        client.set_channel_mute.assert_called_once_with(1, 1)

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_execute_voice_command_set_pan_applies_pan(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        client = MagicMock()
        client.set_channel_pan = MagicMock()
        client.is_connected = True
        server.mixer_client = client
        server.broadcast = AsyncMock()

        await server._execute_voice_command({
            "type": "set_pan",
            "channel": 1,
            "value": -25.0,
        })

        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_executed"
        assert payload["command"] == "set_pan"
        assert payload["channel"] == 1
        assert payload["value"] == -25.0
        client.set_channel_pan.assert_called_once_with(1, -25.0)

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


class TimestampedState(dict):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.owner.state_timestamps[key] = time.time()

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value


class DummyWingLike:
    def __init__(self):
        self.is_connected = True
        self.send = MagicMock()
        self.state_timestamps = {}
        self.state = TimestampedState(self)
        self.state.update({"/ch/1/fdr": -8.0, "/ch/1/in/set/trim": 1.0})
        self.set_channel_fader = MagicMock(return_value=True)
        self.set_channel_gain = MagicMock()
        self.set_eq_band = MagicMock()
        self.set_compressor = MagicMock()
        self.load_snap = MagicMock(return_value=True)
        self.save_snap = MagicMock(return_value=True)
        self.route_output = MagicMock(return_value=True)
        self.route_multiple_outputs = MagicMock(return_value=2)
        self.set_channel_input = MagicMock(return_value=True)
        self.set_channel_alt_input = MagicMock(return_value=True)
        self.set_insert = MagicMock(return_value=True)
        self.set_fx_model = MagicMock(return_value=True)
        self.set_fx_on = MagicMock(return_value=True)
        self.set_fx_mix = MagicMock(return_value=True)
        self.set_fx_parameter = MagicMock(return_value=True)
        self.get_channel_fader = MagicMock(return_value=-8.0)
        self.set_eq_on = MagicMock(return_value=True)
        self.get_eq_band_gain = MagicMock(return_value=0.0)
        self.set_eq_band_gain = MagicMock(return_value=True)
        self.set_compressor_on = MagicMock(return_value=True)
        self.set_compressor_threshold = MagicMock(return_value=True)
        self.get_compressor_gain = MagicMock(return_value=0.0)
        self.set_compressor_gain = MagicMock(return_value=True)
        self.set_channel_mute = MagicMock(return_value=True)
        self.set_channel_pan = MagicMock(return_value=True)
        self.set_channel_phase_invert = MagicMock(return_value=True)
        self.set_channel_delay = MagicMock(return_value=True)
        self.get_write_gate_status = MagicMock(return_value={"state": "unconfigured"})


class DummyAutoEQController:
    def __init__(self, mixer_client=None, bleed_service=None):
        self.mixer_client = mixer_client
        self.bleed_service = bleed_service
        self.start_calls = []

    def start(self, **kwargs):
        self.start_calls.append(kwargs)
        return True

    def apply_to_mixer(self):
        return True


@pytest.fixture
def dummy_wing_server():
    with patch("server.WingClient", new=DummyWingLike), \
         patch("server.EnhancedOSCClient", new=DummyWingLike), \
         patch("server.BleedService"), \
         patch.object(AutoMixerServer, "_load_config", return_value={}):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        server.connection_mode = "wing"
        server.mixer_client = DummyWingLike()
        yield server


@pytest.fixture
def minimal_server():
    with patch("server.BleedService"), patch.object(AutoMixerServer, "_load_config", return_value={}):
        yield AutoMixerServer(ws_host="localhost", ws_port=8765)


class TestWingDeploymentWriteBoundary:

    @staticmethod
    def _decode_last_message(websocket):
        assert websocket.send.await_count >= 1
        raw = websocket.send.await_args_list[-1].args[0]
        return json.loads(raw)

    def test_manual_write_supervision_reports_mode_limited_live_kinds(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"

        status = server.get_manual_write_supervision_status()
        assert status["approved_live_write_kinds"] == ["fader", "gain", "mute"]

        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        status = server.get_manual_write_supervision_status()
        assert status["approved_live_write_kinds"] == []

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_manual_write_supervision_matches_offline_unlocked_auto_scope(
        self,
        _mock_bleed: object,
        _mock_load_config: object,
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        server.connection_mode = "offline"
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server.mixer_client = DummyWingLike()

        status = server.get_manual_write_supervision_status()
        assert status["approved_live_write_kinds"] == ["fader", "gain"]

    def test_supported_manual_write_kinds_remain_in_sync(self):
        assert AutoMixerServer._get_supported_manual_write_kinds() == tuple(
            SUPPORTED_MANUAL_LIVE_WRITE_KINDS
        )

    def test_parse_manual_write_address_supports_supported_paths(self, minimal_server):
        assert minimal_server._parse_manual_write_address("/ch/12/fdr") == (12, "fader")
        assert minimal_server._parse_manual_write_address("/ch/12/in/set/trim") == (12, "gain")
        assert minimal_server._parse_manual_write_address("  /ch/12/mute  ") == (12, "mute")
        assert minimal_server._parse_manual_write_address("/ch/12/pan") == (12, "pan")
        assert minimal_server._parse_manual_write_address(b"/ch/13/fdr") == (13, "fader")
        assert minimal_server._parse_manual_write_address(Path("/ch/15/fdr")) == (15, "fader")
        class CustomPathLike(os.PathLike):
            def __fspath__(self):
                return "/ch/16/fdr"

        assert minimal_server._parse_manual_write_address(CustomPathLike()) == (16, "fader")

        assert minimal_server._parse_manual_write_address(
            bytearray(b"/ch/14/in/set/trim")
        ) == (14, "gain")

    def test_parse_manual_write_address_rejects_unknown_paths(self, minimal_server):
        assert minimal_server._parse_manual_write_address("/ch/12/eq") is None
        assert minimal_server._parse_manual_write_address("/bus/1/fdr") is None
        assert minimal_server._parse_manual_write_address("/ch/not-an-int/fdr") is None
        assert minimal_server._parse_manual_write_address("") is None
        assert minimal_server._parse_manual_write_address(None) is None
        assert minimal_server._parse_manual_write_address("/ch/12/fdr/") is None
        assert minimal_server._parse_manual_write_address(b"/ch/\xff/fdr") is None

    def test_handle_external_fader_update_tracks_fader_and_notifies_realtime_controller(self, minimal_server):
        controller = MagicMock()
        controller.realtime_enabled = True
        minimal_server.auto_fader_controller = controller

        minimal_server._handle_external_fader_update("/ch/5/fdr", (-9.75,))

        assert minimal_server.last_fader_values[5] == -9.75
        controller.handle_external_fader_change.assert_called_once_with(5, -9.75)

    def test_handle_external_fader_update_ignores_non_fader_write(self, minimal_server):
        controller = MagicMock()
        controller.realtime_enabled = True
        minimal_server.auto_fader_controller = controller

        minimal_server._handle_external_fader_update("/ch/5/in/set/trim", (3.0,))

        assert minimal_server.last_fader_values == {}
        controller.handle_external_fader_change.assert_not_called()

    def test_handle_external_gain_update_tracks_gain_and_notifies_gain_staging_controller(self, minimal_server):
        gain_controller = MagicMock()
        gain_controller.realtime_correction_enabled = True
        minimal_server.gain_staging = gain_controller

        minimal_server._handle_external_fader_update("/ch/5/in/set/trim", (1.75,))

        gain_controller.handle_external_trim_change.assert_called_once_with(5, 1.75)

    def test_handle_external_gain_update_clamps_extreme_gain_values(self, minimal_server):
        gain_controller = MagicMock()
        gain_controller.realtime_correction_enabled = True
        minimal_server.gain_staging = gain_controller

        minimal_server._handle_external_fader_update("/ch/6/in/set/trim", (100.0,))
        minimal_server._handle_external_fader_update("/ch/7/in/set/trim", (-200.0,))

        gain_controller.handle_external_trim_change.assert_any_call(6, 18.0)
        gain_controller.handle_external_trim_change.assert_any_call(7, -18.0)

    def test_handle_external_gain_update_ignores_non_finite_value(self, minimal_server):
        gain_controller = MagicMock()
        gain_controller.realtime_correction_enabled = True
        minimal_server.gain_staging = gain_controller

        minimal_server._handle_external_fader_update("/ch/8/in/set/trim", (float("nan"),))
        minimal_server._handle_external_fader_update("/ch/8/in/set/trim", (float("inf"),))

        assert not gain_controller.handle_external_trim_change.called

    def test_handle_external_fader_update_gracefully_ignores_invalid_value(self, minimal_server):
        minimal_server._handle_external_fader_update("/ch/5/fdr", ("abc",))

        assert minimal_server.last_fader_values == {}

    def test_handle_external_fader_update_without_realtime_controller_does_not_error(self, minimal_server):
        minimal_server.auto_fader_controller = None
        minimal_server._handle_external_fader_update("/ch/8/fdr", (-12.0,))

        assert minimal_server.last_fader_values[8] == -12.0

    def test_handle_external_fader_update_ignores_out_of_range_channel(self, minimal_server):
        controller = MagicMock()
        controller.realtime_enabled = True
        minimal_server.auto_fader_controller = controller

        minimal_server._handle_external_fader_update("/ch/99/fdr", (-10.0,))

        assert 99 not in minimal_server.last_fader_values
        controller.handle_external_fader_change.assert_not_called()

    def test_handle_external_fader_update_ignores_zero_channel(self, minimal_server):
        controller = MagicMock()
        controller.realtime_enabled = True
        minimal_server.auto_fader_controller = controller

        minimal_server._handle_external_fader_update("/ch/0/fdr", (-10.0,))

        assert 0 not in minimal_server.last_fader_values
        controller.handle_external_fader_change.assert_not_called()

    def test_handle_external_fader_update_with_disabled_realtime_controller_skips_controller_callback(self, minimal_server):
        controller = MagicMock()
        controller.realtime_enabled = False
        minimal_server.auto_fader_controller = controller

        minimal_server._handle_external_fader_update("/ch/8/fdr", (-12.0,))

        assert minimal_server.last_fader_values[8] == -12.0
        controller.handle_external_fader_change.assert_not_called()

    def test_handle_external_fader_update_ignores_unknown_address(self, minimal_server):
        controller = MagicMock()
        controller.realtime_enabled = True
        minimal_server.auto_fader_controller = controller

        minimal_server._handle_external_fader_update("/main/1/fdr", (-8.0,))

        assert minimal_server.last_fader_values == {}
        controller.handle_external_fader_change.assert_not_called()

    def test_handle_external_fader_update_without_args_does_not_fail(self, minimal_server):
        minimal_server.auto_fader_controller = MagicMock()
        minimal_server.auto_fader_controller.realtime_enabled = True

        minimal_server._handle_external_fader_update("/ch/8/fdr", ())

        assert minimal_server.last_fader_values == {}

    def test_handle_external_fader_update_survives_controller_exception(self, minimal_server):
        controller = MagicMock()
        controller.realtime_enabled = True
        controller.handle_external_fader_change.side_effect = RuntimeError("osc flood")
        minimal_server.auto_fader_controller = controller

        minimal_server._handle_external_fader_update("/ch/9/fdr", (-8.25,))

        assert minimal_server.last_fader_values[9] == -8.25
        controller.handle_external_fader_change.assert_called_once_with(9, -8.25)

    def test_handle_external_fader_update_uses_first_arg_if_multiple_provided(self, minimal_server):
        controller = MagicMock()
        controller.realtime_enabled = True
        minimal_server.auto_fader_controller = controller

        minimal_server._handle_external_fader_update("/ch/9/fdr", (1.0, 2.0, 3.0))

        assert minimal_server.last_fader_values[9] == 1.0
        controller.handle_external_fader_change.assert_called_once_with(9, 1.0)

    def test_handle_external_fader_update_ignores_non_finite_value(self, minimal_server):
        controller = MagicMock()
        controller.realtime_enabled = True
        minimal_server.auto_fader_controller = controller

        minimal_server._handle_external_fader_update("/ch/10/fdr", (float("nan"),))
        minimal_server._handle_external_fader_update("/ch/10/fdr", (float("inf"),))

        assert 10 not in minimal_server.last_fader_values
        controller.handle_external_fader_change.assert_not_called()

    def test_handle_external_mute_update_tracks_mute_state(self, minimal_server):
        minimal_server._handle_external_fader_update("/ch/12/mute", (1.0,))
        minimal_server._handle_external_fader_update("/ch/13/mute", (0,))
        minimal_server._handle_external_fader_update("/ch/14/mute", ("true",))

        assert minimal_server.last_mute_values[12] is True
        assert minimal_server.last_mute_values[13] is False
        assert minimal_server.last_mute_values[14] is True
        assert 12 not in minimal_server.last_fader_values

    def test_handle_external_mute_update_ignores_invalid_mute_payload(self, minimal_server):
        minimal_server._handle_external_fader_update("/ch/15/mute", (2,))
        minimal_server._handle_external_fader_update("/ch/16/mute", ("bad",))

        assert 15 not in minimal_server.last_mute_values
        assert 16 not in minimal_server.last_mute_values

    def test_handle_external_pan_update_tracks_and_clamps_value(self, minimal_server):
        minimal_server._handle_external_fader_update("/ch/12/pan", (125.0,))
        minimal_server._handle_external_fader_update("/ch/13/pan", (-170.0,))
        minimal_server._handle_external_fader_update("/ch/14/pan", (12.5,))

        assert minimal_server.last_pan_values[12] == 100.0
        assert minimal_server.last_pan_values[13] == -100.0
        assert minimal_server.last_pan_values[14] == 12.5
        assert 12 not in minimal_server.last_fader_values

    def test_handle_external_pan_update_ignores_non_finite_value(self, minimal_server):
        minimal_server._handle_external_fader_update("/ch/15/pan", (float("nan"),))
        minimal_server._handle_external_fader_update("/ch/15/pan", (float("inf"),))

        assert 15 not in minimal_server.last_pan_values

    def test_handle_external_fader_update_clamps_extreme_fader_value(self, minimal_server):
        controller = MagicMock()
        controller.realtime_enabled = True
        minimal_server.auto_fader_controller = controller

        minimal_server._handle_external_fader_update("/ch/10/fdr", (1000.0,))

        assert minimal_server.last_fader_values[10] == 10.0
        controller.handle_external_fader_change.assert_called_once_with(10, 10.0)

        minimal_server._handle_external_fader_update("/ch/11/fdr", (-200.0,))

        assert minimal_server.last_fader_values[11] == -144.0
        controller.handle_external_fader_change.assert_called_with(11, -144.0)

    @pytest.mark.asyncio
    async def test_reset_trim_is_quarantined_for_wing(self, dummy_wing_server):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()

        await server.reset_trim(websocket, [1, 2])

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "reset_trim_result"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == "reset_trim"
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_snapshot_load_is_quarantined_for_wing(self, dummy_wing_server):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = snapshot_handlers.register_handlers(server)

        await handlers["load_snap"](websocket, {"snap_name": "SHOW_A"})

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "load_snap_result"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == "load_snap"
        server.mixer_client.load_snap.assert_not_called()

    @pytest.mark.asyncio
    async def test_snapshot_save_is_quarantined_for_wing(self, dummy_wing_server):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = snapshot_handlers.register_handlers(server)

        await handlers["save_snap"](websocket, {"snap_name": "SHOW_A"})

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "save_snap_result"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == "save_snap"
        server.mixer_client.save_snap.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_auto_eq_forces_auto_apply_off_for_wing_boundary(self, dummy_wing_server):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()

        with patch("server.AutoEQController", DummyAutoEQController):
            await server.start_auto_eq(
                websocket,
                device_id="1",
                channel=3,
                profile="custom",
                auto_apply=True,
            )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "auto_eq_status"
        assert payload["active"] is True
        assert payload["auto_apply"] is False
        assert payload["auto_apply_forced_off"] is True
        assert server.auto_eq_controller.start_calls[-1]["auto_apply"] is False

    @pytest.mark.asyncio
    async def test_apply_eq_correction_is_quarantined_for_wing(self, dummy_wing_server):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()
        controller = MagicMock()
        server.auto_eq_controller = controller

        await server.apply_eq_correction(websocket)

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "auto_eq_apply_result"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == "apply_eq_correction"
        controller.apply_to_mixer.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_fader_requires_supervised_approval_for_wing(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_fader"](websocket, {"channel": 1, "value": -6.0})

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "approval_required"
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_wing_observation_only_blocks_even_approved_manual_write(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server._wing_observation_only = True
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_fader"](
            websocket,
            {"channel": 1, "value": -6.0, "approved": True, "approval_id": "unit-observe"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "wing_observation_only_active"
        assert payload["manual_write_supervision"]["wing_observation_only"] is True
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_fader_routes_approved_write_through_manual_gate(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_fader"](
            websocket,
            {"channel": 1, "value": -6.0, "approved": True, "approval_id": "unit-approval"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_applied"
        assert payload["status"] == "applied"
        assert payload["value_type"] == "fader"
        assert payload["old_value"] == -8.0
        assert payload["pre_read_status"] == "verified_from_cache"
        assert payload["pre_read_source"] == "mixer_state_cache"
        assert payload["readback_status"] == "readback_cache_not_fresh"
        assert payload["readback_value"] == -8.0
        assert payload["readback_cache_freshness"]["status"] == "not_after_write"
        assert payload["readback_cache_freshness"]["updated_after_minimum"] is False
        assert payload["proof_status"] == "partial"
        assert payload["rollback_available"] is True
        assert payload["rollback_value"] == -8.0
        assert payload["rollback_status"] == "not_attempted"
        server.mixer_client.set_channel_fader.assert_called_once_with(1, -6.0)

    @pytest.mark.asyncio
    async def test_websocket_mute_requires_supervised_approval_for_wing(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_mute"](websocket, {"channel": 1, "muted": True})

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "approval_required"
        assert payload["value_type"] == "mute"
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_mute_routes_approved_write_through_manual_gate(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server.mixer_client.state["/ch/1/mute"] = 0
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_mute"](
            websocket,
            {"channel": 1, "muted": True, "approved": True, "approval_id": "unit-mute-approval"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_applied"
        assert payload["status"] == "applied"
        assert payload["value_type"] == "mute"
        assert payload["old_value"] == 0
        assert payload["readback_status"] in {"not_verified", "mismatch_from_cache", "readback_cache_not_fresh"}
        assert payload["readback_value"] == 0
        assert payload["proof_status"] == "partial"
        assert payload["rollback_available"] is True
        assert payload["rollback_value"] == 0
        assert payload["rollback_status"] == "not_attempted"
        server.mixer_client.set_channel_mute.assert_called_once_with(1, True)

    @pytest.mark.asyncio
    async def test_websocket_mute_channel_alias_routes_to_manual_gate(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server.mixer_client.state["/ch/3/mute"] = 0
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["mute_channel"](
            websocket,
            {"channel": 3, "muted": True, "approved": True, "approval_id": "unit-mute-channel"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_applied"
        assert payload["status"] == "applied"
        assert payload["value_type"] == "mute"
        assert payload["requested_value"] is True
        assert payload["old_value"] == 0
        assert payload["proof_status"] == "partial"
        server.mixer_client.set_channel_mute.assert_called_once_with(3, True)

    @pytest.mark.asyncio
    async def test_websocket_mute_channel_alias_requires_supervised_approval_for_wing(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["mute_channel"](
            websocket,
            {"channel": 7, "muted": True},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "approval_required"
        assert payload["value_type"] == "mute"
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_unmute_channel_alias_defaults_to_false_and_routes_to_manual_gate(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server.mixer_client.state["/ch/4/mute"] = 1
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["unmute_channel"](
            websocket,
            {"channel": 4, "approved": True, "approval_id": "unit-unmute-channel"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_applied"
        assert payload["status"] == "applied"
        assert payload["value_type"] == "mute"
        assert payload["requested_value"] is False
        assert payload["old_value"] == 1
        assert payload["proof_status"] == "partial"
        server.mixer_client.set_channel_mute.assert_called_once_with(4, False)

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_auto_mode_unlocked_offline_allows_direct_fader_write(
        self,
        _mock_bleed: object,
        _mock_load_config: object,
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        server.connection_mode = "offline"
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server.mixer_client = DummyWingLike()

        result = await server._apply_manual_console_write(
            websocket=None,
            channel=1,
            value=-6.0,
            value_type="fader",
            source="unit:auto_unlocked_direct_write",
            approved=True,
            approval_id="unit-auto-unlocked",
        )

        assert result["status"] == "applied"
        assert result["value_type"] == "fader"
        assert result["requested_value"] == -6.0
        assert result["old_value"] == -8.0
        assert result["approval_id"] == "unit-auto-unlocked"
        assert result["source"] == "unit:auto_unlocked_direct_write"
        server.mixer_client.set_channel_fader.assert_called_once_with(1, -6.0)

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_auto_mode_offline_requires_approval_for_direct_manual_write(
        self,
        _mock_bleed: object,
        _mock_load_config: object,
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        server.connection_mode = "offline"
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server.mixer_client = DummyWingLike()

        result = await server._apply_manual_console_write(
            websocket=None,
            channel=1,
            value=-6.0,
            value_type="fader",
            source="unit:auto_unlocked_direct_write_unapproved",
            approved=False,
            approval_id=None,
        )

        assert result["status"] == "blocked"
        assert result["reason"] == "approval_required"
        assert result["allowed_live_write_kinds"] == ["fader", "gain"]
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_auto_mode_offline_blocks_mute_kind_even_when_approved(
        self,
        _mock_bleed: object,
        _mock_load_config: object,
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        server.connection_mode = "offline"
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server.mixer_client = DummyWingLike()

        result = await server._apply_manual_console_write(
            websocket=None,
            channel=1,
            value=True,
            value_type="mute",
            source="unit:auto_unlocked_direct_mute",
            approved=True,
            approval_id="unit-auto-unlocked-mute",
        )

        assert result["status"] == "blocked"
        assert result["reason"] == "live_write_kind_not_allowed_in_mode"
        assert result["allowed_live_write_kinds"] == ["fader", "gain"]
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_mode_is_blocked_for_wing_supervised_gate_when_wing_deployment_active(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_fader"](
            websocket,
            {"channel": 1, "value": -6.0, "approved": True, "approval_id": "unit-auto-wing"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "operator_mode_blocks_live_write"
        assert payload["approval_id"] == "unit-auto-wing"
        assert payload["manual_write_supervision"]["deployment_write_boundary_active"] is True
        assert payload["allowed_live_write_kinds"] == []
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_is_blocked_for_auto_mode_with_wing_boundary(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "title": "Lead Vocal: auto fader",
                "channel": 1,
                "value_type": "fader",
                "value": -6.0,
                "reason": "unit",
            },
        )
        create_payload = self._decode_last_message(websocket)
        assert create_payload["type"] == "operator_proposal_created"
        assert create_payload["status"] == "created"
        proposal_id = create_payload["proposal"]["id"]

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                "approved": True,
                "idempotency_key": "unit-auto-boundary",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["apply_result"]["reason"] == "operator_mode_blocks_live_write"
        assert (
            apply_payload["apply_result"]["operator_mode"]["capabilities"]["can_apply_to_console"]
            is False
        )
        assert (
            apply_payload["apply_result"]["operator_mode"]["capabilities"]["allowed_live_write_kinds"]
            == []
        )
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_is_allowed_for_auto_mode_when_wing_boundary_disabled(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.mixer_client.state_timestamps["/ch/1/fdr"] = time.time()
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock(
            return_value={
                "status": "applied",
                "channel": 1,
                "value_type": "fader",
                "effective_value": -9.0,
                "old_value": -8.0,
                "readback_value": -9.0,
                "readback_status": "verified_from_cache",
                "rollback_available": True,
                "rollback_value": -8.0,
            }
        )

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "title": "Lead Vocal: auto fader",
                "channel": 1,
                "value_type": "fader",
                "value": -9.0,
                "reason": "unit",
            },
        )
        create_payload = self._decode_last_message(websocket)
        assert create_payload["type"] == "operator_proposal_created"
        assert create_payload["status"] == "created"
        proposal_id = create_payload["proposal"]["id"]

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                "approved": True,
                "idempotency_key": "unit-auto-noboundary",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_result"
        assert apply_payload["status"] == "applied"
        assert apply_payload["apply_result"]["status"] == "applied"
        assert apply_payload["apply_result"]["channel"] == 1
        assert apply_payload["apply_result"]["value_type"] == "fader"
        server._apply_manual_console_write.assert_awaited_once_with(
            websocket,
            channel=1,
            value=-9.0,
            value_type="fader",
            source=f"operator_proposal:{proposal_id}",
            reason="unit",
            approved=True,
            approval_id=proposal_id,
        )

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_linked_idempotency_key_blocks_duplicate_apply(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.mixer_client.state.update({
            "/ch/1/fdr": -9.0,
            "/ch/2/fdr": -9.0,
        })
        now = time.time()
        server.mixer_client.state_timestamps.update({
            "/ch/1/fdr": now,
            "/ch/2/fdr": now,
        })
        server._apply_manual_console_write = AsyncMock(
            side_effect=[
                {
                    "status": "applied",
                    "success": True,
                    "channel": 1,
                    "value_type": "fader",
                    "effective_value": -9.5,
                    "old_value": -9.0,
                    "readback_value": -9.5,
                    "readback_status": "verified_from_cache",
                    "rollback_available": True,
                    "rollback_value": -8.0,
                },
                {
                    "status": "applied",
                    "success": True,
                    "channel": 2,
                    "value_type": "fader",
                    "effective_value": -9.5,
                    "old_value": -9.0,
                    "readback_value": -9.5,
                    "readback_status": "verified_from_cache",
                    "rollback_available": True,
                    "rollback_value": -9.0,
                },
            ]
        )
        websocket = MagicMock()
        websocket.send = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "title": "Lead Vocal: linked idempotency lock",
                "value_type": "fader",
                "requested_changes": [
                    {"value_type": "fader", "channel": 1, "value": -9.5},
                    {"value_type": "fader", "channel": 2, "value": -9.5},
                ],
                "reason": "unit",
            },
        )
        create_payload = self._decode_last_message(websocket)
        proposal_id = create_payload["proposal"]["id"]

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                "approved": True,
                "idempotency_key": "unit-auto-linked-idemp",
            },
        )
        first_apply_payload = self._decode_last_message(websocket)
        assert first_apply_payload["type"] == "operator_proposal_apply_result"
        assert first_apply_payload["status"] == "applied"
        assert server._apply_manual_console_write.await_count == 2

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                "approved": True,
                "idempotency_key": "unit-auto-linked-idemp",
            },
        )
        second_apply_payload = self._decode_last_message(websocket)
        assert second_apply_payload["type"] == "operator_proposal_apply_blocked"
        assert second_apply_payload["status"] == "blocked"
        assert second_apply_payload["reason"] == "duplicate_idempotency_key"
        assert server._apply_manual_console_write.await_count == 2

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_is_blocked_for_stale_pre_read_when_wing_boundary_disabled(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.mixer_client.state_timestamps["/ch/1/fdr"] = time.time() - 120.0
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "title": "Lead Vocal: auto fader stale cache",
                "channel": 1,
                "value_type": "fader",
                "value": -9.0,
                "reason": "unit",
            },
        )
        create_payload = self._decode_last_message(websocket)
        assert create_payload["type"] == "operator_proposal_created"
        assert create_payload["status"] == "created"
        proposal_id = create_payload["proposal"]["id"]

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                "approved": True,
                "idempotency_key": "unit-auto-stale",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["apply_result"]["reason"] == "pre_read_cache_not_fresh"
        assert apply_payload["apply_result"]["cache_freshness"]["status"] == "stale"
        assert apply_payload["apply_result"]["cache_freshness"]["source"] == "state_timestamps"
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_is_blocked_for_stale_gain_pre_read_when_wing_boundary_disabled(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.mixer_client.state_timestamps["/ch/1/in/set/trim"] = time.time() - 120.0
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "title": "Lead Vocal: auto trim",
                "channel": 1,
                "value_type": "gain",
                "value": -0.3,
                "reason": "unit",
            },
        )
        create_payload = self._decode_last_message(websocket)
        assert create_payload["type"] == "operator_proposal_created"
        assert create_payload["status"] == "created"
        proposal_id = create_payload["proposal"]["id"]

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                "approved": True,
                "idempotency_key": "unit-auto-stale-gain",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["apply_result"]["reason"] == "pre_read_cache_not_fresh"
        assert apply_payload["apply_result"]["cache_freshness"]["status"] == "stale"
        assert apply_payload["apply_result"]["cache_freshness"]["address"] == "/ch/1/in/set/trim"
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_gain.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_is_blocked_when_gain_pre_read_unavailable(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.mixer_client.state = {}
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "title": "Lead Vocal: auto trim read missing",
                "channel": 1,
                "value_type": "gain",
                "value": -0.3,
                "reason": "unit",
            },
        )
        create_payload = self._decode_last_message(websocket)
        assert create_payload["type"] == "operator_proposal_created"
        assert create_payload["status"] == "created"
        proposal_id = create_payload["proposal"]["id"]

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                "approved": True,
                "idempotency_key": "unit-auto-no-read-gain",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["apply_result"]["reason"] == "pre_read_unavailable"
        assert apply_payload["apply_result"]["pre_read_status"] == "readback_unavailable"
        assert apply_payload["apply_result"]["pre_read_source"] == "unavailable"
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_gain.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_is_blocked_for_unsupported_value_type(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "title": "Lead Vocal: unsupported eq proposal",
                "channel": 1,
                "value_type": "eq",
                "value": -2.5,
                "reason": "unit",
            },
        )
        create_payload = self._decode_last_message(websocket)
        assert create_payload["type"] == "operator_proposal_created"
        assert create_payload["status"] == "created"
        proposal_id = create_payload["proposal"]["id"]

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                "approved": True,
                "idempotency_key": "unit-auto-unsupported-type",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["apply_result"]["reason"] == "unsupported_value_type"
        assert (
            apply_payload["apply_result"].get("value_type") == "eq"
            or apply_payload["apply_result"].get("requested_change", {}).get("value_type") == "eq"
        )
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_linked_is_blocked_for_unsupported_value_type(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "title": "Lead Vocal: linked unsupported change",
                "value_type": "fader",
                "requested_changes": [
                    {"value_type": "eq", "channel": 1, "value": -2.5},
                    {"value_type": "fader", "channel": 1, "value": -9.0},
                ],
                "reason": "unit",
            },
        )
        create_payload = self._decode_last_message(websocket)
        assert create_payload["type"] == "operator_proposal_created"
        assert create_payload["status"] == "created"
        proposal_id = create_payload["proposal"]["id"]

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                "approved": True,
                "idempotency_key": "unit-auto-unsupported-linked",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["apply_result"]["reason"] == "unsupported_value_type"
        assert (
            apply_payload["apply_result"].get("value_type") == "eq"
            or apply_payload["apply_result"].get("requested_change", {}).get("value_type") == "eq"
        )
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_linked_is_blocked_when_later_change_pre_read_unavailable(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock()
        server.mixer_client.state = {"/ch/1/fdr": -8.0}

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "title": "Lead Vocal: linked readback missing",
                "value_type": "fader",
                "requested_changes": [
                    {"value_type": "fader", "channel": 1, "value": -8.5},
                    {"value_type": "gain", "channel": 2, "value": 0.5},
                ],
                "reason": "unit",
            },
        )
        create_payload = self._decode_last_message(websocket)
        assert create_payload["type"] == "operator_proposal_created"
        assert create_payload["status"] == "created"
        proposal_id = create_payload["proposal"]["id"]

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                "approved": True,
                "idempotency_key": "unit-auto-linked-missing-read",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["apply_result"]["reason"] == "linked_pre_read_unavailable"
        assert (
            apply_payload["apply_result"].get("value_type") == "gain"
            or apply_payload["apply_result"].get("requested_change", {}).get("value_type") == "gain"
        )
        assert apply_payload["apply_result"]["pre_read_status"] == "readback_unavailable"
        assert apply_payload["apply_result"]["pre_read_source"] == "unavailable"
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_linked_is_rolled_back_when_second_apply_fails(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server.mixer_client.state.update({
            "/ch/1/fdr": -8.0,
            "/ch/2/in/set/trim": 1.0,
        })
        now = time.time()
        server.mixer_client.state_timestamps.update({
            "/ch/1/fdr": now,
            "/ch/2/in/set/trim": now,
        })

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "title": "Lead Vocal: linked rollback failover",
                "value_type": "fader",
                "requested_changes": [
                    {"value_type": "fader", "channel": 1, "value": -9.0},
                    {"value_type": "gain", "channel": 2, "value": 1.0},
                ],
                "reason": "unit",
            },
        )
        create_payload = self._decode_last_message(websocket)
        assert create_payload["type"] == "operator_proposal_created"
        assert create_payload["status"] == "created"
        proposal_id = create_payload["proposal"]["id"]
        server._apply_manual_console_write = AsyncMock(
            side_effect=[
                {
                    "status": "applied",
                    "success": True,
                    "channel": 1,
                    "value_type": "fader",
                    "approval_id": f"{proposal_id}:1",
                    "effective_value": -9.0,
                    "old_value": -8.0,
                    "readback_value": -9.0,
                    "readback_status": "verified_from_cache",
                    "rollback_available": True,
                    "rollback_value": -8.0,
                },
                {
                    "status": "blocked",
                    "success": False,
                    "reason": "mixer_write_failed",
                    "channel": 2,
                    "value_type": "gain",
                    "approval_id": f"{proposal_id}:2",
                    "effective_value": 1.0,
                    "old_value": 1.0,
                    "rollback_available": False,
                },
                {
                    "status": "applied",
                    "success": True,
                    "channel": 1,
                    "value_type": "fader",
                    "effective_value": -8.0,
                    "old_value": -9.0,
                    "readback_value": -8.0,
                    "readback_status": "verified_from_cache",
                    "rollback_available": True,
                    "rollback_value": -9.0,
                    "rollback_of": f"{proposal_id}:1",
                    "rollback": True,
                },
            ]
        )

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                "approved": True,
                "idempotency_key": "unit-auto-linked-fail",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["apply_result"]["reason"] == "linked_apply_failed_rolled_back"
        assert apply_payload["apply_result"]["failed_result"]["reason"] == "mixer_write_failed"
        assert apply_payload["apply_result"]["failed_result"]["status"] == "blocked"
        assert any(
            step["rollback_of"] == f"{proposal_id}:1"
            for step in apply_payload["apply_result"]["rollback_results"]
        )
        assert server._apply_manual_console_write.await_count == 3
        first_call = server._apply_manual_console_write.await_args_list[0].kwargs
        second_call = server._apply_manual_console_write.await_args_list[1].kwargs
        third_call = server._apply_manual_console_write.await_args_list[2].kwargs
        assert first_call["source"] == f"operator_proposal_linked:{proposal_id}:1"
        assert second_call["source"] == f"operator_proposal_linked:{proposal_id}:2"
        assert first_call["approval_id"] == f"{proposal_id}:1"
        assert second_call["approval_id"] == f"{proposal_id}:2"
        assert third_call["source"] == f"operator_proposal_linked_rollback:{proposal_id}"
        assert third_call["approval_id"] == f"rollback:{proposal_id}:1"
        assert third_call["bypass_cooldown"] is True
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_fader_blocks_when_cache_pre_read_is_stale(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server.mixer_client.state_timestamps["/ch/1/fdr"] = time.time() - 120.0
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_fader"](
            websocket,
            {"channel": 1, "value": -6.0, "approved": True, "approval_id": "unit-stale-cache"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "pre_read_cache_not_fresh"
        assert payload["pre_read_status"] == "verified_from_cache"
        assert payload["pre_read_source"] == "mixer_state_cache"
        assert payload["cache_freshness"]["status"] == "stale"
        assert payload["cache_freshness"]["address"] == "/ch/1/fdr"
        assert payload["proof_status"] == "blocked"
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_fader_enforces_cooldown_for_same_wing_target(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server._manual_write_cooldown_sec = 10.0
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_fader"](
            websocket,
            {"channel": 1, "value": -6.0, "approved": True, "approval_id": "unit-approval-1"},
        )
        await handlers["set_fader"](
            websocket,
            {"channel": 1, "value": -5.0, "approved": True, "approval_id": "unit-approval-2"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "cooldown"
        assert payload["cooldown_sec"] == 10.0
        assert payload["cooldown_remaining_sec"] > 0
        server.mixer_client.set_channel_fader.assert_called_once_with(1, -6.0)

    @pytest.mark.asyncio
    async def test_websocket_gain_reports_cache_readback_and_rollback_metadata(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        def apply_gain(channel, value):
            server.mixer_client.state[f"/ch/{channel}/in/set/trim"] = value
            return True

        server.mixer_client.set_channel_gain.side_effect = apply_gain

        await handlers["set_gain"](
            websocket,
            {"channel": 1, "value": 2.0, "approved": True, "approval_id": "unit-gain"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_applied"
        assert payload["status"] == "applied"
        assert payload["value_type"] == "gain"
        assert payload["old_value"] == 1.0
        assert payload["pre_read_value"] == 1.0
        assert payload["pre_read_status"] == "verified_from_cache"
        assert payload["pre_read_source"] == "mixer_state_cache"
        assert payload["readback_value"] == 2.0
        assert payload["readback_status"] == "verified_from_cache"
        assert payload["readback_source"] == "mixer_state_cache"
        assert payload["proof_status"] == "verified"
        assert payload["rollback_available"] is True
        assert payload["rollback_value"] == 1.0
        assert payload["rollback_address"] == "/ch/1/in/set/trim"
        assert payload["rollback_status"] == "not_attempted"
        server.mixer_client.set_channel_gain.assert_called_once_with(1, 2.0)

    @pytest.mark.asyncio
    async def test_websocket_gain_demotes_cache_readback_not_updated_after_write(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server.mixer_client.state = {"/ch/1/in/set/trim": 1.0}
        server.mixer_client.state_timestamps = {"/ch/1/in/set/trim": time.time()}
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        def apply_gain_without_timestamp(channel, value):
            server.mixer_client.state[f"/ch/{channel}/in/set/trim"] = value
            return True

        server.mixer_client.set_channel_gain.side_effect = apply_gain_without_timestamp

        await handlers["set_gain"](
            websocket,
            {"channel": 1, "value": 2.0, "approved": True, "approval_id": "unit-gain-cache-not-updated"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_applied"
        assert payload["status"] == "applied"
        assert payload["readback_value"] == 2.0
        assert payload["readback_status"] == "readback_cache_not_fresh"
        assert payload["readback_cache_freshness"]["status"] == "not_after_write"
        assert payload["readback_cache_freshness"]["updated_after_minimum"] is False
        assert payload["proof_status"] == "partial"
        assert payload["rollback_available"] is True
        server.mixer_client.set_channel_gain.assert_called_once_with(1, 2.0)

    @pytest.mark.asyncio
    async def test_websocket_gain_uses_physical_readback_when_cache_missing(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server.mixer_client.state = {}
        physical = {"/ch/1/in/set/trim": 1.0}

        def query_parameter(address):
            return {"status": "received", "address": address, "value": physical.get(address)}

        def apply_gain(channel, value):
            physical[f"/ch/{channel}/in/set/trim"] = value
            return True

        server.mixer_client.query_parameter = MagicMock(side_effect=query_parameter)
        server.mixer_client.set_channel_gain.side_effect = apply_gain
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_gain"](
            websocket,
            {"channel": 1, "value": 2.0, "approved": True, "approval_id": "unit-gain-physical"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_applied"
        assert payload["status"] == "applied"
        assert payload["old_value"] == 1.0
        assert payload["pre_read_status"] == "verified_from_physical_readback"
        assert payload["pre_read_source"] == "physical_readback"
        assert payload["readback_value"] == 2.0
        assert payload["readback_status"] == "verified_from_physical_readback"
        assert payload["readback_source"] == "physical_readback"
        assert payload["proof_status"] == "verified"
        assert payload["rollback_available"] is True
        assert server.mixer_client.query_parameter.call_count == 2
        server.mixer_client.set_channel_gain.assert_called_once_with(1, 2.0)

    @pytest.mark.asyncio
    async def test_websocket_gain_accepts_wing_trim_quantized_physical_readback(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server.mixer_client.state = {}
        physical = {"/ch/4/in/set/trim": 0.0}

        def query_parameter(address):
            return {"status": "received", "address": address, "value": physical.get(address)}

        def apply_gain(channel, value):
            physical[f"/ch/{channel}/in/set/trim"] = -0.1999988555908203
            return True

        server.mixer_client.query_parameter = MagicMock(side_effect=query_parameter)
        server.mixer_client.set_channel_gain.side_effect = apply_gain
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_gain"](
            websocket,
            {"channel": 4, "value": -0.25, "approved": True, "approval_id": "unit-gain-quantized"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_applied"
        assert payload["status"] == "applied"
        assert payload["old_value"] == 0.0
        assert payload["readback_value"] == -0.1999988555908203
        assert payload["readback_status"] == "verified_from_physical_readback"
        assert payload["proof_status"] == "verified"
        assert payload["rollback_available"] is True
        server.mixer_client.set_channel_gain.assert_called_once_with(4, -0.25)

    @pytest.mark.asyncio
    async def test_query_mixer_parameter_reads_gain_without_writing(self, dummy_wing_server):
        server = dummy_wing_server
        server.mixer_client.state = {}
        server.mixer_client.query_parameter = MagicMock(
            return_value={"status": "received", "address": "/ch/3/in/set/trim", "value": 1.25}
        )
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["query_mixer_parameter"](
            websocket,
            {
                "request_id": "wing-readonly-preflight:ch3:gain",
                "channel": 3,
                "value_type": "gain",
                "address": "/ch/3/in/set/trim",
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "mixer_parameter_value"
        assert payload["request_id"] == "wing-readonly-preflight:ch3:gain"
        assert payload["status"] == "verified_from_physical_readback"
        assert payload["source"] == "physical_readback"
        assert payload["channel"] == 3
        assert payload["value_type"] == "gain"
        assert payload["address"] == "/ch/3/in/set/trim"
        assert payload["value"] == 1.25
        server.mixer_client.query_parameter.assert_called_once_with("/ch/3/in/set/trim")
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_mixer_parameter_reads_mute_without_writing(self, dummy_wing_server):
        server = dummy_wing_server
        server.mixer_client.state = {}
        server.mixer_client.query_parameter = MagicMock(
            return_value={"status": "received", "address": "/ch/3/mute", "value": 1}
        )
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["query_mixer_parameter"](
            websocket,
            {
                "request_id": "wing-readonly-preflight:ch3:mute",
                "channel": 3,
                "value_type": "mute",
                "address": "/ch/3/mute",
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "mixer_parameter_value"
        assert payload["request_id"] == "wing-readonly-preflight:ch3:mute"
        assert payload["status"] == "verified_from_physical_readback"
        assert payload["source"] == "physical_readback"
        assert payload["channel"] == 3
        assert payload["value_type"] == "mute"
        assert payload["address"] == "/ch/3/mute"
        assert payload["value"] == 1
        server.mixer_client.query_parameter.assert_called_once_with("/ch/3/mute")
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_mixer_parameter_blocks_unsupported_address(self, dummy_wing_server):
        server = dummy_wing_server
        server.mixer_client.query_parameter = MagicMock(return_value={"value": -1.0})
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["query_mixer_parameter"](
            websocket,
            {
                "request_id": "wing-readonly-preflight:main:fader",
                "address": "/main/1/fdr",
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "mixer_parameter_read_blocked"
        assert payload["request_id"] == "wing-readonly-preflight:main:fader"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "unsupported_address"
        server.mixer_client.query_parameter.assert_not_called()
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_gain_blocks_when_pre_read_unavailable(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server.mixer_client.state = {}
        server.mixer_client.query_parameter = MagicMock(
            return_value={"status": "timeout", "address": "/ch/1/in/set/trim", "value": None}
        )
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_gain"](
            websocket,
            {"channel": 1, "value": 2.0, "approved": True, "approval_id": "unit-gain-missing-read"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "pre_read_unavailable"
        assert payload["pre_read_status"] == "readback_unavailable"
        assert payload["pre_read_source"] == "physical_readback"
        assert payload["proof_status"] == "blocked"
        server.mixer_client.set_channel_gain.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_fader_blocks_outside_supervised_mode_for_wing(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "assist"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_fader"](
            websocket,
            {"channel": 1, "value": -6.0, "approved": True, "approval_id": "unit-approval"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_blocked"
        assert payload["reason"] == "operator_mode_blocks_live_write"
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_eq_is_quarantined_for_wing_boundary(self, dummy_wing_server):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_eq"](
            websocket,
            {"channel": 1, "band": 1, "freq": 1000.0, "gain": 2.0, "q": 1.2},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "set_eq_result"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == "set_eq"
        server.mixer_client.set_eq_band.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_compressor_is_quarantined_for_wing_boundary(self, dummy_wing_server):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_compressor"](
            websocket,
            {"channel": 1, "params": {"threshold": -18.0, "ratio": "3:1"}},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "set_compressor_result"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == "set_compressor"
        server.mixer_client.set_compressor.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("message_type", "payload", "result_type", "surface", "method_name"),
        [
            (
                "route_output",
                {
                    "output_group": "aux",
                    "output_number": 1,
                    "source_group": "channel",
                    "source_channel": 1,
                },
                "route_output_result",
                "route_output",
                "route_output",
            ),
            (
                "route_multiple_outputs",
                {
                    "output_group": "aux",
                    "start_output": 1,
                    "num_outputs": 2,
                    "source_group": "channel",
                    "start_source_channel": 1,
                },
                "route_multiple_outputs_result",
                "route_multiple_outputs",
                "route_multiple_outputs",
            ),
            (
                "set_channel_input",
                {"channel": 1, "source_group": "local", "source_channel": 1},
                "set_channel_input_result",
                "set_channel_input",
                "set_channel_input",
            ),
            (
                "set_channel_alt_input",
                {"channel": 1, "source_group": "usb", "source_channel": 1},
                "set_channel_alt_input_result",
                "set_channel_alt_input",
                "set_channel_alt_input",
            ),
        ],
    )
    async def test_routing_write_surfaces_are_quarantined_for_wing_boundary(
        self,
        dummy_wing_server,
        message_type,
        payload,
        result_type,
        surface,
        method_name,
    ):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = routing_handlers.register_handlers(server)

        await handlers[message_type](websocket, payload)

        result = self._decode_last_message(websocket)
        assert result["type"] == result_type
        assert result["status"] == "blocked"
        assert result["reason"] == "deployment_write_surface_quarantined"
        assert result["surface"] == surface
        getattr(server.mixer_client, method_name).assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("message_type", "payload", "result_type", "surface", "method_name"),
        [
            (
                "set_insert",
                {"target_type": "channel", "target": 1, "position": "pre", "slot": "FX1", "on": 1},
                "insert_updated",
                "set_insert",
                "set_insert",
            ),
            (
                "set_fx_model",
                {"slot": "FX1", "model": "PLATE"},
                "fx_slot_updated",
                "set_fx_model",
                "set_fx_model",
            ),
            (
                "set_fx_on",
                {"slot": "FX1", "on": 1},
                "fx_slot_updated",
                "set_fx_on",
                "set_fx_on",
            ),
            (
                "set_fx_mix",
                {"slot": "FX1", "mix": 50},
                "fx_slot_updated",
                "set_fx_mix",
                "set_fx_mix",
            ),
            (
                "set_fx_parameter",
                {"slot": "FX1", "parameter": 1, "value": 0.5},
                "fx_parameter_updated",
                "set_fx_parameter",
                "set_fx_parameter",
            ),
        ],
    )
    async def test_fx_write_surfaces_are_quarantined_for_wing_boundary(
        self,
        dummy_wing_server,
        message_type,
        payload,
        result_type,
        surface,
        method_name,
    ):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = fx_handlers.register_handlers(server)

        await handlers[message_type](websocket, payload)

        result = self._decode_last_message(websocket)
        assert result["type"] == result_type
        assert result["status"] == "blocked"
        assert result["reason"] == "deployment_write_surface_quarantined"
        assert result["surface"] == surface
        getattr(server.mixer_client, method_name).assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("surface", "result_type"),
        [
            ("bypass_mixer", "bypass_result"),
            ("reset_eq", "auto_eq_reset_result"),
            ("reset_all_eq", "reset_all_eq_result"),
            ("apply_channel_correction", "multi_channel_apply_result"),
            ("apply_all_corrections", "multi_channel_apply_result"),
            ("apply_phase_corrections", "phase_alignment_apply_result"),
            ("reset_phase_delay", "phase_alignment_reset_result"),
            ("restore_snapshot", "restore_result"),
            ("start_realtime_fader", "auto_fader_status"),
            ("apply_auto_balance", "auto_fader_status"),
            ("start_auto_soundcheck", "auto_soundcheck_status"),
            ("start_auto_compressor", "auto_compressor_status"),
            ("start_auto_compressor_soundcheck", "auto_compressor_status"),
            ("start_auto_compressor_live", "auto_compressor_status"),
            ("set_auto_compressor_profile", "auto_compressor_status"),
            ("set_auto_compressor_manual", "auto_compressor_status"),
        ],
    )
    async def test_server_write_surfaces_are_quarantined_for_wing_boundary(
        self,
        dummy_wing_server,
        surface,
        result_type,
    ):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()

        if surface == "bypass_mixer":
            await server.bypass_mixer(websocket)
        elif surface == "reset_eq":
            await server.reset_eq(websocket, {"channel": 1})
        elif surface == "reset_all_eq":
            await server.reset_all_eq(websocket, {"channels": [1]})
        elif surface == "apply_channel_correction":
            await server.apply_channel_correction(websocket, 1)
        elif surface == "apply_all_corrections":
            await server.apply_all_corrections(websocket)
        elif surface == "apply_phase_corrections":
            await server.apply_phase_corrections(websocket, {1: {"delay_ms": 1.0}})
        elif surface == "reset_phase_delay":
            await server.reset_phase_delay(websocket, [1])
        elif surface == "restore_snapshot":
            await server.restore_snapshot(websocket, "presets/unit_snapshot.json")
        elif surface == "start_realtime_fader":
            await server.start_realtime_fader(websocket, device_id="1", channels=[1])
        elif surface == "apply_auto_balance":
            await server.apply_auto_balance(websocket)
        elif surface == "start_auto_soundcheck":
            await server.start_auto_soundcheck(websocket, device_id="1", channels=[1])
        elif surface == "start_auto_compressor":
            await server.start_auto_compressor(websocket, "1", [1], {1: 1})
        elif surface == "start_auto_compressor_soundcheck":
            await server.start_auto_compressor_soundcheck(websocket)
        elif surface == "start_auto_compressor_live":
            await server.start_auto_compressor_live(websocket)
        elif surface == "set_auto_compressor_profile":
            await server.set_auto_compressor_profile(websocket, 1, "base")
        elif surface == "set_auto_compressor_manual":
            await server.set_auto_compressor_manual(websocket, 1, {"threshold": -18.0})
        else:
            raise AssertionError(f"Unhandled surface {surface}")

        payload = self._decode_last_message(websocket)
        assert payload["type"] == result_type
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == surface
        server.mixer_client.send.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_eq_band.assert_not_called()
        server.mixer_client.set_compressor.assert_not_called()
        server.mixer_client.set_channel_phase_invert.assert_not_called()
        server.mixer_client.set_channel_delay.assert_not_called()

    def test_auto_compressor_reset_helper_is_fail_closed_evidence(self, dummy_wing_server):
        server = dummy_wing_server

        result = server._reset_compressor_on_channel(7)

        assert result["status"] == "blocked"
        assert result["reason"] == "auto_compressor_reset_direct_apply_quarantined"
        assert result["surface"] == "auto_compressor_reset"
        assert result["write_policy"] == "proposal_only_quarantined"
        assert result["live_write_allowed"] is False
        assert result["mutates_mixer"] is False
        assert {
            "cmd": "set_compressor_on",
            "channel": 7,
            "on": 0,
            "semantic": "reset_compressor_disable",
        } in result["blocked_calls"]
        assert {
            "cmd": "set_compressor",
            "channel": 7,
            "threshold": -10.0,
            "ratio": "3.0",
            "attack": 10.0,
            "release": 100.0,
            "gain": 0.0,
            "mix": 100.0,
            "knee": 0,
            "semantic": "reset_compressor_defaults",
        } in result["blocked_calls"]
        server.mixer_client.set_compressor_on.assert_not_called()
        server.mixer_client.set_compressor.assert_not_called()

    @pytest.mark.asyncio
    async def test_restore_snapshot_quarantine_blocks_restore_helper_even_with_valid_snapshot(
        self,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()
        snapshot_path = tmp_path / "unit_snapshot.json"
        snapshot_path.write_text(
            json.dumps({"channels": {"1": {"channel": 1, "fader": -9.0}}}),
            encoding="utf-8",
        )

        with patch(
            "server.restore_from_backup_using_client",
            side_effect=AssertionError("restore helper must stay behind quarantine"),
        ) as restore_helper:
            await server.restore_snapshot(websocket, str(snapshot_path))

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "restore_result"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == "restore_snapshot"
        restore_helper.assert_not_called()

    @pytest.mark.asyncio
    async def test_realtime_gain_correction_is_quarantined_for_wing_boundary(self, dummy_wing_server):
        server = dummy_wing_server
        server.broadcast = AsyncMock()

        await server.start_realtime_correction(device_id="1", channels=[1], channel_mapping={1: 1})

        server.broadcast.assert_awaited_once()
        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "gain_staging_status"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == "start_realtime_correction"
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.send.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("command", "method_name"),
        [
            ({"type": "set_fader", "channel": 1, "value": -6.0}, "set_channel_fader"),
            ({"type": "set_gain", "channel": 1, "value": 2.0}, "set_channel_gain"),
            ({"type": "load_snap", "snap_name": "SHOW_A"}, "load_snap"),
            ({"type": "mute_channel", "channel": 1, "muted": True}, "set_channel_mute"),
            ({"type": "volume_up", "channel": 1, "db": 1}, "set_channel_fader"),
            ({"type": "volume_down", "channel": 1, "db": 1}, "set_channel_fader"),
            ({"type": "eq_on", "channel": 1, "on": 1}, "set_eq_on"),
            ({"type": "eq_band_up", "channel": 1, "band": "2", "db": 1}, "set_eq_band_gain"),
            ({"type": "eq_band_down", "channel": 1, "band": "2", "db": 1}, "set_eq_band_gain"),
            ({"type": "compressor_on", "channel": 1, "on": 1}, "set_compressor_on"),
            ({"type": "compressor_threshold", "channel": 1, "threshold": -18.0}, "set_compressor_threshold"),
            ({"type": "compressor_gain_up", "channel": 1, "db": 1}, "set_compressor_gain"),
            ({"type": "compressor_gain_down", "channel": 1, "db": 1}, "set_compressor_gain"),
            ({"type": "set_pan", "channel": 1, "value": -25.0}, "set_channel_pan"),
        ],
    )
    async def test_voice_write_commands_are_quarantined_for_wing_boundary(
        self,
        dummy_wing_server,
        tmp_path,
        command,
        method_name,
    ):
        server = dummy_wing_server
        server._manual_write_artifacts_dir = tmp_path
        server.broadcast = AsyncMock()

        await server._execute_voice_command(command)

        server.broadcast.assert_awaited_once()
        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == f"voice_command:{command['type']}"
        assert payload["manual_write_supervision"]["deployment_write_boundary_active"] is True
        assert payload["artifact_path"].startswith(str(tmp_path))
        getattr(server.mixer_client, method_name).assert_not_called()

    @pytest.mark.asyncio
    async def test_emergency_stop_blocks_later_manual_writes(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        automation = automation_handlers.register_handlers(server)
        mixer = mixer_handlers.register_handlers(server)

        await automation["emergency_stop"](websocket, {})
        await mixer["set_gain"](
            websocket,
            {"channel": 1, "value": 2.0, "approved": True, "approval_id": "unit-approval"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_blocked"
        assert payload["reason"] == "emergency_stop_active"
        server.mixer_client.set_channel_gain.assert_not_called()

    @pytest.mark.asyncio
    async def test_emergency_reset_requires_explicit_operator_approval(self, dummy_wing_server, tmp_path):
        server = dummy_wing_server
        server._manual_write_artifacts_dir = tmp_path
        websocket = MagicMock()
        websocket.send = AsyncMock()
        automation = automation_handlers.register_handlers(server)

        await automation["emergency_stop"](websocket, {})
        await automation["reset_manual_write_emergency_stop"](websocket, {"reason": "unit_test"})

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_emergency_reset_result"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "operator_approval_required"
        assert payload["manual_write_supervision"]["emergency_stopped"] is True
        assert payload["manual_write_supervision"]["reset_required"] is True
        assert server._manual_write_emergency_stopped is True
        assert (tmp_path / "manual_write_events.jsonl").exists()

    @pytest.mark.asyncio
    async def test_emergency_reset_clears_latch_but_keeps_live_not_ready_until_freeze_review(
        self,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        server._manual_write_artifacts_dir = tmp_path
        server.gain_staging = MagicMock()
        server.auto_fader_controller = MagicMock()
        server.auto_fader_controller.get_status.return_value = {"collecting": True}
        server.auto_fader_controller.get_freeze_status.return_value = {"automation_frozen": True}
        websocket = MagicMock()
        websocket.send = AsyncMock()
        automation = automation_handlers.register_handlers(server)

        await automation["emergency_stop"](websocket, {})
        await automation["reset_manual_write_emergency_stop"](
            websocket,
            {
                "approved": True,
                "approval_id": "operator-reset-unit",
                "reason": "operator_verified_safe_state",
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_emergency_reset_result"
        assert payload["status"] == "reset"
        assert payload["success"] is True
        assert payload["manual_write_supervision"]["emergency_stopped"] is False
        assert payload["manual_write_supervision"]["reset_required"] is False
        assert payload["automation_remains_frozen"] is True
        assert payload["ready_for_live"]["ready"] is False
        assert payload["ready_for_live"]["checks"]["manual_write_emergency_clear"] is True
        assert payload["ready_for_live"]["checks"]["automation_unfrozen"] is False
        assert "automation_frozen_after_emergency_stop" in payload["ready_for_live"]["blocked_reasons"]
        assert server._manual_write_emergency_stopped is False
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_last_manual_write_requires_previous_apply(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["rollback_last_manual_write"](
            websocket,
            {"approved": True, "approval_id": "rollback-unit"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_rollback_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "no_rollback_available"

    @pytest.mark.asyncio
    async def test_rollback_last_manual_write_requires_approval(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server._manual_write_cooldown_sec = 0.0
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        def apply_fader(channel, value):
            server.mixer_client.state[f"/ch/{channel}/fdr"] = value
            return True

        server.mixer_client.set_channel_fader.side_effect = apply_fader
        await handlers["set_fader"](
            websocket,
            {"channel": 1, "value": -6.0, "approved": True, "approval_id": "unit-apply"},
        )
        await handlers["rollback_last_manual_write"](websocket, {"approval_id": "rollback-unit"})

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_rollback_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "approval_required"
        assert payload["rollback"] is True
        assert server.mixer_client.set_channel_fader.call_count == 1

    @pytest.mark.asyncio
    async def test_rollback_last_manual_write_blocks_when_pre_read_times_out(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server._manual_write_cooldown_sec = 0.0
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_fader"](
            websocket,
            {"channel": 1, "value": -6.0, "approved": True, "approval_id": "unit-apply"},
        )

        server.mixer_client.state = {}
        server.mixer_client.query_parameter = MagicMock(
            return_value={"status": "timeout", "address": "/ch/1/fdr", "value": None}
        )

        await handlers["rollback_last_manual_write"](
            websocket,
            {"approved": True, "approval_id": "rollback-timeout"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_rollback_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "pre_read_unavailable"
        assert payload["rollback"] is True
        assert payload["rollback_of"] == "unit-apply"
        assert payload["pre_read_status"] == "readback_unavailable"
        assert payload["pre_read_source"] == "physical_readback"
        assert payload["proof_status"] == "blocked"
        assert server.mixer_client.set_channel_fader.call_count == 1

    @pytest.mark.asyncio
    async def test_rollback_last_manual_write_applies_previous_value_through_gate(self, dummy_wing_server, tmp_path):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server._manual_write_cooldown_sec = 0.0
        server._manual_write_artifacts_dir = tmp_path / "manual_write_gate"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        def apply_fader(channel, value):
            server.mixer_client.state[f"/ch/{channel}/fdr"] = value
            return True

        server.mixer_client.set_channel_fader.side_effect = apply_fader

        await handlers["set_fader"](
            websocket,
            {"channel": 1, "value": -6.0, "approved": True, "approval_id": "unit-apply"},
        )
        await handlers["rollback_last_manual_write"](
            websocket,
            {"approved": True, "approval_id": "rollback-unit"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_rollback_applied"
        assert payload["status"] == "applied"
        assert payload["rollback"] is True
        assert payload["rollback_of"] == "unit-apply"
        assert payload["rollback_status"] == "applied"
        assert payload["requested_value"] == -8.0
        assert payload["effective_value"] == -8.0
        assert payload["old_value"] == -6.0
        assert payload["readback_value"] == -8.0
        assert payload["readback_status"] == "verified_from_cache"
        assert payload["proof_status"] == "verified"
        assert payload["rollback_readback_status"] == "verified_from_cache"
        assert payload["rollback_proof_status"] == "verified"
        assert payload["artifact_path"].endswith("manual_write_events.jsonl")
        assert [item.args for item in server.mixer_client.set_channel_fader.call_args_list] == [
            (1, -6.0),
            (1, -8.0),
        ]

        with open(payload["artifact_path"], encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle]
        assert [event["event_type"] for event in events] == [
            "manual_write_applied",
            "manual_write_rollback_applied",
        ]
        assert events[-1]["result"]["rollback_of"] == "unit-apply"

    @pytest.mark.asyncio
    async def test_rollback_last_manual_write_bypasses_write_cooldown(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server._manual_write_cooldown_sec = 10.0
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        def apply_fader(channel, value):
            server.mixer_client.state[f"/ch/{channel}/fdr"] = value
            return True

        server.mixer_client.set_channel_fader.side_effect = apply_fader

        await handlers["set_fader"](
            websocket,
            {"channel": 1, "value": -6.0, "approved": True, "approval_id": "unit-apply"},
        )
        await handlers["rollback_last_manual_write"](
            websocket,
            {"approved": True, "approval_id": "rollback-unit"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_rollback_applied"
        assert payload["status"] == "applied"
        assert payload["rollback_status"] == "applied"
        assert [item.args for item in server.mixer_client.set_channel_fader.call_args_list] == [
            (1, -6.0),
            (1, -8.0),
        ]

    @pytest.mark.asyncio
    async def test_rollback_last_manual_write_marks_partial_when_readback_not_verified(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server._manual_write_cooldown_sec = 0.0
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        first_apply = {"done": False}

        def apply_then_leave_stale(channel, value):
            if not first_apply["done"]:
                server.mixer_client.state[f"/ch/{channel}/fdr"] = value
                first_apply["done"] = True
            return True

        server.mixer_client.set_channel_fader.side_effect = apply_then_leave_stale

        await handlers["set_fader"](
            websocket,
            {"channel": 1, "value": -6.0, "approved": True, "approval_id": "unit-apply"},
        )
        await handlers["rollback_last_manual_write"](
            websocket,
            {"approved": True, "approval_id": "rollback-unit"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_rollback_applied"
        assert payload["status"] == "applied"
        assert payload["rollback_status"] == "partial"
        assert payload["rollback_readback_status"] == "readback_cache_not_fresh"
        assert payload["rollback_proof_status"] == "partial"
