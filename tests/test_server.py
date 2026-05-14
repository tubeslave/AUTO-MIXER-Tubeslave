"""
Tests for backend/server.py — convert_numpy_types utility and
AutoMixerServer basic initialization.

All tests work without hardware or network (all external connections mocked).
"""

import json
import asyncio
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import handlers.channel_scan_handlers as channel_scan_handlers
import handlers.fader_handlers as fader_handlers

try:
    from server import convert_numpy_types, AutoMixerServer
    from live_apply import LiveApplyPolicy, LiveApplyService
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
    def test_gain_staging_recommendations_prefer_live_input_trim(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.live_input_trim_controller = MagicMock()
        server.live_input_trim_controller.get_status.return_value = {
            "active": True,
            "live_apply_enabled": False,
            "analysis_only_mode": True,
        }
        server.live_input_trim_controller.get_recommendations.return_value = {
            1: {"channel": 1, "delta_db": 1.0}
        }
        server.safe_gain_calibrator = MagicMock()

        payload = server.get_gain_staging_recommendations()

        assert payload["active"] is True
        assert payload["source"] == "live_input_trim_controller"
        assert payload["dry_run"] is True
        assert payload["recommendation_count"] == 1
        assert payload["recommendations"][1]["delta_db"] == 1.0
        server.safe_gain_calibrator.get_recommendations.assert_not_called()

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_apply_auto_balance_defaults_to_unconfirmed_live_apply(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.auto_fader_controller = MagicMock()
        server.auto_fader_controller.get_status.return_value = {
            "auto_balance_result": {1: {"correction": 1.0}},
            "auto_balance_pass": 1,
        }
        messages = []

        async def fake_send_to_client(_websocket, message):
            messages.append(message)

        server.send_to_client = fake_send_to_client
        await server.apply_auto_balance("ws")

        server.auto_fader_controller.set_confirm_live_apply.assert_called_once_with(False)
        server.auto_fader_controller.apply_auto_balance.assert_called_once_with()
        assert messages == []

    @pytest.mark.asyncio
    async def test_fader_handler_apply_auto_balance_requires_explicit_confirmation(self):
        server = MagicMock()
        server.apply_auto_balance = AsyncMock()
        handlers = fader_handlers.register_handlers(server)

        await handlers["apply_auto_balance"]("ws", {})

        server.apply_auto_balance.assert_awaited_once_with(
            "ws",
            confirm_live_apply=False,
        )

    @pytest.mark.asyncio
    async def test_fader_handler_apply_auto_balance_preserves_explicit_confirmation(self):
        server = MagicMock()
        server.apply_auto_balance = AsyncMock()
        handlers = fader_handlers.register_handlers(server)

        await handlers["apply_auto_balance"](
            "ws",
            {"confirm_live_apply": True},
        )

        server.apply_auto_balance.assert_awaited_once_with(
            "ws",
            confirm_live_apply=True,
        )

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_auto_engine_gain_recommendations_default_idle_payload(self, mock_config, mock_bleed):
        server = AutoMixerServer()

        payload = server.get_auto_engine_gain_recommendations()

        assert payload == {
            "active": False,
            "source": "auto_soundcheck_engine",
            "live_apply_enabled": False,
            "dry_run": True,
            "recommendation_count": 0,
            "recommendations": {},
        }

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_safe_gain_ready_payload_requires_manual_review(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.safe_gain_calibrator = MagicMock()
        server.safe_gain_calibrator.get_recommendations.return_value = {
            1: {"channel": 1, "delta_db": 1.5, "dry_run_only": True}
        }

        payload = server._build_safe_gain_ready_payload({
            1: {"suggested_gain_db": 1.5}
        })

        assert payload["type"] == "gain_staging_status"
        assert payload["status_type"] == "safe_gain_ready"
        assert payload["source"] == "safe_gain_calibrator"
        assert payload["review_required"] is True
        assert payload["live_apply_enabled"] is False
        assert payload["dry_run"] is True
        assert payload["recommendation_count"] == 1
        assert payload["message"] == "Analysis complete. Manual review required before any console write."
        assert payload["recommendations"][1]["delta_db"] == 1.5
        assert "apply_summary" not in payload
        server.safe_gain_calibrator.apply_corrections.assert_not_called()

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_gain_staging_status_safe_gain_is_review_only(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.safe_gain_calibrator = MagicMock()
        server.safe_gain_calibrator.get_status.return_value = {
            "state": "ready",
            "learning_progress": 1.0,
            "channels_count": 2,
            "suggestions_ready": True,
            "target_lufs": -18.0,
            "max_peak_limit": -1.0,
            "confirm_live_apply": True,
        }

        payload = server.get_gain_staging_status()

        assert payload["safe_gain_mode"] is True
        assert payload["live_apply_enabled"] is False
        assert payload["analysis_only_mode"] is True
        assert payload["confirm_live_apply"] is True
        assert payload["review_required"] is True

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_live_input_trim_recommendation_surface_default_payload_shape(self, mock_config, mock_bleed):
        server = AutoMixerServer()

        payload = server.get_live_input_trim_recommendations()

        assert payload == {
            "active": False,
            "source": "live_input_trim_controller",
            "live_apply_enabled": False,
            "dry_run": True,
            "recommendation_count": 0,
            "recommendations": {},
        }

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @pytest.mark.asyncio
    async def test_set_live_input_trim_live_apply_reports_confirmation_gated_state(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        server.live_input_trim_controller = MagicMock()
        server.live_input_trim_controller.set_live_apply.return_value = {
            "active": True,
            "live_apply_enabled": True,
            "analysis_only_mode": False,
            "confirm_live_apply": True,
        }
        messages = []

        async def _capture(_websocket, message):
            messages.append(message)

        server.send_to_client = _capture

        await server.set_live_input_trim_live_apply(
            "ws",
            enabled=True,
            analysis_only_mode=False,
            confirm_live_apply=True,
        )

        server.live_input_trim_controller.set_live_apply.assert_called_once_with(
            enabled=True,
            analysis_only_mode=False,
            confirm_live_apply=True,
        )
        assert messages[-1]["type"] == "live_input_trim_status"
        assert messages[-1]["status_type"] == "live_apply_changed"
        assert messages[-1]["live_input_trim_state"]["live_apply_enabled"] is True
        assert messages[-1]["live_input_trim_state"]["analysis_only_mode"] is False
        assert messages[-1]["live_input_trim_state"]["confirm_live_apply"] is True

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @pytest.mark.asyncio
    async def test_reset_all_functions_is_blocked_for_auto_soundcheck_source(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.mixer_client = MagicMock()
        server.mixer_client.is_connected = True
        messages = []

        async def _capture(_websocket, message):
            messages.append(message)

        server.send_to_client = _capture
        server.reset_trim = MagicMock(side_effect=AssertionError("reset_trim should not run"))
        server.reset_all_eq = MagicMock(side_effect=AssertionError("reset_all_eq should not run"))
        server.reset_phase_delay = MagicMock(side_effect=AssertionError("reset_phase_delay should not run"))

        await server.reset_all_functions_to_defaults("ws", [1, 2])

        assert messages[-1]["type"] == "reset_all_functions_result"
        assert messages[-1]["blocked"] is True
        assert messages[-1]["blocked_reason"] == "maintenance_source_not_allowed"

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @pytest.mark.asyncio
    async def test_request_reset_trim_emits_maintenance_payload_on_success(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.mixer_client = MagicMock()
        server.mixer_client.is_connected = True
        messages = []

        async def _capture(_websocket, message):
            messages.append(message)

        async def _fake_reset_trim(_websocket, channels, *, emit_result=True):
            assert emit_result is False
            return {
                "type": "reset_trim_result",
                "success": True,
                "success_count": len(channels),
                "total_count": len(channels),
                "failed_channels": [],
                "results": {channel: {"success": True, "new_trim": 0.0} for channel in channels},
                "message": f"Reset TRIM to 0dB for {len(channels)}/{len(channels)} channels",
            }

        server.send_to_client = _capture
        server.reset_trim = _fake_reset_trim

        result = await server.request_reset_trim(
            "ws",
            [1, 2],
            confirm_maintenance=True,
            rollback_snapshot_path="/tmp/rollback.json",
        )

        assert result.accepted is True
        assert len(messages) == 1
        payload = messages[0]
        assert payload["type"] == "reset_trim_result"
        assert payload["success"] is True
        assert payload["blocked"] is False
        assert payload["send_status"] == "sent"
        assert payload["rollback_plan_present"] is True
        assert payload["audit_id"]

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @pytest.mark.asyncio
    async def test_request_bypass_mixer_blocks_without_confirmation(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.mixer_client = MagicMock()
        server.mixer_client.is_connected = True
        messages = []

        async def _capture(_websocket, message):
            messages.append(message)

        server.send_to_client = _capture
        server.bypass_mixer = MagicMock(side_effect=AssertionError("bypass_mixer should not run"))

        result = await server.request_bypass_mixer("ws")

        assert result.accepted is False
        assert len(messages) == 1
        payload = messages[0]
        assert payload["type"] == "bypass_result"
        assert payload["success"] is False
        assert payload["blocked"] is True
        assert payload["blocked_reason"] == "maintenance_confirmation_required"
        assert payload["rollback_plan_required"] is True

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @pytest.mark.asyncio
    async def test_request_bypass_mixer_emits_maintenance_payload_on_success(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.mixer_client = MagicMock()
        server.mixer_client.is_connected = True
        messages = []

        async def _capture(_websocket, message):
            messages.append(message)

        async def _fake_bypass(_websocket, *, emit_result=True):
            assert emit_result is False
            return {
                "type": "bypass_result",
                "success": True,
                "success_count": 40,
                "total_count": 40,
                "failed_channels": [],
                "message": "Bypass completed: 40/40 channels processed",
            }

        server.send_to_client = _capture
        server.bypass_mixer = _fake_bypass

        result = await server.request_bypass_mixer(
            "ws",
            confirm_maintenance=True,
            rollback_snapshot_path="/tmp/rollback.json",
        )

        assert result.accepted is True
        assert len(messages) == 1
        payload = messages[0]
        assert payload["type"] == "bypass_result"
        assert payload["success"] is True
        assert payload["blocked"] is False
        assert payload["send_status"] == "sent"
        assert payload["rollback_plan_present"] is True
        assert payload["audit_id"]

    @pytest.mark.asyncio
    async def test_bypass_handler_forwards_maintenance_metadata(self):
        class FakeServer:
            def __init__(self):
                self.calls = []

            async def request_bypass_mixer(
                self,
                websocket,
                *,
                dry_run=None,
                confirm_maintenance=False,
                rollback_snapshot_path=None,
            ):
                self.calls.append(
                    (
                        websocket,
                        {
                            "dry_run": dry_run,
                            "confirm_maintenance": confirm_maintenance,
                            "rollback_snapshot_path": rollback_snapshot_path,
                        },
                    )
                )

        server = FakeServer()
        handlers = channel_scan_handlers.register_handlers(server)

        await handlers["bypass_mixer"](
            "ws",
            {
                "dry_run": True,
                "confirm_maintenance": True,
                "rollback_snapshot_path": "/tmp/rollback.json",
            },
        )

        assert server.calls == [
            (
                "ws",
                {
                    "dry_run": True,
                    "confirm_maintenance": True,
                    "rollback_snapshot_path": "/tmp/rollback.json",
                },
            )
        ]


class FakeVoiceMixer:
    def __init__(
        self,
        *,
        fader_result=True,
        gain_result=True,
        last_send_status="sent",
        verify_result=None,
    ):
        self.calls = []
        self.fader_result = fader_result
        self.gain_result = gain_result
        self.last_send_status = last_send_status
        self.verify_result = verify_result

    def set_channel_fader(self, channel, value):
        self.calls.append(("set_channel_fader", int(channel), float(value)))
        return self.fader_result

    def set_channel_gain(self, channel, value):
        self.calls.append(("set_channel_gain", int(channel), float(value)))
        return self.gain_result

    def get_last_send_status(self):
        return self.last_send_status

    def confirm_manual_write(self, operation, channel, desired_value, timeout_ms=300):
        if self.verify_result is not None:
            return self.verify_result
        return {
            "verification_id": "voice-verify",
            "readback_status": "confirmed",
            "confirmed_value": float(desired_value),
            "failure_reason": None,
            "timeout_ms": int(timeout_ms),
        }


class TestVoiceCommandSafety:

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_voice_set_fader_defaults_to_dry_run(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        mixer = FakeVoiceMixer()
        messages = []

        server.mixer_client = mixer

        async def fake_broadcast(message):
            messages.append(message)

        server.broadcast = fake_broadcast

        await server._execute_voice_command({"type": "set_fader", "channel": 4, "value": -6.0})

        assert mixer.calls == []
        assert messages[-1]["type"] == "voice_command_result"
        assert messages[-1]["command"] == "set_fader"
        assert messages[-1]["accepted"] is True
        assert messages[-1]["blocked"] is False
        assert messages[-1]["dry_run_only"] is True
        assert messages[-1]["send_status"] == "not_sent"
        assert messages[-1]["readback_status"] == "not_requested"

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_voice_eq_command_is_blocked_without_console_write(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        mixer = FakeVoiceMixer()
        messages = []

        server.mixer_client = mixer

        async def fake_broadcast(message):
            messages.append(message)

        server.broadcast = fake_broadcast

        await server._execute_voice_command({"type": "eq_on", "channel": 4, "on": 1})

        assert mixer.calls == []
        assert messages[-1]["type"] == "voice_command_result"
        assert messages[-1]["command"] == "eq_on"
        assert messages[-1]["accepted"] is False
        assert messages[-1]["blocked"] is True
        assert messages[-1]["blocked_reason"] == "unsupported_voice_operation"
        assert messages[-1]["send_status"] == "not_sent"

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_voice_set_gain_blocks_without_confirm(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        mixer = FakeVoiceMixer()
        messages = []

        server.mixer_client = mixer

        async def fake_broadcast(message):
            messages.append(message)

        server.broadcast = fake_broadcast

        await server._execute_voice_command(
            {"type": "set_gain", "channel": 4, "value": 2.0, "dry_run": False}
        )

        assert mixer.calls == []
        assert messages[-1]["accepted"] is False
        assert messages[-1]["blocked"] is True
        assert messages[-1]["blocked_reason"] == "confirm_live_apply_required"
        assert messages[-1]["dry_run_only"] is False
        assert messages[-1]["send_status"] == "not_sent"
        assert messages[-1]["readback_status"] == "not_requested"
        assert messages[-1]["message_for_user"] == "Live apply blocked: confirmation required."
        assert messages[-1]["target_type"] == "channel_gain"
        assert messages[-1]["live_apply_source"] == "voice_control"

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_voice_set_gain_allows_confirmed_live_apply(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        mixer = FakeVoiceMixer()
        messages = []

        server.mixer_client = mixer

        async def fake_broadcast(message):
            messages.append(message)

        server.broadcast = fake_broadcast

        await server._execute_voice_command(
            {
                "type": "set_gain",
                "channel": 4,
                "value": 2.0,
                "dry_run": False,
                "confirm_live_apply": True,
            }
        )

        assert mixer.calls == [("set_channel_gain", 4, 2.0)]
        assert messages[-1]["accepted"] is True
        assert messages[-1]["blocked"] is False
        assert messages[-1]["dry_run_only"] is False
        assert messages[-1]["send_status"] == "sent"
        assert messages[-1]["readback_status"] == "confirmed"
        assert messages[-1]["message_for_user"] == "Live apply confirmed by mixer readback."
        assert messages[-1]["verification_id"] == "voice-verify"
        assert messages[-1]["target_type"] == "channel_gain"
    
    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_voice_set_gain_preserves_replay_correlation_id(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        mixer = FakeVoiceMixer()
        messages = []

        server.mixer_client = mixer

        async def fake_broadcast(message):
            messages.append(message)

        server.broadcast = fake_broadcast

        await server._execute_voice_command(
            {
                "type": "set_gain",
                "channel": 4,
                "value": 2.0,
                "dry_run": False,
                "confirm_live_apply": True,
                "replay_correlation_id": "corr::voice::gain::4",
            }
        )

        assert messages[-1]["replay_correlation_id"] == "corr::voice::gain::4"

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_voice_set_fader_marks_readback_timeout_as_not_accepted(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        mixer = FakeVoiceMixer(
            verify_result={
                "verification_id": "voice-timeout",
                "readback_status": "timeout",
                "confirmed_value": None,
                "failure_reason": "readback_timeout",
                "timeout_ms": 300,
            }
        )
        messages = []

        server.mixer_client = mixer

        async def fake_broadcast(message):
            messages.append(message)

        server.broadcast = fake_broadcast

        await server._execute_voice_command(
            {
                "type": "set_fader",
                "channel": 4,
                "value": -6.0,
                "dry_run": False,
                "confirm_live_apply": True,
            }
        )

        assert mixer.calls == [("set_channel_fader", 4, -6.0)]
        assert messages[-1]["accepted"] is False
        assert messages[-1]["blocked"] is True
        assert messages[-1]["blocked_reason"] == "readback_timeout"
        assert messages[-1]["send_status"] == "sent"
        assert messages[-1]["readback_status"] == "timeout"
        assert messages[-1]["message_for_user"] == "Live apply sent, but readback timed out before confirmation."
        assert messages[-1]["verification_id"] == "voice-timeout"
        assert messages[-1]["target_type"] == "channel_fader"

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_voice_set_gain_reports_disconnected_transport(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        mixer = FakeVoiceMixer(gain_result=False, last_send_status="disconnected")
        messages = []

        server.mixer_client = mixer

        async def fake_broadcast(message):
            messages.append(message)

        server.broadcast = fake_broadcast

        await server._execute_voice_command(
            {
                "type": "set_gain",
                "channel": 4,
                "value": 1.5,
                "dry_run": False,
                "confirm_live_apply": True,
            }
        )

        assert mixer.calls == [("set_channel_gain", 4, 1.5)]
        assert messages[-1]["accepted"] is False
        assert messages[-1]["blocked"] is True
        assert messages[-1]["blocked_reason"] == "send_failed"
        assert messages[-1]["send_status"] == "disconnected"
        assert messages[-1]["readback_status"] == "not_requested"
        assert messages[-1]["failure_reason"] == "disconnected"
        assert messages[-1]["message_for_user"] == "Mixer disconnected: command was not sent."
        assert messages[-1]["target_type"] == "channel_gain"

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_voice_set_fader_reports_manual_panic_stop_scope(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        mixer = FakeVoiceMixer()
        messages = []
        server.live_apply_service = LiveApplyService(
            policy=LiveApplyPolicy(
                panic_stop_active=True,
                panic_stop_reason="operator_stop",
            )
        )

        server.mixer_client = mixer

        async def fake_broadcast(message):
            messages.append(message)

        server.broadcast = fake_broadcast

        await server._execute_voice_command(
            {
                "type": "set_fader",
                "channel": 4,
                "value": -6.0,
                "dry_run": False,
                "confirm_live_apply": True,
            }
        )

        assert mixer.calls == []
        assert messages[-1]["accepted"] is False
        assert messages[-1]["blocked"] is True
        assert messages[-1]["blocked_reason"] == "panic_stop_active"
        assert messages[-1]["send_status"] == "not_sent"
        assert messages[-1]["readback_status"] == "not_requested"
        assert messages[-1]["message_for_user"] == "Live apply blocked: panic stop is active for manual live apply."
        assert messages[-1]["panic_stop_active"] is True
        assert messages[-1]["panic_stop_scope"] == "manual_live_apply_only"
        assert messages[-1]["panic_stop_reason"] == "operator_stop"
