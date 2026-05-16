"""
Tests for backend/server.py — convert_numpy_types utility and
AutoMixerServer basic initialization.

All tests work without hardware or network (all external connections mocked).
"""

import json
import sys
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import importlib
from pathlib import Path
from backend.handlers import snapshot_handlers

try:
    from backend.server import convert_numpy_types, AutoMixerServer
except ImportError:
    pytest.skip("backend.server module not importable", allow_module_level=True)

server_module = importlib.import_module("backend.server")
sys.modules.setdefault("server", server_module)


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


class TestDisconnectedAgentHandlers:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("message", "expected_type", "expected_fields"),
        [
            (
                {"type": "get_agent_status"},
                "agent_status",
                {"is_running": False},
            ),
            (
                {"type": "get_pending_actions"},
                "pending_actions",
                {"actions": []},
            ),
            (
                {"type": "approve_action", "index": 3},
                "action_approved",
                {"index": 3, "success": False},
            ),
            (
                {"type": "set_agent_mode", "mode": "live"},
                "agent_mode_changed",
                {"mode": "live", "success": False},
            ),
        ],
    )
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_dead_agent_commands_return_unavailable_responses(
        self,
        mock_bleed,
        mock_config,
        message,
        expected_type,
        expected_fields,
    ):
        server = AutoMixerServer()
        assert not hasattr(server, "mixing_agent")
        server.send_to_client = AsyncMock()

        await server.handle_client_message(AsyncMock(), json.dumps(message))

        server.send_to_client.assert_awaited_once()
        _, payload = server.send_to_client.await_args.args
        assert payload["type"] == expected_type
        assert payload["status"] == "unavailable"
        assert payload["available"] is False
        assert payload["reason"] == "mixing_agent_runtime_not_initialized"
        assert payload["error"] == "Mixing agent runtime is not attached to AutoMixerServer"
        for key, value in expected_fields.items():
            assert payload[key] == value


class DummyWingLike:
    def __init__(self):
        self.set_channel_fader = MagicMock()
        self.set_channel_gain = MagicMock()
        self.supervised_set_channel_fader = MagicMock(side_effect=self._supervised_fader)
        self.supervised_set_channel_gain = MagicMock(side_effect=self._supervised_gain)
        self.load_snap = MagicMock(return_value=True)
        self.save_snap = MagicMock(return_value=True)
        self.configure_write_gate = MagicMock()
        self._gate_state = "disarmed"
        self.arm_write_gate = MagicMock(side_effect=self._arm_gate)
        self.disarm_write_gate = MagicMock(side_effect=self._disarm_gate)
        self.emergency_stop_writes = MagicMock(side_effect=self._emergency_stop_gate)
        self.get_write_gate_status = MagicMock(side_effect=self._get_gate_status)
        self.send = MagicMock()
        self.is_connected = True
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

    def _arm_gate(self, **kwargs):
        self._gate_state = "armed"
        return {"state": self._gate_state}

    def _disarm_gate(self, **kwargs):
        self._gate_state = "disarmed"
        return {"state": self._gate_state}

    def _emergency_stop_gate(self, **kwargs):
        self._gate_state = "emergency_stopped"
        return {"state": self._gate_state}

    def _get_gate_status(self):
        return {"state": self._gate_state}

    def apply_live_change(self, **kwargs):
        value_type = kwargs["value_type"]
        if value_type == "fader":
            return self.supervised_set_channel_fader(
                kwargs["channel"],
                kwargs["value"],
                source=kwargs["source"],
                reason=kwargs["reason"],
                approved=kwargs["approved"],
                approval_id=kwargs.get("approval_id"),
                bypass_cooldown=kwargs.get("bypass_cooldown", False),
            )
        if value_type == "gain":
            return self.supervised_set_channel_gain(
                kwargs["channel"],
                kwargs["value"],
                source=kwargs["source"],
                reason=kwargs["reason"],
                approved=kwargs["approved"],
                approval_id=kwargs.get("approval_id"),
                bypass_cooldown=kwargs.get("bypass_cooldown", False),
            )
        return {"status": "blocked", "reason": "unsupported_value_type"}


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


def write_shadow_report_bundle(
    root: Path,
    *,
    ready: bool = True,
    dry_run_only: bool = True,
    live_mixer_writes: bool = False,
    blocked_reasons: list[str] | None = None,
    disagreement_count: int = 0,
    highest_severity: str = "warning",
) -> Path:
    report_dir = root / "shadow_case"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "shadow_mode_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "replay_shadow_report/v1",
                "generated_at": "2026-05-16T12:00:00Z",
                "manifest_id": "manifest::unit",
                "selected_proposal_id": "corr::operator::unit",
                "disagreement_count": disagreement_count,
                "promotion_readiness_score": 88 if ready else 40,
                "files": {
                    "comparison": "shadow_mode_comparison.json",
                    "rollback": "rollback_simulation_report.json",
                    "readiness": "promotion_readiness_report.json",
                    "confidence_trajectory": "confidence_trajectory.json",
                },
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "shadow_mode_comparison.json").write_text(
        json.dumps(
            {
                "selected_proposal_id": "corr::operator::unit",
                "disagreements": [{} for _ in range(disagreement_count)],
                "evidence": {
                    "dry_run_only": dry_run_only,
                    "live_mixer_writes": live_mixer_writes,
                },
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "promotion_readiness_report.json").write_text(
        json.dumps(
            {
                "ready": ready,
                "promotion_readiness_score": 88 if ready else 40,
                "dry_run_only": dry_run_only,
                "live_mixer_writes": live_mixer_writes,
                "validation_highest_severity": highest_severity,
                "aggregation": {
                    "disagreement_rate": 1.0 if disagreement_count else 0.0,
                    "disagreement_classes": {
                        "by_code": {"operator_selected": disagreement_count} if disagreement_count else {},
                        "by_taxonomy": {"operator_override": disagreement_count} if disagreement_count else {},
                    },
                    "confidence_trend": {
                        "direction": "operator_above_ai" if disagreement_count else "ai_above_operator",
                        "selected_confidence": 0.92 if disagreement_count else 0.95,
                        "operator_vs_ai_delta": 0.08 if disagreement_count else -0.75,
                    },
                    "rollback_frequency": {
                        "event_count": 1,
                        "blocked_event_count": len(blocked_reasons or []),
                        "blocked_event_rate": 1.0 if blocked_reasons else 0.0,
                        "rollback_point_count": 2,
                        "blocked_reason_count": len(blocked_reasons or []),
                    },
                    "audit_coverage": {
                        "complete": True,
                        "passed_check_count": 10,
                        "required_check_count": 10,
                        "coverage_rate": 1.0,
                        "checks": [],
                    },
                },
                "promotion_readiness_gates": [
                    {"key": "dry_run_only", "ok": dry_run_only, "blocking": True},
                    {"key": "no_live_mixer_writes", "ok": not live_mixer_writes, "blocking": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "rollback_simulation_report.json").write_text(
        json.dumps(
            {
                "blocked_reasons": blocked_reasons or [],
            }
        ),
        encoding="utf-8",
    )
    return report_dir


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

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_manual_write_uses_normalized_client_live_apply_contract(
        self,
        mock_bleed_service,
        mock_load_config,
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        server.connection_mode = "mixing_station"
        server._manual_write_supervision_enabled = True
        server._manual_writes_armed = True
        server._manual_write_cooldown_sec = 0.0

        client = MagicMock()
        client.is_connected = True
        client.apply_live_change.return_value = {
            "status": "blocked",
            "channel": 2,
            "value_type": "gain",
            "requested_value": 3.0,
            "effective_value": 3.0,
            "old_value": None,
            "readback_value": None,
            "readback_source": None,
            "source": "manual:test",
            "reason": "pre_read_unavailable",
            "reason_text": "operator approved",
            "approved": True,
            "approval_id": "approval-generic-1",
            "bypass_cooldown": False,
        }
        server.mixer_client = client

        result = await server._apply_manual_console_write(
            None,
            channel=2,
            value=3.0,
            value_type="gain",
            source="manual:test",
            reason="operator approved",
            approved=True,
            approval_id="approval-generic-1",
        )

        assert result["status"] == "blocked"
        assert result["reason"] == "pre_read_unavailable"
        client.apply_live_change.assert_called_once_with(
            channel=2,
            value=3.0,
            value_type="gain",
            source="manual:test",
            reason="operator approved",
            approved=True,
            approval_id="approval-generic-1",
            bypass_cooldown=False,
        )


class TestSupervisedPilotCockpit:

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_supervised_pilot_status_exposes_shadow_and_rollback_ready_states(
        self,
        mock_bleed_service,
        mock_load_config,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        server._shadow_reports_root = tmp_path
        write_shadow_report_bundle(tmp_path, ready=True, highest_severity="warning")
        server.start_manual_write_session(session_label="pilot", reason="unit test")

        await server._apply_manual_console_write(
            None,
            channel=1,
            value=-12.0,
            value_type="fader",
            source="manual:pilot",
            reason="approved pilot write",
            approved=True,
            approval_id="pilot-approval-1",
        )

        status = server.get_supervised_pilot_status()
        states = {item["key"]: item for item in status["states"]}

        assert status["type"] == "supervised_pilot_status"
        assert status["primary_state"] == "rollback_ready"
        assert states["shadow"]["active"] is True
        assert states["confirmed"]["active"] is True
        assert states["rollback_ready"]["active"] is True
        assert states["blocked"]["active"] is False
        assert status["shadow_transport"]["state"] == "shadow_ready"
        assert status["latest_confirmation"]["readback_value"] == -12.0
        assert status["manual_write_session"]["rollback_ready_count"] == 1
        assert status["pilot_ready"] is True
        assert status["gain_fader_runtime"]["selected_runtime"]["id"] == "wing_supervised_soundcheck"
        assert status["gain_fader_runtime"]["deployment_surface"] is True
        assert status["shadow_transport"]["disagreement_rate"] == 0.0
        assert status["shadow_transport"]["audit_coverage"]["coverage_rate"] == 1.0
        assert status["shadow_transport"]["confidence_trend"]["direction"] == "ai_above_operator"

    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    def test_supervised_pilot_status_surfaces_blocked_escalations(
        self,
        mock_bleed_service,
        mock_load_config,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        server._shadow_reports_root = tmp_path
        write_shadow_report_bundle(
            tmp_path,
            ready=False,
            blocked_reasons=["replay_mismatch"],
            highest_severity="error",
        )
        server.set_manual_write_supervision(False, reason="emergency_stop")

        status = server.get_supervised_pilot_status()
        escalation_codes = {item["code"] for item in status["escalations"]}

        assert status["primary_state"] == "blocked"
        assert status["state_flags"]["blocked"] is True
        assert status["shadow_transport"]["state"] == "blocked"
        assert status["gain_fader_runtime"]["selected_runtime"]["id"] == "wing_supervised_soundcheck"
        assert "manual_write_panic_stopped" in escalation_codes
        assert "write_gate_emergency_stopped" in escalation_codes
        assert "shadow_transport_blocked" in escalation_codes
        assert status["pilot_ready"] is False
        assert status["shadow_transport"]["rollback_frequency"]["blocked_event_count"] == 1
        assert status["shadow_transport"]["promotion_readiness_gates"]

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_get_supervised_pilot_status_handler_returns_json_payload(
        self,
        mock_bleed_service,
        mock_load_config,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        server._shadow_reports_root = tmp_path
        write_shadow_report_bundle(tmp_path, ready=True)
        websocket = MagicMock()
        websocket.send = AsyncMock()

        await server._dispatch["get_supervised_pilot_status"](websocket, {})

        raw = websocket.send.await_args_list[-1].args[0]
        payload = json.loads(raw)
        assert payload["type"] == "supervised_pilot_status"
        assert payload["shadow_transport"]["available"] is True
        assert payload["gain_fader_runtime"]["selected_runtime"]["id"] == "wing_supervised_soundcheck"


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


class DummyMultiChannelAutoEQController:
    def __init__(self, mixer_client=None, bleed_service=None):
        self.mixer_client = mixer_client
        self.bleed_service = bleed_service
        self.start_calls = []

    def start_multi_channel(self, **kwargs):
        self.start_calls.append(kwargs)
        return True


class TestWingDeploymentWriteBoundary:

    @staticmethod
    def _decode_last_message(websocket):
        assert websocket.send.await_count >= 1
        raw = websocket.send.await_args_list[-1].args[0]
        return json.loads(raw)

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_reset_trim_is_quarantined_for_wing(
        self,
        mock_bleed_service,
        mock_load_config,
        dummy_wing_server,
    ):
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
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_snapshot_load_is_quarantined_for_wing(
        self,
        mock_bleed_service,
        mock_load_config,
        dummy_wing_server,
    ):
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
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_start_auto_eq_forces_auto_apply_off_for_wing_boundary(
        self,
        mock_bleed_service,
        mock_load_config,
        dummy_wing_server,
    ):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()

        with patch.object(server_module, "AutoEQController", DummyAutoEQController):
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
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_apply_eq_correction_is_quarantined_for_wing(
        self,
        mock_bleed_service,
        mock_load_config,
        dummy_wing_server,
    ):
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
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_start_multi_channel_auto_eq_passes_mode_to_controller(
        self,
        mock_bleed_service,
        mock_load_config,
        dummy_wing_server,
    ):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()

        with patch.object(server_module, "MultiChannelAutoEQController", DummyMultiChannelAutoEQController):
            await server.start_multi_channel_auto_eq(
                websocket,
                device_id="1",
                channels_config=[{"channel": 1, "profile": "custom", "auto_apply": False}],
                mode="live",
            )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "multi_channel_auto_eq_status"
        assert payload["active"] is True
        assert server.multi_channel_auto_eq_controller.start_calls[-1]["mode"] == "live"

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_start_multi_channel_auto_eq_forces_live_apply_into_confirm_safe_defaults(
        self,
        mock_bleed_service,
        mock_load_config,
        dummy_wing_server,
    ):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()

        with patch.object(server_module, "MultiChannelAutoEQController", DummyMultiChannelAutoEQController):
            await server.start_multi_channel_auto_eq(
                websocket,
                device_id="1",
                channels_config=[{"channel": 1, "profile": "custom", "auto_apply": True, "live_apply": True}],
                mode="live",
            )

        payload = self._decode_last_message(websocket)
        normalized = server.multi_channel_auto_eq_controller.start_calls[-1]["channels_config"][0]

        assert payload["type"] == "multi_channel_auto_eq_status"
        assert payload["active"] is True
        assert payload["mode"] == "live"
        assert payload["auto_apply_forced_off"] is True
        assert payload["confirm_defaults_applied"] is True
        assert normalized["auto_apply"] is False
        assert normalized["live_apply"] is True
        assert normalized["apply_mode"] == "confirm"
        assert normalized["requires_confirmation"] is True
        assert normalized["mode"] == "live"
