"""
Tests for backend/server.py — convert_numpy_types utility and
AutoMixerServer basic initialization.

All tests work without hardware or network (all external connections mocked).
"""

import asyncio
import errno
import hashlib
import json
import math
import os
import signal
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import ANY, AsyncMock, patch, MagicMock
from handlers import automation_handlers, fx_handlers, mixer_handlers, routing_handlers, snapshot_handlers
from backend.operator_mode_policy import SUPPORTED_MANUAL_LIVE_WRITE_KINDS
from decision_queue_wire_contract import project_outbound_message
from operator_proposal_queue import (
    OperatorProposalQueue,
    SOUNDCHECK_CORRECTION_REQUIRES_SUPERVISED_TRANSACTION,
)
from operator_proposal_result_store import (
    ResultEvidenceError,
    ResultEvidenceReservationBinding,
    ResultEvidenceStore,
    default_result_store_root,
)
from operator_product_state import build_safety_monitor
from runtime_identity import (
    RUNTIME_CRITICAL_SOURCE_PATHS,
    RUNTIME_IDENTITY_SCHEMA,
    runtime_identity_manifest_sha256,
)
from master_capture_route_proof import MasterCaptureRouteAuthority

try:
    import server as server_module
    from server import convert_numpy_types, AutoMixerServer
except ImportError:
    pytest.skip("server module not importable", allow_module_level=True)
try:
    from ai.agent import AgentMode
except ImportError:
    AgentMode = None


def _configure_unit_master_route_proof(server, tmp_path, proof_id):
    now = time.time()
    launch_sha = "a" * 64
    server._runtime_identity["launch_binding"][
        "prestart_manifest_sha256"
    ] = launch_sha
    server._m110_launch_authority = MagicMock()
    server._m110_launch_authority.source_snapshot_status.return_value = (
        True,
        "m110_live_source_snapshot_current",
    )
    server.connection_mode = "wing"
    owner = object()
    server._ws_auth_required = True
    server._authenticated_websocket_ids.add(id(owner))
    server._unit_route_owner = owner
    server._m110_transport_owner_connection_id = id(owner)
    server._m110_transport_owner_ref = owner
    control_dir = tmp_path / "autopilot"
    server._m110_autopilot_stop_path = control_dir / "STOP"
    server._m110_autopilot_writer_lease_path = (
        control_dir / "WRITER_LEASE" / "owner.json"
    )
    server._m110_autopilot_writer_lease_dir = (
        server._m110_autopilot_writer_lease_path.parent
    )
    server._m110_legacy_autoloop_lock_path = control_dir / ".codex_autoloop.lock"
    server.mixer_client = _unit_connected_wing_client()
    server._master_capture_route_authority = MasterCaptureRouteAuthority(
        proof_id=proof_id,
        issued_at_epoch=now,
        expires_at_epoch=now + 300.0,
        runtime_instance_nonce=server._runtime_identity["instance_nonce"],
        runtime_manifest_sha256=server._runtime_identity["manifest_sha256"],
        launch_manifest_sha256=launch_sha,
        connection_generation=server._mixer_connection_generation,
        transport_generation=server._mixer_connection_generation,
        transport_instance_id=id(server.mixer_client),
        owner_connection_id=id(owner),
        emergency_generation=server._manual_write_emergency_generation,
        console_serial="WING-UNIT",
        firmware_version="3.0.5",
        capture_device="Unit WING USB 48ch",
        capture_device_index=7,
        capture_channel_count=48,
        capture_sample_rate_hz=48000,
        capture_instance_id=0,
        capture_hardware_uid="usb:1397:050b:WING-UNIT214:00100000",
        capture_hardware_binding_sha256="d" * 64,
        capture_usb_serial="WING-UNIT214",
        capture_usb_location_id=1048576,
        master_channels=(47, 48),
        artifact_path=str(tmp_path / f"{proof_id}.json"),
        artifact_sha256="b" * 64,
        evidence_sha256="sha256:" + ("c" * 64),
        _transport_ref=server.mixer_client,
        _capture_ref=None,
        _capture_stream_ref=None,
        _owner_ref=owner,
    )
    server._m110_capture_hardware_identity = MagicMock(
        return_value={
            "verified": True,
            "reason": "physical_wing_usb_capture_bound_to_console",
            "device_name": "Unit WING USB 48ch",
            "device_index": 7,
            "console_serial": "WING-UNIT",
            "hardware_uid": "usb:1397:050b:WING-UNIT214:00100000",
            "binding_sha256": "d" * 64,
            "usb": {
                "vendor_id": 5015,
                "product_id": 1291,
                "serial": "WING-UNIT214",
                "location_id": 1048576,
            },
        }
    )
    return {
        "/$syscfg/$serial": "WING-UNIT",
        "/$syscfg/$firmware": "3.0.5",
        "/io/out/USB/47/grp": "MAIN",
        "/io/out/USB/47/in": 0,
        "/io/out/USB/48/grp": "MAIN",
        "/io/out/USB/48/in": 1,
    }


def _unit_connected_wing_client():
    client = object.__new__(server_module.WingClient)
    client.is_connected = True
    client.query_parameter = MagicMock()
    return client


def _bind_unit_master_route_capture(server, capture):
    server._audio_capture_preview_owner = server._unit_route_owner
    stream = getattr(capture, "_stream", None)
    if stream is None:
        stream = object()
        capture._stream = stream
    server._master_capture_route_authority = replace(
        server._master_capture_route_authority,
        capture_instance_id=id(capture),
        _capture_ref=capture,
        _capture_stream_ref=stream,
    )
    capture.sample_rate = 48_000

    def synchronized(channels, sample_count):
        timeline = np.arange(sample_count, dtype=np.float64) / 48_000.0
        amplitudes = {
            47: 10.0 ** (-12.0 / 20.0),
            48: 10.0 ** (-11.0 / 20.0),
        }
        return {
            "buffers": {
                channel: amplitudes.get(channel, 0.01)
                * np.sin(2.0 * np.pi * 997.0 * timeline)
                for channel in channels
            },
            "captured_sample_count": max(sample_count, 100_000),
        }

    capture.get_synchronized_buffers.side_effect = synchronized


def _unit_audio_capture_progress_statuses():
    base = {
        "running": True,
        "real_input": True,
        "source_type": "sounddevice",
        "callback_age_sec": 0.01,
        "device_name": "Unit WING USB 48ch",
        "device_index": 7,
        "num_channels": 48,
        "sample_rate": 48000,
    }
    return [
        {**base, "captured_sample_count": 1024},
        {**base, "captured_sample_count": 1024},
        {**base, "captured_sample_count": 1024},
        {**base, "captured_sample_count": 2048},
    ]


def _verified_manual_apply_result(**overrides):
    return {
        "status": "applied",
        "success": True,
        "no_console_send_proven": False,
        "physical_verified": True,
        "wal_terminal": "committed",
        "audio_recheck_required": False,
        "audio_recheck_disposition": "not_required_by_manual_write_contract",
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        **overrides,
    }


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("blocker_kind", "expected_blocker"),
    [
        ("cp05", "m110_cp05_hardware_gate_closed"),
        ("source", "m110_live_source_snapshot_drift"),
        ("stop", "autopilot_stop_file_present"),
        ("lease", "autopilot_writer_lease_present"),
    ],
)
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_m110_preconnect_global_rails_block_before_osc_construction(
    mock_config,
    mock_bleed,
    tmp_path,
    blocker_kind,
    expected_blocker,
):
    server = AutoMixerServer()
    owner = object()
    source = server._runtime_identity["source"]
    server._runtime_identity["launch_binding"] = {
        "mode": "m110_attended_runner",
        "head": source["head"],
        "tree": source["tree"],
        "tracked_tree_clean": True,
    }
    server._m110_launch_authority = MagicMock()
    server._m110_launch_authority.source_snapshot_status.return_value = (
        False,
        "m110_live_source_snapshot_drift",
    ) if blocker_kind == "source" else (
        True,
        "m110_live_source_snapshot_current",
    )
    server._ws_auth_required = True
    server._authenticated_websocket_ids.add(id(owner))
    server.operator_mode = "analyze"
    control_dir = tmp_path / "autopilot"
    server._m110_autopilot_stop_path = control_dir / "STOP"
    server._m110_autopilot_writer_lease_path = (
        control_dir / "WRITER_LEASE" / "owner.json"
    )
    server._m110_autopilot_writer_lease_dir = (
        server._m110_autopilot_writer_lease_path.parent
    )
    server._m110_legacy_autoloop_lock_path = control_dir / ".codex_autoloop.lock"
    if blocker_kind == "stop":
        server._m110_autopilot_stop_path.parent.mkdir(parents=True)
        server._m110_autopilot_stop_path.touch()
    if blocker_kind == "lease":
        server._m110_autopilot_writer_lease_dir.mkdir(parents=True)
    server.send_to_client = AsyncMock()
    gate = (
        patch(
            "server.verify_m110_cp05_hardware_gate",
            side_effect=RuntimeError("m110_cp05_hardware_gate_closed"),
        )
        if blocker_kind == "cp05"
        else patch(
            "server.verify_m110_cp05_hardware_gate",
            return_value={"verified": True},
        )
    )

    with gate, patch("server.EnhancedOSCClient") as client_type:
        result = await server.connect_wing(
            "192.168.1.102",
            send_port=2223,
            receive_port=2223,
            observation_only=True,
            record_telemetry=False,
            owner_websocket=owner,
        )

    assert result is False
    client_type.assert_not_called()
    payload = server.send_to_client.await_args.args[1]
    assert payload["reason"] == "m110_preconnect_safety_gate_failed"
    assert expected_blocker in payload["blockers"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message_type", "payload", "response_type"),
    [
        (
            "start_audio_capture_preview",
            {
                "device_id": 7,
                "num_channels": 48,
                "sample_rate": 48000,
                "buffer_seconds": 8.0,
            },
            "audio_capture_status",
        ),
        ("scan_mixers", {"full_scan": True}, "m110_protocol_status"),
        ("auto_connect", {}, "m110_protocol_status"),
        ("connect_dlive", {}, "m110_protocol_status"),
    ],
)
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_m110_ingress_blocks_resource_handlers_before_cp05_or_allowlist(
    mock_config,
    mock_bleed,
    message_type,
    payload,
    response_type,
):
    server = AutoMixerServer()
    owner = object()
    server._runtime_identity["launch_binding"] = {
        "mode": "m110_attended_runner",
    }
    server._ws_auth_required = True
    server._authenticated_websocket_ids.add(id(owner))
    server._verify_m110_cp05_hardware_gate = MagicMock(
        side_effect=RuntimeError("m110_cp05_hardware_gate_closed")
    )
    resource_handler = AsyncMock()
    server._dispatch[message_type] = resource_handler
    server.send_to_client = AsyncMock()

    await server.handle_client_message(
        owner,
        json.dumps({"type": message_type, **payload}),
    )

    resource_handler.assert_not_awaited()
    response = server.send_to_client.await_args.args[1]
    assert response["type"] == response_type
    assert response["status"] == "blocked"
    if message_type == "start_audio_capture_preview":
        assert "m110_cp05_hardware_gate_closed" in response["blockers"]
    else:
        assert "m110_message_type_not_allowlisted" in response["blockers"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message_type",
    ["stop_audio_capture_preview", "disconnect_wing", "disconnect_mixer"],
)
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_m110_ingress_cleanup_remains_available_after_gate_closes(
    mock_config,
    mock_bleed,
    message_type,
):
    server = AutoMixerServer()
    owner = object()
    server._runtime_identity["launch_binding"] = {
        "mode": "m110_attended_runner",
    }
    server._ws_auth_required = True
    server._authenticated_websocket_ids.add(id(owner))
    server._verify_m110_cp05_hardware_gate = MagicMock(
        side_effect=RuntimeError("m110_cp05_hardware_gate_closed")
    )
    cleanup_handler = AsyncMock()
    server._dispatch[message_type] = cleanup_handler

    await server.handle_client_message(
        owner,
        json.dumps({"type": message_type}),
    )

    cleanup_handler.assert_awaited_once_with(owner, {"type": message_type})


# ---------------------------------------------------------------------------
# AutoMixerServer initialization tests (with mocks)
# ---------------------------------------------------------------------------

class TestAutoMixerServerInit:

    def test_default_websocket_origins_block_untrusted_browser_pages(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTOMIXER_WS_ALLOWED_ORIGINS", None)
            assert AutoMixerServer._build_websocket_allowed_origins() == (
                None,
                "http://127.0.0.1:3000",
                "http://localhost:3000",
            )

    def test_websocket_origins_accept_only_explicit_exact_http_origins(self):
        with patch.dict(
            os.environ,
            {
                "AUTOMIXER_WS_ALLOWED_ORIGINS": (
                    "https://console.example.test, http://localhost:3000"
                )
            },
            clear=False,
        ):
            assert AutoMixerServer._build_websocket_allowed_origins() == (
                None,
                "http://127.0.0.1:3000",
                "http://localhost:3000",
                "https://console.example.test",
            )

        with patch.dict(
            os.environ,
            {"AUTOMIXER_WS_ALLOWED_ORIGINS": "*"},
            clear=False,
        ), pytest.raises(RuntimeError, match=r"exact http\(s\) origins"):
            AutoMixerServer._build_websocket_allowed_origins()

    @pytest.mark.parametrize(
        "origin",
        [
            "https://*.example.test",
            "https://user@example.test",
            "https://example.test?source=unsafe",
            "https://example.test/path",
            "https://example.test#fragment",
            "https://example.test:99999",
            "http://console.example.test",
            "http://192.168.1.20:3000",
        ],
    )
    def test_websocket_origins_reject_non_origin_url_features(self, origin):
        with patch.dict(
            os.environ,
            {"AUTOMIXER_WS_ALLOWED_ORIGINS": origin},
            clear=False,
        ), pytest.raises(RuntimeError, match=r"exact http\(s\) origins"):
            AutoMixerServer._build_websocket_allowed_origins()

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
        assert server._ws_max_clients == 10

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_custom_host_port(self, mock_config, mock_bleed):
        """Direct non-loopback WebSocket binds are refused even with a token."""
        with patch.dict(os.environ, {
            "AUTOMIXER_WS_AUTH_TOKEN": "r" * 43,
            "AUTOMIXER_WS_CLIENT_ID": "automixer-ui",
        }):
            with pytest.raises(RuntimeError, match="Non-loopback WebSocket binds are disabled"):
                AutoMixerServer(ws_host="0.0.0.0", ws_port=9999)
            assert "AUTOMIXER_WS_AUTH_TOKEN" not in os.environ

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_loopback_with_token_requires_auth_and_scrubs_plaintext_environment(
        self,
        mock_config,
        mock_bleed,
    ):
        token = "e" * 43
        with patch.dict(
            os.environ,
            {
                "AUTOMIXER_WS_AUTH_TOKEN": token,
                "AUTOMIXER_WS_CLIENT_ID": "automixer-ui",
            },
        ):
            server = AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)
            assert "AUTOMIXER_WS_AUTH_TOKEN" not in os.environ

        assert server._ws_auth_required is True
        assert token not in server.auth_manager._tokens
        assert token not in repr(server.auth_manager._tokens)

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_loopback_consumes_launch_token_from_one_shot_inherited_fd(
        self,
        mock_config,
        mock_bleed,
    ):
        token = "p" * 43
        read_fd, write_fd = os.pipe()
        os.write(write_fd, token.encode("ascii"))
        os.close(write_fd)
        with patch.dict(
            os.environ,
            {"AUTOMIXER_WS_AUTH_TOKEN_FD": str(read_fd)},
            clear=False,
        ):
            os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
            server = AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)
            assert "AUTOMIXER_WS_AUTH_TOKEN_FD" not in os.environ

        assert server._ws_auth_required is True
        assert token not in repr(server.auth_manager._tokens)
        with pytest.raises(OSError):
            os.fstat(read_fd)

    def test_websocket_launch_token_rejects_regular_file_descriptor(
        self,
        tmp_path,
    ):
        path = tmp_path / "token.txt"
        path.write_text("p" * 43, encoding="ascii")
        fd = os.open(path, os.O_RDONLY)
        with patch.dict(
            os.environ,
            {"AUTOMIXER_WS_AUTH_TOKEN_FD": str(fd)},
            clear=False,
        ):
            os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
            with pytest.raises(RuntimeError, match="one-shot pipe"):
                server_module._consume_configured_ws_auth_token()

        with pytest.raises(OSError):
            os.fstat(fd)

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_m110_rejects_plaintext_environment_websocket_token(
        self,
        mock_config,
        mock_bleed,
    ):
        authority = MagicMock()
        with patch.dict(
            os.environ,
            {"AUTOMIXER_WS_AUTH_TOKEN": "p" * 43},
            clear=False,
        ), patch(
            "server.consume_m110_attended_launch_authority",
            return_value=({"mode": "m110_attended_runner"}, authority),
        ):
            os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN_FD", None)
            with pytest.raises(RuntimeError, match="separate inherited pipe"):
                AutoMixerServer()

        authority.close.assert_called_once_with()

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_m110_rejects_shared_authority_and_websocket_token_fd(
        self,
        mock_config,
        mock_bleed,
    ):
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        with patch.dict(
            os.environ,
            {
                "AUTOMIXER_WS_AUTH_TOKEN_FD": str(read_fd),
                "AUTOMIXER_M110_AUTHORITY_FD": str(read_fd),
            },
            clear=False,
        ):
            with pytest.raises(RuntimeError, match="distinct inherited pipes"):
                AutoMixerServer()

        with pytest.raises(OSError):
            os.fstat(read_fd)

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_electron_fd_launcher_token_remains_valid_for_process_lifetime(
        self,
        mock_config,
        mock_bleed,
    ):
        token = "l" * 43
        read_fd, write_fd = os.pipe()
        os.write(write_fd, token.encode("ascii"))
        os.close(write_fd)
        with patch.dict(
            os.environ,
            {
                "AUTOMIXER_WS_AUTH_TOKEN_FD": str(read_fd),
                "AUTOMIXER_WS_TOKEN_PROCESS_LIFETIME": "true",
                "AUTOMIXER_PACKAGED_APP": "true",
                "AUTOMIXER_WS_CLIENT_ID": "automixer-ui",
            },
            clear=False,
        ):
            os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
            server = AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)

        websocket = MagicMock()
        websocket.recv = AsyncMock(return_value=json.dumps({
            "type": "authenticate",
            "token": token,
            "client_id": "automixer-ui",
        }))
        websocket.close = AsyncMock()

        assert await server._authenticate_websocket(websocket) is True
        assert server._ws_token_process_lifetime is True
        assert id(websocket) not in server._authenticated_websocket_expiry
        websocket.close.assert_not_awaited()

    def test_launch_token_pipe_deadline_covers_partial_payload_without_eof(self):
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b"partial")
        started = time.monotonic()
        try:
            with (
                patch.dict(
                    os.environ,
                    {"AUTOMIXER_WS_AUTH_TOKEN_FD": str(read_fd)},
                    clear=False,
                ),
                patch.object(
                    server_module,
                    "WS_AUTH_TOKEN_PIPE_TIMEOUT_SECONDS",
                    0.02,
                ),
            ):
                os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
                with pytest.raises(RuntimeError, match="Timed out"):
                    server_module._consume_configured_ws_auth_token()
        finally:
            os.close(write_fd)

        assert time.monotonic() - started < 0.5
        with pytest.raises(OSError):
            os.fstat(read_fd)

    def test_repo_dotenv_cannot_activate_plaintext_control_token(
        self,
        tmp_path,
    ):
        (tmp_path / ".env").write_text(
            "AUTOMIXER_WS_AUTH_TOKEN=" + ("d" * 43) + "\n",
            encoding="utf-8",
        )
        with (
            patch.object(server_module, "_REPO_ROOT", str(tmp_path)),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
            server_module._load_repo_dotenv()
            assert "AUTOMIXER_WS_AUTH_TOKEN" not in os.environ

    def test_m005_bound_backend_ignores_all_untracked_repo_dotenv_inputs(
        self,
        tmp_path,
    ):
        poisoned = {
            "AUTOMIXER_OPERATOR_MODE": "auto",
            "AUTOMIXER_AUTO_MODE_UNLOCKED": "true",
            "AUTOMIXER_WING_OBSERVATION_MODE": "false",
            "AUTOMIXER_WING_DEPLOYMENT_WRITE_BOUNDARY_ENABLED": "false",
            "AUTOMIXER_WING_REPLAY_READINESS_DIR": "/tmp/poisoned-readiness",
        }
        (tmp_path / ".env").write_text(
            "".join(f"{name}={value}\n" for name, value in poisoned.items()),
            encoding="utf-8",
        )
        with (
            patch.object(server_module, "_REPO_ROOT", str(tmp_path)),
            patch.dict(
                os.environ,
                {"AUTOMIXER_M005_LAUNCH_NONCE": "a" * 32},
                clear=False,
            ),
        ):
            for name in poisoned:
                os.environ.pop(name, None)
            server_module._load_repo_dotenv()
            assert all(name not in os.environ for name in poisoned)

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_loopback_rejects_ambiguous_environment_and_fd_token_sources(
        self,
        mock_config,
        mock_bleed,
    ):
        read_fd, write_fd = os.pipe()
        try:
            with patch.dict(
                os.environ,
                {
                    "AUTOMIXER_WS_AUTH_TOKEN": "a" * 43,
                    "AUTOMIXER_WS_AUTH_TOKEN_FD": str(read_fd),
                },
                clear=False,
            ):
                with pytest.raises(RuntimeError, match="exactly one WebSocket token"):
                    AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_loopback_without_token_remains_cli_compatible(
        self,
        mock_config,
        mock_bleed,
    ):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
            server = AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)

        assert server._ws_auth_required is False

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_runtime_identity_binds_process_nonce_and_exact_source_backend(
        self,
        mock_config,
        mock_bleed,
    ):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
            server = AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)
        identity = server.get_runtime_identity_status()

        assert identity["schema"] == RUNTIME_IDENTITY_SCHEMA
        assert identity["runtime_kind"] == "source"
        assert len(identity["instance_nonce"]) == 32
        assert identity["process"]["instance_nonce"] == identity["instance_nonce"]
        assert identity["process"]["started_at_utc"] == identity["started_at_utc"]
        assert identity["process"]["pid"] == os.getpid()
        assert identity["websocket"] == {
            "host": "127.0.0.1",
            "port": 8765,
            "loopback_only": True,
        }
        assert identity["read_only_identity"] is True
        assert identity["live_authority_granted"] is False
        assert identity["launch_binding"]["mode"] == "unbound"
        assert identity["launch_binding"]["proof_eligible"] is False
        assert [row["path"] for row in identity["code"]["files"]] == sorted(
            RUNTIME_CRITICAL_SOURCE_PATHS
        )
        assert identity["source"]["head"]
        assert identity["source"]["tree"]
        assert identity["manifest_sha256"] == runtime_identity_manifest_sha256(identity)
        server_path = Path(__file__).resolve().parents[1] / "backend" / "server.py"
        server_bytes = server_path.read_bytes()
        assert identity["backend"] == {
            "path": "backend/server.py",
            "exists": True,
            "size": len(server_bytes),
            "sha256": hashlib.sha256(server_bytes).hexdigest(),
        }

        server.send_to_client = AsyncMock()
        websocket = object()
        await server._dispatch["get_runtime_identity"](
            websocket,
            {"challenge_nonce": "b" * 32},
        )
        challenged = {**identity, "request_challenge_nonce": "b" * 32}
        server.send_to_client.assert_awaited_once_with(websocket, challenged)

        mutated = server.get_runtime_identity_status()
        mutated["process"]["pid"] = -1
        assert server.get_runtime_identity_status()["process"]["pid"] == os.getpid()

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_runtime_identity_accepts_only_complete_preprocess_launch_binding(
        self,
        mock_config,
        mock_bleed,
    ):
        binding = {
            "AUTOMIXER_M005_LAUNCH_NONCE": "c" * 32,
            "AUTOMIXER_M005_PRESTART_PROVENANCE_SCHEMA": (
                "automixer.m005_current_release_provenance.v1"
            ),
            "AUTOMIXER_M005_PRESTART_PROVENANCE_SHA256": "d" * 64,
            "AUTOMIXER_M005_PRESTART_PROVENANCE_SIZE": "1234",
        }
        with patch.dict(os.environ, binding, clear=False):
            server = AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)

        assert server._runtime_identity["launch_binding"] == {
            "mode": "proof_runner",
            "launch_nonce": "c" * 32,
            "prestart_provenance_schema": (
                "automixer.m005_current_release_provenance.v1"
            ),
            "prestart_provenance_size": 1234,
            "prestart_provenance_sha256": "d" * 64,
            "bound_before_process_start": True,
            "proof_eligible": True,
        }
        for key in binding:
            assert key not in os.environ

        invalid = dict(binding)
        invalid["AUTOMIXER_M005_LAUNCH_NONCE"] = "not-a-nonce"
        with patch.dict(os.environ, invalid, clear=False):
            with pytest.raises(RuntimeError, match="Invalid M005 process-start"):
                AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_remote_bind_without_token_fails_closed(self, mock_config, mock_bleed):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
            with pytest.raises(RuntimeError, match="Non-loopback WebSocket binds are disabled"):
                AutoMixerServer(ws_host="0.0.0.0", ws_port=9999)

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_token_authenticated_loopback_websocket_authenticates_first_frame(self, mock_config, mock_bleed):
        token = "r" * 43
        with patch.dict(os.environ, {
            "AUTOMIXER_WS_AUTH_TOKEN": token,
            "AUTOMIXER_WS_CLIENT_ID": "automixer-ui",
        }):
            server = AutoMixerServer(ws_host="127.0.0.1", ws_port=9999)
        websocket = MagicMock()
        websocket.recv = AsyncMock(return_value=json.dumps({
            "type": "authenticate",
            "token": token,
            "client_id": "automixer-ui",
        }))
        websocket.send = AsyncMock()
        websocket.close = AsyncMock()

        assert await server._authenticate_websocket(websocket) is True
        websocket.close.assert_not_awaited()
        sent = json.loads(websocket.send.await_args.args[0])
        assert sent == {"type": "authentication_status", "authenticated": True}

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_token_authenticated_loopback_websocket_rejects_bad_token(self, mock_config, mock_bleed):
        with patch.dict(os.environ, {
            "AUTOMIXER_WS_AUTH_TOKEN": "r" * 43,
            "AUTOMIXER_WS_CLIENT_ID": "automixer-ui",
        }):
            server = AutoMixerServer(ws_host="127.0.0.1", ws_port=9999)
        websocket = MagicMock()
        websocket.recv = AsyncMock(return_value=json.dumps({
            "type": "authenticate",
            "token": "wrong-token",
            "client_id": "automixer-ui",
        }))
        websocket.close = AsyncMock()

        assert await server._authenticate_websocket(websocket) is False
        websocket.close.assert_awaited_once_with(code=1008, reason="authentication_failed")

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_websocket_auth_rejects_non_string_token_without_exception(
        self,
        mock_config,
        mock_bleed,
    ):
        with patch.dict(os.environ, {
            "AUTOMIXER_WS_AUTH_TOKEN": "r" * 43,
            "AUTOMIXER_WS_CLIENT_ID": "automixer-ui",
        }):
            server = AutoMixerServer(ws_host="127.0.0.1", ws_port=9999)
        websocket = MagicMock()
        websocket.recv = AsyncMock(return_value=json.dumps({
            "type": "authenticate",
            "token": ["not", "a", "token"],
            "client_id": "automixer-ui",
        }))
        websocket.close = AsyncMock()

        assert await server._authenticate_websocket(websocket) is False
        websocket.close.assert_awaited_once_with(code=1008, reason="authentication_failed")

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_websocket_auth_attempt_bucket_limits_invalid_token_flood(
        self,
        mock_config,
        mock_bleed,
    ):
        with patch.dict(os.environ, {
            "AUTOMIXER_WS_AUTH_TOKEN": "r" * 43,
            "AUTOMIXER_WS_CLIENT_ID": "automixer-ui",
        }):
            server = AutoMixerServer(ws_host="127.0.0.1", ws_port=9999)
        exhausted = server_module.TokenBucket(1e-9, 1)
        exhausted._tokens = 0.0
        server._ws_auth_attempt_bucket = exhausted
        websocket = MagicMock()
        websocket.recv = AsyncMock()
        websocket.close = AsyncMock()

        assert await server._authenticate_websocket(websocket) is False
        websocket.recv.assert_not_awaited()
        websocket.close.assert_awaited_once_with(
            code=1013,
            reason="authentication_rate_limited",
        )

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_loopback_token_authenticates_before_client_registration(
        self,
        mock_config,
        mock_bleed,
    ):
        with patch.dict(
            os.environ,
            {
                "AUTOMIXER_WS_AUTH_TOKEN": "e" * 43,
                "AUTOMIXER_WS_CLIENT_ID": "automixer-ui",
            },
        ):
            server = AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)
        websocket = MagicMock()
        websocket.recv = AsyncMock(
            return_value=json.dumps(
                {
                    "type": "authenticate",
                    "token": "e" * 43,
                    "client_id": "automixer-ui",
                }
            )
        )
        websocket.send = AsyncMock()
        websocket.close = AsyncMock()

        assert await server._authenticate_websocket(websocket) is True
        assert json.loads(websocket.send.await_args.args[0]) == {
            "type": "authentication_status",
            "authenticated": True,
        }

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_loopback_auth_rejects_weak_token(self, mock_config, mock_bleed):
        with patch.dict(os.environ, {
            "AUTOMIXER_WS_AUTH_TOKEN": "too-short",
            "AUTOMIXER_WS_CLIENT_ID": "automixer-ui",
        }):
            with pytest.raises(RuntimeError, match="43-character base64url"):
                AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)

    @pytest.mark.parametrize("ttl", ["nan", "inf", "0", "0.01", "24.1", "invalid"])
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_loopback_auth_rejects_unbounded_ttl(
        self,
        mock_config,
        mock_bleed,
        ttl,
    ):
        with patch.dict(os.environ, {
            "AUTOMIXER_WS_AUTH_TOKEN": "t" * 43,
            "AUTOMIXER_WS_CLIENT_ID": "automixer-ui",
            "AUTOMIXER_WS_TOKEN_TTL_HOURS": ttl,
        }):
            with pytest.raises(RuntimeError, match="1 minute to 24 hours"):
                AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_websocket_auth_rejects_missing_client_id(self, mock_config, mock_bleed):
        token = "t" * 43
        with patch.dict(os.environ, {
            "AUTOMIXER_WS_AUTH_TOKEN": token,
            "AUTOMIXER_WS_CLIENT_ID": "automixer-ui",
        }):
            server = AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)
        websocket = MagicMock()
        websocket.recv = AsyncMock(return_value=json.dumps({
            "type": "authenticate",
            "token": token,
        }))
        websocket.close = AsyncMock()

        assert await server._authenticate_websocket(websocket) is False
        websocket.close.assert_awaited_once_with(
            code=1008,
            reason="authentication_failed",
        )

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_authenticated_websocket_closes_at_session_expiry(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)
        websocket = MagicMock()
        websocket.close = AsyncMock()
        expires_at = time.time() - 1
        server._authenticated_websocket_expiry[id(websocket)] = expires_at

        await server._expire_authenticated_websocket(websocket, expires_at)

        websocket.close.assert_awaited_once_with(
            code=1008,
            reason="authentication_expired",
        )

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_disconnect_stops_only_websocket_owned_audio_preview(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)
        websocket = object()
        capture = MagicMock()
        capture.running = True
        server.audio_capture = capture
        server._audio_capture_preview_owner = websocket
        server.connected_clients.add(websocket)

        await server.unregister_client(websocket)

        capture.stop.assert_called_once_with()
        assert server.audio_capture is None
        assert server._audio_capture_preview_owner is None

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_ws_port_cleanup_requires_explicit_operator_opt_in(self, mock_config, mock_bleed):
        """Port cleanup is destructive and must not be enabled by default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTOMIXER_FORCE_CLEANUP_WS_PORT", None)
            server = AutoMixerServer(ws_host="localhost", ws_port=8765)
            assert server._ws_port_force_cleanup_enabled() is False

            os.environ["AUTOMIXER_FORCE_CLEANUP_WS_PORT"] = "true"
            assert server._ws_port_force_cleanup_enabled() is True

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_ws_port_pid_lookup_uses_argv_without_shell(self, mock_config, mock_bleed):
        """The lsof probe must not invoke a shell or interpolate the port into a command string."""
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return subprocess.CompletedProcess(args, 0, stdout="123\nnot-a-pid\n456\n", stderr="")

        with patch("server.subprocess.run", side_effect=fake_run):
            pids = server._pids_listening_on_port("8765")

        assert pids == [123, 456]
        assert captured["args"] == ["lsof", "-ti", ":8765"]
        assert "shell" not in captured["kwargs"]
        assert captured["kwargs"]["check"] is False
        assert captured["kwargs"]["capture_output"] is True
        assert captured["kwargs"]["text"] is True
        assert captured["kwargs"]["timeout"] == 2.0

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_ws_port_pid_lookup_times_out_fail_closed(self, mock_config, mock_bleed):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)

        with patch(
            "server.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["lsof"], timeout=2.0),
        ):
            assert server._pids_listening_on_port("8765") == []

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_ws_port_force_cleanup_skips_current_pid(self, mock_config, mock_bleed):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)

        with (
            patch.object(server, "_pids_listening_on_port", return_value=[os.getpid(), 222, 222]),
            patch("server.os.kill") as mock_kill,
        ):
            server._force_cleanup_ws_port()

        mock_kill.assert_called_once_with(222, signal.SIGKILL)

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_start_refuses_automatic_port_kill_without_opt_in(self, mock_config, mock_bleed):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        error = OSError()
        error.errno = errno.EADDRINUSE

        with (
            patch.dict(os.environ, {"AUTOMIXER_FORCE_CLEANUP_WS_PORT": "false"}, clear=False),
            patch("server.websockets.serve", side_effect=error),
            pytest.raises(RuntimeError, match="Refusing automatic kill"),
        ):
            await server.start()

    def test_entrypoint_returns_nonzero_after_fatal_startup_error(self):
        fake_server = MagicMock()
        fake_server.start.return_value = object()

        with (
            patch.dict(
                os.environ,
                {
                    "AUTOMIXER_WS_HOST": "127.0.0.1",
                    "AUTOMIXER_WS_PORT": "8765",
                },
                clear=False,
            ),
            patch("server.atexit.register"),
            patch("server.AutoMixerServer", return_value=fake_server),
            patch("server.asyncio.run", side_effect=RuntimeError("fatal bind failure")),
        ):
            assert server_module.main() == 1

        fake_server.cleanup_all_controllers.assert_called_once_with()
        assert server_module._server_instance is None

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_start_passes_fail_closed_origins_to_websocket_server(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        fake_server = MagicMock()
        fake_server.close = MagicMock()
        fake_server.wait_closed = AsyncMock()
        captured = {}

        async def fake_serve(*args, **kwargs):
            captured.update(kwargs)
            server._shutdown_event.set()
            return fake_server

        with patch("server.websockets.serve", side_effect=fake_serve):
            await server.start()

        assert captured["origins"] == server._ws_allowed_origins
        assert "https://evil.example" not in captured["origins"]
        assert captured["max_size"] == 256 * 1024
        assert captured["max_queue"] == 16
        fake_server.close.assert_called_once_with()

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_start_retry_preserves_websocket_transport_bounds(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        bind_error = OSError()
        bind_error.errno = errno.EADDRINUSE
        fake_server = MagicMock()
        fake_server.close = MagicMock()
        fake_server.wait_closed = AsyncMock()
        calls = []

        async def fake_serve(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise bind_error
            server._shutdown_event.set()
            return fake_server

        with (
            patch.dict(
                os.environ,
                {"AUTOMIXER_FORCE_CLEANUP_WS_PORT": "true"},
                clear=False,
            ),
            patch.object(server, "_force_cleanup_ws_port"),
            patch("server.asyncio.sleep", new=AsyncMock()),
            patch("server.websockets.serve", side_effect=fake_serve),
        ):
            await server.start()

        assert len(calls) == 2
        for kwargs in calls:
            assert kwargs["origins"] == server._ws_allowed_origins
            assert kwargs["max_size"] == 256 * 1024
            assert kwargs["max_queue"] == 16
        fake_server.close.assert_called_once_with()

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


_INGRESS_END = object()


class ControlledIngressWebSocket:
    def __init__(self, messages=()):
        self._incoming = asyncio.Queue()
        for message in messages:
            self._incoming.put_nowait(message)
        self._incoming.put_nowait(_INGRESS_END)
        self.sent = []
        self.closed = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self._incoming.get()
        if message is _INGRESS_END:
            raise StopAsyncIteration
        return message

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self, *, code, reason):
        self.closed.append((code, reason))


class StreamingIngressWebSocket(ControlledIngressWebSocket):
    def __init__(self):
        self._incoming = asyncio.Queue()
        self.sent = []
        self.closed = []

    async def feed(self, message):
        await self._incoming.put(message)

    async def finish(self):
        await self._incoming.put(_INGRESS_END)


class TestWebSocketIngressSafety:

    @patch("server.BleedService")
    @patch.object(
        AutoMixerServer,
        "_load_config",
        return_value={"websocket": {"max_clients": 2}},
    )
    def test_configured_max_clients_is_loaded(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        assert server._ws_max_clients == 2

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_client_cap_includes_authentication_in_progress(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        server._ws_max_clients = 1
        authenticating = object()
        rejected = ControlledIngressWebSocket()

        assert server._reserve_websocket_admission(authenticating) is True
        await server.handler(rejected)

        assert rejected.closed == [(1013, "server_capacity")]
        assert rejected not in server.connected_clients
        server._release_websocket_admission(authenticating)

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_pending_authentication_has_separate_small_cap(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        server._ws_max_clients = 10
        pending = [object(), object(), object()]

        assert server._reserve_websocket_admission(pending[0]) is True
        assert server._reserve_websocket_admission(pending[1]) is True
        assert server._reserve_websocket_admission(pending[2]) is False

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_twentieth_rate_reject_closes_with_try_again_later(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        exhausted = server_module.TokenBucket(1e-9, 100)
        exhausted._tokens = 0.0
        server._ws_global_ingress_bucket = exhausted
        websocket = ControlledIngressWebSocket(
            ['{"type":"normal"}'] * 20
        )

        await server.handler(websocket)

        assert websocket.closed == [(1013, "ingress_overload")]
        assert websocket not in server.connected_clients

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_normal_lane_preserves_fifo_handler_order(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        websocket = StreamingIngressWebSocket()
        all_seen = asyncio.Event()
        events = []

        async def handle_message(_websocket, raw):
            message_type = json.loads(raw)["type"]
            if message_type == "one":
                await asyncio.sleep(0)
            events.append(message_type)
            if len(events) == 3:
                all_seen.set()

        server.handle_client_message = AsyncMock(side_effect=handle_message)
        handler_task = asyncio.create_task(server.handler(websocket))

        for message_type in ("one", "two", "three"):
            await websocket.feed(json.dumps({"type": message_type}))
        await asyncio.wait_for(all_seen.wait(), timeout=1)
        await websocket.finish()
        await asyncio.wait_for(handler_task, timeout=1)

        assert events == ["one", "two", "three"]

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_disconnect_does_not_cancel_inflight_normal_handler(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        websocket = StreamingIngressWebSocket()
        started = asyncio.Event()
        release = asyncio.Event()
        completed = asyncio.Event()
        cancelled = False

        async def handle_message(_websocket, _raw):
            nonlocal cancelled
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled = True
                raise
            completed.set()

        server.handle_client_message = AsyncMock(side_effect=handle_message)
        handler_task = asyncio.create_task(server.handler(websocket))
        await websocket.feed('{"type":"slow_transaction"}')
        await asyncio.wait_for(started.wait(), timeout=1)

        await websocket.finish()
        await asyncio.sleep(0)
        assert handler_task.done() is False
        assert cancelled is False

        release.set()
        await asyncio.wait_for(handler_task, timeout=1)

        assert completed.is_set()
        assert cancelled is False

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_emergency_fast_lane_preempts_and_invalidates_stale_fifo(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        websocket = StreamingIngressWebSocket()
        slow_started = asyncio.Event()
        release_slow = asyncio.Event()
        emergency_seen = asyncio.Event()
        after_seen = asyncio.Event()
        events = []

        async def handle_message(_websocket, raw):
            message_type = json.loads(raw)["type"]
            if message_type == "slow":
                events.append("slow-start")
                slow_started.set()
                await release_slow.wait()
                events.append("slow-end")
            elif message_type == "emergency_stop":
                server._manual_write_emergency_stopped = True
                server._manual_write_emergency_generation += 1
                events.append("emergency")
                emergency_seen.set()
            elif message_type == "after":
                events.append("after")
                after_seen.set()
            else:
                events.append(message_type)

        server.handle_client_message = AsyncMock(side_effect=handle_message)
        handler_task = asyncio.create_task(server.handler(websocket))

        await websocket.feed('{"type":"slow"}')
        await asyncio.wait_for(slow_started.wait(), timeout=1)
        await websocket.feed('{"type":"queued-before-stop"}')
        await websocket.feed('{"type":"emergency_stop"}')
        await asyncio.wait_for(emergency_seen.wait(), timeout=1)
        release_slow.set()
        await websocket.feed('{"type":"after"}')
        await asyncio.wait_for(after_seen.wait(), timeout=1)
        await websocket.finish()
        await asyncio.wait_for(handler_task, timeout=1)

        assert events == ["slow-start", "emergency", "slow-end", "after"]
        assert server._ws_ingress_emergency_generation == 1
        assert websocket not in server.connected_clients

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_emergency_bypasses_exhausted_global_bucket(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        exhausted = server_module.TokenBucket(1e-9, 100)
        exhausted._tokens = 0.0
        server._ws_global_ingress_bucket = exhausted
        emergency_seen = asyncio.Event()

        async def handle_message(_websocket, raw):
            assert json.loads(raw)["type"] == "emergency_stop_agent"
            server._manual_write_emergency_stopped = True
            emergency_seen.set()

        server.handle_client_message = AsyncMock(side_effect=handle_message)
        websocket = StreamingIngressWebSocket()
        handler_task = asyncio.create_task(server.handler(websocket))

        await websocket.feed('{"type":"emergency_stop_agent"}')
        await asyncio.wait_for(emergency_seen.wait(), timeout=1)
        await websocket.finish()
        await asyncio.wait_for(handler_task, timeout=1)

        server.handle_client_message.assert_awaited_once()
        assert websocket.closed == []

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_expired_session_drops_buffered_control_before_dispatch(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        websocket = StreamingIngressWebSocket()
        server._authenticated_websocket_expiry[id(websocket)] = time.time() - 1
        server.handle_client_message = AsyncMock()

        handler_task = asyncio.create_task(server.handler(websocket))
        await websocket.feed('{"type":"set_operator_mode","mode":"supervised"}')
        await asyncio.wait_for(handler_task, timeout=1)

        server.handle_client_message.assert_not_awaited()
        assert websocket.closed == [(1008, "authentication_expired")]

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_expired_tombstone_drops_every_already_queued_control_frame(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        websocket = StreamingIngressWebSocket()
        server._authenticated_websocket_expiry[id(websocket)] = time.time() - 1
        server.handle_client_message = AsyncMock()
        queue = asyncio.Queue()
        for payload in (
            '{"type":"set_operator_mode","mode":"supervised"}',
            '{"type":"set_operator_mode","mode":"auto"}',
        ):
            queue.put_nowait(server_module.QueuedIngressMessage(
                payload=payload,
                emergency_generation=server._ws_ingress_emergency_generation,
            ))

        worker = asyncio.create_task(
            server._run_websocket_normal_lane(websocket, queue)
        )
        await asyncio.wait_for(queue.join(), timeout=1)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker

        server.handle_client_message.assert_not_awaited()
        assert websocket.closed == [(1008, "authentication_expired")]
        assert id(websocket) in server._authenticated_websocket_expiry


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
        server.record_master_true_peak_evidence(-6.0)

        status = server.get_ready_for_live_status()

        assert status["ready"] is True
        assert status["checks"]["auto_fader_available"] is True

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_ready_for_live_blocks_without_fresh_master_true_peak(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.mixer_client = MagicMock(is_connected=True)
        server.gain_staging = MagicMock()
        server.auto_fader_controller = MagicMock()
        server.auto_fader_controller.get_status.return_value = {"collecting": True}

        status = server.get_ready_for_live_status()

        assert status["ready"] is False
        assert status["checks"]["no_clipping"] is False
        assert "master_true_peak_unavailable_or_stale" in status["blocked_reasons"]

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_supervised_audio_recheck_measures_master_47_48_with_4x_true_peak(
        self,
        mock_config,
        mock_bleed,
        tmp_path,
    ):
        server = AutoMixerServer()
        _configure_unit_master_route_proof(
            server,
            tmp_path,
            "unit-route-proof",
        )
        expected_route = {
            "/$syscfg/$serial": "WING-UNIT",
            "/$syscfg/$firmware": "3.0.5",
            "/io/out/USB/47/grp": "MAIN",
            "/io/out/USB/47/in": 0,
            "/io/out/USB/48/grp": "MAIN",
            "/io/out/USB/48/in": 1,
        }
        server.mixer_client.query_parameter.side_effect = (
            lambda address, timeout: {
                "status": "received",
                "address": address,
                "value": expected_route[address],
                "confirmed_round_trips": 2,
            }
        )
        server.audio_capture = MagicMock(running=True, num_channels=48)
        server.audio_capture.get_status.side_effect = (
            _unit_audio_capture_progress_statuses()
        )
        server.audio_capture.get_peak.side_effect = lambda channel: {47: -12.0, 48: -11.0}[channel]
        _bind_unit_master_route_capture(server, server.audio_capture)
        operation = MagicMock(family_id="fader", operation_id="fader-1")

        result = await server._supervised_transaction_audio_recheck(operation, {}, {})

        assert result.status == "passed"
        assert result.evidence["master_channels"] == [47, 48]
        assert result.evidence["master_sample_peaks_dbfs"] == [-12.0, -11.0]
        assert result.evidence["master_true_peak_dbtp"] == pytest.approx(
            -11.0,
            abs=0.02,
        )
        assert result.evidence["measurement_source"] == (
            "master_channels_true_peak_4x"
        )
        assert result.evidence["route_proof"]["verified"] is True
        assert result.evidence["physical_route_readback"]["verified"] is True
        assert result.evidence["capture_sample_progressed"] is True

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_m110_feedback_recheck_uses_distinct_fresh_frames(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        capture = MagicMock(
            running=True,
            num_channels=48,
            sample_rate=48_000,
        )
        rng = np.random.default_rng(7)

        def synchronized(channels, sample_count):
            return {
                "buffers": {
                    channel: rng.normal(0.0, 0.001, sample_count)
                    for channel in channels
                },
                "captured_sample_count": 200_000,
            }

        capture.get_synchronized_buffers.side_effect = synchronized
        operation = MagicMock(channel=1, family_id="fader", operation_id="f1")

        result = server._m110_feedback_clear_recheck(
            capture=capture,
            operation=operation,
            additional_channels=(2,),
            minimum_captured_sample_count=150_000,
        )

        assert result.status == "passed"
        assert result.reason == "fresh_feedback_clear_state"
        assert result.evidence["channels"] == [1, 2, 47, 48]
        assert result.evidence["frame_count"] == 12

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_m110_feedback_recheck_blocks_growing_pure_tone(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        capture = MagicMock(
            running=True,
            num_channels=48,
            sample_rate=48_000,
        )
        frame_size = 2048
        frequency = 48_000.0 * 42.0 / frame_size
        timeline = np.arange(frame_size, dtype=np.float64) / 48_000.0
        frames = [
            amplitude
            * np.sin(2.0 * np.pi * frequency * timeline)
            for amplitude in np.linspace(0.02, 0.5, 12)
        ]
        growing_tone = np.concatenate(frames)
        capture.get_synchronized_buffers.return_value = {
            "buffers": {
                1: growing_tone,
                47: growing_tone,
                48: growing_tone,
            },
            "captured_sample_count": 200_000,
        }

        result = server._m110_feedback_clear_recheck(
            capture=capture,
            operation=MagicMock(
                channel=1,
                family_id="fader",
                operation_id="feedback-tone",
            ),
        )

        assert result.status == "failed"
        assert result.reason == "feedback_risk_detected"
        assert result.evidence["feedback_clear"] is False
        assert result.evidence["events"]

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_supervised_audio_recheck_rejects_route_change_after_audio_window(
        self,
        mock_config,
        mock_bleed,
        tmp_path,
    ):
        server = AutoMixerServer()
        route = _configure_unit_master_route_proof(
            server,
            tmp_path,
            "post-audio-route-drift",
        )
        query_count = 0

        def query(address, timeout):
            nonlocal query_count
            query_count += 1
            value = route[address]
            if query_count > 12 and address == "/io/out/USB/47/grp":
                value = "BUS"
            return {
                "status": "received",
                "address": address,
                "value": value,
                "confirmed_round_trips": 2,
            }

        server.mixer_client.query_parameter.side_effect = query
        server.audio_capture = MagicMock(running=True, num_channels=48)
        server.audio_capture.get_status.side_effect = (
            _unit_audio_capture_progress_statuses()
        )
        server.audio_capture.get_peak.side_effect = lambda channel: {
            47: -12.0,
            48: -11.0,
        }[channel]
        _bind_unit_master_route_capture(server, server.audio_capture)

        result = await server._supervised_transaction_audio_recheck(
            MagicMock(family_id="fader", operation_id="post-route-drift"),
            {},
            {},
        )

        assert result.status == "unavailable"
        assert result.reason == "master_capture_route_physical_readback_mismatch"
        assert result.evidence[
            "physical_route_readback_before_audio"
        ]["verified"] is True
        assert result.evidence[
            "physical_route_readback_after_audio"
        ]["verified"] is False
        assert server._latest_master_true_peak_dbtp is None

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_supervised_audio_recheck_uses_active_engine_owned_capture(
        self,
        mock_config,
        mock_bleed,
        tmp_path,
    ):
        server = AutoMixerServer()
        route = _configure_unit_master_route_proof(
            server,
            tmp_path,
            "engine-owned-capture",
        )
        server.mixer_client.query_parameter.side_effect = (
            lambda address, timeout: {
                "status": "received",
                "address": address,
                "value": route[address],
                "confirmed_round_trips": 2,
            }
        )
        capture = MagicMock(running=True, num_channels=48)
        capture.get_status.side_effect = _unit_audio_capture_progress_statuses()
        capture.get_peak.side_effect = lambda channel: {
            47: -12.0,
            48: -11.0,
        }[channel]
        server.audio_capture = None
        server.auto_soundcheck_engine = type(
            "Engine",
            (),
            {"audio_capture": capture},
        )()
        _bind_unit_master_route_capture(server, capture)

        result = await server._supervised_transaction_audio_recheck(
            MagicMock(family_id="fader", operation_id="engine-capture"),
            {},
            {},
        )

        assert result.status == "passed"
        assert result.evidence["capture_owner"] == (
            "active_auto_soundcheck_engine"
        )
        assert result.evidence["master_channels"] == [47, 48]

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_supervised_audio_recheck_rejects_stale_cached_real_input(
        self,
        mock_config,
        mock_bleed,
        tmp_path,
    ):
        server = AutoMixerServer()
        _configure_unit_master_route_proof(
            server,
            tmp_path,
            "stale-input-proof",
        )
        route = {
            "/$syscfg/$serial": "WING-UNIT",
            "/$syscfg/$firmware": "3.0.5",
            "/io/out/USB/47/grp": "MAIN",
            "/io/out/USB/47/in": 0,
            "/io/out/USB/48/grp": "MAIN",
            "/io/out/USB/48/in": 1,
        }
        server.mixer_client.query_parameter.side_effect = (
            lambda address, timeout: {
                "status": "received",
                "address": address,
                "value": route[address],
                "confirmed_round_trips": 2,
            }
        )
        server.audio_capture = MagicMock(running=True, num_channels=48)
        server.audio_capture.get_status.return_value = {
            "running": True,
            "real_input": False,
            "source_type": "sounddevice",
            "callback_age_sec": 60.0,
            "captured_sample_count": 9999,
            "device_name": "Unit WING USB 48ch",
            "device_index": 7,
            "num_channels": 48,
            "sample_rate": 48000,
        }
        _bind_unit_master_route_capture(server, server.audio_capture)
        operation = MagicMock(family_id="fader", operation_id="stale-input")

        result = await server._supervised_transaction_audio_recheck(
            operation,
            {},
            {},
        )

        assert result.status == "unavailable"
        assert result.reason == "master_capture_route_proof_required"
        assert result.evidence["route_verified"] is False
        server.audio_capture.get_peak.assert_not_called()

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_supervised_audio_recheck_rejects_current_usb_route_drift(
        self,
        mock_config,
        mock_bleed,
        tmp_path,
    ):
        server = AutoMixerServer()
        _configure_unit_master_route_proof(
            server,
            tmp_path,
            "route-drift-proof",
        )
        route = {
            "/$syscfg/$serial": "WING-UNIT",
            "/$syscfg/$firmware": "3.0.5",
            "/io/out/USB/47/grp": "BUS",
            "/io/out/USB/47/in": 0,
            "/io/out/USB/48/grp": "MAIN",
            "/io/out/USB/48/in": 1,
        }
        server.mixer_client.query_parameter.side_effect = (
            lambda address, timeout: {
                "status": "received",
                "address": address,
                "value": route[address],
                "confirmed_round_trips": 2,
            }
        )
        server.audio_capture = MagicMock(running=True, num_channels=48)
        server.audio_capture.get_status.return_value = {
            "running": True,
            "real_input": True,
            "source_type": "sounddevice",
            "callback_age_sec": 0.01,
            "captured_sample_count": 2048,
            "device_name": "Unit WING USB 48ch",
            "device_index": 7,
            "num_channels": 48,
            "sample_rate": 48000,
        }
        _bind_unit_master_route_capture(server, server.audio_capture)
        operation = MagicMock(family_id="fader", operation_id="route-drift")

        result = await server._supervised_transaction_audio_recheck(
            operation,
            {},
            {},
        )

        assert result.status == "unavailable"
        assert result.reason == "master_capture_route_physical_readback_mismatch"
        assert result.evidence["route_verified"] is False
        server.audio_capture.get_peak.assert_not_called()

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_supervised_audio_recheck_rejects_env_flag_without_immutable_proof(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        server._master_capture_route_verified = True
        server._master_capture_route_proof_id = "self-asserted-only"
        server.audio_capture = MagicMock(running=True, num_channels=48)
        server.audio_capture.get_peak.side_effect = lambda channel: {
            47: -12.0,
            48: -11.0,
        }[channel]
        operation = MagicMock(family_id="fader", operation_id="fader-1")

        result = await server._supervised_transaction_audio_recheck(
            operation,
            {},
            {},
        )

        assert result.status == "unavailable"
        assert result.reason == "master_capture_route_proof_required"
        assert result.evidence["route_verified"] is False

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_system_emergency_stop_covers_all_active_writer_variants(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.auto_soundcheck_engine = MagicMock()
        legacy_cleanup_finished = asyncio.Event()

        async def legacy_soundcheck():
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0.01)
                legacy_cleanup_finished.set()

        server.auto_soundcheck_task = asyncio.create_task(legacy_soundcheck())
        await asyncio.sleep(0)
        agent = MagicMock()
        server.mixing_agent = agent
        server.multi_channel_auto_eq_controller = MagicMock(spec=["stop_all"])
        server.voice_control = MagicMock(spec=["stop_listening"])
        server.phase_alignment_controller = MagicMock(spec=["stop_analysis"])

        result = await server.activate_system_emergency_stop("unit:all_writers")

        assert result["status"] == "stopped"
        assert server._manual_write_emergency_stopped is True
        assert server._manual_write_emergency_generation == 1
        assert server.auto_soundcheck_task is None
        assert legacy_cleanup_finished.is_set()
        agent.emergency_stop.assert_called_once_with()
        agent.stop.assert_called_once_with()
        server.multi_channel_auto_eq_controller.stop_all.assert_called_once_with()
        server.voice_control.stop_listening.assert_called_once_with()
        server.phase_alignment_controller.stop_analysis.assert_called_once_with()

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_graceful_shutdown_reports_engine_failure_but_closes_everything(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        server._shutdown_event = asyncio.Event()
        server.auto_soundcheck_engine = MagicMock()
        server.auto_soundcheck_engine.stop.side_effect = RuntimeError("thread stuck")
        client = MagicMock()
        client.close = AsyncMock()
        server.connected_clients.add(client)
        server.cleanup_all_controllers = MagicMock()

        await server.graceful_shutdown()

        assert server._shutdown_event.is_set()
        assert server._shutdown_errors == [
            {"step": "auto_soundcheck_engine", "error": "RuntimeError"}
        ]
        server.cleanup_all_controllers.assert_called_once_with()
        client.close.assert_awaited_once_with(1001, "Server shutting down")

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_graceful_shutdown_records_controller_cleanup_failure(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        server._shutdown_event = asyncio.Event()
        server.gain_staging = MagicMock()
        server.gain_staging.stop.side_effect = RuntimeError("gain stop failed")

        await server.graceful_shutdown()

        assert server._shutdown_event.is_set()
        assert {"step": "controller_cleanup", "error": "RuntimeError"} in (
            server._shutdown_errors
        )
        assert server.gain_staging is not None
        assert server._controllers_cleanup_complete is False

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_graceful_shutdown_latches_write_and_ingress_gates_before_first_await(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        server._shutdown_event = asyncio.Event()
        stop_entered = asyncio.Event()
        release_stop = asyncio.Event()

        class BlockingTrainingService:
            async def stop(self):
                stop_entered.set()
                await release_stop.wait()

        server.agent_training_service = BlockingTrainingService()
        manual_generation = server._manual_write_emergency_generation
        ingress_generation = server._ws_ingress_emergency_generation

        shutdown_task = asyncio.create_task(server.graceful_shutdown())
        await asyncio.wait_for(stop_entered.wait(), timeout=1)

        assert server._is_shutting_down is True
        assert server._manual_write_emergency_stopped is True
        assert server._manual_write_emergency_generation == manual_generation + 1
        assert server._ws_ingress_emergency_generation == ingress_generation + 1
        assert server.auto_soundcheck_running is False
        assert server.auto_soundcheck_observe_only is True

        release_stop.set()
        await asyncio.wait_for(shutdown_task, timeout=1)
        assert server._shutdown_errors == []

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_graceful_shutdown_drains_console_write_scope_before_mixer_disconnect(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        server._shutdown_event = asyncio.Event()
        server.agent_training_service = None
        mixer = MagicMock()
        server.mixer_client = mixer
        lock_held = asyncio.Event()
        release_lock = asyncio.Event()

        async def in_flight_console_transaction():
            async with server._console_write_lock:
                lock_held.set()
                await release_lock.wait()

        transaction = asyncio.create_task(in_flight_console_transaction())
        await asyncio.wait_for(lock_held.wait(), timeout=1)
        shutdown_task = asyncio.create_task(server.graceful_shutdown())
        await asyncio.sleep(0)

        assert shutdown_task.done() is False
        mixer.disconnect.assert_not_called()

        release_lock.set()
        await asyncio.wait_for(transaction, timeout=1)
        await asyncio.wait_for(shutdown_task, timeout=1)
        mixer.disconnect.assert_called_once_with()

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_graceful_shutdown_freezes_autonomous_writers_before_client_close_await(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        server._shutdown_event = asyncio.Event()
        server.agent_training_service = None
        server.cleanup_all_controllers = MagicMock()
        gain = MagicMock(spec=["set_automation_frozen", "stop"])
        fader = MagicMock(spec=["set_automation_frozen", "stop"])
        fader_v2 = MagicMock(spec=["set_automation_frozen", "stop"])
        agent = MagicMock(spec=["emergency_stop", "stop"])
        server.gain_staging = gain
        server.auto_fader_controller = fader
        server.auto_fader_v2_controller = fader_v2
        server.mixing_agent = agent
        close_entered = asyncio.Event()
        release_close = asyncio.Event()

        client = MagicMock()

        async def blocked_close(*_args, **_kwargs):
            close_entered.set()
            await release_close.wait()

        client.close = AsyncMock(side_effect=blocked_close)
        server.connected_clients.add(client)

        shutdown_task = asyncio.create_task(server.graceful_shutdown())
        await asyncio.wait_for(close_entered.wait(), timeout=1)

        gain.set_automation_frozen.assert_called_once_with(True)
        fader.set_automation_frozen.assert_called_once_with(True)
        fader_v2.set_automation_frozen.assert_called_once_with(True)
        agent.emergency_stop.assert_called_once_with()
        assert server.agent_auto_apply_enabled is False

        release_close.set()
        await asyncio.wait_for(shutdown_task, timeout=1)

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_emergency_stop_joins_safe_gain_worker_before_success(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        stop_event = threading.Event()
        worker_finished = threading.Event()

        def worker():
            stop_event.wait(1.0)
            worker_finished.set()

        thread = threading.Thread(target=worker, daemon=True)
        calibrator = MagicMock()
        server.safe_gain_calibrator = calibrator
        server._safe_gain_audio_thread_stop = stop_event
        server._safe_gain_audio_thread = thread
        thread.start()

        result = await server.activate_system_emergency_stop("unit:safe_gain")

        assert result["success"] is True
        assert worker_finished.is_set()
        assert thread.is_alive() is False
        calibrator.reset.assert_called_once_with()
        assert server.safe_gain_calibrator is None
        assert server._safe_gain_audio_thread is None
        assert server._safe_gain_audio_thread_stop is None

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_voice_command_after_emergency_latch_never_writes_mixer(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        server.mixer_client = MagicMock()
        server._manual_write_emergency_stopped = True
        server.broadcast = AsyncMock()

        await server._execute_voice_command({
            "type": "set_fader",
            "channel": 1,
            "value": -12.0,
        })

        server.mixer_client.set_channel_fader.assert_not_called()
        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_error"
        assert payload["reason"] == "emergency_stop_active"

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_emergency_voice_stop_keeps_listening_flag_until_stream_cleanup(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()

        class VoiceRuntime:
            def __init__(self):
                self.is_listening = True
                self.stream_open = True

            def stop_listening(self):
                if not self.is_listening:
                    return
                self.stream_open = False
                self.is_listening = False

        voice = VoiceRuntime()
        server.voice_control = voice

        result = await server.activate_system_emergency_stop("unit:voice_stream")

        assert result["success"] is True
        assert voice.stream_open is False
        assert voice.is_listening is False

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_post_latch_normal_message_never_reaches_dispatch(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        dispatched = AsyncMock()
        server._dispatch = {"start_resource": dispatched}
        server._is_shutting_down = True

        await server.handle_client_message(object(), '{"type":"start_resource"}')

        dispatched.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_graceful_shutdown_waits_for_existing_emergency_fast_lane_task(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        server._shutdown_event = asyncio.Event()
        server.agent_training_service = None
        emergency_entered = asyncio.Event()
        release_emergency = asyncio.Event()

        async def emergency_finalizer():
            emergency_entered.set()
            await release_emergency.wait()

        emergency_task = asyncio.create_task(emergency_finalizer())
        server._ws_emergency_fast_lane_task = emergency_task
        await asyncio.wait_for(emergency_entered.wait(), timeout=1)

        shutdown_task = asyncio.create_task(server.graceful_shutdown())
        await asyncio.sleep(0)
        assert shutdown_task.done() is False

        release_emergency.set()
        await asyncio.wait_for(emergency_task, timeout=1)
        await asyncio.wait_for(shutdown_task, timeout=1)
        assert server._shutdown_errors == []

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_duplicate_system_emergency_stop_does_not_advance_generation(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        agent = MagicMock()
        server.mixing_agent = agent

        first = await server.activate_system_emergency_stop("unit:first")
        duplicate = await server.activate_system_emergency_stop("unit:duplicate")

        assert first["coalesced"] is False
        assert duplicate["coalesced"] is True
        assert first["generation"] == duplicate["generation"] == 1
        agent.emergency_stop.assert_called_once_with()
        agent.stop.assert_called_once_with()

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_emergency_stop_does_not_cancel_its_initiating_voice_task(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        initiator = asyncio.current_task()
        assert initiator is not None
        server._voice_command_tasks.add(initiator)
        server.auto_soundcheck_task = initiator
        server.mixing_agent = MagicMock(spec=["emergency_stop", "stop"])
        server.mixing_agent_task = initiator

        result = await server.activate_system_emergency_stop(
            "unit:self-drain",
            initiator_task=initiator,
        )

        assert result["status"] == "stopped"
        assert result["success"] is True
        assert initiator.cancelling() == 0
        assert initiator in server._voice_command_tasks
        assert server.auto_soundcheck_task is initiator
        assert server.mixing_agent_task is initiator

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_concurrent_emergency_stop_coalesces_before_broadcast(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        server.last_soundcheck_readiness = {"run_id": "unit-run"}
        broadcast_started = asyncio.Event()
        release_broadcast = asyncio.Event()

        async def blocked_broadcast(_payload):
            broadcast_started.set()
            await release_broadcast.wait()

        server.broadcast = blocked_broadcast
        controller = MagicMock(spec=["set_automation_frozen", "stop"])
        server.auto_fader_controller = controller
        first_task = asyncio.create_task(
            server.activate_system_emergency_stop("unit:first-concurrent")
        )
        await asyncio.wait_for(broadcast_started.wait(), timeout=1.0)

        duplicate = await server.activate_system_emergency_stop(
            "unit:second-concurrent"
        )

        assert duplicate["status"] == "stopping"
        assert duplicate["success"] is False
        assert duplicate["coalesced"] is True
        assert duplicate["teardown_complete"] is False
        assert server._manual_write_emergency_generation == 1
        controller.stop.assert_not_called()
        release_broadcast.set()
        first = await asyncio.wait_for(first_task, timeout=1.0)
        assert first["status"] == "stopped"
        assert first["generation"] == duplicate["generation"] == 1
        controller.stop.assert_called_once_with()

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_phase_zero_latch_failure_keeps_operator_reset_blocked(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        controller = MagicMock(spec=["set_automation_frozen", "stop"])
        controller.set_automation_frozen.side_effect = RuntimeError(
            "synthetic freeze failure"
        )
        server.auto_fader_controller = controller

        result = await server.activate_system_emergency_stop(
            "unit:latch-failure"
        )

        assert result["status"] == "stop_incomplete"
        assert result["success"] is False
        assert result["teardown_complete"] is False
        assert server._system_emergency_latch_errors == [
            {"step": "auto_fader_freeze", "error": "RuntimeError"}
        ]
        reset = server.reset_manual_write_emergency_stop(
            approved=True,
            approval_id="unit-reset-after-latch-failure",
        )
        assert reset["status"] == "blocked"
        assert reset["reason"] == "emergency_stop_teardown_incomplete"
        assert reset["latch_errors"] == server._system_emergency_latch_errors
        assert server._manual_write_emergency_stopped is True

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_wal_only_latch_does_not_skip_system_emergency_teardown(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        server._manual_write_emergency_stopped = True
        server._manual_write_emergency_generation = 1
        server._system_emergency_teardown_complete = False
        controller = MagicMock(spec=["set_automation_frozen", "stop"])
        server.auto_fader_controller = controller

        result = await server.activate_system_emergency_stop("unit:wal_latch")

        assert result["success"] is True
        assert result["teardown_complete"] is True
        controller.set_automation_frozen.assert_called_once_with(True)
        controller.stop.assert_called_once_with()
        assert server._system_emergency_teardown_complete is True

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_system_emergency_stop_drains_slow_soundcheck_off_event_loop(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        stop_started = threading.Event()
        release_stop = threading.Event()

        def slow_stop():
            stop_started.set()
            assert release_stop.wait(1.0)

        engine = MagicMock()
        engine.stop.side_effect = slow_stop
        server.auto_soundcheck_engine = engine
        server.auto_soundcheck_running = True

        task = asyncio.create_task(
            server.activate_system_emergency_stop("unit:slow-soundcheck")
        )
        try:
            assert await asyncio.to_thread(stop_started.wait, 1.0)
            await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.1)
            assert task.done() is False
            reset = server.reset_manual_write_emergency_stop(
                approved=True,
                approval_id="unit-reset-too-early",
            )
            assert reset["status"] == "blocked"
            assert reset["reason"] == "emergency_stop_teardown_in_progress"
            assert server._manual_write_emergency_stopped is True
        finally:
            release_stop.set()

        result = await asyncio.wait_for(task, timeout=1.0)
        assert result["status"] == "stopped"
        assert server.auto_soundcheck_engine is None
        engine.stop.assert_called_once_with()

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_system_emergency_stop_continues_after_engine_failure_and_can_retry(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        engine = MagicMock()
        engine.stop.side_effect = RuntimeError("engine stop failed")
        server.auto_soundcheck_engine = engine
        server.auto_soundcheck_running = True
        agent = MagicMock()
        server.mixing_agent = agent

        first = await server.activate_system_emergency_stop("unit:engine-failure")

        assert first["status"] == "stop_incomplete"
        assert first["success"] is False
        assert server.auto_soundcheck_engine is engine
        agent.emergency_stop.assert_called_once_with()
        agent.stop.assert_called_once_with()
        reset = server.reset_manual_write_emergency_stop(
            approved=True,
            approval_id="unit-reset-incomplete",
        )
        assert reset["reason"] == "emergency_stop_teardown_incomplete"

        engine.stop.side_effect = None
        retry = await server.activate_system_emergency_stop("unit:retry")

        assert retry["status"] == "stopped"
        assert retry["success"] is True
        assert retry["coalesced"] is True
        assert server.auto_soundcheck_engine is None
        assert engine.stop.call_count == 2

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_system_emergency_controller_stop_does_not_block_event_loop(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        stop_started = threading.Event()
        release_stop = threading.Event()

        def slow_stop():
            stop_started.set()
            assert release_stop.wait(1.0)

        server.gain_staging = MagicMock()
        server.gain_staging.stop.side_effect = slow_stop
        task = asyncio.create_task(server.activate_system_emergency_stop("unit:slow-controller"))
        try:
            assert await asyncio.to_thread(stop_started.wait, 1.0)
            await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.1)
            assert task.done() is False
        finally:
            release_stop.set()

        result = await asyncio.wait_for(task, timeout=1.0)
        assert result["status"] == "stopped"

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
        rollback_path = tmp_path / "rollback.json"
        rollback_path.write_text(
            json.dumps({"channels": {}}),
            encoding="utf-8",
        )

        await server.restore_snapshot(
            "ws",
            str(snapshot_path),
            dry_run=False,
            confirm_maintenance=True,
            rollback_snapshot_path=str(rollback_path),
        )

        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "restore_result",
            "success": False,
            "error": "Snapshot does not contain any restorable channels",
            "path": str(snapshot_path),
        })

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_real_snapshot_handler_signature_and_confirmation_fail_closed(
        self,
        mock_config,
        mock_bleed,
        tmp_path,
    ):
        server = AutoMixerServer()
        server.send_to_client = AsyncMock()
        server.mixer_client = MagicMock(is_connected=True)
        handlers = snapshot_handlers.register_handlers(server)
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(
            json.dumps({"channels": {"1": {"channel": 1, "fader": -9.0}}}),
            encoding="utf-8",
        )

        await handlers["undo_restore_snapshot"](
            "ws",
            {
                "snapshot_path": str(snapshot_path),
                "dry_run": False,
                "confirm_maintenance": "true",
                "rollback_snapshot_path": str(tmp_path / "rollback.json"),
            },
        )

        payload = server.send_to_client.await_args.args[1]
        assert payload["type"] == "restore_result"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "maintenance_confirmation_required"

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
        server.phase_alignment_controller.stop_analysis.assert_called_once()
        server.multi_channel_auto_eq_controller.stop_all.assert_called_once()
        server.auto_fader_controller.stop.assert_called_once()
        server.send_to_client.assert_awaited_once_with("ws", {
            "type": "auto_soundcheck_status",
            "is_running": False,
            "message": "Auto soundcheck stopped"
        })

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_cleanup_all_controllers_retains_failed_resources_and_reports_incomplete(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server.gain_staging = MagicMock()
        server.gain_staging.stop.side_effect = RuntimeError("gain stop failed")
        server.multi_channel_auto_eq_controller = MagicMock()
        server.multi_channel_auto_eq_controller.stop_all.side_effect = RuntimeError("eq stop failed")
        server.phase_alignment_controller = MagicMock()
        server.phase_alignment_controller.stop_analysis.side_effect = RuntimeError("phase stop failed")
        server.auto_fader_controller = MagicMock()
        server.auto_fader_controller.stop.side_effect = RuntimeError("fader stop failed")

        with pytest.raises(RuntimeError, match="Controller cleanup incomplete"):
            server.cleanup_all_controllers()

        assert server.gain_staging is not None
        assert server.multi_channel_auto_eq_controller is not None
        assert server.phase_alignment_controller is not None
        assert server.auto_fader_controller is not None
        assert server._controllers_cleanup_complete is False


class TestAutoMixerServerRuntimeErrors:

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_unauthenticated_network_runtime_is_forced_read_only(
        self,
        mock_config,
        mock_bleed,
    ):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
            server = AutoMixerServer()
        server.operator_mode = "supervised"
        server._auto_mode_unlocked = True
        server._wing_observation_only = False
        server._wing_record_telemetry = True

        server._activate_network_runtime_safety_boundary()

        assert server.operator_mode == "analyze"
        assert server._auto_mode_unlocked is False
        assert server._wing_observation_only is True
        assert server._wing_record_telemetry is False
        assert server.get_operator_mode_status()["unauthenticated_read_only"] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "message_type",
        [
            "set_operator_mode",
            "create_operator_proposal",
            "apply_operator_proposal",
            "disconnect_mixer",
            "set_fader",
            "set_gain",
        ],
    )
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_unauthenticated_source_blocks_control_before_dispatch(
        self,
        mock_config,
        mock_bleed,
        message_type,
    ):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
            server = AutoMixerServer()
        server.send_to_client = AsyncMock()
        handler = AsyncMock()
        server._dispatch[message_type] = handler

        await server.handle_client_message(
            "ws",
            json.dumps(
                {
                    "type": message_type,
                    "mode": "supervised",
                    "approved": True,
                    "approval_id": "forged",
                    "idempotency_key": "forged",
                }
            ),
        )

        handler.assert_not_awaited()
        payload = server.send_to_client.await_args.args[1]
        assert payload["type"] == "control_authentication_required"
        assert payload["request_type"] == message_type
        assert payload["reason"] == "authenticated_control_session_required"
        assert payload["live_authority_granted"] is False

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_unauthenticated_source_allows_only_explicit_observation_connect(
        self,
        mock_config,
        mock_bleed,
    ):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
            server = AutoMixerServer()
        server.send_to_client = AsyncMock()
        handler = AsyncMock()
        server._dispatch["connect_wing"] = handler

        await server.handle_client_message(
            "ws",
            json.dumps({"type": "connect_wing", "ip": "192.0.2.1"}),
        )
        handler.assert_not_awaited()
        assert server.send_to_client.await_args.args[1]["status"] == "blocked"

        server.send_to_client.reset_mock()
        await server.handle_client_message(
            "ws",
            json.dumps(
                {
                    "type": "connect_wing",
                    "ip": "192.0.2.1",
                    "observation_only": True,
                }
            ),
        )
        handler.assert_awaited_once()
        server.send_to_client.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_unauthenticated_source_allows_read_queries_and_meters(
        self,
        mock_config,
        mock_bleed,
    ):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
            server = AutoMixerServer()
        server.send_to_client = AsyncMock()
        get_handler = AsyncMock()
        parameter_handler = AsyncMock()
        meter_handler = AsyncMock()
        server._dispatch["get_dashboard_snapshot"] = get_handler
        server._dispatch["query_mixer_parameter"] = parameter_handler
        server._dispatch["subscribe_meters"] = meter_handler

        await server.handle_client_message(
            "ws",
            '{"type":"get_dashboard_snapshot"}',
        )
        await server.handle_client_message(
            "ws",
            '{"type":"query_mixer_parameter","channel":1,"value_type":"fader"}',
        )
        await server.handle_client_message(
            "ws",
            '{"type":"subscribe_meters"}',
        )

        get_handler.assert_awaited_once()
        parameter_handler.assert_awaited_once()
        meter_handler.assert_awaited_once()
        server.send_to_client.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_unauthenticated_future_get_name_is_blocked_by_default(
        self,
        mock_config,
        mock_bleed,
    ):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
            server = AutoMixerServer()
        server.send_to_client = AsyncMock()
        handler = AsyncMock()
        server._dispatch["get_future_control_surface"] = handler

        await server.handle_client_message(
            "ws",
            '{"type":"get_future_control_surface"}',
        )

        handler.assert_not_awaited()
        assert server.send_to_client.await_args.args[1]["status"] == "blocked"

    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    def test_unauthenticated_read_allowlist_matches_real_dispatch(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        assert server_module.UNAUTHENTICATED_READ_ONLY_MESSAGE_TYPES <= set(
            server._dispatch
        )

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_authenticated_source_control_reaches_dispatch(
        self,
        mock_config,
        mock_bleed,
    ):
        with patch.dict(
            os.environ,
            {"AUTOMIXER_WS_AUTH_TOKEN": "a" * 43},
            clear=False,
        ):
            server = AutoMixerServer()
        handler = AsyncMock()
        server._dispatch["set_operator_mode"] = handler

        await server.handle_client_message(
            "ws",
            '{"type":"set_operator_mode","mode":"supervised"}',
        )

        handler.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_handle_client_message_does_not_info_log_payload(
        self,
        mock_config,
        mock_bleed,
        caplog,
    ):
        server = AutoMixerServer()
        server._ws_auth_required = True
        server._dispatch["safe_noop"] = AsyncMock()
        secret_marker = "must-not-enter-info-logs"
        caplog.set_level("INFO", logger="server")

        await server.handle_client_message(
            "ws",
            json.dumps({"type": "safe_noop", "private": secret_marker}),
        )

        assert secret_marker not in caplog.text

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_handle_client_message_reraises_cancelled_error(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        server._ws_auth_required = True
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
        server._ws_auth_required = True
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
    async def test_execute_voice_command_reports_no_mixer_error_with_authority_denied(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        server.mixer_client = None
        server.broadcast = AsyncMock()
        command = {"type": "set_pan", "channel": 1, "value": 25.0}

        await server._execute_voice_command(command)

        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_error"
        assert payload["status"] == "error"
        assert payload["success"] is False
        assert payload["reason"] == "mixer_not_connected"
        assert payload["surface"] == "voice_command:set_pan"
        assert payload["command"] == command
        assert payload["command_type"] == "set_pan"
        assert payload["channel"] == 1
        assert payload["request"] == command
        assert payload["live_write"] is False
        assert payload["live_authority_granted"] is False
        assert payload["readiness_closure_allowed"] is False

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_execute_voice_command_reports_exception_with_authority_denied(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        client = MagicMock()
        client.is_connected = True
        client.set_channel_pan = MagicMock(side_effect=RuntimeError("pan transport failed"))
        server.mixer_client = client
        server.broadcast = AsyncMock()
        command = {"type": "set_pan", "channel": 1, "value": 25.0}

        await server._execute_voice_command(command)

        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_error"
        assert payload["status"] == "error"
        assert payload["success"] is False
        assert payload["reason"] == "voice_command_exception"
        assert payload["error"] == "pan transport failed"
        assert payload["surface"] == "voice_command:set_pan"
        assert payload["command"] == command
        assert payload["command_type"] == "set_pan"
        assert payload["channel"] == 1
        assert payload["request"] == command
        assert payload["live_write"] is False
        assert payload["live_authority_granted"] is False
        assert payload["readiness_closure_allowed"] is False
        client.set_channel_pan.assert_called_once_with(1, 25.0)

    @pytest.mark.asyncio
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_execute_voice_command_reports_unknown_command_with_authority_denied(
        self,
        mock_config,
        mock_bleed,
    ):
        server = AutoMixerServer()
        client = MagicMock()
        client.is_connected = True
        server.mixer_client = client
        server.broadcast = AsyncMock()
        command = {"type": "raise_master", "channel": 1, "db": 6}

        await server._execute_voice_command(command)

        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_error"
        assert payload["status"] == "error"
        assert payload["success"] is False
        assert payload["reason"] == "unknown_voice_command"
        assert payload["surface"] == "voice_command:raise_master"
        assert payload["command"] == command
        assert payload["command_type"] == "raise_master"
        assert payload["channel"] == 1
        assert payload["request"] == command
        assert payload["live_write"] is False
        assert payload["live_authority_granted"] is False
        assert payload["readiness_closure_allowed"] is False
        assert client.method_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("command", ["raise channel 1", ["set_fader", 1, -6.0]])
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_execute_voice_command_reports_malformed_payload_with_authority_denied(
        self,
        mock_config,
        mock_bleed,
        command,
    ):
        server = AutoMixerServer()
        client = MagicMock()
        client.is_connected = True
        server.mixer_client = client
        server.broadcast = AsyncMock()

        await server._execute_voice_command(command)

        expected_request = {"raw_command": command}
        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_error"
        assert payload["status"] == "error"
        assert payload["success"] is False
        assert payload["reason"] == "malformed_voice_command"
        assert payload["surface"] == "voice_command"
        assert payload["command"] == expected_request
        assert payload["command_type"] == ""
        assert payload["channel"] is None
        assert payload["request"] == expected_request
        assert payload["live_write"] is False
        assert payload["live_authority_granted"] is False
        assert payload["readiness_closure_allowed"] is False
        assert client.method_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("command", "reason", "method_name"),
        [
            ({"type": "set_fader", "value": -6.0}, "missing_channel", "set_channel_fader"),
            ({"type": "set_gain", "value": -2.0}, "missing_channel", "set_channel_gain"),
            ({"type": "set_pan", "value": 25.0}, "missing_channel", "set_channel_pan"),
            ({"type": "mute_channel", "muted": True}, "missing_channel", "set_channel_mute"),
            ({"type": "volume_up", "db": 1}, "missing_channel", "set_channel_fader"),
            ({"type": "volume_down", "db": 1}, "missing_channel", "set_channel_fader"),
            ({"type": "load_snap"}, "missing_snap_name", "load_snap"),
        ],
    )
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_execute_voice_command_reports_invalid_basic_request_without_mixer_write(
        self,
        mock_config,
        mock_bleed,
        command,
        reason,
        method_name,
    ):
        server = AutoMixerServer()
        client = MagicMock()
        client.is_connected = True
        client.send = MagicMock()
        client.set_channel_fader = MagicMock(return_value=True)
        client.set_channel_gain = MagicMock(return_value=True)
        client.set_channel_pan = MagicMock(return_value=True)
        client.set_channel_mute = MagicMock(return_value=True)
        client.load_snap = MagicMock(return_value=True)
        server.mixer_client = client
        server.broadcast = AsyncMock()

        await server._execute_voice_command(command)

        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_error"
        assert payload["status"] == "error"
        assert payload["success"] is False
        assert payload["reason"] == reason
        assert payload["surface"] == f"voice_command:{command['type']}"
        assert payload["command"] == command
        assert payload["command_type"] == command["type"]
        assert payload["request"] == command
        assert payload["live_write"] is False
        assert payload["live_authority_granted"] is False
        assert payload["readiness_closure_allowed"] is False
        getattr(client, method_name).assert_not_called()
        client.send.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("command", "method_name"),
        [
            ({"type": "load_snap", "snap_name": "SHOW_A"}, "load_snap"),
            ({"type": "eq_on", "channel": 1, "on": 1}, "set_eq_on"),
            ({"type": "eq_band_up", "channel": 1, "band": "2", "db": 1}, "set_eq_band_gain"),
            ({"type": "compressor_on", "channel": 1, "on": 1}, "set_compressor_on"),
            (
                {"type": "compressor_threshold", "channel": 1, "threshold": -18.0},
                "set_compressor_threshold",
            ),
            ({"type": "compressor_gain_up", "channel": 1, "db": 1}, "set_compressor_gain"),
        ],
    )
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_execute_voice_command_reports_unsupported_mixer_without_mixer_write(
        self,
        mock_config,
        mock_bleed,
        command,
        method_name,
    ):
        server = AutoMixerServer()
        client = MagicMock()
        client.is_connected = True
        client.load_snap = MagicMock(return_value=True)
        client.set_eq_on = MagicMock(return_value=True)
        client.get_eq_band_gain = MagicMock(return_value=0.0)
        client.set_eq_band_gain = MagicMock(return_value=True)
        client.set_compressor_on = MagicMock(return_value=True)
        client.set_compressor_threshold = MagicMock(return_value=True)
        client.get_compressor_gain = MagicMock(return_value=0.0)
        client.set_compressor_gain = MagicMock(return_value=True)
        server.mixer_client = client
        server.broadcast = AsyncMock()

        await server._execute_voice_command(command)

        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_error"
        assert payload["status"] == "error"
        assert payload["success"] is False
        assert payload["reason"] == "unsupported_mixer_client"
        assert payload["surface"] == f"voice_command:{command['type']}"
        assert payload["command"] == command
        assert payload["command_type"] == command["type"]
        assert payload["request"] == command
        assert payload["live_write"] is False
        assert payload["live_authority_granted"] is False
        assert payload["readiness_closure_allowed"] is False
        getattr(client, method_name).assert_not_called()
        client.get_eq_band_gain.assert_not_called()
        client.get_compressor_gain.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("command", "reason", "method_name"),
        [
            ({"type": "eq_band_up", "channel": 1}, "missing_eq_band", "set_eq_band_gain"),
            (
                {"type": "compressor_threshold", "channel": 1},
                "missing_compressor_threshold",
                "set_compressor_threshold",
            ),
        ],
    )
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_execute_voice_command_reports_missing_supported_fields_without_mixer_write(
        self,
        mock_config,
        mock_bleed,
        command,
        reason,
        method_name,
    ):
        class VoiceMixerStub:
            pass

        server = AutoMixerServer()
        client = VoiceMixerStub()
        client.is_connected = True
        client.get_eq_band_gain = MagicMock(return_value=0.0)
        client.set_eq_band_gain = MagicMock(return_value=True)
        client.set_compressor_threshold = MagicMock(return_value=True)
        server.mixer_client = client
        server.broadcast = AsyncMock()

        with patch("server.WingClient", new=VoiceMixerStub), \
             patch("server.EnhancedOSCClient", new=VoiceMixerStub):
            await server._execute_voice_command(command)

        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_error"
        assert payload["status"] == "error"
        assert payload["success"] is False
        assert payload["reason"] == reason
        assert payload["surface"] == f"voice_command:{command['type']}"
        assert payload["command"] == command
        assert payload["command_type"] == command["type"]
        assert payload["request"] == command
        assert payload["live_write"] is False
        assert payload["live_authority_granted"] is False
        assert payload["readiness_closure_allowed"] is False
        getattr(client, method_name).assert_not_called()
        client.get_eq_band_gain.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("command", "reason", "method_name"),
        [
            ({"type": "set_fader", "channel": 1, "value": "loud"}, "invalid_value", "set_channel_fader"),
            ({"type": "set_gain", "channel": 1, "value": "nan"}, "invalid_value", "set_channel_gain"),
            ({"type": "set_pan", "channel": 1, "value": "left"}, "invalid_value", "set_channel_pan"),
            ({"type": "set_pan", "channel": 1, "value": True}, "invalid_value", "set_channel_pan"),
            ({"type": "volume_up", "channel": 1, "db": "loud"}, "invalid_db", "set_channel_fader"),
            ({"type": "volume_down", "channel": 1, "db": "inf"}, "invalid_db", "set_channel_fader"),
        ],
    )
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_execute_voice_command_reports_invalid_numeric_basic_request_without_mixer_access(
        self,
        mock_config,
        mock_bleed,
        command,
        reason,
        method_name,
    ):
        server = AutoMixerServer()
        client = MagicMock()
        client.is_connected = True
        client.send = MagicMock()
        client.get_channel_fader = MagicMock(return_value=-10.0)
        client.set_channel_fader = MagicMock(return_value=True)
        client.set_channel_gain = MagicMock(return_value=True)
        client.set_channel_pan = MagicMock(return_value=True)
        server.mixer_client = client
        server.broadcast = AsyncMock()

        await server._execute_voice_command(command)

        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_error"
        assert payload["status"] == "error"
        assert payload["success"] is False
        assert payload["reason"] == reason
        assert payload["surface"] == f"voice_command:{command['type']}"
        assert payload["command"] == command
        assert payload["command_type"] == command["type"]
        assert payload["request"] == command
        assert payload["live_write"] is False
        assert payload["live_authority_granted"] is False
        assert payload["readiness_closure_allowed"] is False
        getattr(client, method_name).assert_not_called()
        client.get_channel_fader.assert_not_called()
        client.send.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("command", "reason", "read_method_name", "write_method_name"),
        [
            (
                {"type": "eq_band_up", "channel": 1, "band": "2", "db": "wide"},
                "invalid_db",
                "get_eq_band_gain",
                "set_eq_band_gain",
            ),
            (
                {"type": "compressor_threshold", "channel": 1, "threshold": "low"},
                "invalid_compressor_threshold",
                "get_compressor_gain",
                "set_compressor_threshold",
            ),
            (
                {"type": "compressor_gain_up", "channel": 1, "db": "boost"},
                "invalid_db",
                "get_compressor_gain",
                "set_compressor_gain",
            ),
        ],
    )
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_execute_voice_command_reports_invalid_numeric_wing_request_without_mixer_access(
        self,
        mock_config,
        mock_bleed,
        command,
        reason,
        read_method_name,
        write_method_name,
    ):
        class VoiceMixerStub:
            pass

        server = AutoMixerServer()
        client = VoiceMixerStub()
        client.is_connected = True
        client.get_eq_band_gain = MagicMock(return_value=0.0)
        client.set_eq_band_gain = MagicMock(return_value=True)
        client.get_compressor_gain = MagicMock(return_value=0.0)
        client.set_compressor_threshold = MagicMock(return_value=True)
        client.set_compressor_gain = MagicMock(return_value=True)
        server.mixer_client = client
        server.broadcast = AsyncMock()

        with patch("server.WingClient", new=VoiceMixerStub), \
             patch("server.EnhancedOSCClient", new=VoiceMixerStub):
            await server._execute_voice_command(command)

        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_error"
        assert payload["status"] == "error"
        assert payload["success"] is False
        assert payload["reason"] == reason
        assert payload["surface"] == f"voice_command:{command['type']}"
        assert payload["command"] == command
        assert payload["command_type"] == command["type"]
        assert payload["request"] == command
        assert payload["live_write"] is False
        assert payload["live_authority_granted"] is False
        assert payload["readiness_closure_allowed"] is False
        getattr(client, read_method_name).assert_not_called()
        getattr(client, write_method_name).assert_not_called()

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
    async def test_execute_voice_command_set_pan_clamps_bounds(self, mock_config, mock_bleed):
        server = AutoMixerServer()
        client = MagicMock()
        client.set_channel_pan = MagicMock()
        client.is_connected = True
        server.mixer_client = client
        server.broadcast = AsyncMock()

        await server._execute_voice_command({
            "type": "set_pan",
            "channel": 1,
            "value": 150,
        })

        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_executed"
        assert payload["command"] == "set_pan"
        assert payload["channel"] == 1
        assert payload["value"] == 100.0
        client.set_channel_pan.assert_called_once_with(1, 100.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "command",
            "method_name",
            "expected_call",
            "value_type",
            "requested_value",
        ),
        [
            (
                {"type": "set_fader", "channel": 1, "value": -6.0},
                "set_channel_fader",
                ((1, -6.0),),
                "fader",
                -6.0,
            ),
            (
                {"type": "set_gain", "channel": 1, "value": -2.0},
                "set_channel_gain",
                ((1, -2.0),),
                "gain",
                -2.0,
            ),
            (
                {"type": "set_pan", "channel": 1, "value": 150.0},
                "set_channel_pan",
                ((1, 100.0),),
                "pan",
                100.0,
            ),
            (
                {"type": "mute_channel", "channel": 1, "muted": True},
                "set_channel_mute",
                ((1, 1),),
                "mute",
                True,
            ),
        ],
    )
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_execute_voice_command_reports_failed_mixer_write(
        self,
        mock_config,
        mock_bleed,
        command,
        method_name,
        expected_call,
        value_type,
        requested_value,
    ):
        server = AutoMixerServer()
        client = MagicMock()
        client.is_connected = True
        client.set_channel_fader = MagicMock(return_value=True)
        client.set_channel_gain = MagicMock(return_value=True)
        client.set_channel_pan = MagicMock(return_value=True)
        client.set_channel_mute = MagicMock(return_value=True)
        getattr(client, method_name).return_value = False
        server.mixer_client = client
        server.broadcast = AsyncMock()

        await server._execute_voice_command(command)

        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_blocked"
        assert payload["status"] == "blocked"
        assert payload["success"] is False
        assert payload["reason"] == "mixer_write_failed"
        assert payload["surface"] == f"voice_command:{command['type']}"
        assert payload["command"] == command["type"]
        assert payload["channel"] == command["channel"]
        assert payload["value_type"] == value_type
        assert payload["requested_value"] == requested_value
        assert payload["live_write"] is False
        assert payload["live_authority_granted"] is False
        assert payload["readiness_closure_allowed"] is False
        assert payload["request"] == command
        getattr(client, method_name).assert_called_once_with(*expected_call[0])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("command", "initial_db", "expected_requested_value"),
        [
            ({"type": "volume_up", "channel": 1, "db": 3}, -10.0, -7.0),
            ({"type": "volume_down", "channel": 1, "db": 3}, -10.0, -13.0),
        ],
    )
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_execute_voice_command_volume_write_failure_does_not_advance_cached_fader(
        self,
        mock_config,
        mock_bleed,
        command,
        initial_db,
        expected_requested_value,
    ):
        server = AutoMixerServer()
        client = MagicMock()
        client.is_connected = True
        client.set_channel_fader = MagicMock(return_value=False)
        server.mixer_client = client
        server.broadcast = AsyncMock()
        server.last_fader_values[1] = initial_db

        await server._execute_voice_command(command)

        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_blocked"
        assert payload["status"] == "blocked"
        assert payload["success"] is False
        assert payload["reason"] == "mixer_write_failed"
        assert payload["surface"] == f"voice_command:{command['type']}"
        assert payload["value_type"] == "fader"
        assert payload["requested_value"] == expected_requested_value
        assert payload["live_write"] is False
        assert payload["live_authority_granted"] is False
        assert payload["readiness_closure_allowed"] is False
        assert server.last_fader_values[1] == initial_db
        client.set_channel_fader.assert_called_once_with(1, expected_requested_value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "command",
            "method_name",
            "expected_call",
            "value_type",
            "requested_value",
        ),
        [
            (
                {"type": "eq_on", "channel": 1, "on": 1},
                "set_eq_on",
                (1, 1),
                "eq_on",
                1,
            ),
            (
                {"type": "eq_band_up", "channel": 1, "band": "2", "db": 1},
                "set_eq_band_gain",
                (1, "2g", 1.0),
                "eq_band_gain",
                1.0,
            ),
            (
                {"type": "eq_band_down", "channel": 1, "band": "2", "db": 1},
                "set_eq_band_gain",
                (1, "2g", -1.0),
                "eq_band_gain",
                -1.0,
            ),
            (
                {"type": "compressor_on", "channel": 1, "on": 1},
                "set_compressor_on",
                (1, 1),
                "compressor_on",
                1,
            ),
            (
                {"type": "compressor_threshold", "channel": 1, "threshold": -18.0},
                "set_compressor_threshold",
                (1, -18.0),
                "compressor_threshold",
                -18.0,
            ),
            (
                {"type": "compressor_gain_up", "channel": 1, "db": 1},
                "set_compressor_gain",
                (1, 1.0),
                "compressor_gain",
                1.0,
            ),
            (
                {"type": "compressor_gain_down", "channel": 1, "db": 1},
                "set_compressor_gain",
                (1, -1.0),
                "compressor_gain",
                -1.0,
            ),
        ],
    )
    @patch("server.BleedService")
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    async def test_execute_voice_command_eq_compressor_reports_failed_mixer_write(
        self,
        mock_config,
        mock_bleed,
        command,
        method_name,
        expected_call,
        value_type,
        requested_value,
    ):
        class VoiceMixerStub:
            pass

        server = AutoMixerServer()
        client = VoiceMixerStub()
        client.is_connected = True
        client.set_eq_on = MagicMock(return_value=True)
        client.get_eq_band_gain = MagicMock(return_value=0.0)
        client.set_eq_band_gain = MagicMock(return_value=True)
        client.set_compressor_on = MagicMock(return_value=True)
        client.set_compressor_threshold = MagicMock(return_value=True)
        client.get_compressor_gain = MagicMock(return_value=0.0)
        client.set_compressor_gain = MagicMock(return_value=True)
        getattr(client, method_name).return_value = False
        server.mixer_client = client
        server.broadcast = AsyncMock()

        with patch("server.WingClient", new=VoiceMixerStub), \
             patch("server.EnhancedOSCClient", new=VoiceMixerStub):
            await server._execute_voice_command(command)

        payload = server.broadcast.await_args.args[0]
        assert payload["type"] == "voice_command_blocked"
        assert payload["status"] == "blocked"
        assert payload["success"] is False
        assert payload["reason"] == "mixer_write_failed"
        assert payload["surface"] == f"voice_command:{command['type']}"
        assert payload["command"] == command["type"]
        assert payload["channel"] == command["channel"]
        assert payload["value_type"] == value_type
        assert payload["requested_value"] == requested_value
        assert payload["live_write"] is False
        assert payload["live_authority_granted"] is False
        assert payload["readiness_closure_allowed"] is False
        getattr(client, method_name).assert_called_once_with(*expected_call)

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
        def set_value(address, value):
            self.state[address] = value
            return True

        self.query_parameter = MagicMock(
            side_effect=lambda address: {
                "status": "received",
                "address": address,
                "value": self.state.get(address),
                "confirmed_round_trips": 2,
            }
        )
        self.set_channel_fader = MagicMock(
            wraps=lambda channel, value: set_value(f"/ch/{channel}/fdr", value)
        )
        self.set_channel_gain = MagicMock(
            wraps=lambda channel, value: set_value(f"/ch/{channel}/in/set/trim", value)
        )
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
        self.set_channel_mute = MagicMock(
            wraps=lambda channel, value: set_value(f"/ch/{channel}/mute", int(bool(value)))
        )
        self.set_channel_pan = MagicMock(
            wraps=lambda channel, value: set_value(f"/ch/{channel}/pan", value)
        )
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
def dummy_wing_server(tmp_path):
    with patch("server.WingClient", new=DummyWingLike), \
         patch("server.EnhancedOSCClient", new=DummyWingLike), \
         patch("server.BleedService"), \
         patch.object(AutoMixerServer, "_load_config", return_value={}):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        server.connection_mode = "wing"
        server.mixer_client = DummyWingLike()
        server.operator_proposal_queue = OperatorProposalQueue(
            result_store=ResultEvidenceStore(
                tmp_path / "dummy-wing-operator-proposal-results",
                min_free_bytes=0,
            ),
            result_evidence_lifecycle_ready=True,
        )
        yield server


@pytest.fixture
def minimal_server():
    with patch("server.BleedService"), patch.object(AutoMixerServer, "_load_config", return_value={}):
        yield AutoMixerServer(ws_host="localhost", ws_port=8765)


@pytest.mark.asyncio
async def test_m110_replan_protocol_gate_block_is_runner_safe(minimal_server):
    server = minimal_server
    websocket = object()
    server._runtime_identity.setdefault("launch_binding", {})["mode"] = (
        "m110_attended_runner"
    )
    server._verify_m110_cp05_hardware_gate = MagicMock(
        side_effect=RuntimeError("m110_cp05_hardware_gate_closed")
    )
    server.send_to_client = AsyncMock()
    request = {
        "type": "advance_m110_soundcheck_replan",
        "contract_version": 1,
        "replan_selector": "next_bounded_split_revision",
    }

    blocked = await server._block_m110_protocol_message(
        websocket,
        request["type"],
        request,
    )

    assert blocked is True
    response = server.send_to_client.await_args.args[1]
    assert response["type"] == "m110_soundcheck_replan_result"
    assert response["status"] == "blocked"
    assert response["live_mixer_writes"] is False
    assert response["server_owned_selection"] is True
    assert response["client_targets_accepted"] is False
    assert response["console_mutation_write_attempt_count"] == 0


def test_production_server_bootstraps_result_evidence_lifecycle_when_recovery_is_clear(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "AUTOMIXER_OPERATOR_PROPOSAL_RESULT_DIR",
        str(tmp_path / "clear-result-store"),
    )
    with patch("server.BleedService"), patch.object(
        AutoMixerServer,
        "_load_config",
        return_value={},
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)

    assert server.operator_proposal_queue.result_evidence_contract_ready is True
    assert server.operator_proposal_queue.result_recovery_hazards == ()


def test_production_server_uses_private_user_state_default_result_store(
    monkeypatch,
    tmp_path,
):
    home = tmp_path / "private-home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("AUTOMIXER_OPERATOR_PROPOSAL_RESULT_DIR", raising=False)
    expected_root = default_result_store_root()

    with patch("server.BleedService"), patch.object(
        AutoMixerServer,
        "_load_config",
        return_value={},
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)

    assert server.operator_proposal_queue.result_store.root == expected_root
    assert expected_root.is_relative_to(home)
    assert "AUTO-MIXER" in expected_root.parts
    assert server.operator_proposal_queue.result_evidence_contract_ready is True
    assert expected_root.stat().st_mode & 0o777 == 0o700


def test_production_server_insecure_result_store_override_fails_closed(
    monkeypatch,
    tmp_path,
):
    insecure_parent = tmp_path / "group-writable"
    insecure_parent.mkdir(mode=0o770)
    insecure_parent.chmod(0o770)
    monkeypatch.setenv(
        "AUTOMIXER_OPERATOR_PROPOSAL_RESULT_DIR",
        str(insecure_parent / "results"),
    )

    with patch("server.BleedService"), patch.object(
        AutoMixerServer,
        "_load_config",
        return_value={},
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)

    assert server.operator_proposal_queue.result_evidence_contract_ready is False
    assert server.operator_proposal_queue.result_persistence_failed is True


def test_production_server_keeps_result_evidence_lifecycle_closed_on_recovery_hazard(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "AUTOMIXER_OPERATOR_PROPOSAL_RESULT_DIR",
        str(tmp_path / "hazard-result-store"),
    )
    hazard = {
        "code": "operator_proposal_result_recovery_required",
        "phase": "send_may_have_occurred",
    }
    with patch("server.BleedService"), patch.object(
        AutoMixerServer,
        "_load_config",
        return_value={},
    ), patch.object(
        ResultEvidenceStore,
        "recovery_hazards",
        return_value=(hazard,),
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)

    assert server.operator_proposal_queue.result_evidence_contract_ready is False
    assert server.operator_proposal_queue.result_recovery_hazards == (hazard,)


@pytest.mark.asyncio
async def test_send_to_client_projects_every_generic_payload_once_to_exact_string(
    minimal_server,
):
    websocket = MagicMock()
    websocket.send = AsyncMock()
    payload = {
        "type": "state_update",
        "title": "Вокал 🎙️",
        "levels": np.array([-12.5, -6.0], dtype=np.float32),
    }

    result = await minimal_server.send_to_client(websocket, payload)

    assert result is True
    websocket.send.assert_awaited_once_with(project_outbound_message(payload))


@pytest.mark.asyncio
async def test_send_to_client_degrades_oversized_generic_without_raw_marker(
    minimal_server,
):
    websocket = MagicMock()
    websocket.send = AsyncMock()
    marker = "never-send-this-raw-marker"
    payload = {"type": "bulk_state", "blob": marker + ("x" * 1_000_000)}

    result = await minimal_server.send_to_client(websocket, payload)

    assert result is True
    websocket.send.assert_awaited_once()
    serialized = websocket.send.await_args.args[0]
    parsed = json.loads(serialized)
    assert parsed["type"] == "outbound_message_degraded"
    assert len(serialized.encode("utf-8")) <= 131_072
    assert marker not in serialized


class TestWingDeploymentWriteBoundary:

    @staticmethod
    def _decode_last_message(websocket):
        assert websocket.send.await_count >= 1
        raw = websocket.send.await_args_list[-1].args[0]
        return json.loads(raw)

    @staticmethod
    def _review_binding(server, proposal_id):
        proposal = server.operator_proposal_queue.find(proposal_id=proposal_id)
        assert proposal is not None
        return {
            "expected_review_contract_version": proposal.review_contract_version,
            "expected_review_content_digest": proposal.review_content_digest,
        }

    @staticmethod
    def _enable_result_evidence_lifecycle(server, tmp_path):
        store = ResultEvidenceStore(
            tmp_path / "operator-proposal-results",
            min_free_bytes=0,
        )
        server.operator_proposal_queue = OperatorProposalQueue(
            result_store=store,
            result_evidence_lifecycle_ready=True,
        )
        return store

    @staticmethod
    def _configure_deep_operator_apply(server):
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"

    @staticmethod
    async def _apply_manual_write(
        server,
        *,
        channel=1,
        value=-6.0,
        value_type="fader",
        approval_id="unit-approval",
        **kwargs,
    ):
        return await server._apply_manual_console_write(
            None,
            channel=channel,
            value=value,
            value_type=value_type,
            source="unit:canonical_operator_proposal_path",
            reason="unit canonical proposal path",
            approved=True,
            approval_id=approval_id,
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_operator_proposal_review_binding_and_evidence_gate_block_before_reservation(
        self,
        dummy_wing_server,
    ):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server.connection_mode = "offline"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "id": "review-bound-no-send",
                "title": "Review-bound fader",
                "requested_change": {
                    "value_type": "fader",
                    "channel": 1,
                    "value": -9.0,
                },
            },
        )
        proposal = server.operator_proposal_queue.find(
            proposal_id="review-bound-no-send"
        )

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal.id,
                "approved": True,
                "approval_id": proposal.id,
                "idempotency_key": "missing-review-binding",
            },
        )
        missing = self._decode_last_message(websocket)
        assert missing["reason"] == "proposal_review_contract_version_required"
        assert missing["no_console_send_proven"] is True
        assert missing["idempotency_reserved"] is False
        assert missing["proposal_status_mutation"] is False
        assert proposal.status == "pending"
        assert "missing-review-binding" not in server._operator_apply_idempotency_keys

        proposal.raw["requested_change"]["value"] = -3.0
        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal.id,
                **self._review_binding(server, proposal.id),
                "approved": True,
                "approval_id": proposal.id,
                "idempotency_key": "mutated-review-binding",
            },
        )
        mutated = self._decode_last_message(websocket)
        assert mutated["reason"] == "proposal_review_content_integrity_error"
        assert proposal.status == "pending"
        assert "mutated-review-binding" not in server._operator_apply_idempotency_keys
        server._apply_manual_console_write.assert_not_called()

        proposal.raw["requested_change"]["value"] = -9.0
        server.operator_proposal_queue._result_store = None
        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal.id,
                **self._review_binding(server, proposal.id),
                "approved": True,
                "approval_id": proposal.id,
                "idempotency_key": "evidence-gate-incomplete",
            },
        )
        evidence = self._decode_last_message(websocket)
        assert evidence["reason"] == (
            "operator_proposal_result_evidence_contract_incomplete"
        )
        assert evidence["no_console_send_proven"] is True
        assert proposal.status == "pending"
        assert "evidence-gate-incomplete" not in server._operator_apply_idempotency_keys
        server._apply_manual_console_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_manual_write_deadline_rechecked_after_pre_read_and_before_retry(
        self,
        dummy_wing_server,
    ):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"
        server._record_manual_write_event = MagicMock(return_value=None)
        clock = [100.0]

        async def expiring_pre_read(**_kwargs):
            clock[0] = 102.0
            return {
                "value": -8.0,
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "cache_freshness": {"status": "fresh"},
            }

        server._manual_write_read_value = AsyncMock(side_effect=expiring_pre_read)
        with patch("server.time.time", side_effect=lambda: clock[0]):
            before_send = await server._apply_manual_console_write_unlocked(
                None,
                channel=1,
                value=-9.0,
                value_type="fader",
                source="unit:deadline",
                approved=True,
                approval_id="deadline-before-send",
                approval_expires_at_epoch=101.0,
            )
        assert before_send["reason"] == "operator_approval_expired"
        assert before_send["no_console_send_proven"] is True
        assert before_send["write_attempts"] == 0
        server.mixer_client.set_channel_fader.assert_not_called()

        clock[0] = 100.0
        server._manual_write_read_value = AsyncMock(
            return_value={
                "value": -8.0,
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "cache_freshness": {"status": "fresh"},
            }
        )
        server._append_manual_write_wal = MagicMock(return_value="unit-wal")
        server.activate_system_emergency_stop = AsyncMock()
        server.mixer_client.set_channel_fader = MagicMock(return_value=False)

        async def advance_past_deadline(_delay):
            clock[0] = 102.0

        with patch("server.time.time", side_effect=lambda: clock[0]), patch(
            "server.asyncio.sleep",
            side_effect=advance_past_deadline,
        ):
            after_attempt = await server._apply_manual_console_write_unlocked(
                None,
                channel=1,
                value=-9.0,
                value_type="fader",
                source="unit:deadline",
                approved=True,
                approval_id="deadline-after-attempt",
                approval_expires_at_epoch=101.0,
            )
        assert after_attempt["reason"] == (
            "operator_approval_expired_after_write_attempt"
        )
        assert after_attempt["no_console_send_proven"] is False
        assert after_attempt["write_state_unknown"] is True
        assert after_attempt["manual_review_required"] is True
        assert after_attempt["write_attempts"] == 1
        assert server.mixer_client.set_channel_fader.call_count == 1

        clock[0] = 100.0
        server.mixer_client.set_channel_fader.reset_mock()

        def expire_during_evidence_boundary(_token):
            clock[0] = 102.0
            return {"transitioned": True, "reason": None}

        server.operator_proposal_queue.mark_result_evidence_send_may_have_occurred = (
            MagicMock(side_effect=expire_during_evidence_boundary)
        )
        with patch("server.time.time", side_effect=lambda: clock[0]):
            after_boundary = await server._apply_manual_console_write_unlocked(
                None,
                channel=1,
                value=-9.0,
                value_type="fader",
                source="unit:deadline",
                approved=True,
                approval_id="deadline-after-evidence-boundary",
                approval_expires_at_epoch=101.0,
                result_evidence_reservation_binding=object(),
            )
        assert after_boundary["reason"] == "operator_approval_expired"
        assert after_boundary["no_console_send_proven"] is True
        assert after_boundary["write_attempts"] == 0
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_cross_proposal_evidence_binding_blocks_before_mixer_send(
        self,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        store = self._enable_result_evidence_lifecycle(server, tmp_path)
        server.operator_mode = "supervised"
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"
        server._record_manual_write_event = MagicMock(return_value=None)
        server._manual_write_read_value = AsyncMock(
            return_value={
                "value": -8.0,
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "cache_freshness": {"status": "fresh"},
            }
        )
        server._append_manual_write_wal = MagicMock(return_value="unit-wal")
        server.mixer_client.set_channel_fader.reset_mock()
        mode = server.get_operator_mode_status()
        proposals = []
        reservations = []
        for suffix in ("a", "b"):
            created = server.operator_proposal_queue.create(
                {
                    "id": f"binding-{suffix}",
                    "title": f"binding {suffix}",
                    "requested_change": {
                        "value_type": "fader",
                        "channel": 1,
                        "value": -9.0,
                    },
                },
                mode,
            )
            assert created["status"] == "created"
            proposal = server.operator_proposal_queue.find(
                proposal_id=f"binding-{suffix}"
            )
            proposals.append(proposal)
            reservations.append(
                server.operator_proposal_queue.reserve_result_evidence(proposal)[
                    "reservation_binding"
                ]
            )
        binding_a, binding_b = reservations
        forged = ResultEvidenceReservationBinding(
            token=binding_b.token,
            proposal_id=proposals[0].id,
            review_content_digest=proposals[0].review_content_digest,
        )

        blocked = await server._apply_manual_console_write_unlocked(
            None,
            channel=1,
            value=-9.0,
            value_type="fader",
            source="unit:cross-binding",
            approved=True,
            approval_id="cross-binding",
            result_evidence_reservation_binding=forged,
        )

        assert blocked["reason"] == (
            "operator_proposal_result_evidence_reservation_binding_mismatch"
        )
        assert blocked["no_console_send_proven"] is True
        assert blocked["write_attempts"] == 0
        assert server.operator_proposal_queue.result_persistence_failed is True
        server.mixer_client.set_channel_fader.assert_not_called()
        for binding in (binding_a, binding_b):
            marker = store.root / f".reservation.{binding.token}"
            assert marker.exists()
            assert json.loads(marker.read_text(encoding="utf-8"))["phase"] == (
                "reserved_no_send"
            )

    @pytest.mark.asyncio
    async def test_canonical_manual_success_is_evictable_but_unknown_result_is_not(
        self,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"
        server._record_manual_write_event = MagicMock(return_value=None)
        server._append_manual_write_wal = MagicMock(return_value="unit-wal")

        result = await server._apply_manual_console_write_unlocked(
            None,
            channel=1,
            value=-9.0,
            value_type="fader",
            source="unit:evictable-success",
            approved=True,
            approval_id="evictable-success",
        )

        assert result["status"] == "applied"
        assert result["wal_terminal"] == "committed"
        assert result["physical_verified"] is True
        assert result["audio_recheck_required"] is False
        assert result["audio_recheck_disposition"] == (
            "not_required_by_manual_write_contract"
        )
        assert "audio_recheck_passed" not in result
        store = ResultEvidenceStore(
            tmp_path / "eviction-results",
            max_artifacts=1,
            min_free_bytes=0,
        )
        digest = "sha256:" + ("a" * 64)
        token = store.reserve(
            proposal_id="verified-manual-success",
            review_content_digest=digest,
        )
        record = store.commit(
            ResultEvidenceReservationBinding(
                token=token,
                proposal_id="verified-manual-success",
                review_content_digest=digest,
            ),
            result,
        )
        assert record.pinned is False
        replacement_digest = "sha256:" + ("b" * 64)
        replacement_token = store.reserve(
            proposal_id="replacement",
            review_content_digest=replacement_digest,
        )
        assert not (store.root / record.artifact_name).exists()
        store.release(
            ResultEvidenceReservationBinding(
                token=replacement_token,
                proposal_id="replacement",
                review_content_digest=replacement_digest,
            )
        )

        unknown_digest = "sha256:" + ("c" * 64)
        unknown_token = store.reserve(
            proposal_id="unknown-write",
            review_content_digest=unknown_digest,
        )
        unknown = store.commit(
            ResultEvidenceReservationBinding(
                token=unknown_token,
                proposal_id="unknown-write",
                review_content_digest=unknown_digest,
            ),
            {
                "status": "error",
                "no_console_send_proven": False,
                "write_state_unknown": True,
                "physical_state_restored": False,
                "manual_review_required": True,
            },
        )
        with pytest.raises(ResultEvidenceError) as exc_info:
            store.reserve(
                proposal_id="must-not-evict-unknown",
                review_content_digest="sha256:" + ("d" * 64),
            )
        assert exc_info.value.code == (
            "operator_proposal_result_recovery_required"
        )
        assert (store.root / unknown.artifact_name).exists()

    @pytest.mark.asyncio
    async def test_operator_apply_result_capacity_failure_precedes_claim(
        self,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        self._enable_result_evidence_lifecycle(server, tmp_path)
        self._configure_deep_operator_apply(server)
        server.mixer_client.state["/ch/1/fdr"] = -8.0
        server.mixer_client.state_timestamps["/ch/1/fdr"] = time.time()
        server._apply_manual_console_write = AsyncMock()
        websocket = MagicMock()
        websocket.send = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "id": "capacity-before-claim",
                "title": "capacity before claim",
                "channel": 1,
                "value_type": "fader",
                "value": -8.5,
            },
        )
        proposal = server.operator_proposal_queue.find(
            proposal_id="capacity-before-claim"
        )
        claim = MagicMock(
            wraps=server.operator_proposal_queue.claim_verified_for_apply
        )
        server.operator_proposal_queue.claim_verified_for_apply = claim
        server.operator_proposal_queue.reserve_result_evidence = MagicMock(
            return_value={
                "reserved": False,
                "reason": "operator_proposal_result_evidence_capacity_exhausted",
                "reservation_token": None,
            }
        )

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal.id,
                **self._review_binding(server, proposal.id),
                "approved": True,
                "approval_id": "capacity-approval",
                "idempotency_key": "capacity-idempotency",
            },
        )

        blocked = self._decode_last_message(websocket)
        assert blocked["reason"] == (
            "operator_proposal_result_evidence_capacity_exhausted"
        )
        assert blocked["no_console_send_proven"] is True
        assert blocked["proposal_status_mutation"] is False
        assert proposal.status == "pending"
        assert "capacity-idempotency" not in server._operator_apply_idempotency_keys
        claim.assert_not_called()
        server._apply_manual_console_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_restarted_result_hazard_blocks_handler_before_reserve_claim_or_send(
        self,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        root = tmp_path / "restarted-result-hazard"
        first_store = ResultEvidenceStore(root, min_free_bytes=0)
        review_digest = "sha256:" + ("7" * 64)
        token = first_store.reserve(
            proposal_id="prior-unknown-write",
            review_content_digest=review_digest,
        )
        binding = ResultEvidenceReservationBinding(
            token=token,
            proposal_id="prior-unknown-write",
            review_content_digest=review_digest,
        )
        first_store.mark_send_may_have_occurred(binding)
        prior = first_store.commit(
            binding,
            {
                "status": "error",
                "no_console_send_proven": False,
                "write_state_unknown": True,
                "rollback_state_unknown": True,
                "physical_state_restored": False,
                "manual_review_required": True,
            },
        )
        restarted_store = ResultEvidenceStore(root, min_free_bytes=0)
        server.operator_proposal_queue = OperatorProposalQueue(
            result_store=restarted_store,
            result_evidence_lifecycle_ready=True,
        )
        self._configure_deep_operator_apply(server)
        server._apply_manual_console_write = AsyncMock()
        websocket = MagicMock()
        websocket.send = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "id": "blocked-by-prior-result",
                "title": "must stay pending",
                "channel": 1,
                "value_type": "fader",
                "value": -8.5,
            },
        )
        proposal = server.operator_proposal_queue.find(
            proposal_id="blocked-by-prior-result"
        )
        reserve = MagicMock(
            wraps=server.operator_proposal_queue.reserve_result_evidence
        )
        claim = MagicMock(
            wraps=server.operator_proposal_queue.claim_verified_for_apply
        )
        server.operator_proposal_queue.reserve_result_evidence = reserve
        server.operator_proposal_queue.claim_verified_for_apply = claim

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal.id,
                **self._review_binding(server, proposal.id),
                "approved": True,
                "approval_id": "recovery-blocked-approval",
                "idempotency_key": "recovery-blocked-idempotency",
            },
        )

        blocked = self._decode_last_message(websocket)
        assert blocked["reason"] == (
            "operator_proposal_result_recovery_required"
        )
        assert "global_hazards" not in blocked
        assert blocked["no_console_send_proven"] is True
        assert blocked["result_evidence_reserved"] is False
        assert blocked["proposal_status_mutation"] is False
        assert proposal.status == "pending"
        assert restarted_store.active_reservation_count == 0
        assert (root / prior.artifact_name).exists()
        assert "recovery-blocked-idempotency" not in (
            server._operator_apply_idempotency_keys
        )
        reserve.assert_not_called()
        claim.assert_not_called()
        server._apply_manual_console_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_preclaim_invalid_change_is_non_mutating_and_uses_public_proposal(
        self,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        store = self._enable_result_evidence_lifecycle(server, tmp_path)
        self._configure_deep_operator_apply(server)
        websocket = MagicMock()
        websocket.send = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "id": "preclaim-no-change",
                "title": "no applyable change",
                "kind": "analysis_note",
            },
        )
        proposal = server.operator_proposal_queue.find(
            proposal_id="preclaim-no-change"
        )
        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal.id,
                **self._review_binding(server, proposal.id),
                "approved": True,
                "approval_id": "preclaim-no-change-approval",
                "idempotency_key": "preclaim-no-change-idempotency",
            },
        )

        blocked = self._decode_last_message(websocket)
        assert blocked["reason"] == "proposal_has_no_applyable_requested_change"
        assert blocked["proposal_status_mutation"] is False
        assert blocked["result_evidence_reserved"] is False
        assert proposal.status == "pending"
        assert proposal.result is None
        assert store.active_reservation_count == 0
        assert (
            "preclaim-no-change-idempotency"
            not in server._operator_apply_idempotency_keys
        )

    @pytest.mark.asyncio
    async def test_direct_apply_consumes_bound_result_reservation(
        self,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        store = self._enable_result_evidence_lifecycle(server, tmp_path)
        self._configure_deep_operator_apply(server)
        server.mixer_client.state["/ch/1/fdr"] = -8.0
        server.mixer_client.state_timestamps["/ch/1/fdr"] = time.time()
        websocket = MagicMock()
        websocket.send = AsyncMock()
        seen_tokens = []

        async def apply_with_boundary(*_args, **kwargs):
            binding = kwargs["result_evidence_reservation_binding"]
            seen_tokens.append(binding.token)
            boundary = (
                server.operator_proposal_queue
                .mark_result_evidence_send_may_have_occurred(binding)
            )
            assert boundary["transitioned"] is True
            return _verified_manual_apply_result(
                channel=1,
                value_type="fader",
                effective_value=-8.5,
                old_value=-8.0,
                readback_value=-8.5,
                readback_status="verified_from_physical_readback",
                proof_status="verified",
                rollback_available=True,
                rollback_value=-8.0,
            )

        server._apply_manual_console_write = AsyncMock(
            side_effect=apply_with_boundary
        )
        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "id": "direct-result-lifecycle",
                "title": "direct result lifecycle",
                "channel": 1,
                "value_type": "fader",
                "value": -8.5,
            },
        )
        proposal = server.operator_proposal_queue.find(
            proposal_id="direct-result-lifecycle"
        )

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal.id,
                **self._review_binding(server, proposal.id),
                "approved": True,
                "approval_id": "direct-result-approval",
                "idempotency_key": "direct-result-idempotency",
            },
        )

        applied = self._decode_last_message(websocket)
        assert applied["status"] == "applied"
        assert applied["result_detail_available"] is True
        assert applied["result_digest"].startswith("sha256:")
        assert len(seen_tokens) == 1 and len(seen_tokens[0]) == 32
        assert store.active_reservation_count == 0
        terminal = server.operator_proposal_queue.find(proposal_id=proposal.id)
        evidence = server.operator_proposal_queue.load_result_evidence(
            terminal,
            expected_result_digest=applied["result_digest"],
        )
        assert json.loads(evidence)["status"] == "applied"

        await server._dispatch["get_operator_proposal_result_detail"](
            websocket,
            {
                "contract_version": 2,
                "proposal_id": proposal.id,
                "expected_result_digest": applied["result_digest"],
                "chunk_index": 0,
            },
        )
        detail = self._decode_last_message(websocket)
        assert detail["type"] == "operator_proposal_result_detail"
        assert detail["result_digest"] == applied["result_digest"]

    @pytest.mark.asyncio
    async def test_request_token_is_frozen_before_claim(
        self,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        self._enable_result_evidence_lifecycle(server, tmp_path)
        self._configure_deep_operator_apply(server)
        server.mixer_client.state["/ch/1/fdr"] = -8.0
        server.mixer_client.state_timestamps["/ch/1/fdr"] = time.time()
        server.validate_operator_approval_token = MagicMock(
            side_effect=lambda **facts: (
                facts["token"] == "frozen-token"
                and facts["requested_change"].get("value") == -7.5
            )
        )
        websocket = MagicMock()
        websocket.send = AsyncMock()

        async def apply_with_boundary(*_args, **kwargs):
            server.operator_proposal_queue.mark_result_evidence_send_may_have_occurred(
                kwargs["result_evidence_reservation_binding"]
            )
            return _verified_manual_apply_result(
                channel=1,
                value_type="fader",
                effective_value=-7.5,
                readback_status="verified_from_physical_readback",
                proof_status="verified",
                rollback_available=True,
            )

        server._apply_manual_console_write = AsyncMock(
            side_effect=apply_with_boundary
        )
        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "id": "immutable-request-token",
                "title": "immutable request token",
                "channel": 1,
                "value_type": "fader",
                "value": -7.5,
            },
        )
        proposal = server.operator_proposal_queue.find(
            proposal_id="immutable-request-token"
        )
        request = {
            "proposal_id": proposal.id,
            **self._review_binding(server, proposal.id),
            "approved": True,
            "approval_id": "immutable-request-approval",
            "approval_token": "frozen-token",
            "idempotency_key": "immutable-request-idempotency",
        }
        original_claim = (
            server.operator_proposal_queue.claim_verified_for_apply
        )

        def mutate_request_after_claim(*args, **kwargs):
            claimed = original_claim(*args, **kwargs)
            request["approval_token"] = "mutated-after-claim"
            return claimed

        server.operator_proposal_queue.claim_verified_for_apply = MagicMock(
            side_effect=mutate_request_after_claim
        )

        await server._dispatch["apply_operator_proposal"](websocket, request)

        applied = self._decode_last_message(websocket)
        assert applied["status"] == "applied"
        assert request["approval_token"] == "mutated-after-claim"
        assert server.validate_operator_approval_token.call_args.kwargs["token"] == (
            "frozen-token"
        )

    @pytest.mark.asyncio
    async def test_linked_first_step_no_send_is_retryable_and_consumes_reservation(
        self,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        store = self._enable_result_evidence_lifecycle(server, tmp_path)
        self._configure_deep_operator_apply(server)
        now = time.time()
        server.mixer_client.state.update({"/ch/1/fdr": -8.0, "/ch/2/fdr": -9.0})
        server.mixer_client.state_timestamps.update(
            {"/ch/1/fdr": now, "/ch/2/fdr": now}
        )
        server._apply_manual_console_write = AsyncMock(
            return_value={
                "status": "blocked",
                "success": False,
                "reason": "pre_send_safety_block",
                "no_console_send_proven": True,
                "write_attempts": 0,
            }
        )
        websocket = MagicMock()
        websocket.send = AsyncMock()
        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "id": "linked-first-step-no-send",
                "title": "linked first-step no-send",
                "requested_changes": [
                    {"value_type": "fader", "channel": 1, "value": -8.5},
                    {"value_type": "fader", "channel": 2, "value": -9.5},
                ],
            },
        )
        proposal = server.operator_proposal_queue.find(
            proposal_id="linked-first-step-no-send"
        )

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal.id,
                **self._review_binding(server, proposal.id),
                "approved": True,
                "approval_id": "linked-first-step-approval",
                "idempotency_key": "linked-first-step-idempotency",
            },
        )

        blocked = self._decode_last_message(websocket)
        assert blocked["status"] == "apply_blocked"
        assert blocked["reason"] == (
            "operator_apply_reason_available_in_result_detail"
        )
        assert blocked["no_console_send_proven"] is True
        assert blocked["rollback_attempted"] is False
        assert blocked["idempotency"]["released"] is True
        assert blocked["result_detail_available"] is True
        evidence = server.operator_proposal_queue.load_result_evidence(
            proposal,
            expected_result_digest=blocked["result_digest"],
        )
        full_result = json.loads(evidence)
        assert full_result["reason"] == "pre_send_safety_block"
        assert full_result["no_console_send_proven"] is True
        assert full_result["rollback_results"] == []
        assert proposal.status == "pending"
        assert store.active_reservation_count == 0
        assert "linked-first-step-idempotency" not in (
            server._operator_apply_idempotency_keys
        )
        assert server._apply_manual_console_write.await_count == 1

    @pytest.mark.asyncio
    async def test_real_handler_proven_no_send_releases_wal_key_for_retry_and_restart(
        self,
        dummy_wing_server,
        tmp_path,
        monkeypatch,
    ):
        server = dummy_wing_server
        self._enable_result_evidence_lifecycle(server, tmp_path)
        self._configure_deep_operator_apply(server)
        runtime_dir = tmp_path / "handler-wal-runtime"
        server._console_runtime_dir = runtime_dir
        server._console_write_wal_path = runtime_dir / "manual_write_wal.jsonl"
        server._console_writer_lock_path = runtime_dir / "console_writer.lock"
        server._operator_apply_idempotency_keys = {}
        server._durable_operator_apply_idempotency_keys = {}
        server._durable_operator_proposal_ids = set()
        server._manual_write_prepared_transactions = {}
        server._manual_write_recovery_transactions = {}
        server._manual_write_wal_lifecycles = {}
        server._manual_write_wal_integrity_ok = True
        server._manual_write_emergency_stopped = False
        server._manual_write_cooldown_sec = 0.0
        server._record_manual_write_event = MagicMock(return_value=None)
        server.mixer_client.state["/ch/1/fdr"] = -8.0
        server.mixer_client.state_timestamps["/ch/1/fdr"] = time.time()
        append = server._append_manual_write_wal

        def change_console_after_prepare(phase, transaction_id, **payload):
            path = append(phase, transaction_id, **payload)
            if phase == "prepared":
                server.mixer_client.state["/ch/1/fdr"] = -7.0
            return path

        server._append_manual_write_wal = change_console_after_prepare
        websocket = MagicMock()
        websocket.send = AsyncMock()
        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "id": "handler-wal-retry",
                "title": "handler WAL retry",
                "requested_change": {
                    "value_type": "fader",
                    "channel": 1,
                    "value": -9.0,
                },
            },
        )
        proposal = server.operator_proposal_queue.find(
            proposal_id="handler-wal-retry"
        )
        request = {
            "proposal_id": proposal.id,
            **self._review_binding(server, proposal.id),
            "approved": True,
            "approval_id": "handler-wal-retry",
            "idempotency_key": "handler-wal-retry-key",
        }

        await server._dispatch["apply_operator_proposal"](websocket, request)

        first = self._decode_last_message(websocket)
        assert first["reason"] == (
            "operator_apply_reason_available_in_result_detail"
        )
        first_evidence = server.operator_proposal_queue.load_result_evidence(
            proposal,
            expected_result_digest=first["result_digest"],
        )
        assert json.loads(first_evidence)["reason"] == (
            "manual_write_cas_target_changed_or_unverified"
        )
        assert first["idempotency"]["released"] is True
        assert proposal.status == "pending"
        assert server.mixer_client.set_channel_fader.call_count == 0
        assert "handler-wal-retry-key" not in (
            server._operator_apply_idempotency_keys
        )
        assert "handler-wal-retry-key" not in (
            server._durable_operator_apply_idempotency_keys
        )
        phases = [
            json.loads(line)["phase"]
            for line in server._manual_write_wal_path().read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert phases == ["prepared", "aborted_before_send"]

        server.mixer_client.state["/ch/1/fdr"] = -8.0
        await server._dispatch["apply_operator_proposal"](websocket, request)
        second = self._decode_last_message(websocket)
        assert second.get("reason") != "duplicate_idempotency_key"
        second_evidence = server.operator_proposal_queue.load_result_evidence(
            proposal,
            expected_result_digest=second["result_digest"],
        )
        assert json.loads(second_evidence)["reason"] == (
            "manual_write_cas_target_changed_or_unverified"
        )

        monkeypatch.setenv(
            "AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR",
            str(runtime_dir),
        )
        restarted = AutoMixerServer(ws_host="localhost", ws_port=8766)
        assert "handler-wal-retry-key" not in (
            restarted._durable_operator_apply_idempotency_keys
        )
        assert "handler-wal-retry" not in restarted._durable_operator_proposal_ids
        assert restarted._manual_write_recovery_transactions == {}

    @pytest.mark.asyncio
    async def test_linked_verified_success_persists_evictable_canonical_evidence(
        self,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        store = ResultEvidenceStore(
            tmp_path / "linked-evictable-results",
            max_artifacts=1,
            min_free_bytes=0,
        )
        server.operator_proposal_queue = OperatorProposalQueue(
            result_store=store,
            result_evidence_lifecycle_ready=True,
        )
        self._configure_deep_operator_apply(server)
        now = time.time()
        server.mixer_client.state.update({"/ch/1/fdr": -8.0, "/ch/2/fdr": -9.0})
        server.mixer_client.state_timestamps.update(
            {"/ch/1/fdr": now, "/ch/2/fdr": now}
        )
        call_index = 0

        async def apply_verified(*_args, **kwargs):
            nonlocal call_index
            call_index += 1
            boundary = (
                server.operator_proposal_queue
                .mark_result_evidence_send_may_have_occurred(
                    kwargs["result_evidence_reservation_binding"]
                )
            )
            assert boundary["transitioned"] is True
            channel = kwargs["channel"]
            return {
                "status": "applied",
                "success": True,
                "channel": channel,
                "value_type": "fader",
                "effective_value": kwargs["value"],
                "old_value": -8.0 if channel == 1 else -9.0,
                "readback_value": kwargs["value"],
                "readback_status": "verified_from_physical_readback",
                "proof_status": "verified",
                "rollback_available": True,
                "rollback_value": -8.0 if channel == 1 else -9.0,
                "no_console_send_proven": False,
                "physical_verified": True,
                "wal_terminal": "committed",
                "audio_recheck_required": False,
                "audio_recheck_disposition": (
                    "not_required_by_manual_write_contract"
                ),
                "write_state_unknown": False,
                "rollback_state_unknown": False,
                "manual_review_required": False,
            }

        server._apply_manual_console_write = AsyncMock(side_effect=apply_verified)
        websocket = MagicMock()
        websocket.send = AsyncMock()
        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "id": "linked-evictable-success",
                "title": "linked evictable success",
                "requested_changes": [
                    {"value_type": "fader", "channel": 1, "value": -8.5},
                    {"value_type": "fader", "channel": 2, "value": -9.5},
                ],
            },
        )
        proposal = server.operator_proposal_queue.find(
            proposal_id="linked-evictable-success"
        )
        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal.id,
                **self._review_binding(server, proposal.id),
                "approved": True,
                "approval_id": "linked-evictable-approval",
                "idempotency_key": "linked-evictable-idempotency",
            },
        )

        applied = self._decode_last_message(websocket)
        assert applied["status"] == "applied"
        assert call_index == 2
        assert proposal.result_evidence_pinned is False
        evidence = json.loads(
            server.operator_proposal_queue.load_result_evidence(
                proposal,
                expected_result_digest=proposal.result_digest,
            )
        )
        assert evidence["physical_verified"] is True
        assert evidence["wal_terminal"] == "committed"
        assert evidence["audio_recheck_required"] is False
        assert evidence["audio_recheck_disposition"] == (
            "not_required_by_linked_manual_write_contract"
        )
        replacement_digest = "sha256:" + ("e" * 64)
        replacement_token = store.reserve(
            proposal_id="linked-capacity-replacement",
            review_content_digest=replacement_digest,
        )
        assert not (store.root / f"{proposal.result_digest[7:]}.json").exists()
        store.release(
            ResultEvidenceReservationBinding(
                token=replacement_token,
                proposal_id="linked-capacity-replacement",
                review_content_digest=replacement_digest,
            )
        )

    @pytest.mark.asyncio
    async def test_transaction_apply_consumes_same_preclaim_reservation(
        self,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        store = self._enable_result_evidence_lifecycle(server, tmp_path)
        self._configure_deep_operator_apply(server)
        websocket = MagicMock()
        websocket.send = AsyncMock()
        plan = {
            "transaction_id": "handler-transaction-lifecycle",
            "operations": [
                {
                    "operation_id": "handler-fader-1",
                    "family_id": "fader",
                    "write_kind": "fader",
                    "channel": 1,
                    "target_value": -9.0,
                }
            ],
        }

        async def apply_transaction(_plan, **kwargs):
            boundary = (
                server.operator_proposal_queue
                .mark_result_evidence_send_may_have_occurred(
                    kwargs["result_evidence_reservation_binding"]
                )
            )
            assert boundary["transitioned"] is True
            return _verified_manual_apply_result(
                soundcheck_readiness={"status": "safe_core_ready"},
            )

        server.apply_supervised_soundcheck_transaction = AsyncMock(
            side_effect=apply_transaction
        )
        with patch(
            "operator_proposal_queue._apply_blockers_for_transaction_plan",
            return_value=[],
        ):
            await server._dispatch["create_operator_proposal"](
                websocket,
                {
                    "id": "handler-transaction-proposal",
                    "title": "handler transaction lifecycle",
                    "kind": "supervised_soundcheck_transaction",
                    "supervised_transaction_plan": plan,
                },
            )
            proposal = server.operator_proposal_queue.find(
                proposal_id="handler-transaction-proposal"
            )
            await server._dispatch["apply_operator_proposal"](
                websocket,
                {
                    "proposal_id": proposal.id,
                    **self._review_binding(server, proposal.id),
                    "approved": True,
                    "approval_id": "handler-transaction-approval",
                    "idempotency_key": "handler-transaction-idempotency",
                },
            )

        applied = self._decode_last_message(websocket)
        assert applied["status"] == "applied"
        assert applied["result_detail_available"] is True
        assert store.active_reservation_count == 0
        binding = server.apply_supervised_soundcheck_transaction.await_args.kwargs[
            "result_evidence_reservation_binding"
        ]
        assert isinstance(binding.token, str) and len(binding.token) == 32

    @pytest.mark.asyncio
    async def test_post_boundary_cancellation_persists_unknown_truth_once(
        self,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        store = self._enable_result_evidence_lifecycle(server, tmp_path)
        self._configure_deep_operator_apply(server)
        server.mixer_client.state["/ch/1/fdr"] = -8.0
        server.mixer_client.state_timestamps["/ch/1/fdr"] = time.time()
        server.activate_system_emergency_stop = AsyncMock()
        websocket = MagicMock()
        websocket.send = AsyncMock()

        async def cancel_after_boundary(*_args, **kwargs):
            server.operator_proposal_queue.mark_result_evidence_send_may_have_occurred(
                kwargs["result_evidence_reservation_binding"]
            )
            raise asyncio.CancelledError()

        server._apply_manual_console_write = AsyncMock(
            side_effect=cancel_after_boundary
        )
        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "id": "cancel-after-boundary",
                "title": "cancel after boundary",
                "channel": 1,
                "value_type": "fader",
                "value": -8.5,
            },
        )
        proposal = server.operator_proposal_queue.find(
            proposal_id="cancel-after-boundary"
        )

        with pytest.raises(asyncio.CancelledError):
            await server._dispatch["apply_operator_proposal"](
                websocket,
                {
                    "proposal_id": proposal.id,
                    **self._review_binding(server, proposal.id),
                    "approved": True,
                    "approval_id": "cancel-after-boundary-approval",
                    "idempotency_key": "cancel-after-boundary-idempotency",
                },
            )

        terminal = server.operator_proposal_queue.find(proposal_id=proposal.id)
        assert terminal.status == "apply_error"
        assert terminal.result["no_console_send_proven"] is False
        assert terminal.result["manual_review_required"] is True
        assert terminal.result_detail_available is True
        assert store.active_reservation_count == 0
        assert "cancel-after-boundary-idempotency" in (
            server._operator_apply_idempotency_keys
        )
        server.activate_system_emergency_stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_post_send_result_persistence_failure_survives_restart(
        self,
        dummy_wing_server,
        tmp_path,
    ):
        server = dummy_wing_server
        store = self._enable_result_evidence_lifecycle(server, tmp_path)
        self._configure_deep_operator_apply(server)
        server.mixer_client.state["/ch/1/fdr"] = -8.0
        server.mixer_client.state_timestamps["/ch/1/fdr"] = time.time()
        server.activate_system_emergency_stop = AsyncMock()
        websocket = MagicMock()
        websocket.send = AsyncMock()

        async def applied_after_boundary(*_args, **kwargs):
            server.operator_proposal_queue.mark_result_evidence_send_may_have_occurred(
                kwargs["result_evidence_reservation_binding"]
            )
            return _verified_manual_apply_result()

        server._apply_manual_console_write = AsyncMock(
            side_effect=applied_after_boundary
        )
        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "id": "persistence-failure-after-send",
                "title": "persistence failure after send",
                "channel": 1,
                "value_type": "fader",
                "value": -8.5,
            },
        )
        proposal = server.operator_proposal_queue.find(
            proposal_id="persistence-failure-after-send"
        )
        store._write_atomic = MagicMock(
            side_effect=ResultEvidenceError(
                "operator_proposal_result_evidence_persistence_failed"
            )
        )

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal.id,
                **self._review_binding(server, proposal.id),
                "approved": True,
                "approval_id": "persistence-failure-approval",
                "idempotency_key": "persistence-failure-idempotency",
            },
        )

        failed = self._decode_last_message(websocket)
        assert failed["status"] == "apply_error"
        assert failed["reason"] == (
            "operator_proposal_result_persistence_failed"
        )
        assert failed["write_state_unknown"] is True
        assert failed["manual_review_required"] is True
        assert failed["result_detail_available"] is False
        assert server.operator_proposal_queue.result_persistence_failed is True
        assert store.active_reservation_count == 1
        server.activate_system_emergency_stop.assert_awaited_once()

        restarted_store = ResultEvidenceStore(
            store.root,
            min_free_bytes=0,
        )
        restarted_queue = OperatorProposalQueue(
            result_store=restarted_store,
            result_evidence_lifecycle_ready=True,
        )
        assert restarted_queue.result_persistence_failed is False
        assert restarted_queue.result_evidence_contract_ready is False
        assert restarted_queue.result_recovery_hazards[0]["phase"] == (
            "persistence_failed_after_send"
        )

    @staticmethod
    def _assert_quarantined_operator_correction(payload, method_id, blocked_actions):
        correction = payload["operator_correction"]
        assert correction["source_evidence"]["method_id"] == method_id
        assert correction["read_only_or_dry_run_only"] is True
        assert correction["live_apply_allowed"] is False
        assert correction["queue_import_allowed"] is False
        assert correction["proposed_correction"]["surface"] == payload["surface"]
        assert correction["future_wing_readback_contract"]["required_before_live_promotion"] is True
        assert correction["future_wing_readback_contract"]["blocked_live_actions"] == blocked_actions

    @staticmethod
    def _quarantined_operator_expectation(surface):
        return {
            "start_realtime_correction": (
                "lufs_gain_staging",
                ["preamp_gain_write", "input_trim_write", "gain_up_without_operator_opt_in"],
            ),
            "reset_trim": (
                "lufs_gain_staging",
                ["preamp_gain_write", "input_trim_write", "gain_up_without_operator_opt_in"],
            ),
            "start_realtime_fader": (
                "auto_fader_v1",
                ["unsupervised_fader_write", "fader_above_zero_dbfs"],
            ),
            "apply_auto_balance": (
                "auto_fader_v1",
                ["unsupervised_fader_write", "fader_above_zero_dbfs"],
            ),
            "set_eq": ("auto_eq", ["eq_band_write", "eq_on_write"]),
            "apply_eq_correction": ("auto_eq", ["eq_band_write", "eq_on_write"]),
            "reset_eq": ("auto_eq", ["eq_band_write", "eq_on_write"]),
            "reset_all_eq": ("auto_eq", ["eq_band_write", "eq_on_write"]),
            "apply_channel_correction": ("auto_eq", ["eq_band_write", "eq_on_write"]),
            "apply_all_corrections": ("auto_eq", ["eq_band_write", "eq_on_write"]),
            "set_compressor": (
                "auto_compressor",
                ["compressor_write", "makeup_gain_write"],
            ),
            "start_auto_compressor": (
                "auto_compressor",
                ["compressor_write", "makeup_gain_write"],
            ),
            "start_auto_compressor_soundcheck": (
                "auto_compressor",
                ["compressor_write", "makeup_gain_write"],
            ),
            "start_auto_compressor_live": (
                "auto_compressor",
                ["compressor_write", "makeup_gain_write"],
            ),
            "set_auto_compressor_profile": (
                "auto_compressor",
                ["compressor_write", "makeup_gain_write"],
            ),
            "set_auto_compressor_manual": (
                "auto_compressor",
                ["compressor_write", "makeup_gain_write"],
            ),
            "set_pan": ("auto_panner", ["pan_write", "stereo_width_write"]),
            "apply_phase_corrections": (
                "phase_alignment",
                ["delay_write", "polarity_write"],
            ),
            "reset_phase_delay": (
                "phase_alignment",
                ["delay_write", "polarity_write"],
            ),
            "set_insert": (
                "fx_direct_apply",
                ["fx_parameter_write", "insert_bypass_write", "fx_rack_write"],
            ),
            "set_fx_model": (
                "fx_direct_apply",
                ["fx_parameter_write", "insert_bypass_write", "fx_rack_write"],
            ),
            "set_fx_on": (
                "fx_direct_apply",
                ["fx_parameter_write", "insert_bypass_write", "fx_rack_write"],
            ),
            "set_fx_mix": (
                "fx_direct_apply",
                ["fx_parameter_write", "insert_bypass_write", "fx_rack_write"],
            ),
            "set_fx_parameter": (
                "fx_direct_apply",
                ["fx_parameter_write", "insert_bypass_write", "fx_rack_write"],
            ),
            "voice_command:set_fader": (
                "auto_fader_v1",
                ["unsupervised_fader_write", "fader_above_zero_dbfs"],
            ),
            "voice_command:set_gain": (
                "lufs_gain_staging",
                ["preamp_gain_write", "input_trim_write", "gain_up_without_operator_opt_in"],
            ),
            "voice_command:set_pan": ("auto_panner", ["pan_write", "stereo_width_write"]),
            "voice_command:load_snap": (
                "scene_preset_management_proposal",
                ["scene_load", "scene_save", "preset_load", "preset_save"],
            ),
            "voice_command:mute_channel": (
                "mute_management_proposal",
                ["mute_write", "unmute_write", "mute_group_write"],
            ),
            "voice_command:volume_up": (
                "auto_fader_v1",
                ["unsupervised_fader_write", "fader_above_zero_dbfs"],
            ),
            "voice_command:volume_down": (
                "auto_fader_v1",
                ["unsupervised_fader_write", "fader_above_zero_dbfs"],
            ),
            "voice_command:eq_on": ("auto_eq", ["eq_band_write", "eq_on_write"]),
            "voice_command:eq_band_up": ("auto_eq", ["eq_band_write", "eq_on_write"]),
            "voice_command:eq_band_down": ("auto_eq", ["eq_band_write", "eq_on_write"]),
            "voice_command:compressor_on": (
                "auto_compressor",
                ["compressor_write", "makeup_gain_write"],
            ),
            "voice_command:compressor_threshold": (
                "auto_compressor",
                ["compressor_write", "makeup_gain_write"],
            ),
            "voice_command:compressor_gain_up": (
                "auto_compressor",
                ["compressor_write", "makeup_gain_write"],
            ),
            "voice_command:compressor_gain_down": (
                "auto_compressor",
                ["compressor_write", "makeup_gain_write"],
            ),
        }.get(surface)

    @staticmethod
    def _sequence_execution_payload(value_type, allowed_live_write_kinds=None):
        allowed_live_write_kinds = allowed_live_write_kinds or ["fader", "gain", "pan"]
        return {
            "type": "operator_proposal_apply_sequence_execution_result",
            "status": "applied",
            "summary": {
                "uses_existing_apply_operator_proposal": True,
                "live_apply_entrypoint": "apply_operator_proposal",
                "direct_live_write": False,
                "direct_manual_console_write_called": False,
                "manual_console_write_called_by_sequence_executor": False,
                "idempotency_reserved_by_executor_directly": False,
                "live_widening_allowed": False,
                "queries_console": False,
            },
            "safety": {
                "uses_existing_apply_operator_proposal": True,
                "live_apply_entrypoint": "apply_operator_proposal",
                "does_not_call_manual_console_write_directly": True,
                "allowed_live_write_kinds": allowed_live_write_kinds,
                "direct_live_write": False,
                "direct_manual_console_write_called": False,
                "manual_console_write_called_by_sequence_executor": False,
                "idempotency_reserved_by_executor_directly": False,
                "live_widening_allowed": False,
                "queries_console": False,
            },
            "execution_results": [
                {
                    "proposal_id": "unit-sequence-pan",
                    "status": "applied",
                    "apply_attempted": True,
                    "apply_request": {
                        "type": "apply_operator_proposal",
                        "proposal_id": "unit-sequence-pan",
                        "approved": True,
                        "idempotency_key": "unit-sequence-pan-key",
                    },
                    "apply_result": {
                        "status": "applied",
                        "channel": 1,
                        "value_type": value_type,
                        "effective_value": 0.0,
                        "readback_status": "verified_from_cache",
                        "proof_status": "verified",
                        "rollback_available": True,
                    },
                }
            ],
        }

    @staticmethod
    def _sequence_execution_audit_payload(value_type, allowed_live_write_kinds=None):
        allowed_live_write_kinds = allowed_live_write_kinds or ["fader", "gain", "pan"]
        return {
            "type": "operator_proposal_apply_sequence_execution_audit",
            "status": "passed",
            "summary": {
                "audit_only": True,
                "execution_allowed_by_audit": False,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "live_widening_allowed": False,
                "live_authority_granted": False,
            },
            "safety": {
                "audit_only": True,
                "does_not_call_apply_operator_proposal": True,
                "does_not_call_manual_console_write": True,
                "allowed_live_write_kinds": allowed_live_write_kinds,
                "execution_allowed_by_audit": False,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "live_widening_allowed": False,
                "live_authority_granted": False,
            },
            "audit_rows": [
                {
                    "proposal_id": "unit-sequence-pan",
                    "status": "passed",
                    "value_type": value_type,
                    "channel": 1,
                    "effective_value": 0.0,
                    "idempotency_key": "unit-sequence-pan-key",
                    "readback_status": "verified_from_cache",
                    "proof_status": "verified",
                    "rollback_available": True,
                }
            ],
            "blocked_results": [],
        }

    def test_manual_write_supervision_reports_mode_limited_live_kinds(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"

        status = server.get_manual_write_supervision_status()
        assert status["approved_live_write_kinds"] == ["fader", "gain"]

        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        status = server.get_manual_write_supervision_status()
        assert status["approved_live_write_kinds"] == []

    def test_wing_safety_monitor_filters_manual_scope_to_operator_mode(
        self,
        dummy_wing_server,
    ):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server.get_manual_write_supervision_status = lambda: {
            "approved_live_write_kinds": [
                "fader",
                "gain",
                "pan",
                "mute",
                "trim",
            ]
        }

        monitor = build_safety_monitor(server)

        assert (
            monitor["operator_mode"]["capabilities"]["allowed_live_write_kinds"]
            == ["fader", "gain"]
        )
        assert monitor["live_write_policy"]["approved_live_write_kinds"] == [
            "fader",
            "gain",
        ]

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
        assert status["approved_live_write_kinds"] == ["fader", "gain", "pan"]

    @pytest.mark.asyncio
    async def test_wing_sequence_plan_safety_scope_uses_deployment_boundary(
        self,
        dummy_wing_server,
    ):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()

        await server._dispatch["plan_operator_proposal_apply_sequence"](websocket, {})

        payload = self._decode_last_message(websocket)
        assert payload["operator_mode"]["wing_deployment"] is True
        assert (
            payload["operator_mode"]["capabilities"]["allowed_live_write_kinds"]
            == ["fader", "gain"]
        )
        assert payload["safety"]["allowed_live_write_kinds"] == ["fader", "gain"]
        assert "pan" not in payload["safety"]["allowed_live_write_kinds"]

    @pytest.mark.asyncio
    async def test_offline_sequence_plan_safety_scope_retains_current_proven_kinds(
        self,
        minimal_server,
    ):
        server = minimal_server
        server.connection_mode = "offline"
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()

        await server._dispatch["plan_operator_proposal_apply_sequence"](websocket, {})

        payload = self._decode_last_message(websocket)
        assert payload["operator_mode"]["wing_deployment"] is False
        assert (
            payload["operator_mode"]["capabilities"]["allowed_live_write_kinds"]
            == ["fader", "gain", "pan"]
        )
        assert payload["safety"]["allowed_live_write_kinds"] == [
            "fader",
            "gain",
            "pan",
        ]

    @pytest.mark.asyncio
    async def test_wing_sequence_execution_audit_rejects_pan_scope(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()

        await server._dispatch["audit_operator_proposal_apply_sequence_execution"](
            websocket,
            {
                "execution_result_payload": self._sequence_execution_payload(
                    "pan",
                    allowed_live_write_kinds=["fader", "gain", "pan"],
                )
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "operator_proposal_pipeline_migration_required"
        assert payload["status"] == "migration_required"
        assert payload["reason"] == "server_owned_sequence_state_required"
        assert payload["blocked_handler"] == (
            "audit_operator_proposal_apply_sequence_execution"
        )
        assert payload["no_additional_console_send_proven"] is True
        assert payload["soundcheck_completion_blocked"] is True

    @pytest.mark.asyncio
    async def test_offline_sequence_execution_audit_retains_current_proven_pan_scope(
        self,
        minimal_server,
    ):
        server = minimal_server
        server.connection_mode = "offline"
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()

        await server._dispatch["audit_operator_proposal_apply_sequence_execution"](
            websocket,
            {
                "execution_result_payload": self._sequence_execution_payload(
                    "pan",
                    allowed_live_write_kinds=["fader", "gain", "pan"],
                )
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "operator_proposal_pipeline_migration_required"
        assert payload["status"] == "migration_required"
        assert payload["reason"] == "server_owned_sequence_state_required"
        assert payload["blocked_handler"] == (
            "audit_operator_proposal_apply_sequence_execution"
        )
        assert payload["no_additional_console_send_proven"] is True

    @pytest.mark.asyncio
    async def test_wing_sequence_audit_policy_gate_rejects_pan_scope(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()

        await server._dispatch["gate_operator_proposal_apply_sequence_audit_policy"](
            websocket,
            {
                "execution_audit_payload": self._sequence_execution_audit_payload(
                    "pan",
                    allowed_live_write_kinds=["fader", "gain", "pan"],
                )
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "operator_proposal_pipeline_migration_required"
        assert payload["status"] == "migration_required"
        assert payload["reason"] == "server_owned_sequence_state_required"
        assert payload["blocked_handler"] == (
            "gate_operator_proposal_apply_sequence_audit_policy"
        )
        assert payload["no_additional_console_send_proven"] is True
        assert payload["soundcheck_completion_blocked"] is True

    @pytest.mark.asyncio
    async def test_offline_sequence_audit_policy_gate_retains_current_proven_pan_scope(
        self,
        minimal_server,
    ):
        server = minimal_server
        server.connection_mode = "offline"
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()

        await server._dispatch["gate_operator_proposal_apply_sequence_audit_policy"](
            websocket,
            {
                "execution_audit_payload": self._sequence_execution_audit_payload(
                    "pan",
                    allowed_live_write_kinds=["fader", "gain", "pan"],
                )
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "operator_proposal_pipeline_migration_required"
        assert payload["status"] == "migration_required"
        assert payload["reason"] == "server_owned_sequence_state_required"
        assert payload["blocked_handler"] == (
            "gate_operator_proposal_apply_sequence_audit_policy"
        )
        assert payload["no_additional_console_send_proven"] is True

    @pytest.mark.asyncio
    async def test_wing_manual_gate_filters_pan_from_operator_mode_scope(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()

        result = await server._apply_manual_console_write(
            websocket,
            channel=1,
            value=0.25,
            value_type="pan",
            source="unit:wing_pan_manual_gate",
            approved=True,
            approval_id="unit-wing-pan",
        )

        assert result["status"] == "blocked"
        assert result["reason"] == "live_write_kind_not_allowed_in_mode"
        assert result["allowed_live_write_kinds"] == ["fader", "gain"]
        server.mixer_client.set_channel_pan.assert_not_called()

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
        self._assert_quarantined_operator_correction(
            payload,
            *self._quarantined_operator_expectation("reset_trim"),
        )
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
        self._assert_quarantined_operator_correction(
            payload,
            "scene_preset_management_proposal",
            ["scene_load", "scene_save", "preset_load", "preset_save"],
        )
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
        self._assert_quarantined_operator_correction(
            payload,
            "scene_preset_management_proposal",
            ["scene_load", "scene_save", "preset_load", "preset_save"],
        )
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
        self._assert_quarantined_operator_correction(
            payload,
            *self._quarantined_operator_expectation("apply_eq_correction"),
        )
        controller.apply_to_mixer.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_fader_requires_canonical_operator_proposal_for_wing(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_fader"](websocket, {"channel": 1, "value": -6.0})

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "direct_wing_write_requires_operator_proposal"
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("approved", [1, "true", "false", {}, []])
    async def test_websocket_manual_write_rejects_non_literal_approval(
        self,
        dummy_wing_server,
        approved,
    ):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_fader"](
            websocket,
            {
                "channel": 1,
                "value": -6.0,
                "approved": approved,
                "approval_id": "non-literal-approval",
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["status"] == "blocked"
        assert payload["reason"] == "direct_wing_write_requires_operator_proposal"
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
        assert payload["reason"] == "direct_wing_write_requires_operator_proposal"
        assert payload["manual_write_supervision"]["wing_observation_only"] is True
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_fader_approved_payload_cannot_bypass_operator_proposal(self, dummy_wing_server):
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
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "direct_wing_write_requires_operator_proposal"
        assert payload["value_type"] == "fader"
        server.mixer_client.query_parameter.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_gain_approved_payload_cannot_bypass_operator_proposal(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_gain"](
            websocket,
            {"channel": 1, "value": 12.0, "approved": True, "approval_id": "unit-gain-bypass"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "direct_wing_write_requires_operator_proposal"
        assert payload["value_type"] == "gain"
        server.mixer_client.query_parameter.assert_not_called()
        server.mixer_client.set_channel_gain.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("message_type", "value", "mixer_method"),
        [
            ("set_fader", -6.0, "set_channel_fader"),
            ("set_gain", 12.0, "set_channel_gain"),
        ],
    )
    async def test_direct_wing_write_stays_proposal_only_when_legacy_boundary_flag_is_disabled(
        self,
        dummy_wing_server,
        message_type,
        value,
        mixer_method,
    ):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers[message_type](
            websocket,
            {
                "channel": 1,
                "value": value,
                "approved": True,
                "approval_id": "unit-disabled-flag-bypass",
                "confirm_live_apply": True,
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["status"] == "blocked"
        assert payload["reason"] == "direct_wing_write_requires_operator_proposal"
        server.mixer_client.query_parameter.assert_not_called()
        getattr(server.mixer_client, mixer_method).assert_not_called()

    def test_real_wing_safety_boundary_cannot_be_disabled_by_legacy_flag(
        self,
        dummy_wing_server,
    ):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False

        assert server._is_wing_deployment_write_boundary_active() is True
        status = server.get_operator_mode_status()
        assert status["capabilities"]["can_apply_to_console"] is False
        assert status["capabilities"]["allowed_live_write_kinds"] == []

    @pytest.mark.asyncio
    async def test_websocket_mute_is_quarantined_for_wing_boundary(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_mute"](websocket, {"channel": 1, "muted": True})

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == "set_mute"
        assert payload["value_type"] == "mute"
        assert payload["requested_value"] is True
        self._assert_quarantined_operator_correction(
            payload,
            "mute_management_proposal",
            ["mute_write", "unmute_write", "mute_group_write"],
        )
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_mute_stays_quarantined_even_when_approved(self, dummy_wing_server):
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
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == "set_mute"
        assert payload["value_type"] == "mute"
        assert payload["requested_value"] is True
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_mute_channel_alias_is_quarantined_for_wing_boundary(self, dummy_wing_server):
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
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == "mute_channel"
        assert payload["value_type"] == "mute"
        assert payload["requested_value"] is True
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_mute_channel_alias_quarantine_does_not_depend_on_approval(self, dummy_wing_server):
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
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == "mute_channel"
        assert payload["value_type"] == "mute"
        assert payload["requested_value"] is True
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_unmute_channel_alias_is_quarantined_for_wing_boundary(self, dummy_wing_server):
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
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == "unmute_channel"
        assert payload["value_type"] == "mute"
        assert payload["requested_value"] is False
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("message_type", "payload", "surface", "requested_value"),
        [
            ("set_mute", {"channel": 1, "muted": True}, "set_mute", True),
            ("mute_channel", {"channel": 2}, "mute_channel", True),
            ("unmute_channel", {"channel": 3}, "unmute_channel", False),
        ],
    )
    async def test_websocket_mute_direct_path_reports_failed_mixer_write(
        self,
        dummy_wing_server,
        message_type,
        payload,
        surface,
        requested_value,
    ):
        server = dummy_wing_server
        server.connection_mode = "offline"
        server.mixer_client.set_channel_mute.return_value = False
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers[message_type](websocket, payload)

        result = self._decode_last_message(websocket)
        assert result["type"] == "manual_write_blocked"
        assert result["status"] == "blocked"
        assert result["success"] is False
        assert result["reason"] == "mixer_write_failed"
        assert result["surface"] == surface
        assert result["value_type"] == "mute"
        assert result["requested_value"] is requested_value
        assert result["live_write"] is False
        assert result["live_authority_granted"] is False
        assert result["readiness_closure_allowed"] is False
        server.mixer_client.set_channel_mute.assert_called_once_with(
            payload["channel"],
            requested_value,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("message_type", "payload", "surface", "requested_value"),
        [
            ("set_mute", {"channel": 1, "muted": "maybe"}, "set_mute", "maybe"),
            ("set_mute", {"channel": 1}, "set_mute", None),
            ("unmute_channel", {"channel": 3, "muted": "maybe"}, "unmute_channel", "maybe"),
        ],
    )
    async def test_websocket_mute_direct_path_rejects_invalid_bool_without_mixer_write(
        self,
        dummy_wing_server,
        message_type,
        payload,
        surface,
        requested_value,
    ):
        server = dummy_wing_server
        server.connection_mode = "offline"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers[message_type](websocket, payload)

        result = self._decode_last_message(websocket)
        assert result["type"] == "manual_write_blocked"
        assert result["status"] == "blocked"
        assert result["success"] is False
        assert result["reason"] == "invalid_mute"
        assert result["surface"] == surface
        assert result["value_type"] == "mute"
        assert result["requested_value"] == requested_value
        assert result["live_write"] is False
        assert result["live_authority_granted"] is False
        assert result["readiness_closure_allowed"] is False
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_pan_is_quarantined_for_wing_deployment_boundary(self, dummy_wing_server):
        server = dummy_wing_server
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_pan"](
            websocket,
            {"channel": 1, "value": 12.5, "approved": True, "approval_id": "unit-pan-approval"},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == "set_pan"
        self._assert_quarantined_operator_correction(
            payload,
            *self._quarantined_operator_expectation("set_pan"),
        )
        server.mixer_client.set_channel_pan.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_pan_routes_directly_when_wing_boundary_disabled(self, dummy_wing_server):
        server = dummy_wing_server
        server.connection_mode = "offline"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_pan"](
            websocket,
            {"channel": 1, "pan_percent": 125.0},
        )

        assert websocket.send.await_count == 0
        server.mixer_client.set_channel_pan.assert_called_once_with(1, 100.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw_value", ["left", True, float("inf")])
    async def test_websocket_pan_direct_path_rejects_invalid_value_without_mixer_write(
        self,
        dummy_wing_server,
        raw_value,
    ):
        server = dummy_wing_server
        server.connection_mode = "offline"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_pan"](
            websocket,
            {"channel": 1, "pan_percent": raw_value},
        )

        result = self._decode_last_message(websocket)
        assert result["type"] == "manual_write_blocked"
        assert result["status"] == "blocked"
        assert result["success"] is False
        assert result["reason"] == "invalid_value"
        assert result["surface"] == "set_pan"
        assert result["value_type"] == "pan"
        if type(raw_value) is float and not math.isfinite(raw_value):
            assert result["requested_value"] is None
            assert result["requested_value_omitted"] is True
        else:
            assert result["requested_value"] == raw_value
            assert result["requested_value_omitted"] is False
        assert result["live_write"] is False
        assert result["live_authority_granted"] is False
        assert result["readiness_closure_allowed"] is False
        server.mixer_client.set_channel_pan.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_pan_direct_path_reports_failed_mixer_write(
        self,
        dummy_wing_server,
    ):
        server = dummy_wing_server
        server.connection_mode = "offline"
        server.mixer_client.set_channel_pan.return_value = False
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["set_pan"](
            websocket,
            {"channel": 1, "pan_percent": 125.0},
        )

        result = self._decode_last_message(websocket)
        assert result["type"] == "manual_write_blocked"
        assert result["status"] == "blocked"
        assert result["success"] is False
        assert result["reason"] == "mixer_write_failed"
        assert result["surface"] == "set_pan"
        assert result["value_type"] == "pan"
        assert result["requested_value"] == 100.0
        assert result["live_write"] is False
        assert result["live_authority_granted"] is False
        assert result["readiness_closure_allowed"] is False
        server.mixer_client.set_channel_pan.assert_called_once_with(1, 100.0)

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
        assert result["allowed_live_write_kinds"] == ["fader", "gain", "pan"]
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
        assert result["allowed_live_write_kinds"] == ["fader", "gain", "pan"]
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_manual_write_retries_when_mixer_drops_write(
        self,
        _mock_bleed: object,
        _mock_load_config: object,
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        server.connection_mode = "offline"
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server.mixer_client = DummyWingLike()
        server.mixer_client._osc_throttle_enabled = True
        server.mixer_client._osc_throttle_hz = 1000.0
        attempts = {"count": 0}

        def retry_then_apply(channel, value):
            attempts["count"] += 1
            if attempts["count"] == 1:
                return False
            server.mixer_client.state[f"/ch/{channel}/fdr"] = value
            return True

        server.mixer_client.set_channel_fader = MagicMock(side_effect=retry_then_apply)

        result = await server._apply_manual_console_write(
            websocket=None,
            channel=1,
            value=-6.0,
            value_type="fader",
            source="unit:manual_write_retry",
            approved=True,
            approval_id="unit-manual-write-retry",
        )

        assert result["status"] == "applied"
        assert result["write_attempts"] == 2
        assert server.mixer_client.set_channel_fader.call_count == 2

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_manual_write_blocked_after_exhausting_write_retries(
        self,
        _mock_bleed: object,
        _mock_load_config: object,
    ):
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        server.connection_mode = "offline"
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server.mixer_client = DummyWingLike()
        server.mixer_client._osc_throttle_enabled = True
        server.mixer_client._osc_throttle_hz = 1000.0
        server.mixer_client.set_channel_fader = MagicMock(return_value=False)

        result = await server._apply_manual_console_write(
            websocket=None,
            channel=1,
            value=-6.0,
            value_type="fader",
            source="unit:manual_write_retry_exhausted",
            approved=True,
            approval_id="unit-manual-write-retry-exhausted",
        )

        assert result["status"] == "blocked"
        assert result["reason"] == "mixer_write_failed"
        assert result["write_attempts"] == 3
        assert server.mixer_client.set_channel_fader.call_count == 3

    @pytest.mark.asyncio
    async def test_emergency_generation_stops_a_manual_write_before_retry(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server._manual_write_cooldown_sec = 0.0

        def stop_during_first_attempt(channel, value):
            server._manual_write_emergency_stopped = True
            server._manual_write_emergency_generation += 1
            return False

        server.mixer_client.set_channel_fader.side_effect = stop_during_first_attempt

        result = await server._apply_manual_console_write(
            None,
            channel=1,
            value=-6.0,
            value_type="fader",
            source="unit:emergency_between_retries",
            approved=True,
            approval_id="unit-emergency-between-retries",
        )

        assert result["status"] == "blocked"
        assert result["reason"] == "emergency_stop_active"
        assert server.mixer_client.set_channel_fader.call_count == 1

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_immediate_rollback_survives_same_address_write_throttle(
        self,
        _mock_bleed: object,
        _mock_load_config: object,
    ):
        """Regression for the failed CH3 fader proof (rollback_not_applied).

        WingClient rate-limits repeated writes to the same OSC address and
        returns False for a dropped write. An approved apply followed by an
        immediate rollback to the same address must survive that throttle.
        """
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        server.connection_mode = "offline"
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True

        client = DummyWingLike()
        min_interval_sec = 0.05
        client._osc_throttle_enabled = True
        client._osc_throttle_hz = 1.0 / min_interval_sec
        last_send: dict = {}
        dropped: list = []

        def throttled_set_channel_fader(channel, value):
            now = time.monotonic()
            last = last_send.get(channel)
            if last is not None and (now - last) < min_interval_sec:
                dropped.append((channel, value))
                return False
            last_send[channel] = now
            client.state.update({f"/ch/{channel}/fdr": float(value)})
            return True

        client.set_channel_fader = throttled_set_channel_fader
        server.mixer_client = client

        apply_result = await server._apply_manual_console_write(
            websocket=None,
            channel=1,
            value=-6.0,
            value_type="fader",
            source="unit:throttled_apply",
            approved=True,
            approval_id="unit-throttled-apply",
        )
        assert apply_result["status"] == "applied"
        assert apply_result["rollback_available"] is True

        rollback_result = await server._rollback_last_manual_console_write(
            None,
            source="unit:throttled_rollback",
            approved=True,
            approval_id="unit-throttled-rollback",
            transaction_id=apply_result["transaction_id"],
        )

        assert rollback_result["status"] == "applied"
        assert rollback_result["rollback"] is True
        assert rollback_result["write_attempts"] >= 2
        assert dropped, "expected the immediate rollback write to be throttle-dropped at least once"
        assert client.state["/ch/1/fdr"] == -8.0

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_immediate_rollback_survives_trailing_edge_write_throttle(
        self,
        _mock_bleed: object,
        _mock_load_config: object,
    ):
        """Composition contract with trailing-edge throttle semantics.

        A throttled write may be queued and flushed at last_send + interval
        instead of dropped; the flush itself restarts the throttle window, so
        the manual-write retry delay must span two full intervals or the retry
        is throttled again and the proof misreports mixer_write_failed.
        """
        import threading

        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        server.connection_mode = "offline"
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True

        client = DummyWingLike()
        min_interval_sec = 0.05
        client._osc_throttle_enabled = True
        client._osc_throttle_hz = 1.0 / min_interval_sec
        lock = threading.Lock()
        last_send: dict = {}
        pending: dict = {}
        timers: dict = {}
        deferred_flushes: list = []

        def flush(channel):
            with lock:
                timers.pop(channel, None)
                value = pending.pop(channel, None)
                if value is None:
                    return
                last_send[channel] = time.monotonic()
                client.state.update({f"/ch/{channel}/fdr": float(value)})
                deferred_flushes.append((channel, value))

        def trailing_edge_set_channel_fader(channel, value):
            with lock:
                now = time.monotonic()
                last = last_send.get(channel)
                if channel in timers or (
                    last is not None and now - last < min_interval_sec
                ):
                    pending[channel] = value
                    if channel not in timers:
                        timer = threading.Timer(
                            max(0.0, last + min_interval_sec - now),
                            flush,
                            args=(channel,),
                        )
                        timer.daemon = True
                        timers[channel] = timer
                        timer.start()
                    return False
                last_send[channel] = now
                client.state.update({f"/ch/{channel}/fdr": float(value)})
                return True

        client.set_channel_fader = trailing_edge_set_channel_fader
        server.mixer_client = client

        apply_result = await server._apply_manual_console_write(
            websocket=None,
            channel=1,
            value=-6.0,
            value_type="fader",
            source="unit:trailing_edge_apply",
            approved=True,
            approval_id="unit-trailing-edge-apply",
        )
        assert apply_result["status"] == "applied"

        rollback_result = await server._rollback_last_manual_console_write(
            None,
            source="unit:trailing_edge_rollback",
            approved=True,
            approval_id="unit-trailing-edge-rollback",
            transaction_id=apply_result["transaction_id"],
        )

        assert rollback_result["status"] == "applied"
        assert rollback_result["rollback"] is True
        assert rollback_result["write_attempts"] == 2
        assert deferred_flushes, "expected the first rollback attempt to be queued and flushed"
        assert client.state["/ch/1/fdr"] == -8.0

    @pytest.mark.asyncio
    @patch.object(AutoMixerServer, "_load_config", return_value={})
    @patch("server.BleedService")
    async def test_supervised_proof_sequence_apply_rollback_emergency_stop_blocks_next_apply(
        self,
        _mock_bleed: object,
        _mock_load_config: object,
    ):
        """Mirror the operator proof rerun: apply, immediate rollback through a
        same-address throttle, emergency stop, then a later apply must block."""
        server = AutoMixerServer(ws_host="localhost", ws_port=8765)
        server.connection_mode = "offline"
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True

        client = DummyWingLike()
        min_interval_sec = 0.05
        client._osc_throttle_enabled = True
        client._osc_throttle_hz = 1.0 / min_interval_sec
        last_send: dict = {}

        def throttled_set_channel_fader(channel, value):
            now = time.monotonic()
            last = last_send.get(channel)
            if last is not None and (now - last) < min_interval_sec:
                return False
            last_send[channel] = now
            client.state.update({f"/ch/{channel}/fdr": float(value)})
            return True

        client.set_channel_fader = throttled_set_channel_fader
        server.mixer_client = client

        apply_result = await server._apply_manual_console_write(
            websocket=None,
            channel=1,
            value=-6.0,
            value_type="fader",
            source="unit:proof_sequence_apply",
            approved=True,
            approval_id="unit-proof-sequence-apply",
        )
        assert apply_result["status"] == "applied"

        rollback_result = await server._rollback_last_manual_console_write(
            None,
            source="unit:proof_sequence_rollback",
            approved=True,
            approval_id="unit-proof-sequence-rollback",
            transaction_id=apply_result["transaction_id"],
        )
        assert rollback_result["status"] == "applied"
        assert rollback_result["rollback_status"] in {"applied", "partial"}
        assert client.state["/ch/1/fdr"] == -8.0

        server._manual_write_emergency_stopped = True

        blocked_result = await server._apply_manual_console_write(
            websocket=None,
            channel=1,
            value=-6.0,
            value_type="fader",
            source="unit:proof_sequence_after_emergency",
            approved=True,
            approval_id="unit-proof-sequence-after-emergency",
        )
        assert blocked_result["status"] == "blocked"
        assert blocked_result["reason"] == "emergency_stop_active"
        assert client.state["/ch/1/fdr"] == -8.0

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
        assert payload["reason"] == "direct_wing_write_requires_operator_proposal"
        assert payload["approval_id"] == "unit-auto-wing"
        assert payload["manual_write_supervision"]["deployment_write_boundary_active"] is True
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
                **self._review_binding(server, proposal_id),
                "approved": True,
                "idempotency_key": "unit-auto-boundary",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["reason"] == "operator_proposal_global_hazard_active"
        assert apply_payload["no_console_send_proven"] is True
        assert "global_hazards" not in apply_payload
        operator_mode = server.get_operator_mode_status()
        assert (
            operator_mode["capabilities"]["can_apply_to_console"]
            is False
        )
        assert (
            operator_mode["capabilities"]["allowed_live_write_kinds"]
            == []
        )
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_is_allowed_for_auto_mode_on_non_wing_connection(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"
        server.mixer_client.state_timestamps["/ch/1/fdr"] = time.time()
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock(
            return_value=_verified_manual_apply_result(**{
                "status": "applied",
                "channel": 1,
                "value_type": "fader",
                "effective_value": -9.0,
                "old_value": -8.0,
                "readback_value": -9.0,
                "readback_status": "verified_from_cache",
                "rollback_available": True,
                "rollback_value": -8.0,
            })
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
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "idempotency_key": "unit-auto-noboundary",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_result"
        assert apply_payload["status"] == "applied"
        assert apply_payload["apply_status"] == "applied"
        assert apply_payload["channel"] == 1
        assert apply_payload["value_type"] == "fader"
        assert "apply_result" not in apply_payload
        server._apply_manual_console_write.assert_awaited_once_with(
            websocket,
            channel=1,
            value=-9.0,
            value_type="fader",
            source=f"operator_proposal:{proposal_id}",
            reason="unit",
            approved=True,
            approval_id=proposal_id,
            idempotency_key="unit-auto-noboundary",
            proposal_id=proposal_id,
            expected_current_value=-8.0,
            approval_expires_at_epoch=(
                server.operator_proposal_queue.find(
                    proposal_id=proposal_id
                ).expiration_epoch()
            ),
            result_evidence_reservation_binding=ANY,
        )

    @pytest.mark.asyncio
    async def test_legacy_soundcheck_fader_cannot_bypass_transaction_after_mode_switch(
        self,
        dummy_wing_server,
    ):
        server = dummy_wing_server
        server.operator_mode = "assist"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "id": "legacy-soundcheck-fader",
                "title": "Legacy soundcheck fader",
                "source": "soundcheck_recommendation_bundle",
                "channel": 1,
                "value_type": "fader",
                "value": -9.0,
                "reason": "unmatched soundcheck recommendation",
            },
        )
        created = self._decode_last_message(websocket)
        proposal_id = created["proposal"]["id"]
        assert created["proposal"]["status"] == "pending"
        assert created["proposal"]["can_apply"] is False
        assert created["proposal"]["apply_blocker_codes"] == [
            SOUNDCHECK_CORRECTION_REQUIRES_SUPERVISED_TRANSACTION
        ]

        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"

        await server._dispatch["preflight_operator_proposal_apply_batch"](
            websocket,
            {
                "approved": True,
                "idempotency_key_prefix": "legacy-soundcheck-batch",
            },
        )
        batch = self._decode_last_message(websocket)
        assert batch["summary"]["selected_count"] == 0
        assert batch["summary"]["skipped_analysis_only_ids"] == [proposal_id]

        await server._dispatch["preflight_operator_proposal_apply"](
            websocket,
            {
                "proposal_id": proposal_id,
                "approved": True,
                "idempotency_key": "legacy-soundcheck-preflight",
            },
        )
        preflight = self._decode_last_message(websocket)
        assert preflight["status"] == "blocked"
        assert preflight["reason"] == (
            SOUNDCHECK_CORRECTION_REQUIRES_SUPERVISED_TRANSACTION
        )
        assert preflight["no_console_send_proven"] is True
        assert preflight["idempotency"]["reserved"] is False

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "idempotency_key": "legacy-soundcheck-apply",
            },
        )
        blocked = self._decode_last_message(websocket)
        assert blocked["type"] == "operator_proposal_apply_blocked"
        assert blocked["reason"] == (
            SOUNDCHECK_CORRECTION_REQUIRES_SUPERVISED_TRANSACTION
        )
        assert blocked["proposal_status"] == "pending"
        assert blocked["idempotency_reserved"] is False
        assert blocked["no_console_send_proven"] is True
        assert server.operator_proposal_queue.find(
            proposal_id=proposal_id
        ).status == "pending"
        assert "legacy-soundcheck-apply" not in (
            server._operator_apply_idempotency_keys
        )
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_accepts_matching_wing_surface_contract(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"
        server.mixer_client.state["/ch/1/in/set/trim"] = 0.0
        server.mixer_client.state_timestamps["/ch/1/in/set/trim"] = time.time()
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock(
            return_value=_verified_manual_apply_result(**{
                "status": "applied",
                "channel": 1,
                "value_type": "gain",
                "effective_value": -0.25,
                "old_value": 0.0,
                "readback_value": -0.25,
                "readback_status": "verified_from_cache",
                "rollback_available": True,
                "rollback_value": 0.0,
            })
        )

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "title": "WING proof: gain uses input trim",
                "channel": 1,
                "value_type": "gain",
                "value": -0.25,
                "source": "wing_operator_proposal_validation_runner",
                "reason": "unit",
                "requested_change": {
                    "channel": 1,
                    "value_type": "gain",
                    "value": -0.25,
                    "wing_surface": "input_trim",
                    "wing_address": "/ch/1/in/set/trim",
                    "proof_direction": "negative_input_trim_move_only",
                },
            },
        )
        create_payload = self._decode_last_message(websocket)
        assert create_payload["type"] == "operator_proposal_created"
        proposal_id = create_payload["proposal"]["id"]

        await server._dispatch["preflight_operator_proposal_apply"](
            websocket,
            {
                "proposal_id": proposal_id,
                "approved": True,
                "idempotency_key": "unit-wing-surface-match-preflight",
            },
        )
        preflight_payload = self._decode_last_message(websocket)
        assert preflight_payload["type"] == "operator_proposal_apply_preflight"
        assert preflight_payload["status"] == "ready"
        assert "normalized_change" not in preflight_payload
        assert preflight_payload["proposal"]["id"] == proposal_id

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "idempotency_key": "unit-wing-surface-match",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_result"
        assert apply_payload["status"] == "applied"
        assert apply_payload["apply_status"] == "applied"
        assert apply_payload["value_type"] == "gain"
        assert "apply_result" not in apply_payload
        server._apply_manual_console_write.assert_awaited_once_with(
            websocket,
            channel=1,
            value=-0.25,
            value_type="gain",
            source=f"operator_proposal:{proposal_id}",
            reason="unit",
            approved=True,
            approval_id=proposal_id,
            idempotency_key="unit-wing-surface-match",
            proposal_id=proposal_id,
            expected_current_value=0.0,
            approval_expires_at_epoch=(
                server.operator_proposal_queue.find(
                    proposal_id=proposal_id
                ).expiration_epoch()
            ),
            result_evidence_reservation_binding=ANY,
        )

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_blocks_wing_surface_contract_mismatch(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "title": "WING proof: gain cannot target fader",
                "channel": 1,
                "value_type": "gain",
                "value": -0.25,
                "source": "wing_operator_proposal_validation_runner",
                "reason": "unit",
                "requested_change": {
                    "channel": 1,
                    "value_type": "gain",
                    "value": -0.25,
                    "wing_surface": "channel_fader",
                    "wing_address": "/ch/1/fdr",
                    "proof_direction": "negative_db_move_only",
                },
            },
        )
        create_payload = self._decode_last_message(websocket)
        assert create_payload["type"] == "operator_proposal_created"
        assert create_payload["status"] == "created"
        assert "requested_change" not in create_payload["proposal"]
        proposal_id = create_payload["proposal"]["id"]

        await server._dispatch["preflight_operator_proposal_apply"](
            websocket,
            {
                "proposal_id": proposal_id,
                "approved": True,
                "idempotency_key": "unit-wing-surface-mismatch-preflight",
            },
        )
        preflight_payload = self._decode_last_message(websocket)
        assert preflight_payload["type"] == "operator_proposal_apply_preflight"
        assert preflight_payload["status"] == "blocked"
        assert preflight_payload["reason"] == "wing_surface_contract_mismatch"
        assert "normalized_change" not in preflight_payload
        assert preflight_payload["idempotency"]["reserved"] is False

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "idempotency_key": "unit-wing-surface-mismatch",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["reason"] == "wing_surface_contract_mismatch"
        assert apply_payload["no_console_send_proven"] is True
        assert "apply_result" not in apply_payload
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.parametrize(
        "value_type,address,wing_surface,proof_direction,old_value,target_value",
        [
            ("fader", "/ch/1/fdr", "channel_fader", "negative_db_move_only", -9.0, -8.75),
            ("gain", "/ch/1/in/set/trim", "input_trim", "negative_input_trim_move_only", 0.0, 0.25),
        ],
    )
    @pytest.mark.asyncio
    async def test_operator_proposal_apply_blocks_positive_wing_proof_direction_even_operator_authored(
        self,
        dummy_wing_server,
        value_type,
        address,
        wing_surface,
        proof_direction,
        old_value,
        target_value,
    ):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"
        server.mixer_client.state[address] = old_value
        server.mixer_client.state_timestamps[address] = time.time()
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "title": "WING proof: proof direction stays negative-only",
                "channel": 1,
                "value_type": value_type,
                "value": target_value,
                "source": "wing_operator_proposal_validation_runner",
                "operator_authored": True,
                "reason": "unit",
                "requested_change": {
                    "channel": 1,
                    "value_type": value_type,
                    "value": target_value,
                    "wing_surface": wing_surface,
                    "wing_address": address,
                    "proof_direction": proof_direction,
                },
            },
        )
        create_payload = self._decode_last_message(websocket)
        assert create_payload["type"] == "operator_proposal_created"
        assert create_payload["status"] == "created"
        proposal_id = create_payload["proposal"]["id"]

        await server._dispatch["preflight_operator_proposal_apply"](
            websocket,
            {
                "proposal_id": proposal_id,
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "operator_authored": True,
                "idempotency_key": f"unit-wing-proof-positive-{value_type}-preflight",
            },
        )
        preflight_payload = self._decode_last_message(websocket)
        assert preflight_payload["type"] == "operator_proposal_apply_preflight"
        assert preflight_payload["status"] == "blocked"
        assert preflight_payload["reason"] == "upward_move_requires_operator_authored_proposal"
        assert "delta_db" not in preflight_payload
        assert preflight_payload["normalized_change_count"] == 0

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "operator_authored": True,
                "idempotency_key": f"unit-wing-proof-positive-{value_type}",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["reason"] == "upward_move_requires_operator_authored_proposal"
        assert apply_payload["no_console_send_proven"] is True
        assert "apply_result" not in apply_payload
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_linked_blocks_wing_surface_contract_mismatch_before_write(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"
        now = time.time()
        server.mixer_client.state["/ch/1/fdr"] = -8.0
        server.mixer_client.state_timestamps["/ch/1/fdr"] = now
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock()

        await server._dispatch["create_operator_proposal"](
            websocket,
            {
                "title": "WING proof: linked surface mismatch",
                "value_type": "fader",
                "source": "wing_operator_proposal_validation_runner",
                "requested_changes": [
                    {
                        "value_type": "fader",
                        "channel": 1,
                        "value": -8.25,
                        "wing_surface": "channel_fader",
                        "wing_address": "/ch/1/fdr",
                        "proof_direction": "negative_db_move_only",
                    },
                    {
                        "value_type": "gain",
                        "channel": 2,
                        "value": -0.25,
                        "wing_surface": "channel_fader",
                        "wing_address": "/ch/2/fdr",
                        "proof_direction": "negative_db_move_only",
                    },
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
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "idempotency_key": "unit-linked-wing-surface-mismatch",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["reason"] == "wing_surface_contract_mismatch"
        assert apply_payload["no_console_send_proven"] is True
        assert "apply_result" not in apply_payload
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_linked_idempotency_key_blocks_duplicate_apply(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"
        server.mixer_client.state.update({
            "/ch/1/fdr": -9.0,
            "/ch/2/fdr": -9.0,
        })
        now = time.time()
        server.mixer_client.state_timestamps.update({
            "/ch/1/fdr": now,
            "/ch/2/fdr": now,
        })
        apply_results = [
                {
                    "status": "applied",
                    "success": True,
                    "channel": 1,
                    "value_type": "fader",
                    "effective_value": -9.5,
                    "old_value": -9.0,
                    "readback_value": -9.5,
                    "readback_status": "verified_from_physical_readback",
                    "rollback_available": True,
                    "rollback_value": -8.0,
                    "no_console_send_proven": False,
                    "physical_verified": True,
                    "wal_terminal": "committed",
                    "audio_recheck_required": False,
                    "audio_recheck_disposition": (
                        "not_required_by_manual_write_contract"
                    ),
                    "write_state_unknown": False,
                    "rollback_state_unknown": False,
                    "manual_review_required": False,
                },
                {
                    "status": "applied",
                    "success": True,
                    "channel": 2,
                    "value_type": "fader",
                    "effective_value": -9.5,
                    "old_value": -9.0,
                    "readback_value": -9.5,
                    "readback_status": "verified_from_physical_readback",
                    "rollback_available": True,
                    "rollback_value": -9.0,
                    "no_console_send_proven": False,
                    "physical_verified": True,
                    "wal_terminal": "committed",
                    "audio_recheck_required": False,
                    "audio_recheck_disposition": (
                        "not_required_by_manual_write_contract"
                    ),
                    "write_state_unknown": False,
                    "rollback_state_unknown": False,
                    "manual_review_required": False,
                },
            ]

        async def physical_read(*, address, **_kwargs):
            return {
                "value": server.mixer_client.state.get(address),
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "cache_freshness": {"status": "fresh"},
            }

        async def apply_with_state_update(*_args, **kwargs):
            assert (
                server.operator_proposal_queue
                .mark_result_evidence_send_may_have_occurred(
                    kwargs["result_evidence_reservation_binding"]
                )["transitioned"]
                is True
            )
            result = apply_results.pop(0)
            if result.get("status") == "applied":
                address = server._manual_write_address(
                    kwargs["channel"],
                    kwargs["value_type"],
                )
                server.mixer_client.state[address] = result["effective_value"]
            return result

        server._manual_write_read_value = AsyncMock(side_effect=physical_read)
        server._apply_manual_console_write = AsyncMock(
            side_effect=apply_with_state_update
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
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
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
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "idempotency_key": "unit-auto-linked-idemp",
            },
        )
        second_apply_payload = self._decode_last_message(websocket)
        assert second_apply_payload["type"] == "operator_proposal_apply_blocked"
        assert second_apply_payload["status"] == "apply_blocked"
        assert second_apply_payload["reason"] == "proposal_not_pending"
        assert "unit-auto-linked-idemp" in server._operator_apply_idempotency_keys
        assert server._apply_manual_console_write.await_count == 2

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_is_blocked_for_stale_pre_read_when_wing_boundary_disabled(self, dummy_wing_server, tmp_path):
        server = dummy_wing_server
        self._enable_result_evidence_lifecycle(server, tmp_path)
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"
        server.mixer_client.state_timestamps["/ch/1/fdr"] = time.time() - 120.0
        server.mixer_client.query_parameter = None
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
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "idempotency_key": "unit-auto-stale",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["reason"] == "pre_read_cache_not_fresh"
        assert "apply_result" not in apply_payload
        proposal = server.operator_proposal_queue.find(proposal_id=proposal_id)
        persisted = json.loads(
            server.operator_proposal_queue.load_result_evidence(
                proposal,
                expected_result_digest=apply_payload["result_digest"],
            )
        )
        assert persisted["cache_freshness"]["status"] == "stale"
        assert persisted["cache_freshness"]["source"] == "state_timestamps"
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_is_blocked_for_stale_gain_pre_read_when_wing_boundary_disabled(self, dummy_wing_server, tmp_path):
        server = dummy_wing_server
        self._enable_result_evidence_lifecycle(server, tmp_path)
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"
        server.mixer_client.state_timestamps["/ch/1/in/set/trim"] = time.time() - 120.0
        server.mixer_client.query_parameter = None
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
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "idempotency_key": "unit-auto-stale-gain",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["reason"] == "pre_read_cache_not_fresh"
        assert "apply_result" not in apply_payload
        proposal = server.operator_proposal_queue.find(proposal_id=proposal_id)
        persisted = json.loads(
            server.operator_proposal_queue.load_result_evidence(
                proposal,
                expected_result_digest=apply_payload["result_digest"],
            )
        )
        assert persisted["cache_freshness"]["status"] == "stale"
        assert persisted["cache_freshness"]["address"] == "/ch/1/in/set/trim"
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_gain.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_is_blocked_when_gain_pre_read_unavailable(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"
        server.mixer_client.state = {}
        server.mixer_client.query_parameter = None
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
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "idempotency_key": "unit-auto-no-read-gain",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["reason"] == "pre_read_unavailable"
        assert apply_payload["no_console_send_proven"] is True
        assert "apply_result" not in apply_payload
        server._apply_manual_console_write.assert_not_called()
        server.mixer_client.set_channel_gain.assert_not_called()

    @pytest.mark.asyncio
    async def test_operator_proposal_apply_is_blocked_for_unsupported_value_type(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "auto"
        server._auto_mode_unlocked = True
        server._wing_deployment_write_boundary_enabled = False
        server._wing_observation_only = False
        server.connection_mode = "offline"
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
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "idempotency_key": "unit-auto-unsupported-type",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["reason"] == "unsupported_value_type"
        assert apply_payload["no_console_send_proven"] is True
        assert "apply_result" not in apply_payload
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
        server.connection_mode = "offline"
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
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "idempotency_key": "unit-auto-unsupported-linked",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["reason"] == "unsupported_value_type"
        assert apply_payload["no_console_send_proven"] is True
        assert "apply_result" not in apply_payload
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
        server.connection_mode = "offline"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        server._apply_manual_console_write = AsyncMock()
        server.mixer_client.state = {"/ch/1/fdr": -8.0}
        server.mixer_client.query_parameter = None

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
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "idempotency_key": "unit-auto-linked-missing-read",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["reason"] == "linked_pre_read_unavailable"
        assert apply_payload["no_console_send_proven"] is True
        assert "apply_result" not in apply_payload
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
        server.connection_mode = "offline"
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
        apply_results = [
                {
                    "status": "applied",
                    "success": True,
                    "channel": 1,
                    "value_type": "fader",
                    "approval_id": f"{proposal_id}:1",
                    "effective_value": -9.0,
                    "old_value": -8.0,
                    "readback_value": -9.0,
                    "readback_status": "verified_from_physical_readback",
                    "rollback_available": True,
                    "rollback_value": -8.0,
                    "no_console_send_proven": False,
                    "physical_verified": True,
                    "wal_terminal": "committed",
                    "audio_recheck_required": False,
                    "audio_recheck_disposition": (
                        "not_required_by_manual_write_contract"
                    ),
                    "write_state_unknown": False,
                    "rollback_state_unknown": False,
                    "manual_review_required": False,
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
                    "no_console_send_proven": True,
                    "wal_terminal": "aborted_before_send",
                    "write_state_unknown": False,
                    "rollback_state_unknown": False,
                    "manual_review_required": False,
                },
                {
                    "status": "applied",
                    "success": True,
                    "channel": 1,
                    "value_type": "fader",
                    "effective_value": -8.0,
                    "old_value": -9.0,
                    "readback_value": -8.0,
                    "readback_status": "verified_from_physical_readback",
                    "rollback_available": True,
                    "rollback_value": -9.0,
                    "rollback_of": f"{proposal_id}:1",
                    "rollback": True,
                    "no_console_send_proven": False,
                    "physical_verified": True,
                    "wal_terminal": "committed",
                    "audio_recheck_required": False,
                    "audio_recheck_disposition": (
                        "not_required_by_manual_write_contract"
                    ),
                    "write_state_unknown": False,
                    "rollback_state_unknown": False,
                    "manual_review_required": False,
                },
            ]

        async def physical_read(*, address, **_kwargs):
            return {
                "value": server.mixer_client.state.get(address),
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "cache_freshness": {"status": "fresh"},
            }

        async def apply_with_state_update(*_args, **kwargs):
            assert (
                server.operator_proposal_queue
                .mark_result_evidence_send_may_have_occurred(
                    kwargs["result_evidence_reservation_binding"]
                )["transitioned"]
                is True
            )
            result = apply_results.pop(0)
            if result.get("status") == "applied":
                address = server._manual_write_address(
                    kwargs["channel"],
                    kwargs["value_type"],
                )
                server.mixer_client.state[address] = result["effective_value"]
            return result

        server._manual_write_read_value = AsyncMock(side_effect=physical_read)
        server._apply_manual_console_write = AsyncMock(
            side_effect=apply_with_state_update
        )

        await server._dispatch["apply_operator_proposal"](
            websocket,
            {
                "proposal_id": proposal_id,
                **self._review_binding(server, proposal_id),
                "approved": True,
                "approval_id": proposal_id,
                "idempotency_key": "unit-auto-linked-fail",
            },
        )

        apply_payload = self._decode_last_message(websocket)
        assert apply_payload["type"] == "operator_proposal_apply_blocked"
        assert apply_payload["status"] == "apply_blocked"
        assert apply_payload["reason"] == "linked_apply_failed_rolled_back"
        assert "apply_result" not in apply_payload
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

        proposal = server.operator_proposal_queue.find(proposal_id=proposal_id)
        assert proposal is not None
        assert proposal.result_evidence_pinned is False
        persisted = json.loads(
            server.operator_proposal_queue.load_result_evidence(
                proposal,
                expected_result_digest=proposal.result_digest,
            )
        )
        assert persisted["failed_result"]["reason"] == "mixer_write_failed"
        assert persisted["failed_result"]["status"] == "blocked"
        assert any(
            step["rollback_of"] == f"{proposal_id}:1"
            for step in persisted["rollback_results"]
        )
        for field, expected in {
            "status": "blocked",
            "no_console_send_proven": False,
            "physical_state_restored": True,
            "terminal_after_console_send": True,
            "wal_terminal": "rolled_back",
            "write_state_unknown": False,
            "rollback_state_unknown": False,
            "manual_review_required": False,
        }.items():
            assert persisted[field] == expected
            assert proposal.result[field] == expected

        result_store = server.operator_proposal_queue.result_store
        restarted_store = ResultEvidenceStore(
            result_store.root,
            min_free_bytes=0,
            process_instance_id="b" * 32,
        )
        restarted_queue = OperatorProposalQueue(
            result_store=restarted_store,
            result_evidence_lifecycle_ready=True,
        )
        assert restarted_queue.result_recovery_hazards == ()
        assert restarted_queue.result_evidence_contract_ready is True

    @pytest.mark.asyncio
    async def test_websocket_fader_blocks_when_cache_pre_read_is_stale(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server.mixer_client.state_timestamps["/ch/1/fdr"] = time.time() - 120.0
        server.mixer_client.query_parameter = None
        payload = await self._apply_manual_write(
            server,
            value=-6.0,
            approval_id="unit-stale-cache",
        )
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
        first = await self._apply_manual_write(
            server,
            value=-6.0,
            approval_id="unit-approval-1",
        )
        payload = await self._apply_manual_write(
            server,
            value=-5.0,
            approval_id="unit-approval-2",
        )
        assert first["status"] == "applied"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "cooldown"
        assert payload["no_console_send_proven"] is True
        assert payload["write_state_unknown"] is False
        assert payload["manual_review_required"] is False
        assert payload["write_attempts"] == 0
        assert payload["cooldown_sec"] == 10.0
        assert payload["cooldown_remaining_sec"] > 0
        server.mixer_client.set_channel_fader.assert_called_once_with(1, -6.0)

    @pytest.mark.asyncio
    async def test_websocket_gain_reports_physical_readback_and_rollback_metadata(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        def apply_gain(channel, value):
            server.mixer_client.state[f"/ch/{channel}/in/set/trim"] = value
            return True

        server.mixer_client.set_channel_gain.side_effect = apply_gain

        payload = await self._apply_manual_write(
            server,
            value=2.0,
            value_type="gain",
            approval_id="unit-gain",
        )
        assert payload["status"] == "applied"
        assert payload["value_type"] == "gain"
        assert payload["old_value"] == 1.0
        assert payload["pre_read_value"] == 1.0
        assert payload["pre_read_status"] == "verified_from_physical_readback"
        assert payload["pre_read_source"] == "physical_readback"
        assert payload["readback_value"] == 2.0
        assert payload["readback_status"] == "verified_from_physical_readback"
        assert payload["readback_source"] == "physical_readback"
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
        server.mixer_client.query_parameter = None
        def apply_gain_without_timestamp(channel, value):
            server.mixer_client.state[f"/ch/{channel}/in/set/trim"] = value
            return True

        server.mixer_client.set_channel_gain.side_effect = apply_gain_without_timestamp

        payload = await self._apply_manual_write(
            server,
            value=2.0,
            value_type="gain",
            approval_id="unit-gain-cache-not-updated",
        )
        assert payload["status"] == "sent_unverified"
        assert payload["readback_value"] is None
        assert payload["readback_status"] == "readback_unavailable"
        assert payload["proof_status"] == "partial"
        assert payload["rollback_available"] is True
        assert payload["emergency_stop_activated"] is True
        server.mixer_client.set_channel_gain.assert_called_once_with(1, 2.0)

    @pytest.mark.asyncio
    async def test_websocket_gain_uses_physical_readback_when_cache_missing(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server.mixer_client.state = {}
        physical = {"/ch/1/in/set/trim": 1.0}

        def query_parameter(address):
            return {"status": "received", "address": address, "value": physical.get(address), "confirmed_round_trips": 2}

        def apply_gain(channel, value):
            physical[f"/ch/{channel}/in/set/trim"] = value
            return True

        server.mixer_client.query_parameter = MagicMock(side_effect=query_parameter)
        server.mixer_client.set_channel_gain.side_effect = apply_gain
        payload = await self._apply_manual_write(
            server,
            value=2.0,
            value_type="gain",
            approval_id="unit-gain-physical",
        )
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
    async def test_manual_physical_read_rejects_numeric_value_without_verified_query_contract(
        self,
        dummy_wing_server,
    ):
        server = dummy_wing_server
        server.mixer_client.state = {}

        for raw in (
            {"status": "timeout", "value": -8.0, "confirmed_round_trips": 2},
            {"status": "error", "value": -8.0, "confirmed_round_trips": 2},
            {"status": "received", "value": -8.0},
            {"status": "received", "value": -8.0, "confirmed_round_trips": 1},
            {"status": "received", "value": -8.0, "confirmed_round_trips": True},
            {"status": "received", "value": -8.0, "confirmed_round_trips": "2"},
            {
                "status": "received",
                "address": "/ch/2/fdr",
                "value": -8.0,
                "confirmed_round_trips": 2,
            },
        ):
            server.mixer_client.query_parameter = MagicMock(return_value=raw)
            result = await server._manual_write_read_value(
                channel=1,
                value_type="fader",
                address="/ch/1/fdr",
                prefer_physical=True,
                allow_cache_fallback=False,
            )
            assert result["status"] == "readback_unavailable"
            assert result["source"] == "physical_readback"
            assert result["value"] is None
            assert result["error"] == "physical_query_contract_unverified"

    @pytest.mark.asyncio
    async def test_sync_physical_query_does_not_block_asyncio_event_loop(self, dummy_wing_server):
        server = dummy_wing_server
        server.mixer_client.state = {}
        query_started = threading.Event()
        release_query = threading.Event()

        def blocking_query(address):
            query_started.set()
            release_query.wait(timeout=1.0)
            return {
                "status": "received",
                "address": address,
                "value": -8.0,
                "confirmed_round_trips": 2,
            }

        server.mixer_client.query_parameter = blocking_query
        timer = threading.Timer(0.3, release_query.set)
        timer.start()
        started_at = time.monotonic()
        try:
            read_task = asyncio.create_task(
                server._manual_write_read_value(
                    channel=1,
                    value_type="fader",
                    address="/ch/1/fdr",
                    prefer_physical=True,
                    allow_cache_fallback=False,
                )
            )
            while not query_started.is_set():
                await asyncio.sleep(0.005)

            # If query_parameter runs on the event-loop thread, this checkpoint
            # cannot execute until the 300 ms safety timer releases the query.
            await asyncio.sleep(0)
            assert time.monotonic() - started_at < 0.20

            release_query.set()
            result = await asyncio.wait_for(read_task, timeout=1.0)
        finally:
            release_query.set()
            timer.cancel()

        assert result["status"] == "verified_from_physical_readback"
        assert result["value"] == -8.0

    @pytest.mark.asyncio
    async def test_cancelled_sync_physical_query_drains_worker_before_task_exits(
        self,
        dummy_wing_server,
    ):
        server = dummy_wing_server
        server.mixer_client.state = {}
        query_started = threading.Event()
        release_query = threading.Event()

        def blocking_query(address):
            query_started.set()
            release_query.wait(timeout=1.0)
            return {
                "status": "received",
                "address": address,
                "value": -8.0,
                "confirmed_round_trips": 2,
            }

        server.mixer_client.query_parameter = blocking_query
        task = asyncio.create_task(
            server._manual_write_read_value(
                channel=1,
                value_type="fader",
                address="/ch/1/fdr",
                prefer_physical=True,
                allow_cache_fallback=False,
            )
        )
        try:
            while not query_started.is_set():
                await asyncio.sleep(0.005)
            task.cancel()
            await asyncio.sleep(0.02)
            assert task.done() is False
            release_query.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)
        finally:
            release_query.set()

    @pytest.mark.asyncio
    async def test_websocket_gain_accepts_wing_trim_quantized_physical_readback(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server.mixer_client.state = {}
        physical = {"/ch/4/in/set/trim": 0.0}

        def query_parameter(address):
            return {"status": "received", "address": address, "value": physical.get(address), "confirmed_round_trips": 2}

        def apply_gain(channel, value):
            physical[f"/ch/{channel}/in/set/trim"] = -0.1999988555908203
            return True

        server.mixer_client.query_parameter = MagicMock(side_effect=query_parameter)
        server.mixer_client.set_channel_gain.side_effect = apply_gain
        payload = await self._apply_manual_write(
            server,
            channel=4,
            value=-0.25,
            value_type="gain",
            approval_id="unit-gain-quantized",
        )
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
            return_value={
                "status": "received",
                "address": "/ch/3/in/set/trim",
                "value": 1.25,
                "confirmed_round_trips": 2,
            }
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
    async def test_query_mixer_parameter_never_promotes_cache_after_physical_timeout(
        self, dummy_wing_server
    ):
        server = dummy_wing_server
        server.mixer_client.state = {"/ch/3/fdr": -12.1}
        server.mixer_client.query_parameter = MagicMock(
            return_value={"status": "timeout", "address": "/ch/3/fdr", "value": None}
        )
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["query_mixer_parameter"](
            websocket,
            {
                "request_id": "m005:stale-probe",
                "channel": 3,
                "value_type": "fader",
                "address": "/ch/3/fdr",
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "mixer_parameter_value"
        assert payload["status"] == "readback_unavailable"
        assert payload["source"] == "physical_readback"
        assert payload["value"] is None
        assert payload["raw"]["status"] == "timeout"

    @pytest.mark.asyncio
    async def test_query_mixer_parameter_reads_dsp_health_paths_without_writing(
        self, dummy_wing_server
    ):
        server = dummy_wing_server
        server.mixer_client.state = {}
        server.mixer_client.query_parameter = MagicMock(
            return_value={
                "status": "received",
                "address": "/ch/3/gate/thr",
                "value": -42.0,
                "confirmed_round_trips": 2,
            }
        )
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["query_mixer_parameter"](
            websocket,
            {
                "request_id": "wing-readonly-preflight:ch3:gate_threshold",
                "channel": 3,
                "value_type": "gate_threshold",
                "address": "/ch/3/gate/thr",
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "mixer_parameter_value"
        assert payload["request_id"] == "wing-readonly-preflight:ch3:gate_threshold"
        assert payload["status"] == "verified_from_physical_readback"
        assert payload["source"] == "physical_readback"
        assert payload["channel"] == 3
        assert payload["value_type"] == "gate_threshold"
        assert payload["address"] == "/ch/3/gate/thr"
        assert payload["value"] == -42.0
        server.mixer_client.query_parameter.assert_called_once_with("/ch/3/gate/thr")
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("value_type", "address", "value"),
        [
            ("gate_model", "/ch/3/gate/mdl", "DEQ"),
            ("gate_ratio", "/ch/3/gate/ratio", "2:1"),
            ("gate_attack", "/ch/3/gate/att", 4.0),
            ("gate_release", "/ch/3/gate/rel", 115.0),
            ("gate_filter", "/ch/3/gate/filt", "PEQ"),
            ("gate_gain", "/ch/3/gate/g", -3.5),
            ("gate_frequency", "/ch/3/gate/f", 74.0),
            ("gate_q", "/ch/3/gate/q", 1.82),
            ("gate_mode", "/ch/3/gate/mode", "HIGH"),
        ],
    )
    async def test_query_mixer_parameter_reads_gate_slot_paths_without_writing(
        self,
        dummy_wing_server,
        value_type,
        address,
        value,
    ):
        server = dummy_wing_server
        server.mixer_client.state = {}
        server.mixer_client.query_parameter = MagicMock(
            return_value={
                "status": "received",
                "address": address,
                "value": value,
                "confirmed_round_trips": 2,
            }
        )
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["query_mixer_parameter"](
            websocket,
            {
                "request_id": f"wing-readonly-preflight:ch3:{value_type}",
                "channel": 3,
                "value_type": value_type,
                "address": address,
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "mixer_parameter_value"
        assert payload["request_id"] == f"wing-readonly-preflight:ch3:{value_type}"
        assert payload["status"] == "verified_from_physical_readback"
        assert payload["source"] == "physical_readback"
        assert payload["channel"] == 3
        assert payload["value_type"] == value_type
        assert payload["address"] == address
        assert payload["value"] == value
        server.mixer_client.query_parameter.assert_called_once_with(address)
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_state_returns_unavailable_state_without_mixer_client(self, minimal_server):
        server = minimal_server
        server.connection_mode = "wing"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["get_state"](
            websocket,
            {},
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "state_update"
        assert payload["mode"] == "wing"
        assert payload["state"] == {}
        assert payload["status"] == "state_unavailable"
        assert payload["reason"] == "mixer_client_state_source_unavailable"

    @pytest.mark.asyncio
    async def test_query_mixer_parameter_reads_mute_without_writing(self, dummy_wing_server):
        server = dummy_wing_server
        server.mixer_client.state = {}
        server.mixer_client.query_parameter = MagicMock(
            return_value={
                "status": "received",
                "address": "/ch/3/mute",
                "value": 1,
                "confirmed_round_trips": 2,
            }
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
    async def test_query_mixer_parameter_reads_pan_without_writing(self, dummy_wing_server):
        server = dummy_wing_server
        server.mixer_client.state = {}
        server.mixer_client.query_parameter = MagicMock(
            return_value={
                "status": "received",
                "address": "/ch/3/pan",
                "value": -25.0,
                "confirmed_round_trips": 2,
            }
        )
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["query_mixer_parameter"](
            websocket,
            {
                "request_id": "wing-readonly-preflight:ch3:pan",
                "channel": 3,
                "value_type": "pan",
                "address": "/ch/3/pan",
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "mixer_parameter_value"
        assert payload["request_id"] == "wing-readonly-preflight:ch3:pan"
        assert payload["status"] == "verified_from_physical_readback"
        assert payload["source"] == "physical_readback"
        assert payload["channel"] == 3
        assert payload["value_type"] == "pan"
        assert payload["address"] == "/ch/3/pan"
        assert payload["value"] == -25.0
        server.mixer_client.query_parameter.assert_called_once_with("/ch/3/pan")
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_mixer_parameter_reads_pan_by_address(self, dummy_wing_server):
        server = dummy_wing_server
        server.mixer_client.state = {}
        server.mixer_client.query_parameter = MagicMock(
            return_value={
                "status": "received",
                "address": "/ch/12/pan",
                "value": -40.0,
                "confirmed_round_trips": 2,
            }
        )
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["query_mixer_parameter"](
            websocket,
            {
                "request_id": "wing-readonly-preflight:ch12:pan",
                "channel": 12,
                "address": "/ch/12/pan",
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "mixer_parameter_value"
        assert payload["request_id"] == "wing-readonly-preflight:ch12:pan"
        assert payload["status"] == "verified_from_physical_readback"
        assert payload["source"] == "physical_readback"
        assert payload["channel"] == 12
        assert payload["value_type"] == "pan"
        assert payload["address"] == "/ch/12/pan"
        assert payload["value"] == -40.0
        server.mixer_client.query_parameter.assert_called_once_with("/ch/12/pan")
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()
        server.mixer_client.set_channel_mute.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_mixer_parameter_reads_pan_from_pan_percent_alias(self, dummy_wing_server):
        server = dummy_wing_server
        server.mixer_client.state = {}
        server.mixer_client.query_parameter = MagicMock(
            return_value={
                "status": "received",
                "address": "/ch/8/pan",
                "value": 30.0,
                "confirmed_round_trips": 2,
            }
        )
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        await handlers["query_mixer_parameter"](
            websocket,
            {
                "request_id": "wing-readonly-preflight:ch8:pan-percent",
                "channel": 8,
                "value_type": "pan_percent",
                "address": "/ch/8/pan",
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "mixer_parameter_value"
        assert payload["request_id"] == "wing-readonly-preflight:ch8:pan-percent"
        assert payload["status"] == "verified_from_physical_readback"
        assert payload["source"] == "physical_readback"
        assert payload["channel"] == 8
        assert payload["value_type"] == "pan"
        assert payload["address"] == "/ch/8/pan"
        assert payload["value"] == 30.0
        server.mixer_client.query_parameter.assert_called_once_with("/ch/8/pan")
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
        payload = await self._apply_manual_write(
            server,
            value=2.0,
            value_type="gain",
            approval_id="unit-gain-missing-read",
        )
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
        payload = await self._apply_manual_write(
            server,
            value=-6.0,
            approval_id="unit-approval",
        )
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
        self._assert_quarantined_operator_correction(
            payload,
            *self._quarantined_operator_expectation("set_eq"),
        )
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
        self._assert_quarantined_operator_correction(
            payload,
            *self._quarantined_operator_expectation("set_compressor"),
        )
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
        self._assert_quarantined_operator_correction(
            result,
            "routing_management_proposal",
            ["routing_write", "bus_assignment_write", "matrix_write"],
        )
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
        self._assert_quarantined_operator_correction(
            result,
            *self._quarantined_operator_expectation(surface),
        )
        getattr(server.mixer_client, method_name).assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("message_type", "payload", "result_type", "method_name"),
        [
            (
                "set_insert",
                {"target_type": "channel", "target": 1, "position": "pre", "slot": "FX1", "on": 1},
                "insert_updated",
                "set_insert",
            ),
            (
                "set_fx_model",
                {"slot": "FX1", "model": "PLATE"},
                "fx_slot_updated",
                "set_fx_model",
            ),
            (
                "set_fx_on",
                {"slot": "FX1", "on": 1},
                "fx_slot_updated",
                "set_fx_on",
            ),
            (
                "set_fx_mix",
                {"slot": "FX1", "mix": 50},
                "fx_slot_updated",
                "set_fx_mix",
            ),
            (
                "set_fx_parameter",
                {"slot": "FX1", "parameter": 1, "value": 0.5},
                "fx_parameter_updated",
                "set_fx_parameter",
            ),
        ],
    )
    async def test_fx_write_handlers_fail_closed_when_mixer_write_returns_false(
        self,
        dummy_wing_server,
        message_type,
        payload,
        result_type,
        method_name,
    ):
        server = dummy_wing_server
        server.connection_mode = "offline"
        getattr(server.mixer_client, method_name).return_value = False
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = fx_handlers.register_handlers(server)

        await handlers[message_type](websocket, payload)

        result = self._decode_last_message(websocket)
        assert result["type"] == result_type
        assert result["status"] == "blocked"
        assert result["reason"] == "mixer_write_failed"
        assert result["surface"] == method_name
        assert result["write_result"] is False
        assert result["live_write"] is False
        assert result["live_authority_granted"] is False
        assert result["readiness_closure_allowed"] is False
        getattr(server.mixer_client, method_name).assert_called_once()

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
        expectation = self._quarantined_operator_expectation(surface)
        if expectation:
            self._assert_quarantined_operator_correction(payload, *expectation)
        server.mixer_client.send.assert_not_called()
        server.mixer_client.set_channel_fader.assert_not_called()
        server.mixer_client.set_channel_gain.assert_not_called()
        server.mixer_client.set_eq_band.assert_not_called()
        server.mixer_client.set_compressor.assert_not_called()
        server.mixer_client.set_channel_phase_invert.assert_not_called()
        server.mixer_client.set_channel_delay.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("surface", "result_type", "target_bus", "controller_method"),
        [
            (
                "apply_system_measurement",
                "system_measurement_apply_result",
                "master",
                "apply_corrections",
            ),
            (
                "reset_system_measurement",
                "system_measurement_reset_result",
                "matrix",
                "reset_corrections",
            ),
        ],
    )
    async def test_system_measurement_master_output_writes_are_quarantined_with_operator_contract(
        self,
        dummy_wing_server,
        surface,
        result_type,
        target_bus,
        controller_method,
    ):
        server = dummy_wing_server
        server.system_measurement_controller = MagicMock()
        websocket = MagicMock()
        websocket.send = AsyncMock()

        if surface == "apply_system_measurement":
            await server.apply_system_measurement(websocket, target_bus=target_bus, target_id=1)
        elif surface == "reset_system_measurement":
            await server.reset_system_measurement(websocket, target_bus=target_bus, target_id=1)
        else:
            raise AssertionError(f"Unhandled surface {surface}")

        payload = self._decode_last_message(websocket)
        assert payload["type"] == result_type
        assert payload["status"] == "blocked"
        assert payload["reason"] == "deployment_write_surface_quarantined"
        assert payload["surface"] == surface
        assert payload["request"] == {"target_bus": target_bus, "target_id": 1}
        self._assert_quarantined_operator_correction(
            payload,
            "master_bus_optimization_proposal",
            ["master_bus_write", "limiter_write", "matrix_pa_feed_write"],
        )
        getattr(server.system_measurement_controller, controller_method).assert_not_called()
        server.mixer_client.send.assert_not_called()

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
        self._assert_quarantined_operator_correction(
            payload,
            "scene_preset_management_proposal",
            ["scene_load", "scene_save", "preset_load", "preset_save"],
        )
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
        self._assert_quarantined_operator_correction(
            payload,
            *self._quarantined_operator_expectation("start_realtime_correction"),
        )
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
        self._assert_quarantined_operator_correction(
            payload,
            *self._quarantined_operator_expectation(payload["surface"]),
        )
        assert payload["operator_correction"]["proposed_correction"]["request"] == command
        getattr(server.mixer_client, method_name).assert_not_called()

    @pytest.mark.asyncio
    async def test_emergency_stop_blocks_later_manual_writes(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        websocket = MagicMock()
        websocket.send = AsyncMock()
        automation = automation_handlers.register_handlers(server)

        await automation["emergency_stop"](websocket, {})
        payload = await self._apply_manual_write(
            server,
            value=2.0,
            value_type="gain",
            approval_id="unit-approval",
        )

        assert payload["status"] == "blocked"
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
        apply_payload = await self._apply_manual_write(
            server,
            value=-6.0,
            approval_id="unit-apply",
        )
        await handlers["rollback_last_manual_write"](
            websocket,
            {
                "approval_id": "rollback-unit",
                "transaction_id": apply_payload["transaction_id"],
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_rollback_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "approval_required"
        assert payload["rollback"] is True
        assert server.mixer_client.set_channel_fader.call_count == 1

    @pytest.mark.asyncio
    async def test_rollback_last_manual_write_rejects_string_false_approval(
        self,
        dummy_wing_server,
    ):
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
        apply_payload = await self._apply_manual_write(
            server,
            value=-6.0,
            approval_id="unit-apply",
        )

        await handlers["rollback_last_manual_write"](
            websocket,
            {
                "approved": "false",
                "approval_id": "rollback-string-false",
                "transaction_id": apply_payload["transaction_id"],
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["status"] == "blocked"
        assert payload["reason"] == "approval_required"
        assert server.mixer_client.set_channel_fader.call_count == 1

    @pytest.mark.asyncio
    async def test_rollback_last_manual_write_blocks_when_pre_read_times_out(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server._manual_write_cooldown_sec = 0.0
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        apply_payload = await self._apply_manual_write(
            server,
            value=-6.0,
            approval_id="unit-apply",
        )

        server.mixer_client.state = {}
        server.mixer_client.query_parameter = MagicMock(
            return_value={"status": "timeout", "address": "/ch/1/fdr", "value": None}
        )

        await handlers["rollback_last_manual_write"](
            websocket,
            {
                "approved": True,
                "approval_id": "rollback-timeout",
                "transaction_id": apply_payload["transaction_id"],
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_rollback_blocked"
        assert payload["status"] == "blocked"
        assert payload["reason"] == "rollback_intervening_change_or_unverified_state"
        assert payload["rollback_of_transaction_id"] == apply_payload["transaction_id"]
        assert payload["current_readback_status"] == "readback_unavailable"
        assert payload["current_readback_source"] == "physical_readback"
        assert payload["proof_status"] == "blocked"
        assert server.mixer_client.set_channel_fader.call_count == 1

    @pytest.mark.asyncio
    async def test_rollback_transaction_cas_blocks_intervening_console_change(self, dummy_wing_server):
        server = dummy_wing_server
        server.operator_mode = "supervised"
        server._manual_write_cooldown_sec = 0.0
        websocket = MagicMock()
        websocket.send = AsyncMock()
        handlers = mixer_handlers.register_handlers(server)

        apply_payload = await self._apply_manual_write(
            server,
            value=-6.0,
            approval_id="unit-cas-apply",
        )
        server.mixer_client.state["/ch/1/fdr"] = -5.0

        await handlers["rollback_last_manual_write"](
            websocket,
            {
                "approved": True,
                "approval_id": "unit-cas-rollback",
                "transaction_id": apply_payload["transaction_id"],
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_rollback_blocked"
        assert payload["reason"] == "rollback_intervening_change_or_unverified_state"
        assert payload["current_readback_value"] == -5.0
        assert payload["expected_current_value"] == -6.0
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

        apply_payload = await self._apply_manual_write(
            server,
            value=-6.0,
            approval_id="unit-apply",
        )
        await handlers["rollback_last_manual_write"](
            websocket,
            {
                "approved": True,
                "approval_id": "rollback-unit",
                "transaction_id": apply_payload["transaction_id"],
            },
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
        assert payload["readback_status"] == "verified_from_physical_readback"
        assert payload["proof_status"] == "verified"
        assert payload["rollback_readback_status"] == "verified_from_physical_readback"
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

        apply_payload = await self._apply_manual_write(
            server,
            value=-6.0,
            approval_id="unit-apply",
        )
        await handlers["rollback_last_manual_write"](
            websocket,
            {
                "approved": True,
                "approval_id": "rollback-unit",
                "transaction_id": apply_payload["transaction_id"],
            },
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

        apply_payload = await self._apply_manual_write(
            server,
            value=-6.0,
            approval_id="unit-apply",
        )
        await handlers["rollback_last_manual_write"](
            websocket,
            {
                "approved": True,
                "approval_id": "rollback-unit",
                "transaction_id": apply_payload["transaction_id"],
            },
        )

        payload = self._decode_last_message(websocket)
        assert payload["type"] == "manual_write_rollback_unverified"
        assert payload["status"] == "sent_unverified"
        assert payload["rollback_status"] == "unverified"
        assert payload["rollback_readback_status"] == "not_verified"
        assert payload["rollback_proof_status"] == "partial"
        assert payload["emergency_stop_activated"] is True
