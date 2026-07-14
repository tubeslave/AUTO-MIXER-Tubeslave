from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCHEMA = "automixer.m005_current_release_compatibility_replay.v1"


def _file_identity(path: Path) -> dict[str, Any]:
    value = path.read_bytes()
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": True,
        "size": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }


def _loaded_module_identity(module: Any, expected_relative_path: str) -> dict[str, Any]:
    loaded_path = Path(str(getattr(module, "__file__", ""))).resolve()
    expected_path = (REPO_ROOT / expected_relative_path).resolve()
    assert loaded_path == expected_path
    return _file_identity(loaded_path)


def _find_loaded_module(expected_relative_path: str) -> Any:
    expected_path = (REPO_ROOT / expected_relative_path).resolve()
    matches = []
    for module in tuple(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if module_file and Path(str(module_file)).resolve() == expected_path:
            matches.append(module)
    assert matches
    return matches[0]


class _CaptureWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def send(self, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        assert isinstance(payload, dict)
        self.messages.append(payload)


class _StrictNoConsoleMixer:
    def __init__(self) -> None:
        self.is_connected = True
        self.state = {
            "/ch/3/fdr": -12.1,
            "/ch/3/in/set/trim": 0.0,
        }
        stale = time.time() - 120.0
        self.state_timestamps = {
            "/ch/3/fdr": stale,
            "/ch/3/in/set/trim": stale,
        }
        self.query_parameter = None
        self.writer_calls = {
            "set_channel_fader": 0,
            "set_channel_gain": 0,
            "send": 0,
            "send_message": 0,
        }

    def get_state(self) -> dict[str, Any]:
        return dict(self.state)

    def get_write_gate_status(self) -> dict[str, Any]:
        return {"state": "migration_replay_no_console"}

    def _unexpected_writer(self, name: str) -> None:
        self.writer_calls[name] += 1
        raise AssertionError(f"M005 migration replay reached console writer: {name}")

    def set_channel_fader(self, *_args: Any, **_kwargs: Any) -> bool:
        self._unexpected_writer("set_channel_fader")
        return False

    def set_channel_gain(self, *_args: Any, **_kwargs: Any) -> bool:
        self._unexpected_writer("set_channel_gain")
        return False

    def send(self, *_args: Any, **_kwargs: Any) -> bool:
        self._unexpected_writer("send")
        return False

    def send_message(self, *_args: Any, **_kwargs: Any) -> bool:
        self._unexpected_writer("send_message")
        return False


def _review_binding(server: Any, proposal_id: str) -> dict[str, Any]:
    proposal = server.operator_proposal_queue.find(proposal_id=proposal_id)
    assert proposal is not None
    return {
        "expected_review_contract_version": proposal.review_contract_version,
        "expected_review_content_digest": proposal.review_content_digest,
    }


async def _run_stale_scenario(
    server: Any,
    websocket: _CaptureWebSocket,
    *,
    scenario: str = "stale_cache",
    value_type: str,
    value: float,
    expected_reason: str = "pre_read_cache_not_fresh",
) -> dict[str, Any]:
    before = len(websocket.messages)
    proposal_id = f"m005-migration-replay-{scenario}-{value_type}"
    address = "/ch/3/fdr" if value_type == "fader" else "/ch/3/in/set/trim"
    old_value = float(server.mixer_client.state[address])
    await server._dispatch["create_operator_proposal"](
        websocket,
        {
            "id": proposal_id,
            "title": f"M005 migration stale {value_type}",
            "channel": 3,
            "value_type": value_type,
            "value": value,
            "reason": f"m005_current_release_migration_{scenario}_replay",
        },
    )
    create = websocket.messages[-1]
    assert create["type"] == "operator_proposal_created"
    assert create["proposal"]["id"] == proposal_id

    await server._dispatch["apply_operator_proposal"](
        websocket,
        {
            "proposal_id": proposal_id,
            **_review_binding(server, proposal_id),
            "approved": True,
            "approval_id": f"{proposal_id}-approval",
            "idempotency_key": f"{proposal_id}-key",
        },
    )
    terminal = websocket.messages[-1]
    assert terminal["type"] == "operator_proposal_apply_blocked"
    assert terminal["status"] == "apply_blocked"
    assert terminal["reason"] == expected_reason
    proposal = server.operator_proposal_queue.find(proposal_id=proposal_id)
    assert proposal is not None
    if expected_reason == "pre_read_cache_not_fresh":
        persisted_raw = server.operator_proposal_queue.load_result_evidence(
            proposal,
            expected_result_digest=terminal["result_digest"],
        )
        persisted = json.loads(persisted_raw)
        assert persisted["reason"] == expected_reason
        assert persisted["no_console_send_proven"] is True
        assert persisted["cache_freshness"]["status"] == "stale"
        persisted_result_sha256 = "sha256:" + hashlib.sha256(persisted_raw).hexdigest()
        assert persisted_result_sha256 == terminal["result_digest"]
    else:
        assert terminal.get("result_digest") is None
        assert proposal.result is None
        persisted = None
        persisted_result_sha256 = None
    return {
        "scenario": scenario,
        "value_type": value_type,
        "request_count": len(websocket.messages) - before,
        "proposal_id": proposal_id,
        "request": {
            "channel": 3,
            "address": address,
            "old_value": old_value,
            "requested_value": value,
            "delta": value - old_value,
            "direction": "negative_only",
        },
        "terminal": terminal,
        "persisted_apply_result": persisted,
        "persisted_result_sha256": persisted_result_sha256,
        "no_console_send_proven": True,
    }


@pytest.mark.asyncio
async def test_m005_current_release_migration_replay_emits_bound_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_text = os.environ.get("AUTOMIXER_M005_MIGRATION_REPLAY_OUTPUT")
    output_path = Path(output_text).resolve() if output_text else tmp_path / "replay.json"
    challenge = os.environ.get(
        "AUTOMIXER_M005_MIGRATION_REPLAY_CHALLENGE",
        "0" * 64,
    )
    assert len(challenge) == 64
    assert all(character in "0123456789abcdef" for character in challenge)
    isolated_environment = {
        "console_runtime": "AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR",
        "operator_results": "AUTOMIXER_OPERATOR_PROPOSAL_RESULT_DIR",
        "manual_write_artifacts": "AUTOMIXER_MANUAL_WRITE_ARTIFACTS_DIR",
    }
    isolation_roots: dict[str, str] = {}
    for key, environment_name in isolated_environment.items():
        configured = os.environ.get(environment_name)
        if output_text:
            assert configured
            configured_path = Path(configured).resolve()
        else:
            configured_path = (tmp_path / key).resolve()
            monkeypatch.setenv(environment_name, str(configured_path))
        isolation_roots[key] = str(configured_path)
    monkeypatch.setenv("AUTOMIXER_OPERATOR_MODE", "auto")
    monkeypatch.setenv("AUTOMIXER_AUTO_MODE_UNLOCKED", "true")
    monkeypatch.setenv("AUTOMIXER_WING_DEPLOYMENT_WRITE_BOUNDARY_ENABLED", "false")

    io_guard_call_counts = {
        "socket_inet_construct": 0,
        "socket_bind": 0,
        "socket_connect": 0,
        "socket_connect_ex": 0,
        "socket_listen": 0,
        "socket_send": 0,
        "socket_sendall": 0,
        "socket_sendmsg": 0,
        "socket_sendto": 0,
        "pyaudio_instance": 0,
        "sounddevice_input_stream": 0,
        "audio_capture_start": 0,
    }

    def guarded_call(name: str):
        def fail(*_args: Any, **_kwargs: Any):
            io_guard_call_counts[name] += 1
            raise AssertionError(f"M005 replay attempted forbidden I/O: {name}")

        return fail

    original_socket_init = socket.socket.__init__

    def guarded_socket_init(
        instance: socket.socket,
        family: int = -1,
        type: int = -1,
        proto: int = -1,
        fileno: int | None = None,
    ) -> None:
        if fileno is None and family in {-1, socket.AF_INET, socket.AF_INET6}:
            io_guard_call_counts["socket_inet_construct"] += 1
            raise AssertionError(
                "M005 replay attempted forbidden I/O: socket_inet_construct"
            )
        original_socket_init(instance, family, type, proto, fileno)

    monkeypatch.setattr(socket.socket, "__init__", guarded_socket_init)
    monkeypatch.setattr(socket.socket, "bind", guarded_call("socket_bind"))
    monkeypatch.setattr(socket.socket, "connect", guarded_call("socket_connect"))
    monkeypatch.setattr(
        socket.socket,
        "connect_ex",
        guarded_call("socket_connect_ex"),
    )
    monkeypatch.setattr(socket.socket, "listen", guarded_call("socket_listen"))
    monkeypatch.setattr(socket.socket, "send", guarded_call("socket_send"))
    monkeypatch.setattr(socket.socket, "sendall", guarded_call("socket_sendall"))
    if hasattr(socket.socket, "sendmsg"):
        monkeypatch.setattr(
            socket.socket,
            "sendmsg",
            guarded_call("socket_sendmsg"),
        )
    monkeypatch.setattr(socket.socket, "sendto", guarded_call("socket_sendto"))

    try:
        import sounddevice as sounddevice_module
    except (ImportError, OSError):
        sounddevice_module = None
    if sounddevice_module is not None:
        monkeypatch.setattr(
            sounddevice_module,
            "InputStream",
            guarded_call("sounddevice_input_stream"),
        )
    try:
        import pyaudio as pyaudio_module
    except (ImportError, OSError):
        pyaudio_module = None
    if pyaudio_module is not None:
        monkeypatch.setattr(
            pyaudio_module,
            "PyAudio",
            guarded_call("pyaudio_instance"),
        )

    # All project/backend modules are deliberately imported only after the
    # process-level socket/audio constructors above are sealed.
    import audio_capture as audio_capture_module
    import handlers.product_state_handlers as product_state_handlers_module
    import operator_proposal_queue as operator_proposal_queue_module
    import operator_proposal_result_store as operator_proposal_result_store_module
    import runtime_identity as runtime_identity_module
    import server as server_module
    import tools.migrate_m005_current_release_provenance as migration_module

    AudioCapture = audio_capture_module.AudioCapture
    OperatorProposalQueue = operator_proposal_queue_module.OperatorProposalQueue
    ResultEvidenceStore = operator_proposal_result_store_module.ResultEvidenceStore
    runtime_identity_manifest_sha256 = (
        runtime_identity_module.runtime_identity_manifest_sha256
    )
    AutoMixerServer = server_module.AutoMixerServer
    _worktree_status_audit = migration_module._worktree_status_audit

    monkeypatch.setattr(
        AudioCapture,
        "start",
        guarded_call("audio_capture_start"),
    )

    with (
        patch("server.BleedService"),
        patch.object(AutoMixerServer, "_load_config", return_value={}),
        patch.object(AutoMixerServer, "_init_training_service", return_value=None),
    ):
        server = AutoMixerServer(ws_host="127.0.0.1", ws_port=8765)

    server.operator_mode = "auto"
    server._auto_mode_unlocked = True
    server._wing_deployment_write_boundary_enabled = False
    server._wing_observation_only = False
    server.connection_mode = "offline"
    mixer = _StrictNoConsoleMixer()
    server.mixer_client = mixer
    server.operator_proposal_queue = OperatorProposalQueue(
        result_store=ResultEvidenceStore(
            Path(isolation_roots["operator_results"]),
            min_free_bytes=0,
        ),
        result_evidence_lifecycle_ready=True,
    )
    server._apply_manual_console_write = AsyncMock(
        side_effect=AssertionError("canonical console writer must not be reached")
    )
    websocket = _CaptureWebSocket()

    launch_binding = server._runtime_identity["launch_binding"]
    proof_bound = launch_binding.get("mode") == "proof_runner"
    if output_text:
        assert proof_bound is True
    runtime_request = {
        "type": "get_runtime_identity",
        "challenge_nonce": challenge[:32],
    }
    if proof_bound:
        runtime_request.update(
            {
                "expected_launch_nonce": launch_binding["launch_nonce"],
                "expected_prestart_provenance_sha256": launch_binding[
                    "prestart_provenance_sha256"
                ],
                "expected_prestart_provenance_size": launch_binding[
                    "prestart_provenance_size"
                ],
            }
        )
    await server._dispatch["get_runtime_identity"](websocket, runtime_request)
    runtime_identity = websocket.messages[-1]
    assert runtime_identity["type"] == "runtime_identity"
    assert runtime_identity["request_challenge_nonce"] == challenge[:32]
    assert runtime_identity["manifest_sha256"] == runtime_identity_manifest_sha256(
        runtime_identity
    )

    scenarios = [
        await _run_stale_scenario(
            server,
            websocket,
            value_type="fader",
            value=-12.35,
        ),
        await _run_stale_scenario(
            server,
            websocket,
            value_type="gain",
            value=-0.25,
        ),
    ]
    mixer.is_connected = False
    scenarios.extend(
        [
            await _run_stale_scenario(
                server,
                websocket,
                scenario="disconnected",
                value_type="fader",
                value=-12.35,
                expected_reason="operator_proposal_global_hazard_active",
            ),
            await _run_stale_scenario(
                server,
                websocket,
                scenario="disconnected",
                value_type="gain",
                value=-0.25,
                expected_reason="operator_proposal_global_hazard_active",
            ),
        ]
    )
    assert server._apply_manual_console_write.await_count == 0
    assert all(count == 0 for count in mixer.writer_calls.values())

    conftest_module = _find_loaded_module("tests/conftest.py")
    current_test_module = _find_loaded_module(
        "tests/test_m005_current_release_migration_replay.py"
    )
    loaded_modules = {
        "tools/migrate_m005_current_release_provenance.py": migration_module,
        "backend/server.py": server_module,
        "backend/audio_capture.py": audio_capture_module,
        "backend/handlers/product_state_handlers.py": product_state_handlers_module,
        "backend/operator_proposal_queue.py": operator_proposal_queue_module,
        "backend/operator_proposal_result_store.py": operator_proposal_result_store_module,
        "backend/runtime_identity.py": runtime_identity_module,
        "tests/conftest.py": conftest_module,
        "tests/test_m005_current_release_migration_replay.py": current_test_module,
    }
    module_identities = {
        relative: _loaded_module_identity(module, relative)
        for relative, module in loaded_modules.items()
    }
    report = {
        "schema": REPLAY_SCHEMA,
        "challenge": challenge,
        "child_pid": os.getpid(),
        "proof_owned_process": proof_bound,
        "runtime_identity": runtime_identity,
        "runtime_worktree_status_audit": _worktree_status_audit(),
        "module_identities": module_identities,
        "isolation_roots": isolation_roots,
        "canonical_entrypoints": [
            "create_operator_proposal",
            "apply_operator_proposal",
        ],
        "scenarios": scenarios,
        "writer_call_counts": {
            "_apply_manual_console_write": server._apply_manual_console_write.await_count,
            **mixer.writer_calls,
        },
        "io_guard_call_counts": io_guard_call_counts,
        "no_console_send_proven": True,
        "contacts_backend_transport": False,
        "contacts_wing": False,
        "network_listener_opened": False,
        "audio_device_opened": False,
        "live_authority_granted": False,
        "live_write": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assert not output_path.exists()
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == report
