import importlib
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _extract_frontend_websocket_types() -> set[str]:
    text = (REPO_ROOT / "frontend/src/services/websocket.js").read_text()
    return set(re.findall(r"type:\s*'([^']+)'", text))


def _extract_backend_handler_types() -> set[str]:
    handler_types: set[str] = set()
    for path in (REPO_ROOT / "backend/handlers").glob("*_handlers.py"):
        text = path.read_text()
        handler_types.update(re.findall(r"['\"]([A-Za-z0-9_]+)['\"]\s*:\s*handle_", text))
    return handler_types


def test_backend_server_importable_from_repo_root():
    module = importlib.import_module("backend.server")
    assert hasattr(module, "AutoMixerServer")
    assert hasattr(module, "convert_numpy_types")


# События, существующие только внутри клиента (notifyListeners /
# emitConnectionState в websocket.js) — на бэкенд не отправляются
CLIENT_LOCAL_EVENTS = {"connected", "disconnected", "connection_state"}


def test_frontend_websocket_commands_have_backend_handlers():
    frontend_types = _extract_frontend_websocket_types()
    backend_types = _extract_backend_handler_types()
    assert frontend_types - backend_types - CLIENT_LOCAL_EVENTS == set()


def test_conventions_do_not_recommend_direct_wing_fader_writes():
    conventions = (REPO_ROOT / "docs/CONVENTIONS.md").read_text()
    correct_example = conventions.split("# Неправильно", 1)[0]
    assert "wing_client.set_fader" not in correct_example
    assert "_apply_manual_console_write" in correct_example


@pytest.mark.asyncio
async def test_unimplemented_status_endpoints_advertise_disabled_backend_controls():
    server_module = importlib.import_module("backend.server")

    with patch.object(server_module.AutoMixerServer, "_load_config", return_value={}), patch.object(server_module, "BleedService"):
        server = server_module.AutoMixerServer()

    websocket = object()
    server.send_to_client = AsyncMock()
    methods = [
        ("get_auto_gate_status", "auto_gate_status"),
        ("get_auto_panner_status", "auto_panner_status"),
        ("get_auto_reverb_status", "auto_reverb_status"),
        ("get_auto_effects_status", "auto_effects_status"),
        ("get_cross_adaptive_eq_status", "cross_adaptive_eq_status"),
    ]

    for method_name, event_type in methods:
        await getattr(server, method_name)(websocket, request_id="req-1")
        payload = server.send_to_client.await_args_list[-1].args[1]
        assert payload["type"] == event_type
        assert payload["implemented"] is False
        assert payload["active"] is False
