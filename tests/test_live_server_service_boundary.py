"""Server/service cutover proof without importing unrelated legacy controllers.

Execute the real selected server methods from their AST. The full server still
imports independent legacy controllers outside this bounded soundcheck change;
loading those is neither necessary nor desirable in the offline focused suite.
"""
from __future__ import annotations

import ast
import asyncio
from copy import deepcopy
import logging
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from live_runtime.contracts import LiveMode
from live_runtime.service import LiveSoundcheckService


def server_methods(*names):
    path = ROOT / "backend" / "server.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AutoMixerServer")
    methods = [node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert {node.name for node in methods} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[]))
    scope = {"logger": logging.getLogger(__name__), "asyncio": asyncio, "scan_and_recognize": lambda names: {}}
    exec(compile(module, str(path), "exec"), scope)
    return {name: scope[name] for name in names}


def bind_server(service, *names):
    server = SimpleNamespace(_live_soundcheck_service=service)
    for name, function in server_methods(*names).items():
        setattr(server, name, MethodType(function, server))
    return server


class StatusOnlyService:
    def __init__(self, *, active=True, mode="observe"):
        self.active = active
        self.status = {"state": "running" if active else "idle", "mode": mode,
                       "selected_channels": [1, 3], "channels": {"3": {"peak_db": -8.0, "rms_db": -20.0, "lufs": -22.0}}}
        self.stops = 0

    def is_active(self):
        return self.active

    def get_status(self):
        return deepcopy(self.status)

    def stop(self):
        self.stops += 1
        self.active = False

    @property
    def active_engine(self):
        raise AssertionError("server must not inspect the engine")

    @property
    def mixer_client(self):
        raise AssertionError("server must not obtain live hardware")

    @property
    def audio_capture(self):
        raise AssertionError("server must not obtain live audio ownership")


@pytest.mark.parametrize("mode,observe_only", [("observe", True), ("propose", True), ("freeze", True), ("bench_test", False), ("auto_safe", False)])
def test_sync_reads_status_only_and_preserves_independent_hardware(mode, observe_only):
    service = StatusOnlyService(mode=mode)
    server = bind_server(service, "_sync_runtime_from_live_soundcheck")
    mixer, capture, agent_mixer = object(), object(), object()
    server.mixer_client, server.audio_capture = mixer, capture
    server.connection_mode = "independent"
    server.mixing_agent = SimpleNamespace(mixer=agent_mixer)
    status = server._sync_runtime_from_live_soundcheck()
    assert status["selected_channels"] == [1, 3]
    assert server.auto_soundcheck_running is True
    assert server.auto_soundcheck_observe_only is observe_only
    assert server.mixer_client is mixer and server.audio_capture is capture
    assert server.connection_mode == "independent"
    assert server.mixing_agent.mixer is agent_mixer


def test_sync_without_service_is_read_only_and_idle():
    server = bind_server(None, "_sync_runtime_from_live_soundcheck")
    server.mixer_client = object()
    before = server.mixer_client
    assert server._sync_runtime_from_live_soundcheck() == {}
    assert server.mixer_client is before


def test_inactive_service_clears_stale_display_flags():
    server = bind_server(StatusOnlyService(active=False), "_sync_runtime_from_live_soundcheck")
    server.auto_soundcheck_running = True
    server.auto_soundcheck_observe_only = True
    server._sync_runtime_from_live_soundcheck()
    assert not server.auto_soundcheck_running
    assert not server.auto_soundcheck_observe_only


def test_selected_channels_and_metrics_come_from_service_status():
    service = StatusOnlyService()
    server = bind_server(service, "_sync_runtime_from_live_soundcheck", "_selected_agent_channels", "collect_agent_channel_states")
    server.mixer_client = None
    server.audio_capture = None
    server.config = {}
    server._agent_channel_state = lambda channel, name, preset, metrics: dict(channel=channel, metrics=metrics)
    assert server._selected_agent_channels() == [1, 3]
    states = server.collect_agent_channel_states()
    assert sorted(states) == [1, 3]
    assert states[3]["metrics"]["peak_db"] == -8.0
    assert states[3]["metrics"]["rms_db"] == -20.0


@pytest.mark.parametrize("stop_fails", [False, True])
def test_cleanup_stops_service_before_independent_audio_and_preserves_evidence(stop_fails):
    order = []
    service = StatusOnlyService()
    def stop():
        order.append("live_service")
        if stop_fails:
            raise RuntimeError("teardown failed")
        service.active = False
    service.stop = stop
    server = bind_server(service, "cleanup_all_controllers", "_safe_cleanup_call")
    for name in ("gain_staging", "voice_control", "auto_eq_controller", "multi_channel_auto_eq_controller",
                 "phase_alignment_controller", "system_measurement_controller", "auto_fader_controller",
                 "auto_compressor_controller", "mixing_agent", "mixing_agent_task", "agent_training_service",
                 "auto_soundcheck_task", "auto_soundcheck_websocket"):
        setattr(server, name, None)
    server.auto_soundcheck_running = True
    server.auto_soundcheck_observe_only = True
    server.audio_capture = SimpleNamespace(stop=lambda: order.append("independent_audio"))
    server.mixer_client = SimpleNamespace(disconnect=lambda: order.append("independent_mixer"))
    server.cleanup_all_controllers()
    assert order == ["live_service", "independent_audio", "independent_mixer"]
    # Preserve the service reference for audit/status, including failed teardown.
    assert server._live_soundcheck_service is service
    assert not server.auto_soundcheck_running
    assert not server.auto_soundcheck_observe_only


@pytest.mark.parametrize("owned", [False, True])
def test_status_selection_is_explicit_and_detached_from_request(owned):
    service = LiveSoundcheckService()
    assert service.get_status()["selected_channels"] == []
    service._request = SimpleNamespace(mixer_type="wing", mode=LiveMode.OBSERVE,
                                       selected_channels=[1, 3], capture_bridge=None)
    service._engine = SimpleNamespace(get_status=lambda: {"state": "running", "selected_channels": [48]})
    if owned:
        service._mixer_session = SimpleNamespace(connected=False, client=None)
    result = service.get_status()
    assert result["selected_channels"] == [1, 3]
    result["selected_channels"].append(47)
    assert service._request.selected_channels == [1, 3]
    assert service.get_status()["selected_channels"] == [1, 3]


def test_configured_all_inputs_uses_roles_not_reserved_main_returns():
    service = LiveSoundcheckService()
    service._request = SimpleNamespace(selected_channels=[], capture_bridge=SimpleNamespace(roles={3: "vocal", 1: "kick"}))
    assert service._selected_channel_ids() == [1, 3]
