"""Tests for backend/handlers/soundcheck_handlers.py."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from handlers.soundcheck_handlers import register_handlers


class DummyServer:
    def __init__(self):
        self.config = {
            "mixer": {"type": "wing", "ip": "10.0.0.5", "port": 2223},
            "audio": {"device_name": "Test Device"},
        }
        self.auto_soundcheck_running = False
        self.auto_soundcheck_observe_only = False
        self.auto_soundcheck_engine = None
        self.audio_capture = None
        self._audio_capture_lock = asyncio.Lock()
        self._system_measurement_task = None
        self.sent_messages = []
        self.broadcast_messages = []
        self.imported_soundcheck_bundles = []
        self.send_to_client = AsyncMock(side_effect=self._capture_send)
        self.broadcast = AsyncMock(side_effect=self._capture_broadcast)
        self.start_auto_soundcheck = AsyncMock()

    async def _stop_auto_soundcheck_engine_async(self, *, observe_only):
        async with self._audio_capture_lock:
            engine = self.auto_soundcheck_engine
            self.auto_soundcheck_engine = None
            self.auto_soundcheck_running = False
            self.auto_soundcheck_observe_only = bool(observe_only)
            if engine is not None:
                await asyncio.to_thread(engine.stop)
            return engine is not None

    async def _capture_send(self, websocket, payload):
        self.sent_messages.append((websocket, payload))

    async def _capture_broadcast(self, payload):
        self.broadcast_messages.append(payload)

    def import_soundcheck_recommendations_to_operator_queue(
        self,
        bundle=None,
        *,
        trusted_engine_bundle=False,
    ):
        assert trusted_engine_bundle is True
        self.imported_soundcheck_bundles.append(bundle)
        return {
            "type": "operator_proposals_imported",
            "status": "ok",
            "source": "soundcheck_recommendation_bundle",
            "created_count": len((bundle or {}).get("proposals", [])) if isinstance(bundle, dict) else 0,
            "existing_count": 0,
            "blocked_count": 0,
            "imported_count": len((bundle or {}).get("proposals", [])) if isinstance(bundle, dict) else 0,
            "results": [],
        }


@pytest.mark.asyncio
async def test_get_auto_soundcheck_status_includes_legacy_aliases():
    server = DummyServer()
    handlers = register_handlers(server)

    await handlers["get_auto_soundcheck_status"]("ws", {})

    _, payload = server.sent_messages[-1]
    assert payload["type"] == "auto_soundcheck_status"
    assert payload["is_running"] is False
    assert payload["running"] is False
    assert payload["observe_only"] is False
    assert payload["step_progress"] == 0
    assert payload["progress"] == 0
    assert payload["supervised_transaction_plan_count"] == 0
    assert payload["supervised_transaction_operation_count"] == 0
    assert payload["soundcheck_readiness"]["mix_readiness"] == "soundcheck_incomplete"
    eq_contract = payload["supervised_console_family_registry"]["eq_hpf"]
    assert eq_contract["lifecycle"] == "hardware_proven"
    assert eq_contract["live_allowed_write_kinds"] == []
    assert eq_contract["normal_supervised_apply_allowed"] is False


@pytest.mark.asyncio
async def test_get_auto_soundcheck_status_forwards_session_report_summary():
    server = DummyServer()

    class FakeEngine:
        def get_status(self):
            return {
                "state": "running",
                "autofoh_session_report_summary": "AutoFOH session report: events=4; sent=1; blocked=3; guard_blocks=2",
            }

    server.auto_soundcheck_engine = FakeEngine()
    server.auto_soundcheck_running = True
    handlers = register_handlers(server)

    await handlers["get_auto_soundcheck_status"]("ws", {})

    _, payload = server.sent_messages[-1]
    assert payload["type"] == "auto_soundcheck_status"
    assert payload["autofoh_session_report_summary"].startswith("AutoFOH session report:")


@pytest.mark.asyncio
async def test_status_poll_preserves_aggregate_child_readiness_for_same_run():
    server = DummyServer()
    server.last_soundcheck_readiness = {
        "schema_version": "soundcheck_readiness/v1",
        "run_id": "run-1",
        "plan_revision": 1,
        "plan_fingerprint": "abc",
        "proposal_only": False,
        "mix_readiness": "soundcheck_incomplete",
        "logical_counts": {"total": 2, "applied": 1, "pending": 1},
    }

    class FakeEngine:
        def get_status(self):
            return {
                "state": "running",
                "soundcheck_readiness": {
                    "schema_version": "soundcheck_readiness/v1",
                    "run_id": "run-1",
                    "plan_revision": 1,
                    "plan_fingerprint": "abc",
                    "proposal_only": True,
                    "mix_readiness": "soundcheck_incomplete",
                    "logical_counts": {"total": 2, "applied": 0, "pending": 2},
                },
            }

    server.auto_soundcheck_engine = FakeEngine()
    handlers = register_handlers(server)

    await handlers["get_auto_soundcheck_status"]("ws", {})

    _, payload = server.sent_messages[-1]
    assert payload["soundcheck_readiness"]["proposal_only"] is False
    assert payload["soundcheck_readiness"]["logical_counts"]["applied"] == 1


@pytest.mark.asyncio
async def test_accept_soundcheck_replan_uses_only_server_generated_draft_identity():
    server = DummyServer()
    server.accept_soundcheck_replan_draft = AsyncMock(
        return_value={
            "type": "soundcheck_replan_result",
            "status": "ok",
            "success": True,
            "run_id": "run-1",
            "draft_id": "draft-1",
        }
    )
    handlers = register_handlers(server)

    await handlers["accept_soundcheck_replan"](
        "ws",
        {
            "run_id": "run-1",
            "draft_id": "draft-1",
            "approved": True,
            "approval_id": "operator-replan-1",
            "revised_manifest": {"untrusted": "ignored"},
        },
    )

    server.accept_soundcheck_replan_draft.assert_awaited_once_with(
        run_id="run-1",
        draft_id="draft-1",
        approved=True,
        approval_id="operator-replan-1",
    )
    assert server.sent_messages[-1][1]["status"] == "ok"


@pytest.mark.asyncio
async def test_stage_reanalysis_handler_requires_explicit_server_path():
    server = DummyServer()
    server.reanalyze_soundcheck_stage = AsyncMock(
        return_value={
            "type": "soundcheck_stage_reanalysis_result",
            "status": "ready_for_review",
            "success": True,
            "run_id": "run-1",
            "to_plan_revision": 2,
            "live_authority_changed": False,
        }
    )
    handlers = register_handlers(server)

    await handlers["reanalyze_soundcheck_stage"](
        "ws",
        {
            "run_id": "run-1",
            "approved": True,
            "approval_id": "stage-reanalysis-approval",
        },
    )

    server.reanalyze_soundcheck_stage.assert_awaited_once_with(
        run_id="run-1",
        approved=True,
        approval_id="stage-reanalysis-approval",
    )
    assert server.sent_messages[-1][1]["live_authority_changed"] is False


@pytest.mark.asyncio
async def test_finalize_soundcheck_readiness_is_explicit_and_read_only():
    server = DummyServer()
    server.reconcile_soundcheck_readiness = AsyncMock(
        return_value={
            "run_id": "run-1",
            "mix_readiness": "show_mix_ready",
        }
    )
    handlers = register_handlers(server)

    await handlers["finalize_soundcheck_readiness"](
        "ws",
        {"run_id": "run-1"},
    )

    server.reconcile_soundcheck_readiness.assert_awaited_once_with("run-1")
    payload = server.sent_messages[-1][1]
    assert payload["type"] == "soundcheck_finalize_result"
    assert payload["success"] is True
    assert payload["live_write"] is False
    assert payload["soundcheck_readiness"]["mix_readiness"] == "show_mix_ready"


@pytest.mark.asyncio
async def test_start_auto_soundcheck_handler_starts_new_engine_with_observe_only(monkeypatch):
    server = DummyServer()
    handlers = register_handlers(server)
    created = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.state = type("State", (), {"value": "idle"})()

        def start_async(self):
            self.started = True

    monkeypatch.setattr("handlers.soundcheck_handlers.AutoSoundcheckEngine", FakeEngine)

    await handlers["start_auto_soundcheck"]("ws", {
        "device_id": "dev1",
        "channels": [1, 2],
        "channel_settings": {},
        "channel_mapping": {},
        "timings": {"gain_staging": 10},
        "observe_only": True,
    })

    assert created["selected_channels"] == [1, 2]
    assert created["num_channels"] == 48
    assert created["audio_device_name"] == "dev1"
    assert created["observe_only"] is True
    assert created["auto_apply"] is False
    assert server.auto_soundcheck_running is True


@pytest.mark.asyncio
async def test_start_auto_engine_uses_config_and_wires_callbacks(monkeypatch):
    server = DummyServer()
    handlers = register_handlers(server)
    created = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.state = type("State", (), {"value": "idle"})()
            self.started = False

        def start_async(self):
            self.started = True

    monkeypatch.setattr("handlers.soundcheck_handlers.AutoSoundcheckEngine", FakeEngine)

    await handlers["start_auto_engine"]("ws", {})

    assert created["mixer_type"] == "wing"
    assert created["mixer_ip"] == "10.0.0.5"
    assert created["mixer_port"] == 2223
    assert created["audio_device_name"] == "Test Device"
    assert created["observe_only"] is True
    assert created["auto_apply"] is False
    assert callable(created["on_state_change"])
    assert callable(created["on_channel_update"])

    created["on_state_change"]("running", "Engine started")
    created["on_channel_update"](3, {"preset": "kick"})
    await asyncio.sleep(0)

    assert {"type": "auto_engine_state", "state": "running", "message": "Engine started"} in server.broadcast_messages
    assert {"type": "auto_engine_channel", "channel": 3, "data": {"preset": "kick"}} in server.broadcast_messages


@pytest.mark.asyncio
async def test_start_auto_engine_rejects_error_state_with_live_worker_thread(monkeypatch):
    server = DummyServer()

    class LiveThread:
        def is_alive(self):
            return True

    server.auto_soundcheck_engine = type(
        "FailedEngine",
        (),
        {
            "state": type("State", (), {"value": "error"})(),
            "_engine_thread": LiveThread(),
            "_monitor_thread": None,
        },
    )()
    handlers = register_handlers(server)
    constructor = MagicMock()
    monkeypatch.setattr("handlers.soundcheck_handlers.AutoSoundcheckEngine", constructor)

    await handlers["start_auto_engine"]("ws", {})

    constructor.assert_not_called()
    payload = server.sent_messages[-1][1]
    assert payload["status"] == "blocked"
    assert payload["reason"] == "auto_soundcheck_cleanup_incomplete"
    assert payload["running"] is False
    assert payload["cleanup_complete"] is False


@pytest.mark.asyncio
async def test_start_auto_soundcheck_rejects_running_preview_capture(monkeypatch):
    server = DummyServer()
    server.mixer_client = type("SharedMixer", (), {"is_connected": True})()
    server.audio_capture = type("SharedAudio", (), {"running": True})()
    handlers = register_handlers(server)
    created = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.state = type("State", (), {"value": "idle"})()

        def start_async(self):
            pass

    monkeypatch.setattr("handlers.soundcheck_handlers.AutoSoundcheckEngine", FakeEngine)

    await handlers["start_auto_soundcheck"]("ws", {"channels": [1]})

    assert created == {}
    assert server.auto_soundcheck_running is False
    assert server.sent_messages[-1][1]["reason"] == "audio_capture_in_use_by_preview"


@pytest.mark.asyncio
async def test_start_auto_soundcheck_rejects_active_system_measurement(monkeypatch):
    server = DummyServer()
    active_measurement = asyncio.create_task(asyncio.sleep(10))
    server._system_measurement_task = active_measurement
    handlers = register_handlers(server)
    created = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr("handlers.soundcheck_handlers.AutoSoundcheckEngine", FakeEngine)
    try:
        await handlers["start_auto_soundcheck"]("ws", {"channels": [1]})
    finally:
        active_measurement.cancel()
        await asyncio.gather(active_measurement, return_exceptions=True)

    assert created == {}
    assert server.sent_messages[-1][1]["reason"] == (
        "audio_capture_in_use_by_system_measurement"
    )


@pytest.mark.asyncio
async def test_start_auto_soundcheck_rejects_unbounded_channel_allocation(monkeypatch):
    server = DummyServer()
    handlers = register_handlers(server)
    created = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr("handlers.soundcheck_handlers.AutoSoundcheckEngine", FakeEngine)
    await handlers["start_auto_soundcheck"]("ws", {"channels": [1, 10000]})

    assert created == {}
    assert server.sent_messages[-1][1]["reason"] == "invalid_audio_capture_dimensions"


@pytest.mark.asyncio
async def test_auto_engine_imports_soundcheck_recommendations_on_running_state(monkeypatch):
    server = DummyServer()
    handlers = register_handlers(server)
    bundle = {
        "schema_version": "soundcheck_recommendation_bundle/v1",
        "proposals": [
            {
                "proposal_id": "unit-fader",
                "action_type": "set_fader_db",
                "target": {"mixer_channel": 2},
                "current_state": {"fader_db": -10.0},
                "requested_state": {"fader_db": -8.5},
            }
        ],
    }

    class FakeEngine:
        def __init__(self, **kwargs):
            self.on_state_change = kwargs["on_state_change"]
            self.state = type("State", (), {"value": "idle"})()
            self.started = False
            self.get_soundcheck_recommendation_bundle = MagicMock(return_value=bundle)

        def start_async(self):
            self.started = True

    monkeypatch.setattr("handlers.soundcheck_handlers.AutoSoundcheckEngine", FakeEngine)

    await handlers["start_auto_engine"]("ws", {})
    server.auto_soundcheck_engine.on_state_change("running", "Auto-soundcheck complete, monitoring...")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert server.imported_soundcheck_bundles == [bundle]
    imported = next(
        item
        for item in server.broadcast_messages
        if item.get("type") == "operator_proposals_imported"
    )
    assert imported["status"] == "ok"
    assert imported["source"] == "soundcheck_recommendation_bundle"
    assert imported["created_count"] == 1
    assert imported["existing_count"] == 0
    assert imported["blocked_count"] == 0
    assert imported["imported_count"] == 1
    assert imported["results"] == []
    assert imported["trigger"] == "auto_soundcheck_state"
    assert imported["trigger_state"] == "running"
    assert imported["recommendation_derivation_complete"] is False


@pytest.mark.asyncio
async def test_superseded_engine_cannot_import_after_delayed_preflight(monkeypatch):
    server = DummyServer()
    handlers = register_handlers(server)
    preflight_started = asyncio.Event()
    release_preflight = asyncio.Event()
    bundle = {
        "schema_version": "soundcheck_recommendation_bundle/v1",
        "proposals": [],
        "supervised_transaction_plans": {
            "run_id": "delayed-engine-a",
            "plan_revision": 1,
            "plan_fingerprint": "engine-a",
            "plans": [],
            "unsupported_actions": [],
        },
    }

    async def delayed_preflight(_manifest):
        preflight_started.set()
        await release_preflight.wait()
        return {"assessments": [], "global_blockers": []}

    server.preflight_soundcheck_manifest = delayed_preflight

    class FakeEngine:
        def __init__(self, **kwargs):
            self.on_state_change = kwargs["on_state_change"]
            self.state = type("State", (), {"value": "idle"})()
            self.get_soundcheck_recommendation_bundle = MagicMock(
                return_value=bundle
            )

        def start_async(self):
            pass

    monkeypatch.setattr(
        "handlers.soundcheck_handlers.AutoSoundcheckEngine",
        FakeEngine,
    )
    await handlers["start_auto_engine"]("ws", {})
    engine_a = server.auto_soundcheck_engine
    engine_a.on_state_change("running", "A complete")
    await preflight_started.wait()
    server.auto_soundcheck_engine = object()
    release_preflight.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert server.imported_soundcheck_bundles == []
    blocked = next(
        payload
        for payload in server.broadcast_messages
        if payload.get("reason") == "soundcheck_engine_superseded"
    )
    assert blocked["status"] == "blocked"
