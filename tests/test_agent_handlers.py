"""Tests for backend/handlers/agent_handlers.py."""

import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from handlers.agent_handlers import register_handlers


class FakeMode:
    value = "suggest"


class FakeState:
    mode = FakeMode()
    is_running = False


class FakeAgent:
    def __init__(self):
        self.state = FakeState()
        self.pending = [{"index": 0, "type": "adjust_gain", "channel": 1}]

    def get_status(self):
        return {
            "mode": self.state.mode.value,
            "is_running": self.state.is_running,
            "pending_actions": len(self.pending),
        }

    def get_channel_summary(self):
        return {1: {"instrument": "lead_vocal", "peak_db": -6.0}}

    def get_pending_actions(self):
        return list(self.pending)

    def approve_action(self, idx):
        if idx == 0 and self.pending:
            self.pending.pop(0)
            return True
        return False

    def approve_all_pending(self):
        count = len(self.pending)
        self.pending.clear()
        return count

    def dismiss_action(self, idx):
        if idx == 0 and self.pending:
            self.pending.pop(0)
            return True
        return False

    def dismiss_all_pending(self):
        self.pending.clear()

    def get_action_history(self, limit):
        return []

    def get_action_audit_log(self, limit):
        return [{"status": "applied"}]

    def emergency_stop(self):
        self.state.is_running = False
        self.emergency_stopped = True

    def update_channel_states_batch(self, states):
        self.states = states


class DummyServer:
    def __init__(self):
        self.mixing_agent = None
        self.agent_auto_apply_enabled = False
        self.sent_messages = []
        self.send_to_client = AsyncMock(side_effect=self._capture_send)
        self.collect_agent_channel_states = lambda channels=None: {1: {"channel_id": 1}}

    async def _capture_send(self, websocket, payload):
        self.sent_messages.append((websocket, payload))

    async def init_mixing_agent(self, **kwargs):
        self.init_kwargs = kwargs
        self.mixing_agent = FakeAgent()
        self.mixing_agent.state.is_running = kwargs.get("start", False)
        return self.mixing_agent

    async def stop_mixing_agent(self):
        if self.mixing_agent:
            self.mixing_agent.state.is_running = False


@pytest.mark.asyncio
async def test_get_agent_status_before_init_is_safe():
    server = DummyServer()
    handlers = register_handlers(server)

    await handlers["get_agent_status"]("ws", {})

    _, payload = server.sent_messages[-1]
    assert payload["type"] == "agent_status"
    assert payload["is_running"] is False
    assert payload["error"] == "Agent not initialized"


@pytest.mark.asyncio
async def test_start_agent_initializes_safe_suggest_mode():
    server = DummyServer()
    handlers = register_handlers(server)

    await handlers["start_agent"]("ws", {
        "mode": "suggest",
        "channels": [1, 2],
        "use_llm": True,
        "allow_auto_apply": False,
    })

    assert server.init_kwargs["mode"] == "suggest"
    assert server.init_kwargs["channels"] == [1, 2]
    assert server.init_kwargs["use_llm"] is True
    assert server.init_kwargs["allow_auto_apply"] is False
    assert server.init_kwargs["start"] is True
    _, payload = server.sent_messages[-1]
    assert payload["type"] == "agent_started"
    assert payload["is_running"] is True


@pytest.mark.asyncio
async def test_force_auto_apply_is_rejected_before_agent_initialization():
    server = DummyServer()
    handlers = register_handlers(server)

    await handlers["start_agent"]("ws", {"force_auto_apply": True})

    assert server.mixing_agent is None
    _, payload = server.sent_messages[-1]
    assert payload["status"] == "blocked"
    assert payload["reason"] == "force_auto_apply_removed"


@pytest.mark.asyncio
async def test_approve_action_returns_updated_pending_actions():
    server = DummyServer()
    server.mixing_agent = FakeAgent()
    handlers = register_handlers(server)

    await handlers["approve_action"]("ws", {"index": 0})

    sent_payloads = [payload for _, payload in server.sent_messages]
    assert sent_payloads[-2]["type"] == "action_approved"
    assert sent_payloads[-2]["success"] is True
    assert sent_payloads[-1] == {"type": "pending_actions", "actions": []}


@pytest.mark.asyncio
async def test_emergency_stop_agent_stops_running_agent():
    server = DummyServer()
    server.mixing_agent = FakeAgent()
    server.mixing_agent.state.is_running = True
    handlers = register_handlers(server)

    await handlers["emergency_stop_agent"]("ws", {})

    _, payload = server.sent_messages[-1]
    assert payload["type"] == "agent_emergency_stopped"
    assert payload["is_running"] is False
    assert payload["emergency_stop"] is True
    assert payload["success"] is True
    assert payload["status"] == "stopped"
    assert payload["teardown_complete"] is True
    assert server.mixing_agent.emergency_stopped is True


@pytest.mark.asyncio
async def test_emergency_stop_agent_propagates_incomplete_system_teardown():
    server = DummyServer()
    server.mixing_agent = FakeAgent()
    server.mixing_agent.state.is_running = True
    server._manual_write_emergency_stopped = True

    async def incomplete(_reason):
        return {
            "status": "stop_incomplete",
            "success": False,
            "teardown_complete": False,
            "teardown_errors": [{"step": "engine", "error": "RuntimeError"}],
            "generation": 2,
        }

    server.activate_system_emergency_stop = incomplete
    handlers = register_handlers(server)

    await handlers["emergency_stop_agent"]("ws", {})

    _, payload = server.sent_messages[-1]
    assert payload["status"] == "stop_incomplete"
    assert payload["success"] is False
    assert payload["teardown_complete"] is False
    assert payload["emergency_stop"] is True
    assert payload["is_running"] is True
    assert payload["teardown_errors"] == [
        {"step": "engine", "error": "RuntimeError"}
    ]


@pytest.mark.asyncio
async def test_tokenless_emergency_stop_agent_returns_minimal_acknowledgement():
    server = DummyServer()
    server._ws_auth_required = False
    server._manual_write_emergency_stopped = True

    async def detailed_stop(_reason):
        return {
            "status": "stop_incomplete",
            "success": False,
            "teardown_complete": False,
            "teardown_errors": [{"step": "secret-path", "error": "OSError"}],
            "generation": 7,
        }

    server.activate_system_emergency_stop = detailed_stop
    handlers = register_handlers(server)

    await handlers["emergency_stop_agent"]("ws", {})

    assert server.sent_messages[-1][1] == {
        "type": "agent_emergency_stopped",
        "status": "accepted",
    }
