"""Тесты WS-поверхности голосового помощника (Ф10, контракт CLD-0031)."""

import asyncio

import pytest

from handlers.voice_assistant_handlers import register_handlers


class FakeServer:
    def __init__(self):
        self.sent = []
        self.broadcasts = []
        self.auto_soundcheck_engine = None

    async def send_to_client(self, websocket, payload):
        self.sent.append(payload)

    async def broadcast(self, payload):
        self.broadcasts.append(payload)


class FakeEngine:
    def __init__(self):
        self.voice_assistant = None


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def setup():
    server = FakeServer()
    handlers = register_handlers(server)
    return server, handlers


class TestStatus:
    def test_status_schema_v1(self, setup):
        server, handlers = setup
        run(handlers["get_voice_assistant_status"](None, {}))
        payload = server.sent[-1]
        assert payload["type"] == "voice_assistant_status"
        assert payload["schema"] == "automixer.voice_assistant.v1"
        assert set(payload) == {
            "type", "schema", "lang", "enabled", "available",
            "spoken_log", "backend", "routing",
        }
        assert payload["routing"] == "local_only"
        assert payload["enabled"] is False  # выключен по умолчанию

    def test_default_language_ru(self, setup):
        server, handlers = setup
        run(handlers["get_voice_assistant_status"](None, {}))
        assert server.sent[-1]["lang"] == "ru"


class TestSetLanguage:
    def test_valid_language_changes_and_broadcasts(self, setup):
        server, handlers = setup
        run(handlers["set_voice_language"](None, {"lang": "en"}))
        assert server.sent[-1]["lang"] == "en"
        assert server.broadcasts and server.broadcasts[-1]["lang"] == "en"

    def test_invalid_language_ignored_no_broadcast(self, setup):
        server, handlers = setup
        run(handlers["set_voice_language"](None, {"lang": "de"}))
        assert server.sent[-1]["lang"] == "ru"
        assert server.broadcasts == []

    def test_same_language_no_broadcast(self, setup):
        server, handlers = setup
        run(handlers["set_voice_language"](None, {"lang": "ru"}))
        assert server.broadcasts == []


class TestSetEnabled:
    def test_enable_attaches_to_engine(self, setup):
        server, handlers = setup
        server.auto_soundcheck_engine = FakeEngine()
        run(handlers["set_voice_enabled"](None, {"enabled": True}))
        assert server.sent[-1]["enabled"] is True
        assert server.auto_soundcheck_engine.voice_assistant is server.voice_assistant

    def test_disable_detaches_from_engine(self, setup):
        server, handlers = setup
        server.auto_soundcheck_engine = FakeEngine()
        run(handlers["set_voice_enabled"](None, {"enabled": True}))
        run(handlers["set_voice_enabled"](None, {"enabled": False}))
        assert server.auto_soundcheck_engine.voice_assistant is None

    def test_non_bool_ignored(self, setup):
        server, handlers = setup
        run(handlers["set_voice_enabled"](None, {"enabled": "yes"}))
        assert server.sent[-1]["enabled"] is False
        assert server.broadcasts == []

    def test_enable_without_engine_is_safe(self, setup):
        server, handlers = setup
        run(handlers["set_voice_enabled"](None, {"enabled": True}))
        assert server.sent[-1]["enabled"] is True
