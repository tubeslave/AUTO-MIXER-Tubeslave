"""Tests for backend/ws_transport.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ws_transport import broadcast_json, is_connection_closed_error, send_json, serialize_message


class FakeClosedError(Exception):
    def __init__(self, message="connection closed", code=1001):
        super().__init__(message)
        self.code = code


class FakeWebSocket:
    def __init__(self, fail_with=None):
        self.fail_with = fail_with
        self.sent = []

    async def send(self, payload):
        if self.fail_with is not None:
            raise self.fail_with
        self.sent.append(payload)


def test_is_connection_closed_error_detects_normal_close():
    assert is_connection_closed_error(FakeClosedError()) is True
    assert is_connection_closed_error(Exception("going away")) is True
    assert is_connection_closed_error(Exception("boom")) is False


def test_serialize_message_uses_converter():
    payload = serialize_message({"x": 1}, lambda data: {"wrapped": data["x"]})
    assert payload == '{"wrapped": 1}'


@pytest.mark.asyncio
async def test_send_json_success():
    ws = FakeWebSocket()
    ok = await send_json(ws, {"x": 1}, converter=lambda data: data)
    assert ok is True
    assert ws.sent == ['{"x": 1}']


@pytest.mark.asyncio
async def test_send_json_returns_false_for_closed_connection():
    ws = FakeWebSocket(fail_with=FakeClosedError())
    ok = await send_json(ws, {"x": 1}, converter=lambda data: data)
    assert ok is False


@pytest.mark.asyncio
async def test_broadcast_json_returns_disconnected_clients():
    alive = FakeWebSocket()
    dead = FakeWebSocket(fail_with=FakeClosedError())

    async def sender(client, message):
        return await send_json(client, message, converter=lambda data: data)

    disconnected = await broadcast_json([alive, dead], {"type": "ping"}, sender=sender)
    assert dead in disconnected
    assert alive.sent == ['{"type": "ping"}']
