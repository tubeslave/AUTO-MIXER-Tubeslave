"""Tests for backend/ws_transport.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ws_transport import (
    EMERGENCY_FAST_LANE_TYPES,
    RejectionWindow,
    TokenBucket,
    broadcast_json,
    consume_token_buckets,
    is_connection_closed_error,
    send_json,
    serialize_message,
    websocket_message_type,
)


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


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_token_bucket_enforces_burst_and_refills_at_configured_rate():
    clock = FakeClock()
    bucket = TokenBucket(10, 30, clock=clock)

    assert all(bucket.consume() for _ in range(30))
    assert bucket.consume() is False

    clock.advance(0.1)
    assert bucket.consume() is True
    assert bucket.consume() is False


def test_multi_bucket_consume_is_atomic_when_global_bucket_is_empty():
    clock = FakeClock()
    connection = TokenBucket(10, 30, clock=clock)
    global_bucket = TokenBucket(50, 100, clock=clock)
    for _ in range(100):
        assert global_bucket.consume()

    before = connection.tokens
    assert consume_token_buckets(
        (connection, global_bucket),
        now=clock.now,
    ) is False
    assert connection.tokens == before


def test_rejection_window_closes_on_twentieth_reject_within_five_seconds():
    clock = FakeClock()
    window = RejectionWindow(20, 5.0, clock=clock)

    assert all(window.record() is False for _ in range(19))
    assert window.record() is True

    clock.advance(5.1)
    assert window.record() is False
    assert window.count == 1


@pytest.mark.parametrize("message_type", sorted(EMERGENCY_FAST_LANE_TYPES))
def test_only_exact_emergency_message_types_enter_fast_lane(message_type):
    assert websocket_message_type(f'{{"type":"{message_type}"}}') == message_type
    assert message_type in EMERGENCY_FAST_LANE_TYPES
    assert websocket_message_type(f'{{"type":"{message_type}_later"}}') not in EMERGENCY_FAST_LANE_TYPES


def test_malformed_or_non_object_frames_have_no_fast_lane_type():
    assert websocket_message_type("not-json") is None
    assert websocket_message_type("[]") is None
    assert websocket_message_type('{"type": 123}') is None


def test_is_connection_closed_error_detects_normal_close():
    assert is_connection_closed_error(FakeClosedError()) is True
    assert is_connection_closed_error(FakeClosedError("received 1000 (OK); then sent 1000 (OK)", code=1000)) is True
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
async def test_send_json_returns_none_for_non_disconnect_errors():
    ws = FakeWebSocket(fail_with=ValueError("boom"))
    ok = await send_json(ws, {"x": 1}, converter=lambda data: data)
    assert ok is None


@pytest.mark.asyncio
async def test_broadcast_json_returns_disconnected_clients():
    alive = FakeWebSocket()
    dead = FakeWebSocket(fail_with=FakeClosedError())

    async def sender(client, message):
        return await send_json(client, message, converter=lambda data: data)

    disconnected = await broadcast_json([alive, dead], {"type": "ping"}, sender=sender)
    assert dead in disconnected
    assert alive.sent == ['{"type": "ping"}']


@pytest.mark.asyncio
async def test_broadcast_json_keeps_clients_on_non_disconnect_errors():
    alive = FakeWebSocket()
    flaky = FakeWebSocket(fail_with=ValueError("boom"))

    async def sender(client, message):
        return await send_json(client, message, converter=lambda data: data)

    disconnected = await broadcast_json([alive, flaky], {"type": "ping"}, sender=sender)
    assert disconnected == []
    assert alive.sent == ['{"type": "ping"}']
