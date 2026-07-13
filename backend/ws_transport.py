"""Shared WebSocket transport helpers for the backend server."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import json
import logging
import math
import time
from typing import Any, Awaitable, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)


EMERGENCY_FAST_LANE_TYPES = frozenset(
    {
        "emergency_stop",
        "emergency_stop_agent",
    }
)


@dataclass(frozen=True)
class QueuedIngressMessage:
    """A normal-lane frame bound to the emergency epoch in which it arrived."""

    payload: str | bytes
    emergency_generation: int


class TokenBucket:
    """Small monotonic token bucket used by the post-auth ingress boundary."""

    def __init__(
        self,
        rate_per_second: float,
        capacity: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        rate_per_second = float(rate_per_second)
        capacity = float(capacity)
        if not math.isfinite(rate_per_second) or rate_per_second <= 0:
            raise ValueError("rate_per_second must be finite and positive")
        if not math.isfinite(capacity) or capacity < 1:
            raise ValueError("capacity must be finite and at least one")
        self.rate_per_second = rate_per_second
        self.capacity = capacity
        self._clock = clock
        self._tokens = capacity
        self._last_refill = float(clock())

    @property
    def tokens(self) -> float:
        return self._tokens

    def refill(self, now: float | None = None) -> None:
        timestamp = float(self._clock() if now is None else now)
        if not math.isfinite(timestamp):
            raise ValueError("token bucket time must be finite")
        elapsed = max(0.0, timestamp - self._last_refill)
        if elapsed:
            self._tokens = min(
                self.capacity,
                self._tokens + elapsed * self.rate_per_second,
            )
            self._last_refill = timestamp

    def can_consume(self, amount: float = 1.0) -> bool:
        return self._tokens >= float(amount)

    def debit(self, amount: float = 1.0) -> None:
        amount = float(amount)
        if amount <= 0 or not math.isfinite(amount):
            raise ValueError("token amount must be finite and positive")
        if self._tokens < amount:
            raise ValueError("insufficient tokens")
        self._tokens -= amount

    def consume(self, amount: float = 1.0, *, now: float | None = None) -> bool:
        self.refill(now)
        if not self.can_consume(amount):
            return False
        self.debit(amount)
        return True


def consume_token_buckets(
    buckets: Sequence[TokenBucket],
    *,
    amount: float = 1.0,
    now: float | None = None,
) -> bool:
    """Atomically debit every bucket, or debit none when one is exhausted."""

    if not buckets:
        return True
    timestamp = time.monotonic() if now is None else float(now)
    for bucket in buckets:
        bucket.refill(timestamp)
    if not all(bucket.can_consume(amount) for bucket in buckets):
        return False
    for bucket in buckets:
        bucket.debit(amount)
    return True


class RejectionWindow:
    """Track overload rejects and signal when a sustained threshold is reached."""

    def __init__(
        self,
        threshold: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(threshold, bool) or int(threshold) < 1:
            raise ValueError("threshold must be a positive integer")
        window_seconds = float(window_seconds)
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive")
        self.threshold = int(threshold)
        self.window_seconds = window_seconds
        self._clock = clock
        self._events: deque[float] = deque()

    @property
    def count(self) -> int:
        return len(self._events)

    def record(self, now: float | None = None) -> bool:
        timestamp = float(self._clock() if now is None else now)
        cutoff = timestamp - self.window_seconds
        while self._events and self._events[0] < cutoff:
            self._events.popleft()
        self._events.append(timestamp)
        return len(self._events) >= self.threshold


def websocket_message_type(message: str | bytes) -> str | None:
    """Read only the top-level type used for ingress routing; retain no payload."""

    try:
        data = json.loads(message)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    message_type = data.get("type")
    return message_type if isinstance(message_type, str) else None


def is_connection_closed_error(exc: Exception) -> bool:
    """Return True when the exception looks like a normal closed WS connection."""
    msg = str(exc).lower()
    if (
        "1000" in msg
        or "1001" in msg
        or "going away" in msg
        or "connection closed" in msg
        or "received 1000" in msg
        or "sent 1000" in msg
    ):
        return True
    try:
        return getattr(exc, "code", None) in {1000, 1001}
    except Exception:
        return False


def serialize_message(message: dict, converter: Callable[[Any], Any]) -> str:
    """Convert custom types and serialize a payload for WebSocket transport."""
    return json.dumps(converter(message))


async def send_json(
    websocket: Any,
    message: dict,
    *,
    converter: Callable[[Any], Any],
    logger_: logging.Logger | None = None,
) -> bool | None:
    """Serialize and send one JSON message, returning success/failure."""
    active_logger = logger_ or logger
    try:
        await websocket.send(serialize_message(message, converter))
        return True
    except Exception as exc:
        if is_connection_closed_error(exc):
            active_logger.debug("Send failed (client closed): %s", exc)
            return False
        else:
            active_logger.error("Error sending to client: %s", exc)
            return None


async def broadcast_json(
    clients: Iterable[Any],
    message: dict,
    *,
    sender: Callable[[Any, dict], Awaitable[bool]],
    logger_: logging.Logger | None = None,
) -> list[Any]:
    """Broadcast a message and return the clients that failed due to disconnect."""
    active_logger = logger_ or logger
    clients = list(clients)
    if not clients:
        active_logger.warning("No connected clients to broadcast to")
        return []

    results = await asyncio.gather(
        *[sender(client, message) for client in clients],
        return_exceptions=True,
    )

    disconnected = []
    for client, result in zip(clients, results):
        if isinstance(result, Exception):
            if is_connection_closed_error(result):
                disconnected.append(client)
            else:
                active_logger.error("Error broadcasting to client: %s", result)
            continue
        if result is False:
            disconnected.append(client)
    return disconnected
