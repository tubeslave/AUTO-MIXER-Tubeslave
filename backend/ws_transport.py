"""Shared WebSocket transport helpers for the backend server."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Iterable

logger = logging.getLogger(__name__)


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
