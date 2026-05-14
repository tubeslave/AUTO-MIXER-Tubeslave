"""Fail-closed shadow transport interception for mixer mutation writes."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
import time
from typing import Any, Callable
import uuid


logger = logging.getLogger(__name__)

_OSC_CHANNEL_RE = re.compile(r"/ch/(\d+)")


def _channel_from_address(address: str) -> int | None:
    match = _OSC_CHANNEL_RE.search(address or "")
    return int(match.group(1)) if match else None


@dataclass(slots=True)
class ShadowTransportContext:
    """Request-scoped metadata attached to the next shadow-blocked mutation."""

    audit_id: str = ""
    replay_correlation_id: str = ""
    source: str = ""
    operation: str = ""
    channel: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_event(self, *, client_name: str, address: str, values: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "type": "shadow_would_send",
            "timestamp": time.time(),
            "audit_id": self.audit_id or str(uuid.uuid4()),
            "replay_correlation_id": self.replay_correlation_id,
            "source": self.source,
            "operation": self.operation,
            "channel": self.channel if self.channel is not None else _channel_from_address(address),
            "address": str(address),
            "values": list(values),
            "transport_kind": "mutation",
            "blocked_reason": "shadow_mode_active",
            "send_status": "shadow_blocked",
            "dry_run_only": True,
            "client_name": client_name,
            "metadata": dict(self.metadata),
        }


class ShadowTransportInterceptor:
    """Block mutation writes while preserving read/query traffic."""

    def __init__(
        self,
        *,
        client_name: str,
        enabled: bool = False,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.client_name = client_name
        self.enabled = bool(enabled)
        self.event_sink = event_sink or self._log_event
        self._context = ShadowTransportContext()
        self._last_event: dict[str, Any] | None = None

    def configure(
        self,
        *,
        enabled: bool | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if event_sink is not None:
            self.event_sink = event_sink

    def set_context(
        self,
        *,
        audit_id: str = "",
        replay_correlation_id: str = "",
        source: str = "",
        operation: str = "",
        channel: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._context = ShadowTransportContext(
            audit_id=str(audit_id or ""),
            replay_correlation_id=str(replay_correlation_id or ""),
            source=str(source or ""),
            operation=str(operation or ""),
            channel=None if channel is None else int(channel),
            metadata=dict(metadata or {}),
        )

    def clear_context(self) -> None:
        self._context = ShadowTransportContext()

    def consume_last_event(self) -> dict[str, Any] | None:
        event = self._last_event
        self._last_event = None
        return event

    def intercept(self, address: str, values: tuple[Any, ...]) -> dict[str, Any] | None:
        if not self.enabled or not values:
            return None
        event = self._context.to_event(
            client_name=self.client_name,
            address=address,
            values=values,
        )
        self._last_event = event
        try:
            self.event_sink(event)
        except Exception:
            logger.exception("shadow transport sink failed")
        return event

    @staticmethod
    def _log_event(event: dict[str, Any]) -> None:
        logger.info("shadow_transport %s", event)
