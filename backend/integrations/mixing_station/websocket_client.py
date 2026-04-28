"""WebSocket transport for Mixing Station Desktop JSON API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import websockets

from .models import MixingStationCommand
from .rest_client import TransportResult


@dataclass(frozen=True)
class WebSocketRequest:
    """JSON request shape accepted by the adapter."""

    path: str
    method: str = "SET"
    body: Any = None

    def to_dict(self) -> dict:
        return {"path": self.path, "method": self.method, "body": self.body}


class MixingStationWebSocketClient:
    """Short-lived WebSocket sender for commands and subscriptions."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        *,
        path: str = "/ws",
        timeout: float = 2.0,
        command_path_template: str = "",
    ):
        self.host = host
        self.port = int(port)
        self.path = path if path.startswith("/") else f"/{path}"
        self.timeout = float(timeout)
        self.command_path_template = command_path_template

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}{self.path}"

    def send_command(self, command: MixingStationCommand) -> TransportResult:
        if not self.command_path_template:
            return TransportResult(
                success=False,
                error=(
                    "Mixing Station WebSocket command path is not configured. "
                    "Use the local API Explorer to discover it, then set "
                    "websocket_command_path_template in config/mixing_station.yaml."
                ),
            )
        request = WebSocketRequest(
            path=self.command_path_template.format(data_path=command.data_path),
            method=command.method,
            body={"value": command.value, "format": command.value_format},
        )
        return asyncio.run(self.send_request(request))

    async def send_request(self, request: WebSocketRequest) -> TransportResult:
        try:
            async with websockets.connect(
                self.url,
                open_timeout=self.timeout,
                close_timeout=self.timeout,
            ) as websocket:
                await websocket.send(_json_dumps(request.to_dict()))
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=self.timeout)
                except asyncio.TimeoutError:
                    response = None
            return TransportResult(success=True, data=response)
        except Exception as exc:
            return TransportResult(success=False, error=str(exc))

    def subscribe_to_parameter(self, path: str) -> TransportResult:
        return self._unsupported_subscription(path, "parameter")

    def subscribe_to_metering(self, path: str) -> TransportResult:
        return self._unsupported_subscription(path, "metering")

    @staticmethod
    def _unsupported_subscription(path: str, kind: str) -> TransportResult:
        return TransportResult(
            success=False,
            error=(
                f"Mixing Station WebSocket {kind} subscription for {path!r} "
                "requires API Explorer endpoint confirmation."
            ),
        )


def _json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=True)
