"""OSC fallback transport for Mixing Station Desktop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import MixingStationCommand
from .rest_client import TransportResult


@dataclass(frozen=True)
class OSCMessagePreview:
    """OSC address/value pair generated for tests and dry-run logs."""

    address: str
    value: object


class MixingStationOSCClient:
    """OSC sender using Mixing Station's conceptual `/con/*/{dataPath}` API."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.host = host
        self.port = int(port)

    def build_message(self, command: MixingStationCommand) -> OSCMessagePreview:
        value_kind = "n" if command.value_format == "normalized" else "v"
        data_path = command.data_path.strip("/")
        return OSCMessagePreview(address=f"/con/{value_kind}/{data_path}", value=command.value)

    def send_command(self, command: MixingStationCommand) -> TransportResult:
        message = self.build_message(command)
        try:
            from pythonosc.udp_client import SimpleUDPClient
        except ImportError as exc:
            return TransportResult(success=False, error=f"python-osc unavailable: {exc}")

        try:
            client = SimpleUDPClient(self.host, self.port)
            client.send_message(message.address, message.value)
            return TransportResult(success=True, data={"address": message.address})
        except OSError as exc:
            return TransportResult(success=False, error=str(exc))

    def subscribe_to_parameter(self, path: str) -> TransportResult:
        return _unsupported_subscription(path, "parameter")

    def subscribe_to_metering(self, path: str) -> TransportResult:
        return _unsupported_subscription(path, "metering")


def _unsupported_subscription(path: str, kind: str) -> TransportResult:
    return TransportResult(
        success=False,
        error=f"OSC {kind} subscription for {path!r} is not implemented in the fallback client.",
    )
