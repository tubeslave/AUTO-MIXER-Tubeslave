"""Reusable replay/regression test support for transport-safe mixer tests."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
import re
import socket
from typing import Any


_OSC_CHANNEL_RE = re.compile(r"/ch/(\d+)")


def install_transport_blockers(monkeypatch, *, reason: str = "Real transport is blocked in replay tests") -> None:
    """Fail fast if a test accidentally reaches real socket/pythonosc transport."""

    def _blocked(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError(reason)

    monkeypatch.setattr(socket.socket, "sendto", _blocked)

    try:
        from pythonosc import udp_client
    except ImportError:
        return

    for owner, attr in (
        (udp_client.UDPClient, "send"),
        (udp_client.UDPClient, "send_message"),
        (udp_client.SimpleUDPClient, "send_message"),
    ):
        if hasattr(owner, attr):
            monkeypatch.setattr(owner, attr, _blocked)


@dataclass(frozen=True)
class RecordedTransportEvent:
    address: str
    values: list[Any]
    channel: int | None
    transport_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "values": list(self.values),
            "channel": self.channel,
            "transport_kind": self.transport_kind,
        }


class RecordingSender:
    """Replay sender that records intended OSC sends without touching the network."""

    def __init__(self) -> None:
        self.events: list[RecordedTransportEvent] = []

    def send(self, address: str, *values: Any) -> None:
        self.events.append(
            RecordedTransportEvent(
                address=str(address),
                values=list(values),
                channel=_channel_from_address(address),
                transport_kind="mutation" if values else "query",
            )
        )

    def incident_records(self) -> list[dict[str, Any]]:
        return [
            {
                "event_index": index,
                **event.to_dict(),
            }
            for index, event in enumerate(self.events, start=1)
        ]


class ReplayMixer:
    """Fake mixer that records mutation-capable calls for replay/inventory tests."""

    is_connected = True

    def __init__(
        self,
        *,
        scenario: str,
        channel_names: dict[int, str] | None = None,
        fader_result: bool = True,
        gain_result: bool = True,
        last_send_status: str = "sent",
        verify_result: dict[str, Any] | None = None,
    ):
        self.scenario = scenario
        self.channel_names = dict(channel_names or {})
        self.fader_result = bool(fader_result)
        self.gain_result = bool(gain_result)
        self.last_send_status = str(last_send_status)
        self.verify_result = dict(verify_result) if verify_result is not None else None
        self.fader_values: dict[int, float] = {}
        self.gain_values: dict[int, float] = {}
        self.event_log: list[dict[str, Any]] = []
        self.verify_calls: list[tuple[str, int, float, int]] = []
        self._seq = 0

    def get_channel_fader(self, channel: int) -> float:
        return float(self.fader_values.get(int(channel), -5.0))

    def get_channel_gain(self, channel: int) -> float:
        return float(self.gain_values.get(int(channel), 0.0))

    def get_channel_name(self, channel: int) -> str:
        return self.channel_names.get(int(channel), f"Ch {int(channel)}")

    def get_all_channel_names(self, _max_channels: int) -> dict[int, str]:
        return dict(self.channel_names)

    def set_channel_fader(self, channel: int, value: float) -> bool:
        self.fader_values[int(channel)] = float(value)
        self._record(
            "set_channel_fader",
            channel=channel,
            value=value,
            params={"value": float(value)},
            transport_kind="mutation",
        )
        return self.fader_result

    def set_channel_gain(self, channel: int, value: float) -> bool:
        self.gain_values[int(channel)] = float(value)
        self._record(
            "set_channel_gain",
            channel=channel,
            value=value,
            params={"value": float(value)},
            transport_kind="mutation",
        )
        return self.gain_result

    def set_fader(self, channel: int, value: float) -> bool:
        self.fader_values[int(channel)] = float(value)
        self._record(
            "set_fader",
            channel=channel,
            value=value,
            params={"value": float(value)},
            transport_kind="mutation",
        )
        return True

    def set_gain(self, channel: int, value: float) -> bool:
        self.gain_values[int(channel)] = float(value)
        self._record(
            "set_gain",
            channel=channel,
            value=value,
            params={"value": float(value)},
            transport_kind="mutation",
        )
        return self.gain_result

    def set_eq_band(
        self,
        channel: int,
        band: int,
        freq: float,
        gain: float,
        q: float,
    ) -> bool:
        self._record(
            "set_eq_band",
            channel=channel,
            value=gain,
            params={
                "band": int(band),
                "freq": float(freq),
                "gain": float(gain),
                "q": float(q),
            },
            transport_kind="mutation",
        )
        return True

    def reset_channel_processing(self, channel: int) -> bool:
        self._record(
            "reset_channel_processing",
            channel=channel,
            value=None,
            params={},
            transport_kind="mutation",
        )
        return True

    def send(self, address: str, *values: Any) -> bool:
        transport_kind = "mutation" if values else "query"
        value = values[0] if len(values) == 1 else list(values) if values else None
        self._record(
            "send",
            channel=_channel_from_address(address),
            value=value,
            params={"address": address, "values": list(values)},
            transport_kind=transport_kind,
        )
        return True

    def get_last_send_status(self) -> str:
        return self.last_send_status

    def confirm_manual_write(self, operation: str, channel: int, desired_value: float, timeout_ms: int = 300) -> dict[str, Any]:
        self.verify_calls.append((str(operation), int(channel), float(desired_value), int(timeout_ms)))
        if self.verify_result is not None:
            return dict(self.verify_result)
        return {
            "verification_id": f"verify-{int(channel)}",
            "readback_status": "confirmed",
            "confirmed_value": float(desired_value),
            "failure_reason": None,
            "timeout_ms": int(timeout_ms),
        }

    def mutation_events(self) -> list[dict[str, Any]]:
        return [event for event in self.event_log if event["transport_kind"] == "mutation"]

    def incident_records(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self.event_log]

    def _record(
        self,
        method: str,
        *,
        channel: int | None,
        value: Any,
        params: dict[str, Any],
        transport_kind: str,
    ) -> None:
        self._seq += 1
        source_file, source_function = _source_location()
        self.event_log.append(
            {
                "seq": self._seq,
                "method": method,
                "channel": None if channel is None else int(channel),
                "value": value,
                "params": params,
                "transport_kind": transport_kind,
                "source_file": source_file,
                "source_function": source_function,
                "context": {"scenario": self.scenario},
            }
        )


def _channel_from_address(address: str) -> int | None:
    match = _OSC_CHANNEL_RE.search(address or "")
    return int(match.group(1)) if match else None


def _source_location() -> tuple[str | None, str | None]:
    interesting = {
        "controller.py",
        "auto_fader.py",
        "auto_soundcheck_engine.py",
    }
    for frame_info in inspect.stack()[2:]:
        filename = Path(frame_info.filename).name
        if filename in interesting:
            return frame_info.filename, frame_info.function
    return None, None
