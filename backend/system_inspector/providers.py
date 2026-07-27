"""Read-only provider contracts.

Only snapshot-backed data is supported in Phase 0.  The async shape is kept so
future live collectors can be rate-limited without changing detector APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .model import AudioEndpoint, Device, Subscription, SystemGraph


@dataclass(frozen=True)
class ClockSnapshot:
    status: str
    sample_rate: int | None
    master_device_id: str
    confidence: float
    limitation: str


@dataclass(frozen=True)
class DanteHealthSnapshot:
    status: str
    confidence: float
    limitation: str


class DanteTopologyProvider(ABC):
    @abstractmethod
    async def get_devices(self) -> list[Device]:
        raise NotImplementedError

    @abstractmethod
    async def get_channels(self) -> list[AudioEndpoint]:
        raise NotImplementedError

    @abstractmethod
    async def get_subscriptions(self) -> list[Subscription]:
        raise NotImplementedError

    @abstractmethod
    async def get_clock_status(self) -> ClockSnapshot:
        raise NotImplementedError

    @abstractmethod
    async def get_health(self) -> DanteHealthSnapshot:
        raise NotImplementedError


class DanteSnapshotFileProvider(DanteTopologyProvider):
    """Expose normalized imported data without contacting a Dante network."""

    def __init__(self, graph: SystemGraph, raw_snapshot: Mapping[str, Any]):
        self._devices: Tuple[Device, ...] = tuple(
            device
            for device in graph.devices
            if "dante" in device.kind.lower()
            or "dante" in device.metadata.protocol.lower()
        )
        self._channels = tuple(graph.audio_endpoints)
        self._subscriptions = tuple(graph.subscriptions)
        clock = raw_snapshot.get("clock", {})
        health = raw_snapshot.get("health", {})
        self._clock = ClockSnapshot(
            status=str(clock.get("status", "UNKNOWN"))
            if isinstance(clock, dict)
            else "UNKNOWN",
            sample_rate=_optional_int(clock.get("sample_rate"))
            if isinstance(clock, dict)
            else None,
            master_device_id=str(clock.get("master_device_id", ""))
            if isinstance(clock, dict)
            else "",
            confidence=_confidence(clock),
            limitation=str(clock.get("limitation", "Imported snapshot only."))
            if isinstance(clock, dict)
            else "Imported snapshot only.",
        )
        self._health = DanteHealthSnapshot(
            status=str(health.get("status", "UNKNOWN"))
            if isinstance(health, dict)
            else "UNKNOWN",
            confidence=_confidence(health),
            limitation=str(
                health.get(
                    "limitation",
                    "Imported snapshot cannot prove live Dante health.",
                )
            )
            if isinstance(health, dict)
            else "Imported snapshot cannot prove live Dante health.",
        )

    async def get_devices(self) -> list[Device]:
        return list(self._devices)

    async def get_channels(self) -> list[AudioEndpoint]:
        return list(self._channels)

    async def get_subscriptions(self) -> list[Subscription]:
        return list(self._subscriptions)

    async def get_clock_status(self) -> ClockSnapshot:
        return self._clock

    async def get_health(self) -> DanteHealthSnapshot:
        return self._health


class DanteControllerImportProvider(DanteTopologyProvider):
    """Reserved contract; Controller XML/CSV parsing is not implemented."""

    async def _unavailable(self):
        raise NotImplementedError("dante_controller_import_provider_unavailable")

    get_devices = get_channels = get_subscriptions = get_clock_status = get_health = _unavailable


class DanteManualTopologyProvider(DanteTopologyProvider):
    """Reserved contract; manual topology workflow is not implemented."""

    async def _unavailable(self):
        raise NotImplementedError("dante_manual_topology_provider_unavailable")

    get_devices = get_channels = get_subscriptions = get_clock_status = get_health = _unavailable


class DanteOfficialApiProvider(DanteTopologyProvider):
    """Fail-closed placeholder until an official licensed API is configured."""

    async def _unavailable(self):
        raise NotImplementedError("dante_official_api_provider_unavailable")

    get_devices = get_channels = get_subscriptions = get_clock_status = get_health = _unavailable


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _confidence(value: Any) -> float:
    if not isinstance(value, dict):
        return 0.0
    raw = value.get("confidence", 0.0)
    return (
        float(raw)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool)
        else 0.0
    )

