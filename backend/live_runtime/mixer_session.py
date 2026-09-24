"""Canonical live mixer discovery and physical connection ownership.

This module extracts only transport lifecycle from the legacy
``AutoSoundcheckEngine``. It owns target resolution, connection creation and
teardown, but contains no channel classification, musical policy, AutoFOH
heuristics, safety decisions or mixer writes beyond the client's own connect /
disconnect lifecycle.

The canonical LIVE runtime fails closed when mixer identity is ambiguous. It
never silently defaults an unknown target to dLive and it rejects discovery
results that contradict an explicitly requested mixer type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mixer_discovery import (
    DLIVE_TCP_PORT,
    DLIVE_TLS_PORT,
    WING_OSC_PORT,
    DiscoveredMixer,
    discover_mixer_auto,
)


class LiveMixerSessionError(RuntimeError):
    """Fail-closed mixer discovery/connection lifecycle error."""


def _normalize_mixer_type(value: str | None) -> str | None:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "": None,
        "wing": "wing",
        "wing_rack": "wing",
        "behringer_wing": "wing",
        "dlive": "dlive",
        "d_live": "dlive",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported mixer type: {value!r}")
    return aliases[normalized]


def _default_port(mixer_type: str, tls: bool) -> int:
    if mixer_type == "wing":
        return WING_OSC_PORT
    if mixer_type == "dlive":
        return DLIVE_TLS_PORT if tls else DLIVE_TCP_PORT
    raise ValueError(f"Unsupported mixer type: {mixer_type!r}")


@dataclass(frozen=True)
class LiveMixerConfig:
    """Explicit control-plane connection contract for one live session."""

    mixer_type: str | None = None
    mixer_ip: str | None = None
    mixer_port: int | None = None
    mixer_tls: bool = False
    midi_base_channel: int = 0
    auto_discover: bool = True
    scan_subnet: bool = False
    discovery_timeout_s: float = 2.0
    connect_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        mixer_type = _normalize_mixer_type(self.mixer_type)
        mixer_ip = str(self.mixer_ip or "").strip() or None
        port = self.mixer_port
        if port is not None:
            if isinstance(port, bool) or not (1 <= int(port) <= 65535):
                raise ValueError("mixer_port must be within 1..65535")
            port = int(port)
        if isinstance(self.midi_base_channel, bool):
            raise TypeError("midi_base_channel must be an integer")
        midi_base_channel = int(self.midi_base_channel)
        if midi_base_channel < 0 or midi_base_channel > 15:
            raise ValueError("midi_base_channel must be within 0..15")
        if isinstance(self.discovery_timeout_s, bool) or float(self.discovery_timeout_s) <= 0:
            raise ValueError("discovery_timeout_s must be > 0")
        if isinstance(self.connect_timeout_s, bool) or float(self.connect_timeout_s) <= 0:
            raise ValueError("connect_timeout_s must be > 0")

        object.__setattr__(self, "mixer_type", mixer_type)
        object.__setattr__(self, "mixer_ip", mixer_ip)
        object.__setattr__(self, "mixer_port", port)
        object.__setattr__(self, "mixer_tls", bool(self.mixer_tls))
        object.__setattr__(self, "midi_base_channel", midi_base_channel)
        object.__setattr__(self, "auto_discover", bool(self.auto_discover))
        object.__setattr__(self, "scan_subnet", bool(self.scan_subnet))
        object.__setattr__(self, "discovery_timeout_s", float(self.discovery_timeout_s))
        object.__setattr__(self, "connect_timeout_s", float(self.connect_timeout_s))


@dataclass(frozen=True)
class LiveMixerTarget:
    mixer_type: str
    ip: str
    port: int
    tls: bool = False
    name: str = ""
    discovery_method: str = "explicit"


@dataclass(frozen=True)
class LiveMixerStatus:
    connected: bool
    mixer_type: str | None
    ip: str | None
    port: int | None
    tls: bool
    discovery_method: str | None


def _default_client_factory(target: LiveMixerTarget, midi_base_channel: int) -> Any:
    if target.mixer_type == "wing":
        from wing_client import WingClient

        return WingClient(ip=target.ip, port=target.port)
    if target.mixer_type == "dlive":
        from dlive_client import DLiveClient

        return DLiveClient(
            ip=target.ip,
            port=target.port,
            tls=target.tls,
            midi_base_channel=midi_base_channel,
        )
    raise LiveMixerSessionError(f"Unsupported mixer type: {target.mixer_type!r}")


class LiveMixerSession:
    """Own one physical mixer client from target resolution through teardown.

    Discovery and client construction are injectable so the ownership contract
    can be tested without network or console hardware. ``start()`` is
    idempotent while connected; ``stop()`` disconnects the owned client at most
    once.
    """

    def __init__(
        self,
        config: LiveMixerConfig,
        *,
        discover: Callable[..., DiscoveredMixer | None] = discover_mixer_auto,
        client_factory: Callable[[LiveMixerTarget, int], Any] = _default_client_factory,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not isinstance(config, LiveMixerConfig):
            raise TypeError("config must be LiveMixerConfig")
        self.config = config
        self._discover = discover
        self._client_factory = client_factory
        self._audit_sink = audit_sink
        self._client: Any | None = None
        self._target: LiveMixerTarget | None = None
        self._connected = False

    @property
    def client(self) -> Any | None:
        return self._client

    @property
    def target(self) -> LiveMixerTarget | None:
        return self._target

    @property
    def connected(self) -> bool:
        return self._connected

    def _audit(self, event: str, **payload: Any) -> None:
        if self._audit_sink is not None:
            self._audit_sink({"event": event, **payload})

    def _target_from_explicit(self) -> LiveMixerTarget | None:
        if self.config.mixer_type is None or self.config.mixer_ip is None:
            return None
        mixer_type = self.config.mixer_type
        tls = bool(self.config.mixer_tls) if mixer_type == "dlive" else False
        port = self.config.mixer_port or _default_port(mixer_type, tls)
        return LiveMixerTarget(
            mixer_type=mixer_type,
            ip=self.config.mixer_ip,
            port=port,
            tls=tls,
            discovery_method="explicit",
        )

    def resolve_target(self) -> LiveMixerTarget:
        """Resolve an explicit target first, otherwise use validated discovery."""
        explicit = self._target_from_explicit()
        if explicit is not None:
            return explicit
        if not self.config.auto_discover:
            raise LiveMixerSessionError(
                "Mixer target is incomplete and auto_discover is disabled"
            )

        discovered = self._discover(
            preferred_type=self.config.mixer_type,
            preferred_ip=self.config.mixer_ip,
            scan_subnet=self.config.scan_subnet,
            timeout=self.config.discovery_timeout_s,
        )
        if discovered is None:
            raise LiveMixerSessionError("No mixer discovered for live session")

        discovered_type = _normalize_mixer_type(discovered.mixer_type)
        if discovered_type is None:
            raise LiveMixerSessionError("Discovered mixer has no supported mixer type")
        if (
            self.config.mixer_type is not None
            and discovered_type != self.config.mixer_type
        ):
            raise LiveMixerSessionError(
                "Discovered mixer type contradicts explicit mixer_type: "
                f"expected {self.config.mixer_type}, got {discovered_type}"
            )

        ip = str(discovered.ip or "").strip()
        if not ip:
            raise LiveMixerSessionError("Discovered mixer has no IP address")
        tls = bool(self.config.mixer_tls or getattr(discovered, "tls", False))
        if discovered_type == "wing":
            tls = False

        if self.config.mixer_port is not None:
            port = self.config.mixer_port
        elif self.config.mixer_tls and discovered_type == "dlive":
            port = _default_port(discovered_type, True)
        else:
            discovered_port = int(getattr(discovered, "port", 0) or 0)
            port = discovered_port or _default_port(discovered_type, tls)

        return LiveMixerTarget(
            mixer_type=discovered_type,
            ip=ip,
            port=port,
            tls=tls,
            name=str(getattr(discovered, "name", "") or ""),
            discovery_method=str(
                getattr(discovered, "discovery_method", "") or "discovery"
            ),
        )

    def start(self) -> Any:
        """Resolve and connect one physical client, failing closed on ambiguity."""
        if self._connected and self._client is not None:
            return self._client

        client: Any | None = None
        target: LiveMixerTarget | None = None
        try:
            target = self.resolve_target()
            client = self._client_factory(target, self.config.midi_base_channel)
            if client is None:
                raise LiveMixerSessionError("Mixer client factory returned None")
            result = client.connect(timeout=self.config.connect_timeout_s)
            if result is False:
                raise LiveMixerSessionError(
                    f"Mixer connect returned false for {target.mixer_type}@{target.ip}:{target.port}"
                )
            if getattr(client, "is_connected", True) is False:
                raise LiveMixerSessionError(
                    f"Mixer client is not connected after connect() for {target.ip}"
                )

            self._target = target
            self._client = client
            self._connected = True
            self._audit(
                "live_mixer_connected",
                mixer_type=target.mixer_type,
                ip=target.ip,
                port=target.port,
                tls=target.tls,
                discovery_method=target.discovery_method,
            )
            return client
        except Exception as exc:
            if client is not None:
                try:
                    client.disconnect()
                except Exception:
                    pass
            self._client = None
            self._target = target
            self._connected = False
            self._audit(
                "live_mixer_connect_failed",
                reason=f"{type(exc).__name__}: {exc}",
            )
            if isinstance(exc, LiveMixerSessionError):
                raise
            raise LiveMixerSessionError(
                f"Live mixer startup failed: {type(exc).__name__}: {exc}"
            ) from exc

    def stop(self) -> bool:
        """Disconnect and release the owned client; safe to call repeatedly."""
        client = self._client
        if client is None:
            self._connected = False
            return False

        target = self._target
        self._client = None
        self._connected = False
        try:
            client.disconnect()
        except Exception as exc:
            self._audit(
                "live_mixer_disconnect_failed",
                reason=f"{type(exc).__name__}: {exc}",
            )
            raise LiveMixerSessionError(
                f"Live mixer stop failed: {type(exc).__name__}: {exc}"
            ) from exc

        self._audit(
            "live_mixer_disconnected",
            mixer_type=target.mixer_type if target else None,
            ip=target.ip if target else None,
            port=target.port if target else None,
        )
        return True

    def status(self) -> LiveMixerStatus:
        target = self._target
        return LiveMixerStatus(
            connected=self._connected,
            mixer_type=target.mixer_type if target else None,
            ip=target.ip if target else None,
            port=target.port if target else None,
            tls=bool(target.tls) if target else False,
            discovery_method=target.discovery_method if target else None,
        )
