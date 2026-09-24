"""Temporary read-only seam from live_runtime-owned mixer to legacy soundcheck code.

The canonical owner of physical mixer discovery, connect and disconnect is
``LiveMixerSession``. This seam exists only while legacy
``AutoSoundcheckEngine`` still provides compatibility plumbing around that
transport. It deliberately does not grant legacy decision code write access,
including during BENCH_TEST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

AuditSink = Optional[Callable[[Dict[str, Any]], None]]


class LegacyMixerSeamError(RuntimeError):
    """Raised when external mixer ownership cannot be proven safely."""


@dataclass(frozen=True)
class LegacyMixerSeamStatus:
    bound: bool
    engine_uses_proxy: bool
    physical_client_id: int
    mixer_type: str
    ip: str
    port: int


@dataclass(frozen=True)
class _AttrSnapshot:
    had_instance_value: bool
    value: Any = None


class _ReadOnlyMixerProxy:
    """Expose only read-style mixer APIs while blocking lifecycle and writes."""

    _READ_PREFIXES: Tuple[str, ...] = ("get_", "read_", "query_")
    _SAFE_ATTRIBUTES = frozenset({"is_connected", "ip", "port", "tls"})

    def __init__(self, client: Any, audit_sink: AuditSink = None):
        self._client = client
        self._audit_sink = audit_sink

    def _audit(self, event: str, **payload: Any) -> None:
        if self._audit_sink is None:
            return
        self._audit_sink({"event": event, **payload})

    def _blocked(self, name: str) -> None:
        self._audit("legacy_external_mixer_access_blocked", operation=name)
        raise LegacyMixerSeamError(
            f"Legacy mixer transport is read-only; blocked operation: {name}"
        )

    def connect(self, *args: Any, **kwargs: Any) -> None:
        self._blocked("connect")

    def disconnect(self, *args: Any, **kwargs: Any) -> None:
        self._blocked("disconnect")

    def __getattr__(self, name: str) -> Any:
        if name in self._SAFE_ATTRIBUTES:
            return getattr(self._client, name)
        if name.startswith(self._READ_PREFIXES):
            value = getattr(self._client, name)
            if callable(value):
                return value
            return value
        self._blocked(name)


class LegacyExternalMixerSeam:
    """Bind a live-runtime owned mixer to legacy code without ownership transfer.

    The physical client remains owned by ``LiveMixerSession``. Legacy mixer
    discovery and connection methods are replaced with proof-only bypasses and
    both legacy client slots receive a read-only proxy. This prevents a second
    connection and prevents heuristic legacy code from gaining a mutation path.
    """

    _SNAPSHOT_ATTRS = (
        "_discover_mixer",
        "_connect_mixer",
        "mixer_client",
        "_real_mixer_client",
        "mixer_type",
        "mixer_ip",
        "mixer_port",
    )

    def __init__(self, engine: Any, mixer_session: Any, audit_sink: AuditSink = None):
        self._engine = engine
        self._session = mixer_session
        self._audit_sink = audit_sink
        self._bound = False
        self._proxy: Optional[_ReadOnlyMixerProxy] = None
        self._physical_client: Any = None
        self._target: Any = None
        self._snapshots: Dict[str, _AttrSnapshot] = {}

        if engine is None:
            raise LegacyMixerSeamError("legacy engine is required")
        if not callable(getattr(engine, "_discover_mixer", None)):
            raise LegacyMixerSeamError("legacy engine has no _discover_mixer seam")
        if not callable(getattr(engine, "_connect_mixer", None)):
            raise LegacyMixerSeamError("legacy engine has no _connect_mixer seam")

    @property
    def bound(self) -> bool:
        return self._bound

    def _audit(self, event: str, **payload: Any) -> None:
        if self._audit_sink is None:
            return
        self._audit_sink({"event": event, **payload})

    def _capture_attr(self, name: str) -> _AttrSnapshot:
        values = vars(self._engine)
        return _AttrSnapshot(name in values, values.get(name))

    def _restore_attr(self, name: str, snapshot: _AttrSnapshot) -> None:
        if snapshot.had_instance_value:
            setattr(self._engine, name, snapshot.value)
        elif name in vars(self._engine):
            delattr(self._engine, name)

    def _require_live_owner(self) -> Tuple[Any, Any]:
        if not bool(getattr(self._session, "connected", False)):
            raise LegacyMixerSeamError("canonical LiveMixerSession is not connected")
        client = getattr(self._session, "client", None)
        target = getattr(self._session, "target", None)
        if client is None or target is None:
            raise LegacyMixerSeamError(
                "canonical LiveMixerSession has no connected client/target"
            )
        return client, target

    def _assert_binding_intact(self) -> None:
        if not self._bound or self._proxy is None:
            raise LegacyMixerSeamError("external mixer seam is not bound")
        if not bool(getattr(self._session, "connected", False)):
            raise LegacyMixerSeamError("canonical mixer ownership was lost")
        if getattr(self._session, "client", None) is not self._physical_client:
            raise LegacyMixerSeamError("canonical mixer client changed while bound")
        if getattr(self._session, "target", None) != self._target:
            raise LegacyMixerSeamError("canonical mixer target changed while bound")
        if getattr(self._engine, "mixer_client", None) is not self._proxy:
            raise LegacyMixerSeamError("legacy mixer_client proxy was displaced")
        if getattr(self._engine, "_real_mixer_client", None) is not self._proxy:
            raise LegacyMixerSeamError("legacy _real_mixer_client proxy was displaced")

    def _bypass_discovery(self, *args: Any, **kwargs: Any) -> bool:
        self._assert_binding_intact()
        self._audit("legacy_external_mixer_discovery_bypassed")
        return True

    def _bypass_connection(self, *args: Any, **kwargs: Any) -> bool:
        self._assert_binding_intact()
        self._audit("legacy_external_mixer_connection_bypassed")
        return True

    def bind(self) -> bool:
        if self._bound:
            return False

        client, target = self._require_live_owner()
        if getattr(self._engine, "mixer_client", None) is not None:
            raise LegacyMixerSeamError(
                "legacy mixer_client already exists; mixer ownership is ambiguous"
            )
        if getattr(self._engine, "_real_mixer_client", None) is not None:
            raise LegacyMixerSeamError(
                "legacy _real_mixer_client already exists; mixer ownership is ambiguous"
            )

        self._snapshots = {
            name: self._capture_attr(name) for name in self._SNAPSHOT_ATTRS
        }
        proxy = _ReadOnlyMixerProxy(client, audit_sink=self._audit_sink)
        self._physical_client = client
        self._target = target
        self._proxy = proxy

        try:
            self._engine.mixer_client = proxy
            self._engine._real_mixer_client = proxy
            self._engine.mixer_type = target.mixer_type
            self._engine.mixer_ip = target.ip
            self._engine.mixer_port = target.port
            self._engine._discover_mixer = self._bypass_discovery
            self._engine._connect_mixer = self._bypass_connection
            self._bound = True
            self._audit(
                "legacy_external_mixer_bound",
                mixer_type=target.mixer_type,
                ip=target.ip,
                port=target.port,
            )
            return True
        except Exception:
            for name, snapshot in self._snapshots.items():
                self._restore_attr(name, snapshot)
            self._snapshots = {}
            self._proxy = None
            self._physical_client = None
            self._target = None
            self._bound = False
            raise

    def detach(self) -> bool:
        if not self._bound:
            return False

        self._assert_binding_intact()
        for name, snapshot in self._snapshots.items():
            self._restore_attr(name, snapshot)

        target = self._target
        self._snapshots = {}
        self._proxy = None
        self._physical_client = None
        self._target = None
        self._bound = False
        self._audit(
            "legacy_external_mixer_detached",
            mixer_type=getattr(target, "mixer_type", ""),
            ip=getattr(target, "ip", ""),
            port=getattr(target, "port", 0),
        )
        return True

    def status(self) -> LegacyMixerSeamStatus:
        target = self._target
        proxy = self._proxy
        return LegacyMixerSeamStatus(
            bound=self._bound,
            engine_uses_proxy=bool(
                self._bound
                and proxy is not None
                and getattr(self._engine, "mixer_client", None) is proxy
                and getattr(self._engine, "_real_mixer_client", None) is proxy
            ),
            physical_client_id=id(self._physical_client) if self._physical_client else 0,
            mixer_type=getattr(target, "mixer_type", "") if target else "",
            ip=getattr(target, "ip", "") if target else "",
            port=int(getattr(target, "port", 0)) if target else 0,
        )
