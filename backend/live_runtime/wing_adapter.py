"""WING transport adapter for the authoritative live control plane.

This module is intentionally narrow. It migrates one write family at a time
behind :class:`LiveControlPlane` instead of letting legacy AutoEQ/AutoFader/
AutoFOH controllers write to WING directly.

Migrated surface:
- input-channel fader: ``ch:N / fader_db``
- main fader: ``main:N / fader_db``

Readback is based on a fresh inbound OSC callback, not ``WingClient.state``.
``WingClient.send`` optimistically updates its local state cache for writes, so
reading that cache immediately after a command would falsely prove that the
physical console accepted the value. The adapter therefore subscribes once per
address, sends an explicit query, and waits for an actual WING response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import threading
from typing import Any

from .contracts import ProposedAction


_CHANNEL_TARGET = re.compile(r"^ch:(\d+)$")
_MAIN_TARGET = re.compile(r"^main:(\d+)$")


@dataclass
class _FreshReadSlot:
    event: threading.Event = field(default_factory=threading.Event)
    value: Any = None


class WingWriteAdapter:
    """Translate typed live actions into WING OSC transport operations.

    Fader writes are migrated for channels and mains. Unsupported parameters and
    target families fail closed so BENCH_TEST authorization cannot accidentally
    turn into an unreviewed protocol write.
    """

    FADER_MIN_DB = -144.0
    FADER_MAX_DB = 10.0

    def __init__(self, client: Any, *, readback_timeout: float = 0.35):
        if readback_timeout <= 0:
            raise ValueError("readback_timeout must be > 0")
        self._client = client
        self._readback_timeout = float(readback_timeout)
        self._slots: dict[str, _FreshReadSlot] = {}
        self._slots_lock = threading.Lock()

    @staticmethod
    def _extract_actual(args: tuple[Any, ...]) -> Any:
        """Mirror WING response semantics without trusting the local cache."""
        if len(args) >= 3:
            return args[2]
        if len(args) == 1:
            return args[0]
        if not args:
            return None
        return args

    @classmethod
    def _address_for(cls, action: ProposedAction) -> str:
        if action.parameter != "fader_db":
            raise NotImplementedError(
                f"WingWriteAdapter has not migrated parameter {action.parameter!r} yet"
            )

        channel_match = _CHANNEL_TARGET.fullmatch(action.target)
        if channel_match:
            channel = int(channel_match.group(1))
            if not 1 <= channel <= 40:
                raise ValueError(f"WING channel out of range: {channel}")
            return f"/ch/{channel}/fdr"

        main_match = _MAIN_TARGET.fullmatch(action.target)
        if main_match:
            main = int(main_match.group(1))
            if not 1 <= main <= 4:
                raise ValueError(f"WING main out of range: {main}")
            return f"/main/{main}/fdr"

        raise ValueError(f"Unsupported WING target: {action.target!r}")

    def _slot_for(self, address: str) -> _FreshReadSlot:
        with self._slots_lock:
            slot = self._slots.get(address)
            if slot is not None:
                return slot

            slot = _FreshReadSlot()

            def on_update(_address: str, *args: Any, _slot: _FreshReadSlot = slot) -> None:
                _slot.value = self._extract_actual(tuple(args))
                _slot.event.set()

            self._client.subscribe(address, on_update)
            self._slots[address] = slot
            return slot

    def read_value(self, action: ProposedAction) -> Any:
        """Query WING and require a fresh physical-console response."""
        address = self._address_for(action)
        slot = self._slot_for(address)
        slot.event.clear()

        sent = self._client.send(address)
        if sent is False:
            raise RuntimeError(f"WING query transport failed for {address}")
        if not slot.event.wait(self._readback_timeout):
            raise TimeoutError(f"No fresh WING readback for {address}")
        return slot.value

    def write_value(self, action: ProposedAction) -> Any:
        """Write one migrated fader action through WING OSC."""
        address = self._address_for(action)
        if isinstance(action.value, bool) or not isinstance(action.value, (int, float)):
            raise TypeError("fader_db value must be numeric")
        value = float(action.value)
        if not math.isfinite(value):
            raise ValueError("fader_db value must be finite")
        if not self.FADER_MIN_DB <= value <= self.FADER_MAX_DB:
            raise ValueError(
                f"fader_db outside WING range {self.FADER_MIN_DB}..{self.FADER_MAX_DB}: {value}"
            )

        sent = self._client.send(address, value)
        if sent is False:
            raise RuntimeError(f"WING write transport failed for {address}")
        return sent
