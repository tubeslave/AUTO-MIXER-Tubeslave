"""WING transport adapter for the authoritative live control plane.

This module is intentionally narrow. It migrates one write family at a time
behind :class:`LiveControlPlane` instead of letting legacy AutoEQ/AutoFader/
AutoFOH controllers write to WING directly.

Migrated surfaces:
- input-channel fader: ``ch:N / fader_db``
- main fader: ``main:N / fader_db``
- channel PEQ band gain: ``ch:N / eq_gain_db`` with an explicit
  :class:`EqBandLocator`
- relative fader/EQ-gain proposals use the same addresses; the control plane
  resolves them to absolute values before the write.

Readback is based on fresh inbound OSC callbacks, not ``WingClient.state``.
``WingClient.send`` optimistically updates its local state cache for writes, so
reading that cache immediately after a command would falsely prove that the
physical console accepted the value. EQ writes additionally re-check the band's
frequency/Q fingerprint before mutation, so a stale band number fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import threading
from typing import Any

from .contracts import EqBandLocator, ProposedAction


_CHANNEL_TARGET = re.compile(r"^ch:(\d+)$")
_MAIN_TARGET = re.compile(r"^main:(\d+)$")
_FADER_PARAMETERS = {"fader_db", "fader_delta_db"}
_EQ_GAIN_PARAMETERS = {"eq_gain_db", "eq_gain_delta_db"}
_OUTPUT_GROUPS = {"MOD", "AUX", "CRD", "AES", "USB"}


@dataclass(frozen=True)
class OutputRouteReadback:
    """Fresh physical WING output-route observation."""

    output_group: str
    output_number: int
    source_group: str
    source_channel: int


@dataclass
class _FreshReadSlot:
    event: threading.Event = field(default_factory=threading.Event)
    value: Any = None


class WingWriteAdapter:
    """Translate typed live actions into WING OSC transport operations.

    Only explicitly migrated parameter/target families are accepted. Unsupported
    actions fail closed so BENCH_TEST authorization cannot accidentally turn
    into an unreviewed protocol write.
    """

    FADER_MIN_DB = -144.0
    FADER_MAX_DB = 10.0
    EQ_GAIN_MIN_DB = -15.0
    EQ_GAIN_MAX_DB = 15.0
    EQ_FREQ_MIN_HZ = 20.0
    EQ_FREQ_MAX_HZ = 20000.0
    EQ_Q_MIN = 0.44
    EQ_Q_MAX = 10.0

    def __init__(
        self,
        client: Any,
        *,
        readback_timeout: float = 0.35,
        eq_frequency_tolerance_ratio: float = 0.02,
        eq_q_tolerance: float = 0.08,
    ):
        if readback_timeout <= 0:
            raise ValueError("readback_timeout must be > 0")
        if eq_frequency_tolerance_ratio < 0:
            raise ValueError("eq_frequency_tolerance_ratio must be >= 0")
        if eq_q_tolerance < 0:
            raise ValueError("eq_q_tolerance must be >= 0")
        self._client = client
        self._readback_timeout = float(readback_timeout)
        self._eq_frequency_tolerance_ratio = float(eq_frequency_tolerance_ratio)
        self._eq_q_tolerance = float(eq_q_tolerance)
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

    @staticmethod
    def _numeric(value: Any, *, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite")
        return number

    @classmethod
    def _channel_number(cls, action: ProposedAction) -> int:
        match = _CHANNEL_TARGET.fullmatch(action.target)
        if not match:
            raise ValueError(f"Channel EQ requires ch:N target, got {action.target!r}")
        channel = int(match.group(1))
        if not 1 <= channel <= 40:
            raise ValueError(f"WING channel out of range: {channel}")
        return channel

    @classmethod
    def _validated_eq_locator(cls, action: ProposedAction) -> EqBandLocator:
        locator = action.eq_locator
        if locator is None:
            raise ValueError(f"{action.parameter} requires an explicit EqBandLocator")
        if not 1 <= locator.band <= 4:
            raise ValueError(f"WING EQ band out of range: {locator.band}")
        frequency = cls._numeric(locator.frequency_hz, label="EQ locator frequency_hz")
        q = cls._numeric(locator.q, label="EQ locator q")
        if not cls.EQ_FREQ_MIN_HZ <= frequency <= cls.EQ_FREQ_MAX_HZ:
            raise ValueError(
                f"EQ locator frequency outside WING range {cls.EQ_FREQ_MIN_HZ}..{cls.EQ_FREQ_MAX_HZ}: {frequency}"
            )
        if not cls.EQ_Q_MIN <= q <= cls.EQ_Q_MAX:
            raise ValueError(
                f"EQ locator q outside WING range {cls.EQ_Q_MIN}..{cls.EQ_Q_MAX}: {q}"
            )
        return locator

    @classmethod
    def _address_for(cls, action: ProposedAction) -> str:
        if action.parameter in _EQ_GAIN_PARAMETERS:
            channel = cls._channel_number(action)
            locator = cls._validated_eq_locator(action)
            return f"/ch/{channel}/eq/{locator.band}g"

        if action.parameter not in _FADER_PARAMETERS:
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

    def _query_address(self, address: str) -> Any:
        slot = self._slot_for(address)
        slot.event.clear()
        sent = self._client.send(address)
        if sent is False:
            raise RuntimeError(f"WING query transport failed for {address}")
        if not slot.event.wait(self._readback_timeout):
            raise TimeoutError(f"No fresh WING readback for {address}")
        return slot.value

    def read_output_route(self, output_group: str, output_number: int) -> OutputRouteReadback:
        """Read one physical output route from fresh WING callbacks.

        This deliberately bypasses ``WingClient.get_output_routing()`` because
        that legacy helper waits a fixed interval and then consults the shared
        state cache. PATCH_VERIFY needs proof from the current query response,
        not a possibly stale cached route.
        """
        if not isinstance(output_group, str):
            raise TypeError("WING output_group must be a string")
        group = output_group.strip().upper()
        if group not in _OUTPUT_GROUPS:
            raise ValueError(f"Unsupported WING output group: {output_group!r}")
        if isinstance(output_number, bool) or not isinstance(output_number, int):
            raise TypeError("WING output_number must be an integer")
        if not 1 <= output_number <= 48:
            raise ValueError(f"WING output number out of range: {output_number}")

        base = f"/io/out/{group}/{output_number - 1}"
        source_group_raw = self._query_address(f"{base}/grp")
        if not isinstance(source_group_raw, str):
            raise TypeError("WING output source group readback must be a string")
        source_group = source_group_raw.strip().upper()
        if not source_group:
            raise ValueError("WING output source group readback is empty")

        source_channel_number = self._numeric(
            self._query_address(f"{base}/in"),
            label="WING output source channel readback",
        )
        if not source_channel_number.is_integer():
            raise ValueError(
                f"WING output source channel readback must be integral: {source_channel_number}"
            )

        return OutputRouteReadback(
            output_group=group,
            output_number=output_number,
            source_group=source_group,
            source_channel=int(source_channel_number),
        )

    def read_eq_locators(self, channel: int) -> list[EqBandLocator]:
        """Read fresh frequency/Q fingerprints for all four channel PEQ bands.

        This is a read-only migration primitive for the realtime EQ locator
        selector.  It deliberately does not consult ``WingClient.state`` and it
        does not read or change gain, so choosing a candidate band cannot mutate
        the console or accidentally validate an optimistic local cache value.
        """
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise TypeError("WING channel must be an integer")
        if not 1 <= channel <= 40:
            raise ValueError(f"WING channel out of range: {channel}")

        locators: list[EqBandLocator] = []
        for band in range(1, 5):
            frequency = self._numeric(
                self._query_address(f"/ch/{channel}/eq/{band}f"),
                label="WING EQ frequency readback",
            )
            q = self._numeric(
                self._query_address(f"/ch/{channel}/eq/{band}q"),
                label="WING EQ Q readback",
            )
            if not self.EQ_FREQ_MIN_HZ <= frequency <= self.EQ_FREQ_MAX_HZ:
                raise ValueError(
                    f"WING EQ frequency outside range on band {band}: {frequency}"
                )
            if not self.EQ_Q_MIN <= q <= self.EQ_Q_MAX:
                raise ValueError(f"WING EQ Q outside range on band {band}: {q}")
            locators.append(EqBandLocator(band=band, frequency_hz=frequency, q=q))
        return locators

    def _assert_eq_locator_matches(self, action: ProposedAction) -> None:
        """Require the physical band's current F/Q to match the proposed locator."""
        if action.parameter not in _EQ_GAIN_PARAMETERS:
            return
        channel = self._channel_number(action)
        locator = self._validated_eq_locator(action)
        actual_frequency = self._numeric(
            self._query_address(f"/ch/{channel}/eq/{locator.band}f"),
            label="WING EQ frequency readback",
        )
        actual_q = self._numeric(
            self._query_address(f"/ch/{channel}/eq/{locator.band}q"),
            label="WING EQ Q readback",
        )
        frequency_tolerance = max(
            1.0,
            abs(float(locator.frequency_hz)) * self._eq_frequency_tolerance_ratio,
        )
        if abs(actual_frequency - float(locator.frequency_hz)) > frequency_tolerance:
            raise ValueError(
                "WING EQ locator frequency mismatch: "
                f"expected {locator.frequency_hz}, read {actual_frequency}"
            )
        if abs(actual_q - float(locator.q)) > self._eq_q_tolerance:
            raise ValueError(
                f"WING EQ locator Q mismatch: expected {locator.q}, read {actual_q}"
            )

    def read_value(self, action: ProposedAction) -> Any:
        """Query WING and require a fresh physical-console response."""
        address = self._address_for(action)
        self._assert_eq_locator_matches(action)
        return self._query_address(address)

    def write_value(self, action: ProposedAction) -> Any:
        """Write one resolved migrated action through WING OSC."""
        address = self._address_for(action)

        if action.parameter == "fader_delta_db":
            raise ValueError(
                "fader_delta_db must be resolved to absolute fader_db by LiveControlPlane before write"
            )
        if action.parameter == "eq_gain_delta_db":
            raise ValueError(
                "eq_gain_delta_db must be resolved to absolute eq_gain_db by LiveControlPlane before write"
            )

        value = self._numeric(action.value, label=action.parameter)
        if action.parameter == "fader_db":
            if not self.FADER_MIN_DB <= value <= self.FADER_MAX_DB:
                raise ValueError(
                    f"fader_db outside WING range {self.FADER_MIN_DB}..{self.FADER_MAX_DB}: {value}"
                )
        elif action.parameter == "eq_gain_db":
            self._assert_eq_locator_matches(action)
            if not self.EQ_GAIN_MIN_DB <= value <= self.EQ_GAIN_MAX_DB:
                raise ValueError(
                    f"eq_gain_db outside WING range {self.EQ_GAIN_MIN_DB}..{self.EQ_GAIN_MAX_DB}: {value}"
                )
        else:  # pragma: no cover - _address_for already fails closed
            raise NotImplementedError(action.parameter)

        sent = self._client.send(address, value)
        if sent is False:
            raise RuntimeError(f"WING write transport failed for {address}")
        return sent
