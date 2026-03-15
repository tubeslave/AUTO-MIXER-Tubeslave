"""
Thread-safe mixer state management.

Provides ThreadSafeMixerState with asyncio.Lock for concurrent access,
copy-on-read snapshots, and asyncio.Queue for producer/consumer patterns.
"""

import asyncio
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class StateSnapshot:
    """Immutable snapshot of mixer state at a point in time."""
    timestamp: float
    channels: Dict[int, Dict[str, Any]]
    main: Dict[str, Any]
    buses: Dict[int, Dict[str, Any]]
    fx: Dict[int, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_channel(self, ch: int) -> Dict[str, Any]:
        return self.channels.get(ch, {})

    def get_channel_param(self, ch: int, param: str, default: Any = None) -> Any:
        return self.channels.get(ch, {}).get(param, default)


class ThreadSafeMixerState:
    """
    Thread-safe mixer state with asyncio locks and copy-on-read semantics.

    All mutations go through set_* methods that acquire the lock.
    Reads return deep copies to prevent data races.
    """

    def __init__(self, num_channels: int = 40, num_buses: int = 16, num_fx: int = 8):
        self._lock = asyncio.Lock()
        self._num_channels = num_channels
        self._num_buses = num_buses
        self._num_fx = num_fx

        # Internal mutable state
        self._channels: Dict[int, Dict[str, Any]] = {
            ch: {
                "fader": -144.0,
                "mute": 0,
                "name": "",
                "pan": 0.0,
                "gain": 0.0,
                "eq_on": 0,
                "comp_on": 0,
                "gate_on": 0,
                "solo": 0,
            }
            for ch in range(1, num_channels + 1)
        }
        self._main: Dict[str, Any] = {
            "fader": 0.0,
            "mute": 0,
        }
        self._buses: Dict[int, Dict[str, Any]] = {
            b: {"fader": -144.0, "mute": 0, "name": ""}
            for b in range(1, num_buses + 1)
        }
        self._fx: Dict[int, Dict[str, Any]] = {
            f: {"model": "", "on": 0, "params": {}}
            for f in range(1, num_fx + 1)
        }
        self._version = 0
        self._last_update = time.time()
        self._dirty_channels: Set[int] = set()

    async def snapshot(self) -> StateSnapshot:
        """Create an immutable deep-copy snapshot of current state."""
        async with self._lock:
            return StateSnapshot(
                timestamp=time.time(),
                channels=copy.deepcopy(self._channels),
                main=copy.deepcopy(self._main),
                buses=copy.deepcopy(self._buses),
                fx=copy.deepcopy(self._fx),
                metadata={
                    "version": self._version,
                    "num_channels": self._num_channels,
                },
            )

    async def set_channel_param(self, ch: int, param: str, value: Any):
        """Set a single channel parameter."""
        async with self._lock:
            if ch in self._channels:
                self._channels[ch][param] = value
                self._dirty_channels.add(ch)
                self._version += 1
                self._last_update = time.time()

    async def set_channel_params(self, ch: int, params: Dict[str, Any]):
        """Set multiple channel parameters atomically."""
        async with self._lock:
            if ch in self._channels:
                self._channels[ch].update(params)
                self._dirty_channels.add(ch)
                self._version += 1
                self._last_update = time.time()

    async def get_channel_param(self, ch: int, param: str, default: Any = None) -> Any:
        """Get a single channel parameter (copy)."""
        async with self._lock:
            val = self._channels.get(ch, {}).get(param, default)
            if isinstance(val, (dict, list)):
                return copy.deepcopy(val)
            return val

    async def get_channel(self, ch: int) -> Dict[str, Any]:
        """Get full channel state (deep copy)."""
        async with self._lock:
            return copy.deepcopy(self._channels.get(ch, {}))

    async def set_main_param(self, param: str, value: Any):
        """Set a main bus parameter."""
        async with self._lock:
            self._main[param] = value
            self._version += 1
            self._last_update = time.time()

    async def get_main(self) -> Dict[str, Any]:
        """Get main bus state (deep copy)."""
        async with self._lock:
            return copy.deepcopy(self._main)

    async def set_bus_param(self, bus: int, param: str, value: Any):
        """Set a bus parameter."""
        async with self._lock:
            if bus in self._buses:
                self._buses[bus][param] = value
                self._version += 1
                self._last_update = time.time()

    async def get_bus(self, bus: int) -> Dict[str, Any]:
        """Get bus state (deep copy)."""
        async with self._lock:
            return copy.deepcopy(self._buses.get(bus, {}))

    async def set_fx_param(self, fx_num: int, param: str, value: Any):
        """Set an FX parameter."""
        async with self._lock:
            if fx_num in self._fx:
                if param == "params" and isinstance(value, dict):
                    self._fx[fx_num]["params"].update(value)
                else:
                    self._fx[fx_num][param] = value
                self._version += 1
                self._last_update = time.time()

    async def get_dirty_channels(self) -> Set[int]:
        """Get and clear the set of channels modified since last call."""
        async with self._lock:
            dirty = self._dirty_channels.copy()
            self._dirty_channels.clear()
            return dirty

    async def get_version(self) -> int:
        """Get current state version number."""
        async with self._lock:
            return self._version

    async def bulk_update_from_osc(self, osc_state: Dict[str, Any]):
        """Update internal state from raw OSC state dict."""
        async with self._lock:
            for address, value in osc_state.items():
                self._apply_osc_value(address, value)
            self._version += 1
            self._last_update = time.time()

    def _apply_osc_value(self, address: str, value: Any):
        """Parse an OSC address and apply the value to internal state."""
        parts = address.strip("/").split("/")
        if len(parts) < 2:
            return

        if parts[0] == "ch":
            try:
                ch = int(parts[1])
            except (ValueError, IndexError):
                return
            if ch not in self._channels:
                return

            if len(parts) == 3:
                param_map = {
                    "fdr": "fader", "fader": "fader",
                    "mute": "mute", "pan": "pan",
                    "name": "name", "solo": "solo",
                }
                param = param_map.get(parts[2])
                if param:
                    self._channels[ch][param] = value
                    self._dirty_channels.add(ch)

        elif parts[0] == "main" and len(parts) >= 3:
            param_map = {"fdr": "fader", "mute": "mute"}
            param = param_map.get(parts[2])
            if param:
                self._main[param] = value


class MixerStateQueue:
    """
    Async producer/consumer queue for mixer state updates.

    Producers push state changes; consumers process them in order.
    Uses asyncio.Queue with optional max size for backpressure.
    """

    def __init__(self, maxsize: int = 1000):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._consumers: List[asyncio.Task] = []
        self._running = False

    async def put(self, event_type: str, data: Dict[str, Any]):
        """Put a state update event into the queue."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("MixerStateQueue full, dropping oldest event")
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(event)

    async def get(self) -> Dict[str, Any]:
        """Get next event from the queue (blocks until available)."""
        return await self._queue.get()

    def get_nowait(self) -> Optional[Dict[str, Any]]:
        """Non-blocking get."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    async def drain(self) -> List[Dict[str, Any]]:
        """Drain all pending events."""
        events = []
        while not self._queue.empty():
            try:
                events.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events
