"""
Thread-safe mixer state management.

Provides ThreadSafeMixerState with threading.Lock for concurrent access,
copy-on-read snapshots, StateUpdateQueue with pub/sub for state updates.
"""

import asyncio
import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class UpdateType(Enum):
    PARAM_SET = "param_set"
    BATCH = "batch"
    CHANNEL_RESET = "channel_reset"
    FULL_RESET = "full_reset"


@dataclass
class StateUpdate:
    """A single state update event."""
    update_type: UpdateType
    channel_id: Optional[int] = None
    param: Optional[str] = None
    value: Any = None
    params: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)

    def __repr__(self):
        parts = [f"StateUpdate({self.update_type.name}"]
        if self.channel_id is not None:
            parts.append(f"ch={self.channel_id}")
        if self.param is not None:
            parts.append(f"param={self.param!r}")
        if self.value is not None:
            parts.append(f"value={self.value!r}")
        return ", ".join(parts) + ")"


@dataclass
class _Subscriber:
    """Internal subscriber record."""
    sub_id: int
    channels: Optional[Set[int]]
    update_types: Optional[Set[UpdateType]]
    queue: asyncio.Queue

    def matches(self, update: StateUpdate) -> bool:
        if update.update_type == UpdateType.FULL_RESET:
            return True
        if self.update_types is not None and update.update_type not in self.update_types:
            return False
        if self.channels is not None and update.channel_id is not None:
            return update.channel_id in self.channels
        return True


class StateUpdateQueue:
    """
    Pub/sub queue for mixer state updates.

    Subscribers register with optional channel and type filters.
    """

    def __init__(self, maxsize: int = 1000):
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._subscribers: Dict[int, _Subscriber] = {}
        self._next_id = 1

    def subscribe(
        self,
        channels: Optional[Set[int]] = None,
        update_types: Optional[Set[UpdateType]] = None,
    ) -> int:
        with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            self._subscribers[sub_id] = _Subscriber(
                sub_id=sub_id,
                channels=channels,
                update_types=update_types,
                queue=asyncio.Queue(maxsize=self._maxsize),
            )
            return sub_id

    def unsubscribe(self, sub_id: int) -> bool:
        with self._lock:
            return self._subscribers.pop(sub_id, None) is not None

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def update_subscription(
        self,
        sub_id: int,
        channels: Optional[Set[int]] = None,
        update_types: Optional[Set[UpdateType]] = None,
    ) -> bool:
        with self._lock:
            sub = self._subscribers.get(sub_id)
            if sub is None:
                return False
            if channels is not None:
                sub.channels = channels
            if update_types is not None:
                sub.update_types = update_types
            return True

    def publish(self, update: StateUpdate) -> int:
        delivered = 0
        with self._lock:
            subs = list(self._subscribers.values())
        for sub in subs:
            if sub.matches(update):
                try:
                    sub.queue.put_nowait(update)
                    delivered += 1
                except asyncio.QueueFull:
                    try:
                        sub.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        sub.queue.put_nowait(update)
                        delivered += 1
                    except asyncio.QueueFull:
                        pass
        return delivered

    async def get_update(self, sub_id: int, timeout: float = 1.0) -> Optional[StateUpdate]:
        with self._lock:
            sub = self._subscribers.get(sub_id)
        if sub is None:
            raise ValueError(f"Unknown subscriber {sub_id}")
        try:
            return await asyncio.wait_for(sub.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def drain(self, sub_id: int) -> List[StateUpdate]:
        with self._lock:
            sub = self._subscribers.get(sub_id)
        if sub is None:
            return []
        updates = []
        while not sub.queue.empty():
            try:
                updates.append(sub.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return updates

    def subscriber_info(self) -> List[Dict[str, Any]]:
        with self._lock:
            info = []
            for sub in self._subscribers.values():
                info.append({
                    "id": sub.sub_id,
                    "channels": sorted(sub.channels) if sub.channels else None,
                    "update_types": [t.value for t in sub.update_types] if sub.update_types else None,
                    "pending": sub.queue.qsize(),
                })
            return info


class ThreadSafeMixerState:
    """
    Thread-safe mixer state with threading.Lock and copy-on-read semantics.

    Provides both synchronous and async APIs.
    """

    def __init__(self, num_channels: int = 0, num_buses: int = 0, num_fx: int = 0):
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._channels: Dict[int, Dict[str, Any]] = {}
        self._queue: Optional[StateUpdateQueue] = None
        self._version = 0
        self._last_update = time.time()

    def attach_queue(self, queue: StateUpdateQueue):
        self._queue = queue

    def detach_queue(self):
        self._queue = None

    def _publish(self, update: StateUpdate):
        if self._queue:
            self._queue.publish(update)

    # ── Core (sync) operations ──────────────────────────────────

    def _do_set_channel_param(self, ch: int, param: str, value: Any):
        with self._lock:
            if ch not in self._channels:
                self._channels[ch] = {}
            self._channels[ch][param] = value
            self._version += 1
            self._last_update = time.time()
        self._publish(StateUpdate(
            update_type=UpdateType.PARAM_SET,
            channel_id=ch, param=param, value=value,
        ))

    def _do_get_channel_param(self, ch: int, param: str, default: Any = None) -> Any:
        with self._lock:
            val = self._channels.get(ch, {}).get(param, default)
            if isinstance(val, (dict, list)):
                return copy.deepcopy(val)
            return val

    def _do_get_channel(self, ch: int) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._channels.get(ch, {}))

    def _do_get_snapshot(self) -> Dict[int, Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._channels)

    def _do_batch_update(self, ch: int, params: Dict[str, Any]):
        with self._lock:
            if ch not in self._channels:
                self._channels[ch] = {}
            self._channels[ch].update(params)
            self._version += 1
            self._last_update = time.time()
        self._publish(StateUpdate(
            update_type=UpdateType.BATCH,
            channel_id=ch, params=params,
        ))

    def _do_reset_channel(self, ch: int):
        with self._lock:
            self._channels.pop(ch, None)
            self._version += 1
            self._last_update = time.time()
        self._publish(StateUpdate(
            update_type=UpdateType.CHANNEL_RESET,
            channel_id=ch,
        ))

    def _do_reset_all(self):
        with self._lock:
            self._channels.clear()
            self._version += 1
            self._last_update = time.time()
        self._publish(StateUpdate(
            update_type=UpdateType.FULL_RESET,
        ))

    # ── Sync API (for thread-safe direct access) ────────────────

    def set_channel_param(self, ch: int, param: str, value: Any):
        """Set a channel parameter (sync)."""
        self._do_set_channel_param(ch, param, value)

    def get_channel_param(self, ch: int, param: str, default: Any = None) -> Any:
        return self._do_get_channel_param(ch, param, default)

    def get_channel(self, ch: int) -> Dict[str, Any]:
        return self._do_get_channel(ch)

    def get_snapshot(self) -> Dict[int, Dict[str, Any]]:
        return self._do_get_snapshot()

    def batch_update(self, ch: int, params: Dict[str, Any]):
        self._do_batch_update(ch, params)

    def reset_channel(self, ch: int):
        self._do_reset_channel(ch)

    def reset_all(self):
        self._do_reset_all()

    def channel_ids(self) -> List[int]:
        with self._lock:
            return sorted(self._channels.keys())

    def channel_count(self) -> int:
        with self._lock:
            return len(self._channels)

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    # ── Async API ────────────────────────────────────────────────

    async def async_set_channel_param(self, ch: int, param: str, value: Any):
        self._do_set_channel_param(ch, param, value)

    async def async_get_channel_param(self, ch: int, param: str, default: Any = None) -> Any:
        return self._do_get_channel_param(ch, param, default)

    async def async_get_snapshot(self) -> Dict[int, Dict[str, Any]]:
        return self._do_get_snapshot()

    async def async_batch_update(self, ch: int, params: Dict[str, Any]):
        self._do_batch_update(ch, params)

    async def snapshot(self):
        """Async snapshot (legacy compat)."""
        return self._do_get_snapshot()

    async def set_channel_params(self, ch: int, params: Dict[str, Any]):
        """Async batch update (legacy compat)."""
        self._do_batch_update(ch, params)

    async def get_version(self) -> int:
        return self.version

    async def get_dirty_channels(self) -> Set[int]:
        return set()


class MixerStateQueue:
    """
    Async producer/consumer queue for mixer state updates.
    Legacy compat wrapper around StateUpdateQueue.
    """

    def __init__(self, maxsize: int = 1000):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    async def put(self, event_type: str, data: Dict[str, Any]):
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(event)

    async def get(self) -> Dict[str, Any]:
        return await self._queue.get()

    def get_nowait(self) -> Optional[Dict[str, Any]]:
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
        events = []
        while not self._queue.empty():
            try:
                events.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events
