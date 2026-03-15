"""
Thread-Safe Mixer State Management for AUTO MIXER Tubeslave.

Provides ``ThreadSafeMixerState`` — a concurrency-safe wrapper around the
per-channel mixer state dict — and ``StateUpdateQueue``, an asyncio-based
producer/consumer queue with per-channel subscription support.

Supports both synchronous (``threading.Lock``) and asynchronous
(``asyncio.Lock``) access patterns so the module works in threaded OSC
callbacks as well as the async web-socket server.

All public read methods return **deep copies** (copy-on-read) so that
consumers never hold references into the live state.
"""

import asyncio
import copy
import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State update descriptor
# ---------------------------------------------------------------------------

class UpdateType(enum.Enum):
    """Describes what kind of mutation occurred."""
    PARAM_SET = "param_set"
    BATCH = "batch"
    CHANNEL_RESET = "channel_reset"
    FULL_RESET = "full_reset"


@dataclass
class StateUpdate:
    """Immutable description of a single state change, pushed into the queue."""
    update_type: UpdateType
    channel_id: Optional[int] = None       # None for full-reset
    param: Optional[str] = None            # e.g. "eq.1g" or "fdr"
    value: Any = None                      # New value
    previous_value: Any = None             # Old value (for undo)
    timestamp: float = field(default_factory=time.monotonic)
    batch_params: Optional[Dict[str, Any]] = None  # For batch updates

    def __repr__(self) -> str:
        if self.update_type == UpdateType.BATCH:
            n = len(self.batch_params) if self.batch_params else 0
            return f"StateUpdate(BATCH ch={self.channel_id}, {n} params)"
        return (
            f"StateUpdate({self.update_type.value} ch={self.channel_id} "
            f"{self.param}={self.value})"
        )


# ---------------------------------------------------------------------------
# ThreadSafeMixerState
# ---------------------------------------------------------------------------

class ThreadSafeMixerState:
    """
    Concurrency-safe dict-like store for per-channel mixer parameters.

    Internal layout::

        _state = {
            1: {"fdr": -5.0, "mute": 0, "eq.1g": 0.0, ...},
            2: { ... },
            ...
        }

    Thread safety
    -------------
    * ``threading.Lock`` protects synchronous callers (OSC callbacks, etc.).
    * ``asyncio.Lock`` protects async callers (WebSocket handlers, etc.).
    * Every public read returns a **deep copy** so callers cannot mutate the
      live state.

    Usage::

        state = ThreadSafeMixerState()
        state.set_channel_param(1, "fdr", -5.0)
        value = state.get_channel_param(1, "fdr")        # -5.0  (deep copy)
        snap  = state.get_snapshot()                       # full deep copy
        state.batch_update(1, {"fdr": 0.0, "mute": 0})   # atomic
    """

    def __init__(self) -> None:
        self._state: Dict[int, Dict[str, Any]] = {}
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        # Optional update queue (connected via attach_queue)
        self._queue: Optional["StateUpdateQueue"] = None
        logger.debug("ThreadSafeMixerState created")

    # -- Queue integration -------------------------------------------------

    def attach_queue(self, queue: "StateUpdateQueue") -> None:
        """Attach a ``StateUpdateQueue`` so every mutation is automatically published."""
        self._queue = queue
        logger.info("StateUpdateQueue attached to ThreadSafeMixerState")

    def detach_queue(self) -> None:
        """Detach the update queue."""
        self._queue = None

    # -- Synchronous API ---------------------------------------------------

    def set_channel_param(self, channel: int, param: str, value: Any) -> None:
        """Set a single parameter for *channel* (thread-safe, synchronous)."""
        with self._sync_lock:
            prev = self._raw_get(channel, param)
            self._raw_set(channel, param, value)
        self._publish(StateUpdate(
            update_type=UpdateType.PARAM_SET,
            channel_id=channel,
            param=param,
            value=value,
            previous_value=prev,
        ))

    def get_channel_param(self, channel: int, param: str,
                          default: Any = None) -> Any:
        """Return a deep copy of a single parameter (thread-safe, synchronous)."""
        with self._sync_lock:
            val = self._raw_get(channel, param)
        if val is None:
            return default
        return copy.deepcopy(val)

    def get_channel(self, channel: int) -> Dict[str, Any]:
        """Return a deep copy of all parameters for *channel*."""
        with self._sync_lock:
            ch_data = self._state.get(channel)
        if ch_data is None:
            return {}
        return copy.deepcopy(ch_data)

    def get_snapshot(self) -> Dict[int, Dict[str, Any]]:
        """Return a complete deep copy of the entire state."""
        with self._sync_lock:
            return copy.deepcopy(self._state)

    def batch_update(self, channel: int, params: Dict[str, Any]) -> None:
        """
        Atomically apply multiple parameter changes for *channel*.

        All writes happen under a single lock acquisition so no reader
        can observe a partial batch.
        """
        with self._sync_lock:
            prev_values: Dict[str, Any] = {}
            for param, value in params.items():
                prev_values[param] = self._raw_get(channel, param)
                self._raw_set(channel, param, value)
        self._publish(StateUpdate(
            update_type=UpdateType.BATCH,
            channel_id=channel,
            batch_params=params,
            previous_value=prev_values,
        ))

    def reset_channel(self, channel: int) -> None:
        """Remove all state for *channel*."""
        with self._sync_lock:
            self._state.pop(channel, None)
        self._publish(StateUpdate(
            update_type=UpdateType.CHANNEL_RESET,
            channel_id=channel,
        ))

    def reset_all(self) -> None:
        """Remove all state for all channels."""
        with self._sync_lock:
            self._state.clear()
        self._publish(StateUpdate(update_type=UpdateType.FULL_RESET))

    def channel_ids(self) -> List[int]:
        """Return sorted list of channels that have state."""
        with self._sync_lock:
            return sorted(self._state.keys())

    def channel_count(self) -> int:
        """Return the number of channels with state."""
        with self._sync_lock:
            return len(self._state)

    # -- Async API ---------------------------------------------------------

    async def async_set_channel_param(self, channel: int, param: str,
                                      value: Any) -> None:
        """Async version of ``set_channel_param``."""
        async with self._async_lock:
            with self._sync_lock:
                prev = self._raw_get(channel, param)
                self._raw_set(channel, param, value)
        self._publish(StateUpdate(
            update_type=UpdateType.PARAM_SET,
            channel_id=channel,
            param=param,
            value=value,
            previous_value=prev,
        ))

    async def async_get_channel_param(self, channel: int, param: str,
                                      default: Any = None) -> Any:
        """Async version of ``get_channel_param``."""
        async with self._async_lock:
            with self._sync_lock:
                val = self._raw_get(channel, param)
        if val is None:
            return default
        return copy.deepcopy(val)

    async def async_get_snapshot(self) -> Dict[int, Dict[str, Any]]:
        """Async version of ``get_snapshot``."""
        async with self._async_lock:
            with self._sync_lock:
                return copy.deepcopy(self._state)

    async def async_batch_update(self, channel: int,
                                 params: Dict[str, Any]) -> None:
        """Async version of ``batch_update``."""
        async with self._async_lock:
            with self._sync_lock:
                prev_values: Dict[str, Any] = {}
                for param, value in params.items():
                    prev_values[param] = self._raw_get(channel, param)
                    self._raw_set(channel, param, value)
        self._publish(StateUpdate(
            update_type=UpdateType.BATCH,
            channel_id=channel,
            batch_params=params,
            previous_value=prev_values,
        ))

    # -- Raw (unlocked) helpers --------------------------------------------

    def _raw_get(self, channel: int, param: str) -> Any:
        ch_data = self._state.get(channel)
        if ch_data is None:
            return None
        return ch_data.get(param)

    def _raw_set(self, channel: int, param: str, value: Any) -> None:
        if channel not in self._state:
            self._state[channel] = {}
        self._state[channel][param] = value

    # -- Queue publish -----------------------------------------------------

    def _publish(self, update: StateUpdate) -> None:
        """Push an update into the attached queue (non-blocking, best-effort)."""
        if self._queue is None:
            return
        try:
            self._queue.publish(update)
        except Exception:
            logger.warning("Failed to publish state update: %s", update, exc_info=True)


# ---------------------------------------------------------------------------
# StateUpdateQueue — asyncio producer/consumer with subscriptions
# ---------------------------------------------------------------------------

class StateUpdateQueue:
    """
    Async queue that distributes ``StateUpdate`` objects to subscribers.

    Producers call ``publish(update)`` (sync-safe, non-blocking).
    Consumers ``subscribe()`` to receive updates for specific channels (or all).

    Architecture::

        OSC callback  ──►  ThreadSafeMixerState.set_channel_param()
                               │
                               ▼
                          StateUpdateQueue.publish()
                               │
                       ┌───────┼───────┐
                       ▼       ▼       ▼
                    sub_A   sub_B   sub_C   (asyncio.Queue per subscriber)

    Usage::

        queue = StateUpdateQueue()
        state = ThreadSafeMixerState()
        state.attach_queue(queue)

        # Consumer coroutine
        sub_id = queue.subscribe(channels={1, 2})
        async for update in queue.iter_updates(sub_id):
            print(update)

        # Cleanup
        queue.unsubscribe(sub_id)
    """

    def __init__(self, maxsize: int = 1024) -> None:
        self._maxsize = maxsize
        self._subscribers: Dict[int, _Subscriber] = {}
        self._next_id = 0
        self._lock = threading.Lock()  # protects _subscribers and _next_id
        logger.debug("StateUpdateQueue created (maxsize=%d)", maxsize)

    # -- Subscribe / unsubscribe -------------------------------------------

    def subscribe(self, channels: Optional[Set[int]] = None,
                  update_types: Optional[Set[UpdateType]] = None) -> int:
        """
        Create a new subscription and return its integer ID.

        Args:
            channels: Set of channel IDs to receive updates for.
                      ``None`` means all channels.
            update_types: Set of ``UpdateType`` values to filter on.
                          ``None`` means all types.
        """
        with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            self._subscribers[sub_id] = _Subscriber(
                sub_id=sub_id,
                channels=channels,
                update_types=update_types,
                queue=asyncio.Queue(maxsize=self._maxsize),
            )
        logger.debug(
            "Subscriber %d created (channels=%s, types=%s)",
            sub_id,
            channels or "ALL",
            update_types or "ALL",
        )
        return sub_id

    def unsubscribe(self, sub_id: int) -> bool:
        """Remove a subscription.  Returns ``True`` if it existed."""
        with self._lock:
            removed = self._subscribers.pop(sub_id, None)
        if removed is not None:
            logger.debug("Subscriber %d removed", sub_id)
            return True
        return False

    def update_subscription(self, sub_id: int,
                            channels: Optional[Set[int]] = None,
                            update_types: Optional[Set[UpdateType]] = None) -> bool:
        """Change the filter for an existing subscription."""
        with self._lock:
            sub = self._subscribers.get(sub_id)
            if sub is None:
                return False
            sub.channels = channels
            sub.update_types = update_types
        return True

    # -- Publish -----------------------------------------------------------

    def publish(self, update: StateUpdate) -> int:
        """
        Push *update* to all matching subscribers (non-blocking).

        Returns the number of subscribers that received the update.
        Drops the update silently for any subscriber whose queue is full.
        """
        delivered = 0
        with self._lock:
            subscribers = list(self._subscribers.values())
        for sub in subscribers:
            if not sub.matches(update):
                continue
            try:
                sub.queue.put_nowait(update)
                delivered += 1
            except asyncio.QueueFull:
                logger.warning(
                    "Subscriber %d queue full — dropping update %s",
                    sub.sub_id, update,
                )
        return delivered

    # -- Consume -----------------------------------------------------------

    async def get_update(self, sub_id: int,
                         timeout: Optional[float] = None) -> Optional[StateUpdate]:
        """
        Wait for the next update for subscriber *sub_id*.

        Args:
            sub_id: Subscription ID from ``subscribe()``.
            timeout: Maximum seconds to wait.  ``None`` = wait forever.

        Returns:
            The next ``StateUpdate``, or ``None`` on timeout.
        """
        with self._lock:
            sub = self._subscribers.get(sub_id)
        if sub is None:
            raise ValueError(f"Unknown subscriber ID: {sub_id}")
        try:
            if timeout is None:
                return await sub.queue.get()
            return await asyncio.wait_for(sub.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def iter_updates(self, sub_id: int):
        """
        Async iterator that yields ``StateUpdate`` objects for *sub_id*.

        The iterator runs indefinitely until the subscription is removed or
        the coroutine is cancelled.

        Usage::

            async for update in queue.iter_updates(sub_id):
                handle(update)
        """
        while True:
            with self._lock:
                sub = self._subscribers.get(sub_id)
            if sub is None:
                return  # subscription was removed
            try:
                update = await sub.queue.get()
                yield update
            except asyncio.CancelledError:
                return

    async def drain(self, sub_id: int) -> List[StateUpdate]:
        """Return all currently queued updates for *sub_id* without blocking."""
        with self._lock:
            sub = self._subscribers.get(sub_id)
        if sub is None:
            return []
        updates: List[StateUpdate] = []
        while not sub.queue.empty():
            try:
                updates.append(sub.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return updates

    # -- Diagnostics -------------------------------------------------------

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def subscriber_info(self) -> List[Dict[str, Any]]:
        """Return diagnostic info for all subscribers."""
        with self._lock:
            subs = list(self._subscribers.values())
        return [
            {
                "sub_id": s.sub_id,
                "channels": sorted(s.channels) if s.channels else "ALL",
                "update_types": (
                    [t.value for t in s.update_types]
                    if s.update_types else "ALL"
                ),
                "queue_size": s.queue.qsize(),
                "queue_maxsize": s.queue.maxsize,
            }
            for s in subs
        ]


# ---------------------------------------------------------------------------
# Internal subscriber record
# ---------------------------------------------------------------------------

@dataclass
class _Subscriber:
    sub_id: int
    channels: Optional[Set[int]]
    update_types: Optional[Set[UpdateType]]
    queue: asyncio.Queue

    def matches(self, update: StateUpdate) -> bool:
        """Return True if this subscriber wants *update*."""
        # Full reset always delivered
        if update.update_type == UpdateType.FULL_RESET:
            return True
        # Filter by update type
        if self.update_types is not None and update.update_type not in self.update_types:
            return False
        # Filter by channel
        if self.channels is not None and update.channel_id is not None:
            if update.channel_id not in self.channels:
                return False
        return True
