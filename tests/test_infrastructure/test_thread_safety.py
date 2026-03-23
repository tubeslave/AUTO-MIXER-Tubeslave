"""
Tests for backend/thread_safety.py — ThreadSafeMixerState, StateUpdateQueue,
UpdateType, StateUpdate, concurrent access, and async API.

All tests work without hardware or network.
"""

import asyncio
import threading
import pytest

try:
    from thread_safety import (
        ThreadSafeMixerState,
        StateUpdateQueue,
        StateUpdate,
        UpdateType,
        _Subscriber,
    )
except ImportError:
    pytest.skip("thread_safety module not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    """Fresh ThreadSafeMixerState instance."""
    return ThreadSafeMixerState()


@pytest.fixture
def queue():
    """Fresh StateUpdateQueue instance."""
    return StateUpdateQueue(maxsize=128)


@pytest.fixture
def state_with_queue():
    """ThreadSafeMixerState with an attached StateUpdateQueue."""
    s = ThreadSafeMixerState()
    q = StateUpdateQueue(maxsize=128)
    s.attach_queue(q)
    return s, q


# ---------------------------------------------------------------------------
# UpdateType / StateUpdate tests
# ---------------------------------------------------------------------------

class TestUpdateType:

    def test_enum_values(self):
        assert UpdateType.PARAM_SET.value == "param_set"
        assert UpdateType.BATCH.value == "batch"
        assert UpdateType.CHANNEL_RESET.value == "channel_reset"
        assert UpdateType.FULL_RESET.value == "full_reset"

    def test_state_update_repr(self):
        u = StateUpdate(
            update_type=UpdateType.PARAM_SET,
            channel_id=1,
            param="fdr",
            value=-5.0,
        )
        r = repr(u)
        assert "PARAM_SET" in r or "param_set" in r
        assert "fdr" in r


# ---------------------------------------------------------------------------
# ThreadSafeMixerState synchronous API tests
# ---------------------------------------------------------------------------

class TestThreadSafeMixerStateSyncAPI:

    def test_set_and_get_param(self, state):
        state.set_channel_param(1, "fdr", -5.0)
        assert state.get_channel_param(1, "fdr") == -5.0

    def test_get_nonexistent_param_returns_default(self, state):
        result = state.get_channel_param(99, "fdr", default=-144.0)
        assert result == -144.0

    def test_get_channel_returns_dict(self, state):
        state.set_channel_param(1, "fdr", -5.0)
        state.set_channel_param(1, "mute", 0)
        ch = state.get_channel(1)
        assert isinstance(ch, dict)
        assert ch["fdr"] == -5.0
        assert ch["mute"] == 0

    def test_get_channel_empty(self, state):
        ch = state.get_channel(42)
        assert ch == {}

    def test_get_snapshot_deep_copy(self, state):
        """get_snapshot should return a deep copy."""
        state.set_channel_param(1, "fdr", -5.0)
        snap = state.get_snapshot()
        snap[1]["fdr"] = 999.0
        assert state.get_channel_param(1, "fdr") == -5.0

    def test_batch_update(self, state):
        state.batch_update(1, {"fdr": -5.0, "mute": 1, "pan": 25.0})
        assert state.get_channel_param(1, "fdr") == -5.0
        assert state.get_channel_param(1, "mute") == 1
        assert state.get_channel_param(1, "pan") == 25.0

    def test_reset_channel(self, state):
        state.set_channel_param(1, "fdr", -5.0)
        state.reset_channel(1)
        assert state.get_channel(1) == {}

    def test_reset_all(self, state):
        state.set_channel_param(1, "fdr", -5.0)
        state.set_channel_param(2, "fdr", -10.0)
        state.reset_all()
        assert state.channel_count() == 0

    def test_channel_ids(self, state):
        state.set_channel_param(3, "fdr", 0.0)
        state.set_channel_param(1, "fdr", 0.0)
        state.set_channel_param(5, "fdr", 0.0)
        ids = state.channel_ids()
        assert ids == [1, 3, 5]

    def test_channel_count(self, state):
        assert state.channel_count() == 0
        state.set_channel_param(1, "fdr", 0.0)
        state.set_channel_param(2, "fdr", 0.0)
        assert state.channel_count() == 2


# ---------------------------------------------------------------------------
# ThreadSafeMixerState async API tests
# ---------------------------------------------------------------------------

class TestThreadSafeMixerStateAsyncAPI:

    @pytest.mark.asyncio
    async def test_async_set_and_get(self, state):
        await state.async_set_channel_param(1, "fdr", -8.0)
        val = await state.async_get_channel_param(1, "fdr")
        assert val == -8.0

    @pytest.mark.asyncio
    async def test_async_get_default(self, state):
        val = await state.async_get_channel_param(99, "fdr", default=-144.0)
        assert val == -144.0

    @pytest.mark.asyncio
    async def test_async_get_snapshot(self, state):
        state.set_channel_param(1, "fdr", -5.0)
        snap = await state.async_get_snapshot()
        assert 1 in snap
        assert snap[1]["fdr"] == -5.0

    @pytest.mark.asyncio
    async def test_async_batch_update(self, state):
        await state.async_batch_update(1, {"fdr": -3.0, "mute": 0})
        assert state.get_channel_param(1, "fdr") == -3.0
        assert state.get_channel_param(1, "mute") == 0


# ---------------------------------------------------------------------------
# Queue integration tests
# ---------------------------------------------------------------------------

class TestQueueIntegration:

    def test_attach_detach_queue(self, state, queue):
        state.attach_queue(queue)
        assert state._queue is queue
        state.detach_queue()
        assert state._queue is None

    def test_set_param_publishes_to_queue(self, state_with_queue):
        s, q = state_with_queue
        sub_id = q.subscribe()
        s.set_channel_param(1, "fdr", -5.0)
        # The update should be in the subscriber's queue
        with threading.Lock():
            sub = q._subscribers.get(sub_id)
        assert sub is not None
        assert not sub.queue.empty()

    def test_batch_update_publishes_to_queue(self, state_with_queue):
        s, q = state_with_queue
        sub_id = q.subscribe()
        s.batch_update(1, {"fdr": -5.0, "mute": 0})
        sub = q._subscribers.get(sub_id)
        assert sub is not None
        assert not sub.queue.empty()

    def test_reset_channel_publishes_to_queue(self, state_with_queue):
        s, q = state_with_queue
        sub_id = q.subscribe()
        s.set_channel_param(1, "fdr", 0.0)
        # Drain existing
        while not q._subscribers[sub_id].queue.empty():
            q._subscribers[sub_id].queue.get_nowait()
        s.reset_channel(1)
        assert not q._subscribers[sub_id].queue.empty()


# ---------------------------------------------------------------------------
# StateUpdateQueue tests
# ---------------------------------------------------------------------------

class TestStateUpdateQueue:

    def test_subscribe_returns_id(self, queue):
        sub_id = queue.subscribe()
        assert isinstance(sub_id, int)

    def test_subscribe_increments_id(self, queue):
        id1 = queue.subscribe()
        id2 = queue.subscribe()
        assert id2 > id1

    def test_unsubscribe(self, queue):
        sub_id = queue.subscribe()
        assert queue.unsubscribe(sub_id) is True
        assert queue.unsubscribe(sub_id) is False

    def test_subscriber_count(self, queue):
        assert queue.subscriber_count() == 0
        id1 = queue.subscribe()
        assert queue.subscriber_count() == 1
        queue.unsubscribe(id1)
        assert queue.subscriber_count() == 0

    def test_publish_delivers_to_subscribers(self, queue):
        sub_id = queue.subscribe()
        update = StateUpdate(
            update_type=UpdateType.PARAM_SET,
            channel_id=1,
            param="fdr",
            value=-5.0,
        )
        delivered = queue.publish(update)
        assert delivered == 1

    def test_publish_with_channel_filter(self, queue):
        sub_ch1 = queue.subscribe(channels={1})
        sub_ch2 = queue.subscribe(channels={2})
        update = StateUpdate(
            update_type=UpdateType.PARAM_SET,
            channel_id=1,
            param="fdr",
            value=-5.0,
        )
        delivered = queue.publish(update)
        # Only sub_ch1 should get it
        assert delivered == 1
        assert not queue._subscribers[sub_ch1].queue.empty()
        assert queue._subscribers[sub_ch2].queue.empty()

    def test_publish_with_type_filter(self, queue):
        sub = queue.subscribe(update_types={UpdateType.BATCH})
        update = StateUpdate(
            update_type=UpdateType.PARAM_SET,
            channel_id=1,
            param="fdr",
            value=-5.0,
        )
        delivered = queue.publish(update)
        assert delivered == 0

    def test_full_reset_always_delivered(self, queue):
        """FULL_RESET should be delivered regardless of channel filter."""
        sub_id = queue.subscribe(channels={1})
        update = StateUpdate(update_type=UpdateType.FULL_RESET)
        delivered = queue.publish(update)
        assert delivered == 1

    def test_update_subscription(self, queue):
        sub_id = queue.subscribe(channels={1})
        result = queue.update_subscription(sub_id, channels={1, 2})
        assert result is True
        result = queue.update_subscription(999, channels={1})
        assert result is False

    def test_subscriber_info(self, queue):
        queue.subscribe(channels={1, 2})
        info = queue.subscriber_info()
        assert len(info) == 1
        assert info[0]["channels"] == [1, 2]

    @pytest.mark.asyncio
    async def test_get_update(self, queue):
        sub_id = queue.subscribe()
        update = StateUpdate(
            update_type=UpdateType.PARAM_SET,
            channel_id=1,
            param="fdr",
            value=-5.0,
        )
        queue.publish(update)
        result = await queue.get_update(sub_id, timeout=1.0)
        assert result is not None
        assert result.value == -5.0

    @pytest.mark.asyncio
    async def test_get_update_timeout(self, queue):
        sub_id = queue.subscribe()
        result = await queue.get_update(sub_id, timeout=0.01)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_update_unknown_subscriber(self, queue):
        with pytest.raises(ValueError):
            await queue.get_update(999, timeout=0.01)

    @pytest.mark.asyncio
    async def test_drain(self, queue):
        sub_id = queue.subscribe()
        for i in range(5):
            queue.publish(StateUpdate(
                update_type=UpdateType.PARAM_SET,
                channel_id=1,
                param="fdr",
                value=float(i),
            ))
        updates = await queue.drain(sub_id)
        assert len(updates) == 5

    @pytest.mark.asyncio
    async def test_drain_empty(self, queue):
        sub_id = queue.subscribe()
        updates = await queue.drain(sub_id)
        assert updates == []

    @pytest.mark.asyncio
    async def test_drain_unknown_subscriber(self, queue):
        updates = await queue.drain(999)
        assert updates == []


# ---------------------------------------------------------------------------
# _Subscriber.matches tests
# ---------------------------------------------------------------------------

class TestSubscriberMatches:

    def test_matches_all(self):
        sub = _Subscriber(sub_id=0, channels=None, update_types=None,
                          queue=asyncio.Queue())
        update = StateUpdate(update_type=UpdateType.PARAM_SET, channel_id=1)
        assert sub.matches(update) is True

    def test_matches_channel_filter(self):
        sub = _Subscriber(sub_id=0, channels={1, 2}, update_types=None,
                          queue=asyncio.Queue())
        assert sub.matches(StateUpdate(update_type=UpdateType.PARAM_SET, channel_id=1)) is True
        assert sub.matches(StateUpdate(update_type=UpdateType.PARAM_SET, channel_id=3)) is False

    def test_matches_type_filter(self):
        sub = _Subscriber(sub_id=0, channels=None,
                          update_types={UpdateType.BATCH},
                          queue=asyncio.Queue())
        assert sub.matches(StateUpdate(update_type=UpdateType.BATCH, channel_id=1)) is True
        assert sub.matches(StateUpdate(update_type=UpdateType.PARAM_SET, channel_id=1)) is False

    def test_full_reset_always_matches(self):
        sub = _Subscriber(sub_id=0, channels={1},
                          update_types={UpdateType.PARAM_SET},
                          queue=asyncio.Queue())
        assert sub.matches(StateUpdate(update_type=UpdateType.FULL_RESET)) is True


# ---------------------------------------------------------------------------
# Concurrent access test
# ---------------------------------------------------------------------------

class TestConcurrentAccess:

    def test_concurrent_writes_no_corruption(self, state):
        """Multiple threads writing should not corrupt state."""
        errors = []

        def writer(ch, n):
            try:
                for i in range(n):
                    state.set_channel_param(ch, "fdr", float(i))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(ch, 100))
            for ch in range(1, 5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert state.channel_count() == 4
