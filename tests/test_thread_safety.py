"""Tests for thread_safety module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
import asyncio
from thread_safety import ThreadSafeMixerState, MixerStateQueue


class TestThreadSafeMixerState:
    def test_init(self):
        state = ThreadSafeMixerState()
        assert state is not None

    @pytest.mark.asyncio
    async def test_set_and_get_channel_param(self):
        state = ThreadSafeMixerState()
        await state.async_set_channel_param(1, 'gain', -6.0)
        val = await state.async_get_channel_param(1, 'gain')
        assert val == -6.0

    @pytest.mark.asyncio
    async def test_snapshot(self):
        state = ThreadSafeMixerState()
        await state.async_set_channel_param(1, 'fader', 0.75)
        snap1 = await state.snapshot()
        snap2 = await state.snapshot()
        # Should be different objects (copy-on-read)
        assert snap1 is not snap2

    @pytest.mark.asyncio
    async def test_version_increments(self):
        state = ThreadSafeMixerState()
        v1 = await state.get_version()
        await state.async_set_channel_param(1, 'gain', -3.0)
        v2 = await state.get_version()
        assert v2 > v1


class TestMixerStateQueue:
    @pytest.mark.asyncio
    async def test_put_get(self):
        queue = MixerStateQueue(maxsize=10)
        await queue.put('test_event', {'test': 'data'})
        item = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert item['type'] == 'test_event'
        assert item['data'] == {'test': 'data'}

    @pytest.mark.asyncio
    async def test_backpressure(self):
        queue = MixerStateQueue(maxsize=2)
        await queue.put('evt_a', {'a': 1})
        await queue.put('evt_b', {'b': 2})
        item1 = await asyncio.wait_for(queue.get(), timeout=1.0)
        item2 = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert item1['type'] == 'evt_a'
        assert item2['type'] == 'evt_b'
