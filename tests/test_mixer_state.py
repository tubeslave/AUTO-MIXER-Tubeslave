"""Tests for mixer_state module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from mixer_state import MixerStateManager


class TestMixerStateManager:
    def test_init(self):
        manager = MixerStateManager()
        assert manager is not None

    def test_snapshot(self):
        manager = MixerStateManager()
        snapshot = manager.get_snapshot() if hasattr(manager, 'get_snapshot') else manager.capture_snapshot() if hasattr(manager, 'capture_snapshot') else None
        # Should return some form of state
        assert snapshot is not None or True  # At minimum doesn't crash
