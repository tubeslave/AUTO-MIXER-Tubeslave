"""Tests for osc_manager module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from osc_manager import OSCManager


class TestOSCManager:
    def test_init(self):
        manager = OSCManager()
        assert manager is not None

    def test_throttle_setting(self):
        manager = OSCManager(throttle_hz=20.0)
        assert manager.throttle_hz == 20.0
