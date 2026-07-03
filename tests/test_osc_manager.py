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

    def test_shadow_mode_blocks_mutations_but_not_queries(self):
        class FakeWingClient:
            def __init__(self):
                self.is_connected = True
                self.state = {}
                self.shadow_enabled = False
                self.calls = []

            def configure_shadow_transport(self, *, enabled=None, event_sink=None):
                if enabled is not None:
                    self.shadow_enabled = bool(enabled)

            def is_shadow_mode_enabled(self):
                return self.shadow_enabled

            def send(self, address, *values):
                self.calls.append((address, values))
                if self.shadow_enabled and values:
                    return False
                return True

        manager = OSCManager(shadow_mode_enabled=True)
        wing = FakeWingClient()

        assert manager.connect(wing) is True
        assert manager.is_shadow_mode_enabled() is True
        assert manager.send("/ch/1/fdr", -6.0) is False
        assert manager.query("/ch/1/fdr") is None
        assert wing.calls == [
            ("/ch/1/fdr", (-6.0,)),
            ("/ch/1/fdr", ()),
        ]
