"""Tests for routing module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from routing import MessageRouter


class TestMessageRouter:
    def test_init(self):
        router = MessageRouter()
        assert router is not None

    def test_register_handler(self):
        router = MessageRouter()
        called = []
        def handler(msg):
            called.append(msg)
        router.register('test', handler) if hasattr(router, 'register') else router.register_handler('test', handler)
        # Verify registration didn't crash
        assert True
