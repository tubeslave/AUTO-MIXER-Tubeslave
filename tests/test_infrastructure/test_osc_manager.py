"""
Tests for backend/osc_manager.py — OSCSubscription pattern matching,
OSCManager initialization, stats, and subscription management.

All tests work without hardware or network (no actual connections).
"""

import pytest

try:
    from osc_manager import OSCSubscription, OSCManager
except ImportError:
    pytest.skip("osc_manager module not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manager():
    """An OSCManager instance (not connected, no network)."""
    return OSCManager(ip="127.0.0.1", port=2223, rate_limit_hz=50.0)


# ---------------------------------------------------------------------------
# OSCSubscription pattern matching tests
# ---------------------------------------------------------------------------

class TestOSCSubscription:

    def test_exact_match(self):
        sub = OSCSubscription("/ch/1/fdr", lambda *a: None)
        assert sub.matches("/ch/1/fdr") is True

    def test_exact_no_match(self):
        sub = OSCSubscription("/ch/1/fdr", lambda *a: None)
        assert sub.matches("/ch/2/fdr") is False

    def test_wildcard_star(self):
        sub = OSCSubscription("/ch/*/fdr", lambda *a: None)
        assert sub.matches("/ch/1/fdr") is True
        assert sub.matches("/ch/25/fdr") is True
        assert sub.matches("/ch/1/mute") is False

    def test_wildcard_question_mark(self):
        sub = OSCSubscription("/ch/?/fdr", lambda *a: None)
        assert sub.matches("/ch/1/fdr") is True
        assert sub.matches("/ch/10/fdr") is False

    def test_complex_pattern(self):
        sub = OSCSubscription("/ch/*/eq/*", lambda *a: None)
        assert sub.matches("/ch/1/eq/1g") is True
        assert sub.matches("/ch/5/eq/on") is True
        assert sub.matches("/ch/1/fdr") is False

    def test_pattern_stored(self):
        cb = lambda *a: None
        sub = OSCSubscription("/ch/*/fdr", cb)
        assert sub.pattern == "/ch/*/fdr"
        assert sub.callback is cb


# ---------------------------------------------------------------------------
# OSCManager initialization tests
# ---------------------------------------------------------------------------

class TestOSCManagerInit:

    def test_default_values(self, manager):
        assert manager.ip == "127.0.0.1"
        assert manager.port == 2223
        assert manager.rate_limit_hz == 50.0
        assert manager.is_connected is False

    def test_custom_values(self):
        m = OSCManager(ip="10.0.0.1", port=9000, rate_limit_hz=100.0,
                       health_timeout_sec=30.0)
        assert m.ip == "10.0.0.1"
        assert m.port == 9000
        assert m.rate_limit_hz == 100.0
        assert m.health_timeout_sec == 30.0

    def test_repr(self, manager):
        r = repr(manager)
        assert "127.0.0.1" in r
        assert "2223" in r
        assert "disconnected" in r


# ---------------------------------------------------------------------------
# OSCManager stats tests
# ---------------------------------------------------------------------------

class TestOSCManagerStats:

    def test_get_stats_disconnected(self, manager):
        stats = manager.get_stats()
        assert stats["connected"] is False
        assert stats["messages_sent"] == 0
        assert stats["messages_received"] == 0
        assert stats["subscriptions"] == 0

    def test_stats_include_queue_size(self, manager):
        stats = manager.get_stats()
        assert "queue_size" in stats
        assert stats["queue_size"] == 0


# ---------------------------------------------------------------------------
# OSCManager subscription management tests
# ---------------------------------------------------------------------------

class TestOSCManagerSubscriptions:

    def test_subscribe(self, manager):
        received = []

        def cb(addr, *args):
            received.append((addr, args))

        manager.subscribe("/ch/*/fdr", cb)
        assert len(manager._subscriptions) == 1

    def test_unsubscribe(self, manager):
        def cb(addr, *args):
            pass

        manager.subscribe("/ch/*/fdr", cb)
        assert manager.unsubscribe("/ch/*/fdr", cb) is True
        assert len(manager._subscriptions) == 0

    def test_unsubscribe_wrong_callback(self, manager):
        def cb1(addr, *args):
            pass

        def cb2(addr, *args):
            pass

        manager.subscribe("/ch/*/fdr", cb1)
        assert manager.unsubscribe("/ch/*/fdr", cb2) is False
        assert len(manager._subscriptions) == 1

    def test_unsubscribe_wrong_pattern(self, manager):
        def cb(addr, *args):
            pass

        manager.subscribe("/ch/*/fdr", cb)
        assert manager.unsubscribe("/ch/1/fdr", cb) is False

    def test_multiple_subscriptions(self, manager):
        def cb1(*a): pass
        def cb2(*a): pass

        manager.subscribe("/ch/*/fdr", cb1)
        manager.subscribe("/ch/*/mute", cb2)
        assert len(manager._subscriptions) == 2


# ---------------------------------------------------------------------------
# OSCManager global listener tests
# ---------------------------------------------------------------------------

class TestOSCManagerGlobalListeners:

    def test_add_global_listener(self, manager):
        def listener(*a): pass
        manager.add_global_listener(listener)
        assert listener in manager._global_listeners

    def test_remove_global_listener(self, manager):
        def listener(*a): pass
        manager.add_global_listener(listener)
        assert manager.remove_global_listener(listener) is True
        assert listener not in manager._global_listeners

    def test_remove_nonexistent_listener(self, manager):
        def listener(*a): pass
        assert manager.remove_global_listener(listener) is False


# ---------------------------------------------------------------------------
# OSCManager send while disconnected
# ---------------------------------------------------------------------------

class TestOSCManagerSendDisconnected:

    def test_send_returns_false(self, manager):
        assert manager.send("/ch/1/fdr", -5.0) is False

    def test_query_returns_false(self, manager):
        assert manager.query("/ch/1/fdr") is False

    def test_send_batch_returns_zero(self, manager):
        messages = [("/ch/1/fdr", (-5.0,)), ("/ch/2/fdr", (0.0,))]
        assert manager.send_batch(messages) == 0


# ---------------------------------------------------------------------------
# OSCManager health callbacks
# ---------------------------------------------------------------------------

class TestOSCManagerHealthCallbacks:

    def test_on_connected_callback_stored(self, manager):
        def cb():
            pass
        manager.on_connected(cb)
        assert manager._on_connected is cb

    def test_on_disconnected_callback_stored(self, manager):
        def cb():
            pass
        manager.on_disconnected(cb)
        assert manager._on_disconnected is cb


# ---------------------------------------------------------------------------
# OSCManager dispatch tests
# ---------------------------------------------------------------------------

class TestOSCManagerDispatch:

    def test_dispatch_to_subscriber(self, manager):
        received = []

        def cb(addr, *args):
            received.append((addr, args))

        manager.subscribe("/ch/1/fdr", cb)
        manager._dispatch("/ch/1/fdr", -5.0)
        assert len(received) == 1
        assert received[0] == ("/ch/1/fdr", (-5.0,))

    def test_dispatch_wildcard(self, manager):
        received = []

        def cb(addr, *args):
            received.append(addr)

        manager.subscribe("/ch/*/fdr", cb)
        manager._dispatch("/ch/1/fdr", -5.0)
        manager._dispatch("/ch/2/fdr", 0.0)
        manager._dispatch("/ch/1/mute", 1)  # should NOT match
        assert len(received) == 2

    def test_dispatch_to_global_listener(self, manager):
        received = []

        def listener(addr, *args):
            received.append(addr)

        manager.add_global_listener(listener)
        manager._dispatch("/anything", 42)
        assert len(received) == 1
        assert received[0] == "/anything"

    def test_dispatch_callback_error_does_not_crash(self, manager):
        """A callback error should not prevent other callbacks from running."""
        received = []

        def bad_cb(addr, *args):
            raise ValueError("boom")

        def good_cb(addr, *args):
            received.append(addr)

        manager.subscribe("/ch/*/fdr", bad_cb)
        manager.add_global_listener(good_cb)
        # Should not raise
        manager._dispatch("/ch/1/fdr", -5.0)
        assert len(received) == 1
